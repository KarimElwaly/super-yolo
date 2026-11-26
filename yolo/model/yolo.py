from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import ListConfig, OmegaConf
from torch import nn

from yolo.config.config import ModelConfig, YOLOLayer
from yolo.tools.dataset_preparation import prepare_weight
from yolo.utils.logger import logger
from yolo.utils.module_utils import get_layer_map

ACTIVATION_LAYER_TYPES = {
    "Conv",
    "ELAN",
    "RepConv",
    "Bottleneck",
    "RepNCSP",
    "RepNCSPELAN",
    "AConv",
    "ADown",
    "SPPELAN",
    "SPPCSPConv",
    "Detection",
    "MultiheadDetection",
    "Segmentation",
    "MultiheadSegmentation",
    "Classification",
    "DConv",
    "RepNCSPELAND",
}


class YOLO(nn.Module):
    """
    A preliminary YOLO (You Only Look Once) model class still under development.

    Parameters:
        model_cfg: Configuration for the YOLO model. Expected to define the layers,
                   parameters, and any other relevant configuration details.
    """

    def __init__(self, model_cfg: ModelConfig, class_num: int = 80):
        super(YOLO, self).__init__()
        self.num_classes = class_num
        self.layer_map = get_layer_map()  # Get the map Dict[str: Module]
        self.model: List[YOLOLayer] = nn.ModuleList()
        self.reg_max = getattr(model_cfg.anchor, "reg_max", 16)
        self.default_activation = getattr(model_cfg, "activation", None)
        self.build_model(model_cfg.model)

    def build_model(self, model_arch: Dict[str, List[Dict[str, Dict[str, Dict]]]]):
        self.layer_index = {}
        output_dim, layer_idx = [3], 1
        logger.info(f":tractor: Building YOLO")
        for arch_name in model_arch:
            if model_arch[arch_name]:
                logger.info(f"  :building_construction:  Building {arch_name}")
            for layer_idx, layer_spec in enumerate(model_arch[arch_name], start=layer_idx):
                layer_type, layer_info = next(iter(layer_spec.items()))
                layer_args = dict(layer_info.get("args", {}) or {})

                if (
                    self.default_activation is not None
                    and layer_type in ACTIVATION_LAYER_TYPES
                    and "activation" not in layer_args
                ):
                    layer_args["activation"] = self.default_activation

                # Get input source
                source = self.get_source_idx(layer_info.get("source", -1), layer_idx)

                # Find in channels
                if any(module in layer_type for module in ["Conv", "ELAN", "ADown", "AConv", "CBLinear"]):
                    layer_args["in_channels"] = output_dim[source]
                if any(module in layer_type for module in ["Detection", "Segmentation", "Classification"]):
                    if isinstance(source, list):
                        layer_args["in_channels"] = [output_dim[idx] for idx in source]
                    else:
                        layer_args["in_channel"] = output_dim[source]
                    layer_args["num_classes"] = self.num_classes
                    layer_args["reg_max"] = self.reg_max

                # create layers
                layer = self.create_layer(layer_type, source, layer_info, arch_section=arch_name, **layer_args)
                self.model.append(layer)

                if layer.tags:
                    if layer.tags in self.layer_index:
                        raise ValueError(f"Duplicate tag '{layer_info['tags']}' found.")
                    self.layer_index[layer.tags] = layer_idx

                out_channels = self.get_out_channels(layer_type, layer_args, output_dim, source)
                output_dim.append(out_channels)
                setattr(layer, "out_c", out_channels)
            layer_idx += 1

    def forward(self, x, external: Optional[Dict] = None, shortcut: Optional[str] = None):
        y = {0: x, **(external or {})}
        output = dict()
        for index, layer in enumerate(self.model, start=1):
            if isinstance(layer.source, list):
                model_input = [y[idx] for idx in layer.source]
            else:
                model_input = y[layer.source]

            external_input = {source_name: y[source_name] for source_name in layer.external}

            x = layer(model_input, **external_input)
            y[-1] = x
            if layer.usable:
                y[index] = x
            if layer.output:
                output[layer.tags] = x
                if layer.tags == shortcut:
                    return output
        return output

    def get_out_channels(self, layer_type: str, layer_args: dict, output_dim: list, source: Union[int, list]):
        if "out_channels" in layer_args:
            return layer_args["out_channels"]
        if layer_type == "CBFuse":
            return output_dim[source[-1]]
        if isinstance(source, int):
            return output_dim[source]
        if isinstance(source, list):
            return sum(output_dim[idx] for idx in source)

    def get_source_idx(self, source: Union[ListConfig, str, int], layer_idx: int):
        if isinstance(source, ListConfig):
            return [self.get_source_idx(index, layer_idx) for index in source]
        if isinstance(source, str):
            source = self.layer_index[source]
        if source < -1:
            source += layer_idx
        if source > 0:  # Using Previous Layer's Output
            self.model[source - 1].usable = True
        return source

    def create_layer(
        self,
        layer_type: str,
        source: Union[int, list],
        layer_info: Dict,
        arch_section: Optional[str] = None,
        **kwargs,
    ) -> YOLOLayer:
        if layer_type in self.layer_map:
            layer = self.layer_map[layer_type](**kwargs)
            setattr(layer, "layer_type", layer_type)
            setattr(layer, "source", source)
            setattr(layer, "in_c", kwargs.get("in_channels", None))
            setattr(layer, "output", layer_info.get("output", False))
            setattr(layer, "tags", layer_info.get("tags", None))
            setattr(layer, "external", layer_info.get("external", []))
            setattr(layer, "usable", 0)
            setattr(layer, "arch_section", arch_section)
            return layer
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def strip_auxiliary_branches(self) -> Dict[str, int]:
        """Remove training-only auxiliary (PGI) layers so exported graphs contain zero aux ops."""
        aux_indices = {
            idx
            for idx, layer in enumerate(self.model, start=1)
            if getattr(layer, "arch_section", None) == "auxiliary" or getattr(layer, "tags", None) == "AUX"
        }
        if not aux_indices:
            return {"removed_layers": 0, "removed_params": 0, "remaining_layers": len(self.model)}

        for idx, layer in enumerate(self.model, start=1):
            if idx in aux_indices:
                continue
            sources = layer.source if isinstance(layer.source, list) else [layer.source]
            for src in sources:
                if src in aux_indices:
                    raise RuntimeError(f"Retained layer {idx} ({layer.tags}) depends on auxiliary layer {src}")

        kept_layers: List[nn.Module] = []
        index_remap: Dict[int, int] = {0: 0, -1: -1}
        removed_params = 0
        for old_idx, layer in enumerate(self.model, start=1):
            if old_idx in aux_indices:
                removed_params += sum(p.numel() for p in layer.parameters())
                continue
            new_idx = len(kept_layers) + 1
            index_remap[old_idx] = new_idx
            kept_layers.append(layer)

        new_layer_index: Dict[str, int] = {}
        for new_idx, layer in enumerate(kept_layers, start=1):
            if isinstance(layer.source, list):
                layer.source = [index_remap[s] for s in layer.source]
            else:
                layer.source = index_remap[layer.source]
            if getattr(layer, "tags", None):
                new_layer_index[layer.tags] = new_idx

        self.model = nn.ModuleList(kept_layers)
        self.layer_index = new_layer_index
        removed_count = len(aux_indices)
        logger.info(
            f":scissors: Stripped {removed_count} auxiliary layer(s) ({removed_params:,} params); "
            f"{len(kept_layers)} inference layer(s) remain"
        )
        return {
            "removed_layers": removed_count,
            "removed_params": removed_params,
            "remaining_layers": len(kept_layers),
        }

    def load_partial_weights(self, weights: Union[str, Path, OrderedDict], strict: bool = False) -> Dict[str, Any]:
        """Load shape-matched layers from a checkpoint into this model (strict=False)."""
        return load_partial_weights(self, weights, strict=strict)

    def _maybe_adapt_head_class_neck(self, weights: Dict[str, torch.Tensor]) -> None:
        """Align Detection.class_conv neck width when loading official v1.0-alpha checkpoints (e.g. v9-t.pt class_neck=80)."""
        from yolo.model.module import Conv, MultiheadDetection

        adapted = False
        for layer_idx, layer in enumerate(self.model):
            if not isinstance(layer, MultiheadDetection):
                continue
            for head_idx, head in enumerate(layer.heads):
                c0_key = f"{layer_idx}.heads.{head_idx}.class_conv.0.conv.weight"
                c2_key = f"{layer_idx}.heads.{head_idx}.class_conv.2.weight"
                if c0_key not in weights or c2_key not in weights:
                    continue
                ckpt_neck, ckpt_in_c = int(weights[c0_key].shape[0]), int(weights[c0_key].shape[1])
                ckpt_cls = int(weights[c2_key].shape[0])
                cur_in_c = int(head.class_conv[0].conv.in_channels)
                cur_neck = int(head.class_conv[0].conv.out_channels)
                cur_cls = int(head.class_conv[2].out_channels)
                if ckpt_cls == cur_cls and ckpt_in_c == cur_in_c and ckpt_neck != cur_neck:
                    act_name = getattr(head, "activation_name", "SiLU")
                    head.class_conv = nn.Sequential(
                        Conv(cur_in_c, ckpt_neck, 3, activation=act_name),
                        Conv(ckpt_neck, ckpt_neck, 3, activation=act_name),
                        nn.Conv2d(ckpt_neck, cur_cls, 1),
                    )
                    adapted = True
        if adapted:
            logger.info(":wrench: Adapted Detection class_conv neck width to match checkpoint head dimensions")

    def save_load_weights(self, weights: Union[Path, OrderedDict]):
        """
        Update the model's weights with the provided weights.

        args:
            weights: A OrderedDict containing the new weights.
        """
        if isinstance(weights, Path):
            weights = torch.load(weights, map_location=torch.device("cpu"), weights_only=False)
        if "state_dict" in weights:
            weights = {name.removeprefix("model.model."): key for name, key in weights["state_dict"].items()}
        self._maybe_adapt_head_class_neck(weights)
        model_state_dict = self.model.state_dict()

        # TODO1: autoload old version weight
        # TODO2: weight transform if num_class difference

        error_dict = {"Mismatch": set(), "Not Found": set()}
        for model_key, model_weight in model_state_dict.items():
            if model_key not in weights:
                error_dict["Not Found"].add(tuple(model_key.split(".")[:-2]))
                continue
            if model_weight.shape != weights[model_key].shape:
                error_dict["Mismatch"].add(tuple(model_key.split(".")[:-2]))
                continue
            model_state_dict[model_key] = weights[model_key]

        for error_name, error_set in error_dict.items():
            error_dict = dict()
            for layer_idx, *layer_name in error_set:
                if layer_idx not in error_dict:
                    error_dict[layer_idx] = [".".join(layer_name)]
                else:
                    error_dict[layer_idx].append(".".join(layer_name))
            for layer_idx, layer_name in error_dict.items():
                layer_name.sort()
                logger.warning(f":warning: Weight {error_name} for Layer {layer_idx}: {', '.join(layer_name)}")

        self.model.load_state_dict(model_state_dict)


def _extract_state_dict(weights: Union[str, Path, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if isinstance(weights, (str, Path)):
        raw = torch.load(Path(weights), map_location=torch.device("cpu"), weights_only=False)
    else:
        raw = weights
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]

    normalized: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if not isinstance(v, torch.Tensor):
            continue
        clean_k = k
        for prefix in ("model.model.", "ema.model.", "model."):
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix) :]
                break
        normalized[clean_k] = v
    return normalized


def load_partial_weights(
    model: YOLO,
    weights: Union[str, Path, Dict[str, Any]],
    strict: bool = False,
) -> Dict[str, Any]:
    """Load shape-matched layers from a pretrained checkpoint into an upgraded YOLO model."""
    src_state = _extract_state_dict(weights)
    target_state = model.model.state_dict()

    matched: Dict[str, torch.Tensor] = {}
    mismatched: List[str] = []
    missing_only: List[str] = []

    for key, target_tensor in target_state.items():
        if key not in src_state:
            missing_only.append(key)
            continue
        if src_state[key].shape != target_tensor.shape:
            mismatched.append(key)
            continue
        matched[key] = src_state[key].to(dtype=target_tensor.dtype)

    model.model.load_state_dict(matched, strict=strict)

    reinitialized = mismatched + missing_only
    logger.info(
        f":package: Partial weight transfer: {len(matched)}/{len(target_state)} tensors loaded, "
        f"{len(mismatched)} shape-mismatched, {len(missing_only)} newly initialized "
        f"({len(reinitialized)} total re-initialized)"
    )
    return {
        "loaded": len(matched),
        "skipped_mismatch": len(mismatched),
        "missing": len(missing_only),
        "reinitialized": len(reinitialized),
        "total_target": len(target_state),
        "mismatch_keys": mismatched,
        "reinitialized_keys": reinitialized,
    }


def strip_auxiliary_branches(model: YOLO) -> Dict[str, int]:
    """Module-level wrapper around `YOLO.strip_auxiliary_branches()`."""
    return model.strip_auxiliary_branches()


def create_model(
    model_cfg: ModelConfig,
    weight_path: Union[bool, str, Path] = True,
    class_num: int = 80,
    partial_load: bool = False,
) -> YOLO:
    """Constructs and returns a model from a Dictionary configuration file.

    Args:
        config_file (dict): The configuration file of the model.

    Returns:
        YOLO: An instance of the model defined by the given configuration.
    """
    OmegaConf.set_struct(model_cfg, False)
    model = YOLO(model_cfg, class_num)
    if weight_path:
        if weight_path == True:
            weight_path = Path("weights") / f"{model_cfg.name}.pt"
        elif isinstance(weight_path, str):
            weight_path = Path(weight_path)

        if not weight_path.exists():
            logger.info(f"🌐 Weight {weight_path} not found, try downloading")
            prepare_weight(weight_path=weight_path)
        if weight_path.exists():
            if partial_load:
                load_partial_weights(model, weight_path, strict=False)
            else:
                model.save_load_weights(weight_path)
            logger.info(":white_check_mark: Success load model & weight")
    else:
        logger.info(":white_check_mark: Success load model")
    return model
