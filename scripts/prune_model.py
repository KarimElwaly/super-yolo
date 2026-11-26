"""Standalone structured channel pruning CLI for YOLOv9 models using MIT-licensed torch-pruning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set

import torch
import torch_pruning as tp
from omegaconf import OmegaConf
from torch import nn

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from yolo.model.module import (
    ADown,
    Anchor2Vec,
    CBLinear,
    DConv,
    Detection,
    ELAN,
    IDetection,
    MultiheadDetection,
    RepNCSPELAN,
)
from yolo.model.yolo import YOLO, create_model


class _PruneTraceWrapper(nn.Module):
    """Flattens YOLO dict outputs into a tuple of tensors for torch-pruning graph tracing."""

    def __init__(self, yolo_model: YOLO):
        super().__init__()
        self.yolo_model = yolo_model

    def forward(self, x: torch.Tensor):
        out = self.yolo_model(x)
        tensors: List[torch.Tensor] = []
        for branch_key in ("Main", "AUX"):
            if branch_key not in out:
                continue
            for head_out in out[branch_key]:
                if isinstance(head_out, (tuple, list)):
                    tensors.extend(head_out)
                else:
                    tensors.append(head_out)
        return tuple(tensors)


def _collect_protected_layers(model: YOLO) -> List[nn.Module]:
    """Protect detection heads and internal tensor-chunk/split boundaries from pruning."""
    ignored: List[nn.Module] = []
    seen: Set[int] = set()

    def _add(mod: nn.Module) -> None:
        if id(mod) not in seen:
            ignored.append(mod)
            seen.add(id(mod))
        for sub in mod.modules():
            if id(sub) not in seen:
                ignored.append(sub)
                seen.add(id(sub))

    for mod in model.modules():
        if isinstance(mod, (Detection, IDetection, MultiheadDetection, Anchor2Vec, CBLinear, DConv)):
            _add(mod)
        elif isinstance(mod, (ELAN, RepNCSPELAN)):
            # ELAN/RepNCSPELAN splits `conv1(x).chunk(2, 1)` into concat; keep chunk producer width fixed.
            _add(mod.conv1)
        elif isinstance(mod, ADown):
            _add(mod)
    return ignored


def _refresh_yolo_channel_metadata(model: YOLO) -> None:
    for layer in model.model:
        if hasattr(layer, "conv") and isinstance(layer.conv, nn.Conv2d):
            layer.in_c = layer.conv.in_channels
            layer.out_c = layer.conv.out_channels


def prune_yolo_model(
    model: YOLO,
    *,
    pruning_ratio: float = 0.20,
    importance_type: str = "l1",
    image_size: int = 640,
    round_to: int = 4,
    strip_aux: bool = False,
) -> dict:
    """Apply structured channel pruning in-place on `model` and return parameter/MACs statistics."""
    if not (0.0 < pruning_ratio < 1.0):
        raise ValueError(f"pruning_ratio must be in (0, 1), got {pruning_ratio}")

    if strip_aux:
        model.strip_auxiliary_branches()

    model.eval()
    example_inputs = torch.randn(1, 3, image_size, image_size)
    wrapper = _PruneTraceWrapper(model)

    base_macs, base_params = tp.utils.count_ops_and_params(wrapper, example_inputs)

    if importance_type == "l1":
        importance = tp.importance.MagnitudeImportance(p=1)
    elif importance_type == "taylor":
        importance = tp.importance.GroupTaylorImportance()
    else:
        raise ValueError(f"Unsupported importance type: {importance_type}")

    ignored_layers = _collect_protected_layers(model)

    pruner = tp.pruner.MagnitudePruner(
        wrapper,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=1,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        round_to=round_to,
    )

    if importance_type == "taylor":
        model.train()
        wrapper.zero_grad(set_to_none=True)
        outputs = wrapper(example_inputs)
        loss = sum(t.pow(2).mean() for t in outputs)
        loss.backward()
        model.eval()

    pruner.step()
    wrapper.zero_grad(set_to_none=True)
    _refresh_yolo_channel_metadata(model)

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(wrapper, example_inputs)
    with torch.no_grad():
        test_out = model(example_inputs)
    assert "Main" in test_out and len(test_out["Main"]) > 0

    param_red_pct = (1.0 - pruned_params / max(base_params, 1)) * 100.0
    macs_red_pct = (1.0 - pruned_macs / max(base_macs, 1)) * 100.0

    return {
        "base_params": int(base_params),
        "pruned_params": int(pruned_params),
        "param_reduction_pct": round(param_red_pct, 2),
        "base_macs": int(base_macs),
        "pruned_macs": int(pruned_macs),
        "macs_reduction_pct": round(macs_red_pct, 2),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Structured channel pruning for YOLOv9 using torch-pruning.")
    parser.add_argument("--model", required=True, help="Model variant name (e.g. v9-t-opt) or path to model YAML.")
    parser.add_argument("--weights", type=Path, help="Input checkpoint .pt/.ckpt (never overwritten).")
    parser.add_argument("--output", type=Path, required=True, help="Output path for standalone pruned checkpoint.")
    parser.add_argument("--class-num", type=int, default=80, help="Number of classes (default: 80).")
    parser.add_argument("--pruning-ratio", type=float, default=0.20, help="Channel pruning ratio (default: 0.20).")
    parser.add_argument("--importance", choices=["l1", "taylor"], default="l1", help="Importance criterion.")
    parser.add_argument("--image-size", type=int, default=640, help="Input spatial resolution for tracing.")
    parser.add_argument("--round-to", type=int, default=4, help="Round remaining channels to multiples of N (default: 4).")
    parser.add_argument("--strip-aux", action="store_true", help="Strip auxiliary PGI branch before pruning.")
    args = parser.parse_args(argv)

    if args.weights and args.weights.resolve() == args.output.resolve():
        raise ValueError("Refusing to overwrite input --weights; specify a distinct --output path.")

    model_cfg_path = Path(args.model)
    if not model_cfg_path.is_file():
        model_cfg_path = project_root / "yolo" / "config" / "model" / f"{args.model}.yaml"
    if not model_cfg_path.is_file():
        raise FileNotFoundError(f"Model config YAML not found: {model_cfg_path}")

    model_cfg = OmegaConf.load(model_cfg_path)
    model = create_model(
        model_cfg,
        weight_path=str(args.weights) if args.weights else None,
        class_num=args.class_num,
        partial_load=False,
    )

    stats = prune_yolo_model(
        model,
        pruning_ratio=args.pruning_ratio,
        importance_type=args.importance,
        image_size=args.image_size,
        round_to=args.round_to,
        strip_aux=args.strip_aux,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pruned_model": model.cpu(),
        "state_dict": model.cpu().state_dict(),
        "model_name": model_cfg.name,
        "class_num": args.class_num,
        "pruning_ratio": args.pruning_ratio,
        "importance": args.importance,
        "image_size": args.image_size,
        "stats": stats,
    }
    torch.save(payload, args.output)

    print(
        f"[OK] Pruned {model_cfg.name}: "
        f"Params {stats['base_params']:,} -> {stats['pruned_params']:,} (-{stats['param_reduction_pct']:.2f}%), "
        f"MACs {stats['base_macs']:,} -> {stats['pruned_macs']:,} (-{stats['macs_reduction_pct']:.2f}%)"
    )
    print(f"[OK] Saved standalone pruned checkpoint -> {args.output}")
    print(
        "Recovery fine-tuning command:\n"
        f"  .venv\\Scripts\\python.exe yolo/lazy.py task=train model={model_cfg.name} "
        f"weight={args.output.as_posix()} image_size=[{args.image_size},{args.image_size}] "
        f"task.epoch=50 task.optimizer.args.lr=0.002"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
