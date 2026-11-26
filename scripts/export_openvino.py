"""Export YOLOv9 PyTorch models (vanilla, -opt, -p2, or pruned) to ONNX & OpenVINO FP16/INT8."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import nncf
import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import openvino as ov
import torch
import yaml
from omegaconf import OmegaConf
from torch import nn

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from inference.detector_ov import letterbox_bgr
from yolo.model.yolo import YOLO, create_model
from yolo.utils.bounding_box_utils import generate_anchors


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class ExportYOLOWrapper(nn.Module):
    """Static-shape export wrapper stripping AUX branches and decoding (boxes, scores)."""

    def __init__(self, yolo_model: YOLO, image_size: Tuple[int, int], strides: Sequence[int]):
        super().__init__()
        self.yolo_model = yolo_model
        anchor_grid, scaler = generate_anchors(list(image_size), list(strides))
        self.register_buffer("anchor_grid", anchor_grid.float().unsqueeze(0))  # [1, N, 2]
        self.register_buffer("scaler", scaler.float().view(1, -1, 1))  # [1, N, 1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.yolo_model(x)
        main_heads = out["Main"]
        cls_list: List[torch.Tensor] = []
        vec_list: List[torch.Tensor] = []
        for cls_x, _anc_x, vec_x in main_heads:
            cls_list.append(cls_x.flatten(2).transpose(1, 2))
            vec_list.append(vec_x.flatten(2).transpose(1, 2))
        all_cls = torch.cat(cls_list, dim=1)
        all_vec = torch.cat(vec_list, dim=1)

        pred_ltrb = all_vec * self.scaler
        lt, rb = pred_ltrb.split(2, dim=-1)
        boxes = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)
        scores = torch.sigmoid(all_cls)
        return boxes, scores


def _resolve_model_cfg(model_arg: str):
    p = Path(model_arg)
    if not p.is_file():
        p = project_root / "yolo" / "config" / "model" / f"{model_arg}.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"Model YAML config not found: {model_arg} ({p})")
    return OmegaConf.load(p)


def _resolve_class_info(
    dataset_arg: Optional[str],
    class_num_arg: Optional[int],
    class_names_arg: Optional[str],
    ckpt_class_num: Optional[int] = None,
) -> Tuple[int, List[str]]:
    if dataset_arg:
        ds_path = Path(dataset_arg)
        if not ds_path.is_file():
            ds_path = project_root / "yolo" / "config" / "dataset" / f"{dataset_arg}.yaml"
        if not ds_path.is_file():
            raise FileNotFoundError(f"Dataset YAML not found: {dataset_arg} ({ds_path})")
        with open(ds_path, "r", encoding="utf-8") as fh:
            ds_cfg = yaml.safe_load(fh) or {}
        names = [str(x) for x in ds_cfg.get("class_list", [])]
        cnum = int(ds_cfg.get("class_num", len(names) or 80))
        return cnum, names

    if class_names_arg:
        p = Path(class_names_arg)
        if p.is_file():
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
            names = [str(x) for x in (raw.get("class_list", raw) if isinstance(raw, dict) else raw)]
        else:
            names = [c.strip() for c in class_names_arg.split(",") if c.strip()]
        return int(class_num_arg or len(names)), names

    cnum = int(class_num_arg or ckpt_class_num or 80)
    if cnum == 80:
        coco_yaml = project_root / "yolo" / "config" / "dataset" / "coco.yaml"
        if coco_yaml.is_file():
            with open(coco_yaml, "r", encoding="utf-8") as fh:
                ds_cfg = yaml.safe_load(fh) or {}
            names = [str(x) for x in ds_cfg.get("class_list", [])]
            if len(names) == 80:
                return 80, names
    return cnum, [f"class_{i}" for i in range(cnum)]


def _load_yolo_for_export(
    model_arg: str,
    weights_arg: Optional[str],
    class_num: int,
) -> Tuple[YOLO, Any, str]:
    model_cfg = _resolve_model_cfg(model_arg)
    weight_source = "random_init"

    if weights_arg and weights_arg.lower() not in {"none", "false", "null"}:
        if weights_arg.lower() in {"true", "auto"}:
            model = create_model(model_cfg, weight_path=True, class_num=class_num, partial_load=False)
            weight_source = f"weights/{model_cfg.name}.pt"
        else:
            wpath = Path(weights_arg)
            if not wpath.is_file():
                raise FileNotFoundError(f"Checkpoint file not found: {wpath}")
            ckpt = torch.load(wpath, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "pruned_model" in ckpt:
                model = ckpt["pruned_model"]
                weight_source = str(wpath)
            else:
                model = create_model(model_cfg, weight_path=wpath, class_num=class_num, partial_load=False)
                weight_source = str(wpath)
    else:
        model = create_model(model_cfg, weight_path=None, class_num=class_num, partial_load=False)

    model = model.cpu().eval()
    return model, model_cfg, weight_source


def _infer_strides(model: YOLO, model_cfg: Any, image_size: Tuple[int, int]) -> List[int]:
    if hasattr(model_cfg, "anchor") and hasattr(model_cfg.anchor, "strides") and model_cfg.anchor.strides:
        return [int(s) for s in model_cfg.anchor.strides]
    w, h = image_size
    dummy = torch.zeros(1, 3, h, w)
    with torch.no_grad():
        out = model(dummy)
    return [w // head[0].shape[-1] for head in out["Main"]]


def inspect_onnx_no_aux(onnx_path: Path, expected_max_convs: int) -> Dict[str, Any]:
    """Verify simplified ONNX model has valid structure, 2 outputs, and zero auxiliary PGI nodes."""
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    graph = model.graph
    if len(graph.output) != 2:
        raise AssertionError(f"Expected 2 outputs (boxes, scores) in exported ONNX, got {len(graph.output)}")

    forbidden_tokens = ("auxiliary", "/aux", ".aux.", "cblinear", "cbfuse")
    bad_names: List[str] = []
    conv_count = 0
    for node in graph.node:
        if node.op_type == "Conv":
            conv_count += 1
        combined = " ".join([node.name, *node.input, *node.output]).lower()
        if any(tok in combined for tok in forbidden_tokens):
            bad_names.append(node.name or str(node.output))

    if bad_names:
        raise AssertionError(f"Found auxiliary PGI nodes in exported ONNX: {bad_names[:10]}")
    if conv_count > expected_max_convs:
        raise AssertionError(
            f"ONNX Conv count ({conv_count}) exceeds stripped PyTorch Conv count ({expected_max_convs}); "
            "auxiliary branch may not have been stripped."
        )

    return {
        "onnx_nodes": len(graph.node),
        "onnx_convs": conv_count,
        "expected_max_convs": expected_max_convs,
        "outputs": [o.name for o in graph.output],
    }


def _collect_protected_head_ops(ov_model: ov.Model) -> List[str]:
    """Trace backward from model outputs to protect final Detection head Convs and decode tail."""
    protected_names: Set[str] = set()
    visited_ids: Set[int] = set()
    queue: List[ov.Node] = [out.get_node() for out in ov_model.outputs]

    while queue:
        node = queue.pop(0)
        nid = id(node)
        if nid in visited_ids:
            continue
        visited_ids.add(nid)

        op_type = node.get_type_name()
        name = node.get_friendly_name()
        if name:
            protected_names.add(name)

        # Stop tracing deeper past the terminal 1x1 detection Convs (`anchor_conv.2` / `class_conv.2`),
        # while continuing past internal `anc2vec` Conv3d / Constant / Reshape ops.
        is_conv = op_type in {"Convolution", "GroupConvolution", "ConvolutionBackpropData"}
        is_anc2vec = "anc2vec" in name.lower()
        if is_conv and not is_anc2vec:
            continue

        for inp in node.input_values():
            parent = inp.get_node()
            if parent.get_type_name() != "Parameter":
                queue.append(parent)

    # Also include any explicitly named final head/decode ops
    for op in ov_model.get_ordered_ops():
        lname = op.get_friendly_name().lower()
        if any(k in lname for k in ("anc2vec", "anchor_conv.2", "class_conv.2", "sigmoid")):
            if op.get_friendly_name():
                protected_names.add(op.get_friendly_name())

    return sorted(protected_names)


def _attach_embedded_preprocess(ov_model: ov.Model, image_size: Tuple[int, int], dynamic_spatial: bool = True) -> ov.Model:
    """Fuse uint8 NHWC BGR -> RGB resize/normalize/NCHW PrePostProcessor into OpenVINO IR."""
    ppp = ov.preprocess.PrePostProcessor(ov_model)
    inp = ppp.input()
    tensor_builder = (
        inp.tensor()
        .set_element_type(ov.Type.u8)
        .set_layout(ov.Layout("NHWC"))
        .set_color_format(ov.preprocess.ColorFormat.BGR)
    )
    if dynamic_spatial:
        tensor_builder.set_spatial_dynamic_shape()
        inp.preprocess().convert_color(ov.preprocess.ColorFormat.RGB).resize(
            ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR
        ).convert_element_type(ov.Type.f32).scale(255.0)
    else:
        w, h = image_size
        tensor_builder.set_shape([1, h, w, 3])
        inp.preprocess().convert_color(ov.preprocess.ColorFormat.RGB).convert_element_type(ov.Type.f32).scale(255.0)

    inp.model().set_layout(ov.Layout("NCHW"))
    return ppp.build()


def _load_calibration_tensors(
    calib_dir: Path,
    image_size: Tuple[int, int],
    max_samples: int,
) -> List[np.ndarray]:
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration image directory not found: {calib_dir}")
    img_paths = sorted(
        p for p in calib_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )[:max_samples]
    if not img_paths:
        raise FileNotFoundError(f"No calibration images found in {calib_dir}")

    tensors: List[np.ndarray] = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        tensor, _, _ = letterbox_bgr(img, target_size=image_size, return_uint8_nhwc=False)
        tensors.append(tensor)
    if not tensors:
        raise RuntimeError(f"Failed to read any valid images from {calib_dir}")
    return tensors


def export_pipeline(
    *,
    model_arg: str,
    weights_arg: Optional[str] = None,
    output_dir: Path,
    name: Optional[str] = None,
    dataset_arg: Optional[str] = None,
    class_num_arg: Optional[int] = None,
    class_names_arg: Optional[str] = None,
    image_size: Tuple[int, int] = (640, 640),
    opset: int = 17,
    simplify: bool = True,
    int8: bool = False,
    calib_dir: Optional[Path] = None,
    calib_samples: int = 250,
    protect_heads: bool = False,
    embed_preprocess: bool = False,
    static_preprocess: bool = False,
) -> Dict[str, Any]:
    """Run full PyTorch -> stripped ONNX -> OpenVINO FP16/INT8 + PrePostProcessor export."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_class_num = None
    if weights_arg and weights_arg.lower() not in {"none", "false", "null", "true", "auto"}:
        wp = Path(weights_arg)
        if wp.is_file():
            raw_ckpt = torch.load(wp, map_location="cpu", weights_only=False)
            if isinstance(raw_ckpt, dict) and "class_num" in raw_ckpt:
                ckpt_class_num = int(raw_ckpt["class_num"])

    class_num, class_names = _resolve_class_info(dataset_arg, class_num_arg, class_names_arg, ckpt_class_num)
    yolo_model, model_cfg, weight_source = _load_yolo_for_export(model_arg, weights_arg, class_num)

    strip_stats = yolo_model.strip_auxiliary_branches()
    strides = _infer_strides(yolo_model, model_cfg, image_size)
    wrapper = ExportYOLOWrapper(yolo_model, image_size=image_size, strides=strides).eval()

    w, h = int(image_size[0]), int(image_size[1])
    dummy = torch.rand(1, 3, h, w, dtype=torch.float32)
    with torch.no_grad():
        pt_boxes, pt_scores = wrapper(dummy)
    pt_boxes_np = pt_boxes.numpy()
    pt_scores_np = pt_scores.numpy()

    stem = name or f"{model_cfg.name}_{'int8' if int8 else 'fp16'}"
    onnx_path = output_dir / f"{stem}.onnx"

    export_kwargs = dict(
        input_names=["images"],
        output_names=["boxes", "scores"],
        opset_version=opset,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(wrapper, dummy, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrapper, dummy, str(onnx_path), **export_kwargs)

    if simplify:
        onnx_model = onnx.load(str(onnx_path))
        onnx_sim, ok = onnxsim.simplify(onnx_model)
        if ok:
            onnx.save(onnx_sim, str(onnx_path))
        else:
            print("[WARN] onnxsim simplification returned ok=False; keeping unsimplified ONNX.")

    expected_convs = sum(1 for m in wrapper.yolo_model.modules() if isinstance(m, (nn.Conv2d, nn.Conv3d)))
    onnx_info = inspect_onnx_no_aux(onnx_path, expected_max_convs=expected_convs)

    # Numerical parity: ONNX Runtime vs PyTorch
    ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outs = ort_sess.run(None, {"images": dummy.numpy()})
    ort_boxes, ort_scores = (ort_outs[0], ort_outs[1]) if ort_outs[0].shape[-1] == 4 else (ort_outs[1], ort_outs[0])
    onnx_box_diff = float(np.max(np.abs(pt_boxes_np - ort_boxes)))
    onnx_score_diff = float(np.max(np.abs(pt_scores_np - ort_scores)))
    if not (np.allclose(pt_boxes_np, ort_boxes, atol=1e-2, rtol=1e-3) and np.allclose(pt_scores_np, ort_scores, atol=1e-3, rtol=1e-3)):
        raise AssertionError(f"ONNX vs PyTorch parity failed: box_diff={onnx_box_diff}, score_diff={onnx_score_diff}")

    # Convert ONNX -> OpenVINO IR
    ov_model = ov.convert_model(str(onnx_path))

    # Check OpenVINO FP16 IR parity before optional INT8 / PPP
    core = ov.Core()
    fp16_tmp_xml = output_dir / f"{stem}_parity_tmp.xml"
    ov.save_model(ov_model, str(fp16_tmp_xml), compress_to_fp16=True)
    cm_fp16 = core.compile_model(str(fp16_tmp_xml), "CPU")
    ov_fp16_res = list(cm_fp16({cm_fp16.inputs[0].any_name: dummy.numpy()}).values())
    ov_boxes, ov_scores = (ov_fp16_res[0], ov_fp16_res[1]) if ov_fp16_res[0].shape[-1] == 4 else (ov_fp16_res[1], ov_fp16_res[0])
    fp16_box_diff = float(np.max(np.abs(pt_boxes_np - ov_boxes)))
    fp16_score_diff = float(np.max(np.abs(pt_scores_np - ov_scores)))
    fp16_tmp_xml.unlink(missing_ok=True)
    fp16_tmp_xml.with_suffix(".bin").unlink(missing_ok=True)

    protected_ops_count = 0
    if int8:
        if calib_dir is None:
            raise ValueError("--calib-dir is required when --int8 is enabled.")
        calib_tensors = _load_calibration_tensors(calib_dir, image_size=(w, h), max_samples=calib_samples)
        input_name = ov_model.inputs[0].any_name
        calib_dataset = nncf.Dataset(calib_tensors, lambda item: {input_name: item})

        ignored_scope = None
        if protect_heads:
            protected_names = _collect_protected_head_ops(ov_model)
            protected_ops_count = len(protected_names)
            ignored_scope = nncf.IgnoredScope(names=protected_names)

        ov_model = nncf.quantize(
            ov_model,
            calibration_dataset=calib_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            ignored_scope=ignored_scope,
            subset_size=len(calib_tensors),
        )

    dynamic_spatial = bool(embed_preprocess and not static_preprocess)
    if embed_preprocess:
        ov_model = _attach_embedded_preprocess(ov_model, image_size=(w, h), dynamic_spatial=dynamic_spatial)

    xml_path = output_dir / f"{stem}.xml"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=not int8)

    meta = {
        "model_xml": xml_path.name,
        "model_variant": str(model_cfg.name),
        "precision": "INT8" if int8 else "FP16",
        "input_size": [w, h],
        "strides": strides,
        "embedded_preprocess": bool(embed_preprocess),
        "dynamic_spatial": dynamic_spatial,
        "protect_heads": bool(protect_heads and int8),
        "protected_ops_count": protected_ops_count,
        "class_num": class_num,
        "class_names": class_names,
        "source_weights": weight_source,
        "onnx_path": onnx_path.name,
        "stripped_aux": strip_stats,
        "onnx_info": onnx_info,
        "parity": {
            "onnx_max_box_diff": round(onnx_box_diff, 6),
            "onnx_max_score_diff": round(onnx_score_diff, 6),
            "ov_fp16_max_box_diff": round(fp16_box_diff, 6),
            "ov_fp16_max_score_diff": round(fp16_score_diff, 6),
        },
    }
    stem_meta_path = xml_path.with_suffix(".meta.json")
    dir_meta_path = output_dir / "export_meta.json"
    meta_text = json.dumps(meta, indent=2)
    stem_meta_path.write_text(meta_text, encoding="utf-8")
    dir_meta_path.write_text(meta_text, encoding="utf-8")

    return {
        "xml_path": str(xml_path),
        "onnx_path": str(onnx_path),
        "meta_path": str(stem_meta_path),
        "meta": meta,
    }


def _parse_imgsz(raw: str) -> Tuple[int, int]:
    parts = [int(p.strip()) for p in raw.lower().replace("x", ",").split(",") if p.strip()]
    if len(parts) == 1:
        return parts[0], parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"Invalid --imgsz format: {raw} (use e.g. 640 or 640,480)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export YOLOv9 to ONNX & OpenVINO FP16/INT8 IR.")
    parser.add_argument("--model", required=True, help="Model name (e.g. v9-t, v9-t-opt, v9-t-opt-p2) or YAML path.")
    parser.add_argument("--weights", default=None, help="Path to .pt/.ckpt, 'auto' to download, or omit for random init.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save ONNX, IR XML/BIN, and export_meta.json.")
    parser.add_argument("--name", default=None, help="Output file stem (defaults to <model>_<precision>).")
    parser.add_argument("--dataset", default=None, help="Hydra dataset name or YAML path for class names/count.")
    parser.add_argument("--class-num", type=int, default=None, help="Override number of classes.")
    parser.add_argument("--class-names", default=None, help="Comma-separated class names or YAML/JSON path.")
    parser.add_argument("--imgsz", default="640", help="Input size W or W,H (default: 640).")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17).")
    parser.add_argument("--no-simplify", action="store_true", help="Disable onnxsim graph simplification.")
    parser.add_argument("--int8", action="store_true", help="Run NNCF INT8 PTQ quantization.")
    parser.add_argument("--calib-dir", type=Path, default=None, help="Calibration images directory for NNCF INT8.")
    parser.add_argument("--calib-samples", type=int, default=250, help="Number of calibration samples (default: 250).")
    parser.add_argument("--protect-heads", action="store_true", help="Keep detection output heads in higher precision via nncf.IgnoredScope.")
    parser.add_argument("--embed-preprocess", action="store_true", help="Fuse OpenVINO PrePostProcessor (uint8 NHWC BGR input).")
    parser.add_argument("--static-preprocess", action="store_true", help="Use fixed [1,H,W,3] input shape with PrePostProcessor instead of dynamic spatial resize.")
    args = parser.parse_args(argv)

    result = export_pipeline(
        model_arg=args.model,
        weights_arg=args.weights,
        output_dir=args.output_dir,
        name=args.name,
        dataset_arg=args.dataset,
        class_num_arg=args.class_num,
        class_names_arg=args.class_names,
        image_size=_parse_imgsz(args.imgsz),
        opset=args.opset,
        simplify=not args.no_simplify,
        int8=args.int8,
        calib_dir=args.calib_dir,
        calib_samples=args.calib_samples,
        protect_heads=args.protect_heads,
        embed_preprocess=args.embed_preprocess,
        static_preprocess=args.static_preprocess,
    )
    meta = result["meta"]
    print(
        f"[OK] Exported {meta['model_variant']} ({meta['precision']}, PPP={meta['embedded_preprocess']}) -> {result['xml_path']}\n"
        f"     Stripped AUX: {meta['stripped_aux']['removed_layers']} layers ({meta['stripped_aux']['removed_params']:,} params)\n"
        f"     ONNX Convs: {meta['onnx_info']['onnx_convs']} | Parity: "
        f"ONNX box/score diff=({meta['parity']['onnx_max_box_diff']:.2e}, {meta['parity']['onnx_max_score_diff']:.2e}), "
        f"OV FP16 diff=({meta['parity']['ov_fp16_max_box_diff']:.2e}, {meta['parity']['ov_fp16_max_score_diff']:.2e})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
