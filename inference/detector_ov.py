"""Production-ready OpenVINO detector adapter supporting FP16/INT8 and PrePostProcessor IR models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import openvino as ov
import supervision as sv


def letterbox_bgr(
    image_bgr: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    pad_color: Tuple[int, int, int] = (114, 114, 114),
    return_uint8_nhwc: bool = False,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize and pad a BGR image preserving aspect ratio (matching YOLOv9 PadAndResize).

    Args:
        image_bgr: Input HWC uint8 BGR image.
        target_size: (target_w, target_h).
        pad_color: BGR padding fill color.
        return_uint8_nhwc: If True, return [1, H, W, 3] uint8 BGR; else [1, 3, H, W] float32 RGB in [0, 1].
    """
    orig_h, orig_w = image_bgr.shape[:2]
    target_w, target_h = int(target_size[0]), int(target_size[1])

    scale = min(target_w / max(orig_w, 1), target_h / max(orig_h, 1))
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    if (new_w, new_h) != (orig_w, orig_h):
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)
    else:
        resized = image_bgr

    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color,
    )

    if return_uint8_nhwc:
        tensor = np.ascontiguousarray(padded[None, ...], dtype=np.uint8)
        return tensor, float(scale), (pad_left, pad_top)

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) * (1.0 / 255.0)
    tensor = np.ascontiguousarray(chw[None, ...], dtype=np.float32)
    return tensor, float(scale), (pad_left, pad_top)


def _load_export_meta(model_xml_path: Path) -> Dict[str, Any]:
    stem_meta = model_xml_path.with_suffix(".meta.json")
    dir_meta = model_xml_path.parent / "export_meta.json"
    for cand in (stem_meta, dir_meta):
        if cand.is_file():
            with open(cand, "r", encoding="utf-8") as fh:
                return json.load(fh)
    return {}


class OpenVINODetector:
    """Model-agnostic OpenVINO detector returning `supervision.Detections`."""

    def __init__(
        self,
        model_xml_path: Union[str, Path],
        num_threads: int = 4,
        num_streams: int = 1,
        performance_hint: str = "LATENCY",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "CPU",
        max_det: int = 300,
        letterbox_embedded: Optional[bool] = None,
    ) -> None:
        self.model_xml_path = Path(model_xml_path)
        if not self.model_xml_path.is_file():
            raise FileNotFoundError(f"OpenVINO IR XML not found: {self.model_xml_path}")

        self.meta = _load_export_meta(self.model_xml_path)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_det = int(max_det)
        self.device = str(device)
        self.num_threads = int(num_threads)
        self.num_streams = int(num_streams)
        self.performance_hint = str(performance_hint)
        if self.num_threads < 1:
            raise ValueError(f"num_threads must be positive, got {self.num_threads}")
        if self.num_streams < 1:
            raise ValueError(f"num_streams must be positive, got {self.num_streams}")

        self.core = ov.Core()
        ov_model = self.core.read_model(str(self.model_xml_path))

        # Infer layout & embedded preprocess settings from metadata and/or IR tensor properties
        in_port = ov_model.inputs[0]
        in_pshape = in_port.get_partial_shape()
        in_type = in_port.get_element_type()
        in_dims = [d.get_length() if d.is_static else -1 for d in in_pshape]

        self.embedded_preprocess = bool(self.meta.get("embedded_preprocess", in_type == ov.Type.u8))
        self.dynamic_spatial = len(in_dims) == 4 and (in_dims[1] == -1 or in_dims[2] == -1)

        if "input_size" in self.meta:
            w, h = self.meta["input_size"]
            self.input_size = (int(w), int(h))
        elif len(in_dims) == 4 and not self.dynamic_spatial:
            if self.embedded_preprocess or in_dims[3] == 3:
                self.input_size = (int(in_dims[2]), int(in_dims[1]))
            else:
                self.input_size = (int(in_dims[3]), int(in_dims[2]))
        else:
            self.input_size = (640, 640)

        # If embedded PPP uses static shape [1, H, W, 3] or caller requests letterbox, preserve aspect ratio
        if letterbox_embedded is None:
            self.letterbox_embedded = self.embedded_preprocess and not self.dynamic_spatial
        else:
            self.letterbox_embedded = bool(letterbox_embedded)

        self.class_names: List[str] = list(self.meta.get("class_names", []))

        compile_cfg: Dict[str, Any] = {}
        if self.device.upper().startswith("CPU"):
            compile_cfg["INFERENCE_NUM_THREADS"] = str(self.num_threads)
            compile_cfg["PERFORMANCE_HINT"] = self.performance_hint
            compile_cfg["NUM_STREAMS"] = str(self.num_streams)

        self.compiled_model = self.core.compile_model(ov_model, self.device, compile_cfg)
        self.infer_request = self.compiled_model.create_infer_request()
        self.input_name = self.compiled_model.inputs[0].any_name

    def prepare_input(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Prepare raw BGR uint8 frame for OpenVINO inference and return coordinate unscaling info."""
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR frame, got shape {None if frame_bgr is None else frame_bgr.shape}")

        orig_h, orig_w = frame_bgr.shape[:2]
        in_w, in_h = self.input_size

        if self.embedded_preprocess:
            if self.letterbox_embedded:
                tensor, scale, (pad_l, pad_t) = letterbox_bgr(
                    frame_bgr, target_size=self.input_size, return_uint8_nhwc=True
                )
                geom = {"mode": "letterbox", "scale": scale, "pad": (pad_l, pad_t), "orig_wh": (orig_w, orig_h)}
            else:
                if not self.dynamic_spatial and (orig_w, orig_h) != (in_w, in_h):
                    resized = cv2.resize(frame_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
                    tensor = np.ascontiguousarray(resized[None, ...], dtype=np.uint8)
                else:
                    tensor = np.ascontiguousarray(frame_bgr[None, ...], dtype=np.uint8)
                geom = {
                    "mode": "stretch",
                    "sx": orig_w / float(in_w),
                    "sy": orig_h / float(in_h),
                    "orig_wh": (orig_w, orig_h),
                }
            return tensor, geom

        tensor, scale, (pad_l, pad_t) = letterbox_bgr(
            frame_bgr, target_size=self.input_size, return_uint8_nhwc=False
        )
        geom = {"mode": "letterbox", "scale": scale, "pad": (pad_l, pad_t), "orig_wh": (orig_w, orig_h)}
        return tensor, geom

    def infer_raw(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run OpenVINO inference on an already-prepared tensor and return (boxes [B,N,4], scores [B,N,C])."""
        outputs = self.infer_request.infer({self.input_name: input_tensor}, share_outputs=True)
        tensors = list(outputs.values())
        return self._order_raw_outputs(tensors)

    @staticmethod
    def _order_raw_outputs(tensors: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if len(tensors) != 2:
            raise RuntimeError(f"Expected 2 outputs (boxes, scores) from OpenVINO IR, got {len(tensors)}")
        # Identify boxes ([B, N, 4]) vs scores ([B, N, C]) safely even if output order varies
        t0, t1 = tensors[0], tensors[1]
        if t0.shape[-1] == 4 and t1.shape[-1] != 4:
            return t0, t1
        if t1.shape[-1] == 4 and t0.shape[-1] != 4:
            return t1, t0
        return t0, t1

    def postprocess(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        geom: Dict[str, Any],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> sv.Detections:
        """Apply confidence thresholding, class-aware cv2.dnn.NMSBoxes, and rescale boxes to original pixels."""
        conf_th = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        iou_th = self.iou_threshold if iou_threshold is None else float(iou_threshold)

        b_boxes = boxes[0] if boxes.ndim == 3 else boxes  # [N, 4] in input pixel xyxy
        b_scores = scores[0] if scores.ndim == 3 else scores  # [N, C]

        if b_boxes.size == 0 or b_scores.size == 0:
            return sv.Detections.empty()

        # Vectorized multi-label confidence filtering across [N, C]
        eligible_anchor_idx = np.flatnonzero(np.max(b_scores, axis=1) >= conf_th)
        if eligible_anchor_idx.size == 0:
            return sv.Detections.empty()
        local_anchor_idx, cls_idx = np.where(b_scores[eligible_anchor_idx] >= conf_th)
        anchor_idx = eligible_anchor_idx[local_anchor_idx]
        if anchor_idx.size == 0:
            return sv.Detections.empty()

        cand_scores = b_scores[anchor_idx, cls_idx].astype(np.float32)
        cand_boxes = b_boxes[anchor_idx].astype(np.float32)
        cand_cls = cls_idx.astype(np.int64)

        # Keep top candidates prior to NMS for predictable CPU overhead
        if cand_scores.size > 3000:
            top_order = np.argsort(cand_scores)[::-1][:3000]
            cand_scores = cand_scores[top_order]
            cand_boxes = cand_boxes[top_order]
            cand_cls = cand_cls[top_order]

        # Class-aware NMS using cv2.dnn.NMSBoxes with class coordinate offset
        max_wh = 7680.0
        offsets = cand_cls.astype(np.float32) * max_wh
        x1 = cand_boxes[:, 0]
        y1 = cand_boxes[:, 1]
        w = np.maximum(0.0, cand_boxes[:, 2] - cand_boxes[:, 0])
        h = np.maximum(0.0, cand_boxes[:, 3] - cand_boxes[:, 1])

        nms_boxes_xywh = np.stack([x1 + offsets, y1 + offsets, w, h], axis=1).tolist()
        keep = cv2.dnn.NMSBoxes(nms_boxes_xywh, cand_scores.tolist(), conf_th, iou_th)

        if keep is None or len(keep) == 0:
            return sv.Detections.empty()

        keep_idx = np.asarray(keep, dtype=np.int64).reshape(-1)
        # Sort descending by confidence and cap at max_det
        keep_idx = keep_idx[np.argsort(cand_scores[keep_idx])[::-1][: self.max_det]]

        final_boxes = cand_boxes[keep_idx].copy()
        final_scores = cand_scores[keep_idx]
        final_cls = cand_cls[keep_idx]

        # Rescale to original image pixel space
        orig_w, orig_h = geom["orig_wh"]
        if geom["mode"] == "letterbox":
            pad_l, pad_t = geom["pad"]
            scale = geom["scale"]
            final_boxes[:, [0, 2]] = (final_boxes[:, [0, 2]] - pad_l) / scale
            final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - pad_t) / scale
        else:
            final_boxes[:, [0, 2]] *= geom["sx"]
            final_boxes[:, [1, 3]] *= geom["sy"]

        final_boxes[:, [0, 2]] = np.clip(final_boxes[:, [0, 2]], 0.0, float(orig_w))
        final_boxes[:, [1, 3]] = np.clip(final_boxes[:, [1, 3]], 0.0, float(orig_h))

        data: Dict[str, Any] = {}
        if self.class_names:
            data["class_name"] = np.array(
                [self.class_names[c] if c < len(self.class_names) else str(c) for c in final_cls],
                dtype=object,
            )

        return sv.Detections(
            xyxy=final_boxes.astype(np.float32),
            confidence=final_scores.astype(np.float32),
            class_id=final_cls.astype(int),
            data=data,
        )

    def detect(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> sv.Detections:
        """Run full preprocess -> OpenVINO inference -> NMS -> coordinate unscaling on a BGR frame."""
        tensor, geom = self.prepare_input(frame_bgr)
        boxes, scores = self.infer_raw(tensor)
        return self.postprocess(boxes, scores, geom, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

    def __call__(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> sv.Detections:
        return self.detect(frame_bgr, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
