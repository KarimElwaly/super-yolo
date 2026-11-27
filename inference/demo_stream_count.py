"""Reference real-time stream counting pipeline: threaded capture + OpenVINODetector + ByteTrack + Line/PolygonZone."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import openvino as ov
import supervision as sv
from scipy.optimize import linear_sum_assignment

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from inference.detector_ov import OpenVINODetector


def _bbox_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union


@dataclass
class _Track:
    track_id: int
    xyxy: np.ndarray
    class_id: int
    confidence: float
    velocity: np.ndarray
    lost_frames: int = 0

    def predict(self) -> None:
        self.xyxy = self.xyxy + self.velocity

    def update(self, xyxy: np.ndarray, class_id: int, confidence: float) -> None:
        delta = xyxy - self.xyxy
        self.velocity = 0.5 * self.velocity + 0.5 * delta
        self.xyxy = xyxy.copy()
        self.class_id = int(class_id)
        self.confidence = float(confidence)
        self.lost_frames = 0


class SimpleByteTrack:
    """Lightweight MIT-clean two-stage ByteTrack association using NumPy + SciPy Hungarian matching."""

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.3,
        max_lost_frames: int = 30,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_lost_frames = max_lost_frames
        self._next_id = 1
        self._tracks: List[_Track] = []

    def _associate(
        self, track_indices: List[int], det_boxes: np.ndarray, iou_thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not track_indices or len(det_boxes) == 0:
            return [], list(track_indices), list(range(len(det_boxes)))
        t_boxes = np.stack([self._tracks[i].xyxy for i in track_indices], axis=0)
        iou = _bbox_iou_matrix(t_boxes, det_boxes)
        row_ind, col_ind = linear_sum_assignment(1.0 - iou)
        matched, unmatched_t, unmatched_d = [], set(track_indices), set(range(len(det_boxes)))
        for r, c in zip(row_ind, col_ind):
            if iou[r, c] >= iou_thresh:
                ti = track_indices[r]
                matched.append((ti, int(c)))
                unmatched_t.discard(ti)
                unmatched_d.discard(int(c))
        return matched, sorted(unmatched_t), sorted(unmatched_d)

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        for trk in self._tracks:
            trk.predict()

        n_det = len(detections)
        if n_det == 0:
            for trk in self._tracks:
                trk.lost_frames += 1
            self._tracks = [t for t in self._tracks if t.lost_frames <= self.max_lost_frames]
            return sv.Detections.empty()

        confs = detections.confidence if detections.confidence is not None else np.ones(n_det, dtype=np.float32)
        cls_ids = detections.class_id if detections.class_id is not None else np.zeros(n_det, dtype=int)

        high_mask = confs >= self.high_thresh
        low_mask = (~high_mask) & (confs >= self.low_thresh)
        high_idx = np.where(high_mask)[0]
        low_idx = np.where(low_mask)[0]

        all_track_idx = list(range(len(self._tracks)))
        matched_high, unmatched_t, unmatched_high_d = self._associate(
            all_track_idx, detections.xyxy[high_idx], self.match_thresh
        )

        det_to_track_id: Dict[int, int] = {}
        for ti, local_di in matched_high:
            di = int(high_idx[local_di])
            self._tracks[ti].update(detections.xyxy[di], int(cls_ids[di]), float(confs[di]))
            det_to_track_id[di] = self._tracks[ti].track_id

        # Second stage: recover occluded/low-confidence detections against remaining tracks
        matched_low, still_unmatched_t, _ = self._associate(
            unmatched_t, detections.xyxy[low_idx], self.match_thresh
        )
        for ti, local_di in matched_low:
            di = int(low_idx[local_di])
            self._tracks[ti].update(detections.xyxy[di], int(cls_ids[di]), float(confs[di]))
            det_to_track_id[di] = self._tracks[ti].track_id

        for ti in still_unmatched_t:
            self._tracks[ti].lost_frames += 1

        for local_di in unmatched_high_d:
            di = int(high_idx[local_di])
            trk = _Track(
                track_id=self._next_id,
                xyxy=detections.xyxy[di].copy(),
                class_id=int(cls_ids[di]),
                confidence=float(confs[di]),
                velocity=np.zeros(4, dtype=np.float32),
            )
            self._next_id += 1
            self._tracks.append(trk)
            det_to_track_id[di] = trk.track_id

        self._tracks = [t for t in self._tracks if t.lost_frames <= self.max_lost_frames]

        kept_indices = sorted(det_to_track_id.keys())
        if not kept_indices:
            return sv.Detections.empty()

        tracked = detections[kept_indices]
        tracked.tracker_id = np.array([det_to_track_id[i] for i in kept_indices], dtype=int)
        return tracked


def create_tracker(conf_threshold: float = 0.25):
    if hasattr(sv, "ByteTrack"):
        return sv.ByteTrack(track_activation_threshold=max(0.4, conf_threshold))
    return SimpleByteTrack(high_thresh=max(0.4, conf_threshold), low_thresh=0.1)


class ThreadedFrameGrabber:
    """Reads video/RTSP frames on a background thread, optionally dropping stale frames for low latency."""

    def __init__(self, source: str, drop_stale: bool = True, max_queue_size: int = 16) -> None:
        cap_src = int(source) if str(source).isdigit() else str(source)
        self.cap = cv2.VideoCapture(cap_src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video/stream source: {source}")
        raw_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.source_fps: float = raw_fps if 1.0 <= raw_fps <= 120.0 else 30.0
        self.drop_stale = bool(drop_stale)
        self.max_queue_size = max(1, int(max_queue_size))
        self._lock = Lock()
        self._stop = Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._queue: List[np.ndarray] = []
        self._ended = False
        self._thread = Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            if not self.drop_stale:
                with self._lock:
                    queue_full = len(self._queue) >= self.max_queue_size
                if queue_full:
                    time.sleep(0.005)
                    continue
            ok, frame = self.cap.read()
            if not ok:
                with self._lock:
                    self._ended = True
                break
            with self._lock:
                if self.drop_stale:
                    self._latest_frame = frame
                else:
                    self._queue.append(frame)

    def read(self, timeout_s: float = 2.0) -> Tuple[bool, Optional[np.ndarray]]:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self._lock:
                if self.drop_stale and self._latest_frame is not None:
                    frame = self._latest_frame
                    self._latest_frame = None
                    return True, frame
                if not self.drop_stale and self._queue:
                    return True, self._queue.pop(0)
                if self._ended:
                    return False, None
            time.sleep(0.002)
        return False, None

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


class _OrderedAsyncInference:
    """Run bounded asynchronous inference and return detections in submission order."""

    def __init__(self, detector: OpenVINODetector, num_requests: int) -> None:
        self.detector = detector
        self.queue = ov.AsyncInferQueue(detector.compiled_model, num_requests)
        self._condition = Condition()
        self._completed: Dict[int, Tuple[np.ndarray, sv.Detections, Optional[Exception]]] = {}
        self.queue.set_callback(self._on_complete)

    def submit(self, frame_index: int, frame: np.ndarray) -> None:
        input_tensor, geom = self.detector.prepare_input(frame)
        self.queue.start_async(
            {self.detector.input_name: input_tensor},
            userdata=(frame_index, frame, geom, input_tensor),
            share_inputs=True,
        )

    def _on_complete(self, request: ov.InferRequest, userdata: Any) -> None:
        frame_index, frame, geom, _input_tensor = userdata
        detections = sv.Detections.empty()
        error: Optional[Exception] = None
        try:
            outputs = [request.get_output_tensor(i).data for i in range(len(self.detector.compiled_model.outputs))]
            boxes, scores = self.detector._order_raw_outputs(outputs)
            detections = self.detector.postprocess(boxes, scores, geom)
        except Exception as exc:
            error = exc
        with self._condition:
            self._completed[frame_index] = (frame, detections, error)
            self._condition.notify_all()

    def take(self, frame_index: int) -> Tuple[np.ndarray, sv.Detections]:
        with self._condition:
            self._condition.wait_for(lambda: frame_index in self._completed)
            frame, detections, error = self._completed.pop(frame_index)
        if error is not None:
            raise RuntimeError(f"Asynchronous inference failed for frame {frame_index}") from error
        return frame, detections

    def wait_all(self) -> None:
        self.queue.wait_all()


def _resolve_class_filter(
    target_classes: Optional[List[str]],
    class_names: List[str],
) -> Optional[set[int]]:
    """Resolve CLI class names/IDs into a set of integer class IDs, or None for all classes."""
    if not target_classes:
        return None
    name_to_id = {name.strip().lower(): idx for idx, name in enumerate(class_names)}
    resolved: set[int] = set()
    for raw in target_classes:
        for token in str(raw).split(","):
            item = token.strip()
            if not item:
                continue
            if item.lstrip("-").isdigit():
                resolved.add(int(item))
            elif item.lower() in name_to_id:
                resolved.add(name_to_id[item.lower()])
            else:
                raise ValueError(f"Unknown class '{item}'. Available: {class_names}")
    return resolved or None


def run_stream_pipeline(
    *,
    model_xml: Path,
    source: str,
    num_threads: int = 4,
    async_requests: int = 4,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    detect_every_n: int = 1,
    max_frames: int = 0,
    output_video: Optional[Path] = None,
    display: bool = False,
    drop_stale: bool = False,
    target_classes: Optional[List[str]] = None,
    show_zones: bool = True,
    use_tracker: bool = True,
) -> Dict[str, Any]:
    if async_requests < 1:
        raise ValueError(f"async_requests must be positive, got {async_requests}")
    detector = OpenVINODetector(
        model_xml_path=model_xml,
        num_threads=num_threads,
        num_streams=async_requests,
        performance_hint="THROUGHPUT" if async_requests > 1 else "LATENCY",
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    allowed_ids = _resolve_class_filter(target_classes, detector.class_names)
    tracker = create_tracker(conf_threshold=conf_threshold) if use_tracker else None

    grabber = ThreadedFrameGrabber(source, drop_stale=drop_stale)
    ok, first_frame = grabber.read(timeout_s=5.0)
    if not ok or first_frame is None:
        grabber.release()
        raise RuntimeError(f"No frames received from source: {source}")

    h, w = first_frame.shape[:2]
    line_zone = sv.LineZone(start=sv.Point(0, h // 2), end=sv.Point(w, h // 2)) if show_zones else None
    polygon = np.array(
        [[int(w * 0.2), int(h * 0.2)], [int(w * 0.8), int(h * 0.2)], [int(w * 0.8), int(h * 0.8)], [int(w * 0.2), int(h * 0.8)]],
        dtype=np.int64,
    )
    poly_zone = sv.PolygonZone(polygon=polygon) if show_zones else None

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    line_annotator = sv.LineZoneAnnotator() if show_zones else None
    poly_annotator = sv.PolygonZoneAnnotator(zone=poly_zone) if poly_zone is not None else None

    writer = None
    if output_video is not None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), grabber.source_fps, (w, h))

    frame_idx = 0
    next_async_frame_idx = 0
    pending_frames: List[Tuple[np.ndarray, bool]] = []
    async_inference = _OrderedAsyncInference(detector, async_requests) if async_requests > 1 else None
    total_detections = 0
    unique_track_ids: set[int] = set()
    last_detections = sv.Detections.empty()
    t_start = time.perf_counter()
    previous_cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    stop_requested = False

    def process_frame(frame: np.ndarray, raw_dets: Optional[sv.Detections]) -> None:
        nonlocal last_detections, total_detections, stop_requested
        if raw_dets is not None:
            if allowed_ids is not None and len(raw_dets) > 0 and raw_dets.class_id is not None:
                keep = np.isin(raw_dets.class_id, sorted(allowed_ids))
                raw_dets = raw_dets[keep]
            last_detections = tracker.update_with_detections(raw_dets) if tracker is not None else raw_dets
            total_detections += len(raw_dets)
            if last_detections.tracker_id is not None:
                unique_track_ids.update(int(t) for t in last_detections.tracker_id.tolist())

        if line_zone is not None:
            line_zone.trigger(last_detections)
        if poly_zone is not None:
            poly_zone.trigger(last_detections)

        annotated = box_annotator.annotate(scene=frame.copy(), detections=last_detections)
        labels = [
            (
                f"#{tid} " if tracker is not None and tid != -1 else ""
            )
            + f"{detector.class_names[int(cid)] if 0 <= int(cid) < len(detector.class_names) else f'cls:{cid}'} {conf:.2f}"
            for tid, cid, conf in zip(
                last_detections.tracker_id if last_detections.tracker_id is not None else [-1] * len(last_detections),
                last_detections.class_id if last_detections.class_id is not None else [0] * len(last_detections),
                last_detections.confidence if last_detections.confidence is not None else [0.0] * len(last_detections),
            )
        ]
        if len(last_detections) > 0:
            annotated = label_annotator.annotate(scene=annotated, detections=last_detections, labels=labels)
        if line_annotator is not None and line_zone is not None:
            annotated = line_annotator.annotate(frame=annotated, line_counter=line_zone)
        if poly_annotator is not None:
            annotated = poly_annotator.annotate(scene=annotated)

        if writer is not None:
            writer.write(annotated)
        if display:
            cv2.imshow("YOLOv9 OpenVINO Stream Counter", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_requested = True

    try:
        frame: Optional[np.ndarray] = first_frame
        while frame is not None and not stop_requested:
            should_detect = frame_idx % max(1, detect_every_n) == 0
            if async_inference is None:
                process_frame(frame, detector.detect(frame) if should_detect else None)
            else:
                if len(pending_frames) >= async_requests:
                    old_frame, old_detect = pending_frames.pop(0)
                    old_result = async_inference.take(next_async_frame_idx) if old_detect else None
                    if old_detect:
                        next_async_frame_idx += 1
                    process_frame(old_result[0] if old_result else old_frame, old_result[1] if old_result else None)
                if should_detect:
                    async_inference.submit(next_async_frame_idx + sum(1 for _, do_detect in pending_frames if do_detect), frame)
                pending_frames.append((frame, should_detect))

            frame_idx += 1
            if max_frames > 0 and frame_idx >= max_frames:
                break
            ok, frame = grabber.read(timeout_s=2.0)
            if not ok:
                break
        if async_inference is not None:
            async_inference.wait_all()
            if not stop_requested:
                for old_frame, old_detect in pending_frames:
                    old_result = async_inference.take(next_async_frame_idx) if old_detect else None
                    if old_detect:
                        next_async_frame_idx += 1
                    process_frame(old_result[0] if old_result else old_frame, old_result[1] if old_result else None)
    finally:
        if async_inference is not None:
            async_inference.wait_all()
        grabber.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        cv2.setNumThreads(previous_cv_threads)

    elapsed = max(time.perf_counter() - t_start, 1e-6)
    return {
        "frames_processed": frame_idx,
        "fps": round(frame_idx / elapsed, 2),
        "total_detections": int(total_detections),
        "unique_track_ids": len(unique_track_ids),
        "line_in_count": int(line_zone.in_count) if line_zone is not None else 0,
        "line_out_count": int(line_zone.out_count) if line_zone is not None else 0,
        "polygon_current_count": int(poly_zone.current_count) if poly_zone is not None else 0,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Real-time OpenVINO YOLOv9 stream detection, tracking & zone counting.")
    parser.add_argument("--model-xml", type=Path, required=True, help="Path to exported OpenVINO .xml IR model.")
    parser.add_argument("--source", required=True, help="Video file path, RTSP URL, or webcam index (e.g. 0).")
    parser.add_argument("--threads", type=int, default=4, help="OpenVINO CPU inference threads (default: 4).")
    parser.add_argument("--async-requests", type=int, default=4, help="Maximum concurrent inference requests (default: 4; use 1 for sequential inference).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45).")
    parser.add_argument("--classes", nargs="+", default=None, help="Filter detections by class ID(s) or name(s), e.g. --classes 0 or --classes Person.")
    parser.add_argument("--detect-every-n", type=int, default=1, help="Run detector every N frames (default: 1).")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = until stream ends).")
    parser.add_argument("--output", type=Path, default=None, help="Optional output annotated MP4 path.")
    parser.add_argument("--display", action="store_true", help="Display live OpenCV window.")
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument("--drop-stale-frames", action="store_true", help="Drop stale capture frames to minimize live-stream latency.")
    frame_group.add_argument("--no-drop-frames", action="store_false", dest="drop_stale_frames", help="Process every captured frame (default).")
    parser.set_defaults(drop_stale_frames=False)
    parser.add_argument("--no-zones", action="store_true", help="Disable LineZone (in/out line) and PolygonZone overlays.")
    parser.add_argument("--no-track", action="store_true", help="Disable ByteTrack ID tracking and render raw detections directly.")
    args = parser.parse_args(argv)

    summary = run_stream_pipeline(
        model_xml=args.model_xml,
        source=args.source,
        num_threads=args.threads,
        async_requests=args.async_requests,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        detect_every_n=args.detect_every_n,
        max_frames=args.max_frames,
        output_video=args.output,
        display=args.display,
        drop_stale=args.drop_stale_frames,
        target_classes=args.classes,
        show_zones=not args.no_zones,
        use_tracker=not args.no_track,
    )
    print(f"[OK] Stream summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
