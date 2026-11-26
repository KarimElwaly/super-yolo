"""Reference real-time stream counting pipeline: threaded capture + OpenVINODetector + ByteTrack + Line/PolygonZone."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
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

    def __init__(self, source: str, drop_stale: bool = True) -> None:
        cap_src = int(source) if str(source).isdigit() else str(source)
        self.cap = cv2.VideoCapture(cap_src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video/stream source: {source}")
        self.drop_stale = bool(drop_stale)
        self._lock = Lock()
        self._stop = Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._queue: List[np.ndarray] = []
        self._ended = False
        self._thread = Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
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


def run_stream_pipeline(
    *,
    model_xml: Path,
    source: str,
    num_threads: int = 4,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    detect_every_n: int = 1,
    max_frames: int = 0,
    output_video: Optional[Path] = None,
    display: bool = False,
    drop_stale: bool = True,
) -> Dict[str, Any]:
    detector = OpenVINODetector(
        model_xml_path=model_xml,
        num_threads=num_threads,
        performance_hint="LATENCY",
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    tracker = create_tracker(conf_threshold=conf_threshold)

    grabber = ThreadedFrameGrabber(source, drop_stale=drop_stale)
    ok, first_frame = grabber.read(timeout_s=5.0)
    if not ok or first_frame is None:
        grabber.release()
        raise RuntimeError(f"No frames received from source: {source}")

    h, w = first_frame.shape[:2]
    line_zone = sv.LineZone(start=sv.Point(0, h // 2), end=sv.Point(w, h // 2))
    polygon = np.array(
        [[int(w * 0.2), int(h * 0.2)], [int(w * 0.8), int(h * 0.2)], [int(w * 0.8), int(h * 0.8)], [int(w * 0.2), int(h * 0.8)]],
        dtype=np.int64,
    )
    poly_zone = sv.PolygonZone(polygon=polygon)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    line_annotator = sv.LineZoneAnnotator()
    poly_annotator = sv.PolygonZoneAnnotator(zone=poly_zone)

    writer = None
    if output_video is not None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))

    frame_idx = 0
    last_detections = sv.Detections.empty()
    t_start = time.perf_counter()

    try:
        frame: Optional[np.ndarray] = first_frame
        while frame is not None:
            if frame_idx % max(1, detect_every_n) == 0:
                raw_dets = detector.detect(frame)
                last_detections = tracker.update_with_detections(raw_dets)

            line_zone.trigger(last_detections)
            poly_zone.trigger(last_detections)

            annotated = box_annotator.annotate(scene=frame.copy(), detections=last_detections)
            labels = [
                f"#{tid} cls:{cid} {conf:.2f}"
                for tid, cid, conf in zip(
                    last_detections.tracker_id if last_detections.tracker_id is not None else [-1] * len(last_detections),
                    last_detections.class_id if last_detections.class_id is not None else [0] * len(last_detections),
                    last_detections.confidence if last_detections.confidence is not None else [0.0] * len(last_detections),
                )
            ]
            if len(last_detections) > 0:
                annotated = label_annotator.annotate(scene=annotated, detections=last_detections, labels=labels)
            annotated = line_annotator.annotate(frame=annotated, line_counter=line_zone)
            annotated = poly_annotator.annotate(scene=annotated)

            if writer is not None:
                writer.write(annotated)
            if display:
                cv2.imshow("YOLOv9 OpenVINO Stream Counter", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if max_frames > 0 and frame_idx >= max_frames:
                break
            ok, frame = grabber.read(timeout_s=2.0)
            if not ok:
                break
    finally:
        grabber.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - t_start, 1e-6)
    return {
        "frames_processed": frame_idx,
        "fps": round(frame_idx / elapsed, 2),
        "line_in_count": int(line_zone.in_count),
        "line_out_count": int(line_zone.out_count),
        "polygon_current_count": int(poly_zone.current_count),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Real-time OpenVINO YOLOv9 stream detection, tracking & zone counting.")
    parser.add_argument("--model-xml", type=Path, required=True, help="Path to exported OpenVINO .xml IR model.")
    parser.add_argument("--source", required=True, help="Video file path, RTSP URL, or webcam index (e.g. 0).")
    parser.add_argument("--threads", type=int, default=4, help="OpenVINO CPU inference threads (default: 4).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45).")
    parser.add_argument("--detect-every-n", type=int, default=1, help="Run detector every N frames (default: 1).")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = until stream ends).")
    parser.add_argument("--output", type=Path, default=None, help="Optional output annotated MP4 path.")
    parser.add_argument("--display", action="store_true", help="Display live OpenCV window.")
    parser.add_argument("--no-drop-frames", action="store_true", help="Process every frame sequentially instead of dropping stale frames.")
    args = parser.parse_args(argv)

    summary = run_stream_pipeline(
        model_xml=args.model_xml,
        source=args.source,
        num_threads=args.threads,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        detect_every_n=args.detect_every_n,
        max_frames=args.max_frames,
        output_video=args.output,
        display=args.display,
        drop_stale=not args.no_drop_frames,
    )
    print(f"[OK] Stream summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
