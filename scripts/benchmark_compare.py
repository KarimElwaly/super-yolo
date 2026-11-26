"""Automated A/B OpenVINO IR benchmark suite with locked CPU controls and Go/No-Go acceptance gates."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from inference.detector_ov import OpenVINODetector


def _load_manifest(manifest_path: Optional[Path], model_specs: Optional[List[str]]) -> Tuple[str, List[Dict[str, Any]]]:
    if manifest_path is not None:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            entries = list(raw.get("models", []))
            baseline_name = raw.get("baseline")
        elif isinstance(raw, list):
            entries = list(raw)
            baseline_name = None
        else:
            raise ValueError(f"Unsupported manifest structure in {manifest_path}")
    elif model_specs:
        entries = []
        baseline_name = None
        for idx, spec in enumerate(model_specs):
            if "=" in spec:
                name, xml_path = spec.split("=", 1)
            else:
                xml_path = spec
                name = Path(xml_path).stem
            entries.append({"name": name.strip(), "model_xml": xml_path.strip(), "baseline": idx == 0})
    else:
        raise ValueError("Provide either --manifest or --models.")

    if not entries:
        raise ValueError("No models defined in manifest / --models.")

    if baseline_name is None:
        explicit = [e["name"] for e in entries if e.get("baseline")]
        baseline_name = explicit[0] if explicit else entries[0]["name"]

    return baseline_name, entries


def _evaluate_coco_accuracy(
    detector: OpenVINODetector,
    val_images_dir: Path,
    val_ann_path: Path,
    max_eval_images: int,
    eval_conf: float,
    eval_iou: float,
    eval_max_det: int,
) -> Dict[str, Optional[float]]:
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(val_ann_path))

    sorted_cat_ids = sorted(coco_gt.getCatIds())
    img_records = sorted(coco_gt.dataset.get("images", []), key=lambda r: int(r["id"]))
    if max_eval_images > 0:
        img_records = img_records[:max_eval_images]

    orig_max_det = detector.max_det
    detector.max_det = eval_max_det

    coco_results: List[Dict[str, Any]] = []
    eval_img_ids: List[int] = []
    try:
        for idx, img_info in enumerate(img_records, start=1):
            img_path = val_images_dir / img_info["file_name"]
            if not img_path.is_file():
                continue
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            eval_img_ids.append(int(img_info["id"]))
            dets = detector.detect(frame, conf_threshold=eval_conf, iou_threshold=eval_iou)
            if len(dets) == 0:
                continue

            xyxy = dets.xyxy
            confs = dets.confidence
            cls_ids = dets.class_id
            for box, score, cidx in zip(xyxy, confs, cls_ids):
                c_int = int(cidx)
                if c_int < 0 or c_int >= len(sorted_cat_ids):
                    continue
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0.0 or h <= 0.0:
                    continue
                coco_results.append(
                    {
                        "image_id": int(img_info["id"]),
                        "category_id": int(sorted_cat_ids[c_int]),
                        "bbox": [round(x1, 3), round(y1, 3), round(w, 3), round(h, 3)],
                        "score": float(score),
                    }
                )
            if idx % 250 == 0:
                print(f"  [eval] {detector.model_xml_path.stem}: {idx}/{len(img_records)} images...", flush=True)
    finally:
        detector.max_det = orig_max_det

    if not eval_img_ids:
        raise FileNotFoundError(f"No matching validation images found in {val_images_dir}")

    if not coco_results:
        return {
            "evaluated_images": len(eval_img_ids),
            "detections": 0,
            "map_50_95": 0.0,
            "map_50": 0.0,
            "ap_small": 0.0,
            "ap_medium": 0.0,
            "ap_large": 0.0,
        }

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = eval_img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "evaluated_images": len(eval_img_ids),
        "detections": len(coco_results),
        "map_50_95": round(float(stats[0]) * 100.0, 3) if stats[0] >= 0 else None,
        "map_50": round(float(stats[1]) * 100.0, 3) if stats[1] >= 0 else None,
        "ap_small": round(float(stats[3]) * 100.0, 3) if stats[3] >= 0 else None,
        "ap_medium": round(float(stats[4]) * 100.0, 3) if stats[4] >= 0 else None,
        "ap_large": round(float(stats[5]) * 100.0, 3) if stats[5] >= 0 else None,
    }


def _load_speed_frames(
    val_images_dir: Optional[Path],
    fallback_wh: Tuple[int, int],
    max_frames: int = 16,
) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    if val_images_dir is not None and val_images_dir.is_dir():
        for p in sorted(val_images_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                img = cv2.imread(str(p))
                if img is not None:
                    frames.append(img)
                if len(frames) >= max_frames:
                    break
    if not frames:
        w, h = fallback_wh
        rng = np.random.default_rng(42)
        frames.append(rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8))
    return frames


def _get_peak_rss_bytes(proc: psutil.Process, fallback_rss: int) -> int:
    mem = proc.memory_info()
    peak_bytes = getattr(mem, "peak_wset", None)
    if peak_bytes is None:
        try:
            import resource

            peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except Exception:
            peak_bytes = mem.rss
    return max(int(peak_bytes), int(fallback_rss))


def benchmark_single_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Measure accuracy, latency, CPU efficiency, and RSS/disk footprint for one OpenVINO IR model."""
    xml_path = Path(cfg["model_xml"]).resolve()
    bin_path = xml_path.with_suffix(".bin")
    disk_bytes = xml_path.stat().st_size + (bin_path.stat().st_size if bin_path.is_file() else 0)
    disk_mb = disk_bytes / (1024.0 * 1024.0)

    proc = psutil.Process(os.getpid())
    detector = OpenVINODetector(
        model_xml_path=xml_path,
        num_threads=int(cfg["threads"]),
        performance_hint=str(cfg["performance_hint"]),
        conf_threshold=float(cfg["bench_conf"]),
        iou_threshold=float(cfg["bench_iou"]),
        device="CPU",
        letterbox_embedded=cfg.get("letterbox_embedded"),
    )

    val_images_dir = Path(cfg["val_images"]).resolve() if cfg.get("val_images") else None
    val_ann_path = Path(cfg["val_ann"]).resolve() if cfg.get("val_ann") else None

    speed_frames = _load_speed_frames(val_images_dir, detector.input_size)
    warmup_iters = int(cfg["warmup"])
    timed_iters = int(cfg["iters"])

    peak_rss = proc.memory_info().rss
    for i in range(warmup_iters):
        detector.detect(speed_frames[i % len(speed_frames)])
        if i % 10 == 0:
            peak_rss = max(peak_rss, proc.memory_info().rss)

    latencies_ms: List[float] = []
    cpu_start = proc.cpu_times()
    wall_start = time.perf_counter()

    for i in range(timed_iters):
        frame = speed_frames[i % len(speed_frames)]
        t0 = time.perf_counter()
        detector.detect(frame)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if i % 25 == 0:
            peak_rss = max(peak_rss, proc.memory_info().rss)

    wall_elapsed = max(time.perf_counter() - wall_start, 1e-6)
    cpu_end = proc.cpu_times()
    peak_rss = max(peak_rss, proc.memory_info().rss)

    cpu_seconds = max(0.0, (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system))
    cpu_sec_per_100 = (cpu_seconds / max(timed_iters, 1)) * 100.0
    avg_cpu_pct = (cpu_seconds / wall_elapsed) * 100.0

    lat_arr = np.asarray(latencies_ms, dtype=np.float64)
    lat_mean = float(np.mean(lat_arr))
    lat_p50 = float(np.percentile(lat_arr, 50))
    lat_p95 = float(np.percentile(lat_arr, 95))
    fps = 1000.0 / max(lat_mean, 1e-6)

    accuracy: Dict[str, Optional[float]] = {
        "evaluated_images": 0,
        "detections": 0,
        "map_50_95": None,
        "map_50": None,
        "ap_small": None,
        "ap_medium": None,
        "ap_large": None,
    }
    if val_images_dir and val_ann_path:
        accuracy = _evaluate_coco_accuracy(
            detector=detector,
            val_images_dir=val_images_dir,
            val_ann_path=val_ann_path,
            max_eval_images=int(cfg["max_eval_images"]),
            eval_conf=float(cfg["eval_conf"]),
            eval_iou=float(cfg["eval_iou"]),
            eval_max_det=int(cfg["eval_max_det"]),
        )
        peak_rss = max(peak_rss, proc.memory_info().rss)
    peak_rss = _get_peak_rss_bytes(proc, peak_rss)

    return {
        "name": cfg["name"],
        "model_xml": str(xml_path),
        "precision": detector.meta.get("precision", "UNKNOWN"),
        "model_variant": detector.meta.get("model_variant", "UNKNOWN"),
        "embedded_preprocess": detector.embedded_preprocess,
        "letterbox_embedded": detector.letterbox_embedded,
        "input_size": list(detector.input_size),
        "threads": int(cfg["threads"]),
        "performance_hint": str(cfg["performance_hint"]),
        "warmup": warmup_iters,
        "iters": timed_iters,
        "latency_mean_ms": round(lat_mean, 3),
        "latency_p50_ms": round(lat_p50, 3),
        "latency_p95_ms": round(lat_p95, 3),
        "fps": round(fps, 2),
        "avg_cpu_pct": round(avg_cpu_pct, 2),
        "cpu_sec_per_100_frames": round(cpu_sec_per_100, 4),
        "peak_rss_mb": round(peak_rss / (1024.0 * 1024.0), 2),
        "disk_size_mb": round(disk_mb, 2),
        **accuracy,
    }


def evaluate_go_no_go(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate Go/No-Go acceptance gates comparing candidate against vanilla baseline."""
    failures: List[str] = []

    base_map = baseline.get("map_50_95")
    cand_map = candidate.get("map_50_95")
    map_delta = None
    if base_map is not None and cand_map is not None:
        map_delta = round(cand_map - base_map, 3)
        if map_delta < -0.5:
            failures.append(f"mAP@0.5:0.95 delta {map_delta:+.3f}% < -0.500%")

    base_aps = baseline.get("ap_small")
    cand_aps = candidate.get("ap_small")
    aps_delta = None
    if base_aps is not None and cand_aps is not None:
        aps_delta = round(cand_aps - base_aps, 3)
        if aps_delta < -1.0:
            failures.append(f"AP_small delta {aps_delta:+.3f}% < -1.000%")

    base_p95 = baseline["latency_p95_ms"]
    cand_p95 = candidate["latency_p95_ms"]
    p95_speedup_pct = ((base_p95 - cand_p95) / max(base_p95, 1e-6)) * 100.0

    base_cpu = baseline["cpu_sec_per_100_frames"]
    cand_cpu = candidate["cpu_sec_per_100_frames"]
    cpu_reduction_pct = ((base_cpu - cand_cpu) / max(base_cpu, 1e-6)) * 100.0

    if p95_speedup_pct < 25.0 and cpu_reduction_pct < 25.0:
        failures.append(
            f"Neither Latency P95 speedup ({p95_speedup_pct:+.2f}%) nor CPU-sec/100f reduction "
            f"({cpu_reduction_pct:+.2f}%) reached >= +25.0%"
        )

    base_rss = baseline["peak_rss_mb"]
    cand_rss = candidate["peak_rss_mb"]
    rss_ratio_pct = (cand_rss / max(base_rss, 1e-6)) * 100.0
    if rss_ratio_pct > 115.0:
        failures.append(f"Peak RSS {rss_ratio_pct:.1f}% of baseline > 115.0%")

    verdict = "PASS (Adopt Upgraded)" if not failures else "ROLLBACK (Keep Vanilla)"
    return {
        "candidate": candidate["name"],
        "baseline": baseline["name"],
        "verdict": verdict,
        "passed": len(failures) == 0,
        "deltas": {
            "map_50_95_delta_pp": map_delta,
            "ap_small_delta_pp": aps_delta,
            "latency_p95_reduction_pct": round(p95_speedup_pct, 2),
            "cpu_sec_per_100_reduction_pct": round(cpu_reduction_pct, 2),
            "peak_rss_pct_of_baseline": round(rss_ratio_pct, 2),
        },
        "failed_criteria": failures,
    }


def _fmt_metric(val: Optional[float], suffix: str = "") -> str:
    if val is None:
        return "N/A"
    return f"{val:.2f}{suffix}"


def render_markdown_report(
    results: List[Dict[str, Any]],
    comparisons: List[Dict[str, Any]],
    baseline_name: str,
    controls: Dict[str, Any],
) -> str:
    lines: List[str] = [
        "# OpenVINO A/B Benchmark Report",
        "",
        "## Locked Execution Controls",
        f"- **Baseline:** `{baseline_name}`",
        f"- **INFERENCE_NUM_THREADS:** `{controls['threads']}` | **PERFORMANCE_HINT:** `{controls['performance_hint']}`",
        f"- **Warmup / Timed Iterations:** `{controls['warmup']}` / `{controls['iters']}`",
        f"- **Max Eval Images:** `{controls['max_eval_images'] or 'ALL'}`",
        "",
        "## Model Metrics",
        "| Model | Variant | Precision | PPP | mAP@0.5:0.95 | mAP@0.5 | AP_small | AP_med | AP_large | Mean (ms) | P50 (ms) | P95 (ms) | FPS | CPU % | CPU-s/100f | Peak RSS (MB) | Disk (MB) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['model_variant']} | {r['precision']} | {'Yes' if r['embedded_preprocess'] else 'No'} | "
            f"{_fmt_metric(r['map_50_95'], '%')} | {_fmt_metric(r['map_50'], '%')} | "
            f"{_fmt_metric(r['ap_small'], '%')} | {_fmt_metric(r['ap_medium'], '%')} | {_fmt_metric(r['ap_large'], '%')} | "
            f"{r['latency_mean_ms']:.2f} | {r['latency_p50_ms']:.2f} | {r['latency_p95_ms']:.2f} | {r['fps']:.2f} | "
            f"{r['avg_cpu_pct']:.1f}% | {r['cpu_sec_per_100_frames']:.3f} | {r['peak_rss_mb']:.1f} | {r['disk_size_mb']:.2f} |"
        )

    lines.extend(["", "## Automated Go / No-Go Verdict"])
    for cmp_item in comparisons:
        d = cmp_item["deltas"]
        map_d = f"{d['map_50_95_delta_pp']:+.3f} pp" if d["map_50_95_delta_pp"] is not None else "N/A"
        aps_d = f"{d['ap_small_delta_pp']:+.3f} pp" if d["ap_small_delta_pp"] is not None else "N/A"
        lines.append(f"### `{cmp_item['candidate']}` vs `{cmp_item['baseline']}` -> **{cmp_item['verdict']}**")
        lines.append(
            f"- mAP@0.5:0.95 delta: `{map_d}` | AP_small delta: `{aps_d}` | "
            f"P95 latency speedup: `{d['latency_p95_reduction_pct']:+.2f}%` | "
            f"CPU-s/100f reduction: `{d['cpu_sec_per_100_reduction_pct']:+.2f}%` | "
            f"Peak RSS vs baseline: `{d['peak_rss_pct_of_baseline']:.2f}%`"
        )
        if cmp_item["failed_criteria"]:
            lines.append("- Failed criteria:")
            for fail in cmp_item["failed_criteria"]:
                lines.append(f"  - {fail}")
    lines.append("")
    return "\n".join(lines)


def run_benchmark_suite(
    *,
    manifest_path: Optional[Path] = None,
    model_specs: Optional[List[str]] = None,
    val_images: Optional[Path] = None,
    val_ann: Optional[Path] = None,
    max_eval_images: int = 0,
    threads: int = 4,
    performance_hint: str = "LATENCY",
    warmup: int = 50,
    iters: int = 300,
    bench_conf: float = 0.25,
    bench_iou: float = 0.45,
    eval_conf: float = 0.001,
    eval_iou: float = 0.65,
    eval_max_det: int = 300,
    output_md: Path = Path("benchmark_report.md"),
    output_json: Path = Path("benchmark_report.json"),
    isolated: bool = True,
) -> Dict[str, Any]:
    baseline_name, entries = _load_manifest(manifest_path, model_specs)
    results: List[Dict[str, Any]] = []

    for entry in entries:
        worker_cfg = {
            "name": entry["name"],
            "model_xml": str(entry["model_xml"]),
            "letterbox_embedded": entry.get("letterbox_embedded"),
            "val_images": str(val_images) if val_images else None,
            "val_ann": str(val_ann) if val_ann else None,
            "max_eval_images": int(max_eval_images),
            "threads": int(threads),
            "performance_hint": performance_hint,
            "warmup": int(warmup),
            "iters": int(iters),
            "bench_conf": float(bench_conf),
            "bench_iou": float(bench_iou),
            "eval_conf": float(eval_conf),
            "eval_iou": float(eval_iou),
            "eval_max_det": int(eval_max_det),
        }
        if isolated:
            with tempfile.TemporaryDirectory() as tmpd:
                cfg_file = Path(tmpd) / "worker_cfg.json"
                out_file = Path(tmpd) / "worker_out.json"
                cfg_file.write_text(json.dumps(worker_cfg), encoding="utf-8")
                cmd = [sys.executable, str(Path(__file__).resolve()), "--worker-cfg", str(cfg_file), "--worker-out", str(out_file)]
                proc = subprocess.run(cmd, check=False)
                if proc.returncode != 0 or not out_file.is_file():
                    raise RuntimeError(f"Benchmark worker failed for {entry['name']} (exit={proc.returncode})")
                res = json.loads(out_file.read_text(encoding="utf-8"))
        else:
            res = benchmark_single_model(worker_cfg)
        res["compare_to"] = entry.get("compare_to")
        results.append(res)

    by_name = {r["name"]: r for r in results}
    if baseline_name not in by_name:
        raise ValueError(f"Baseline '{baseline_name}' not found among benchmarked models: {list(by_name)}")
    baseline_res = by_name[baseline_name]

    comparisons = []
    for r in results:
        ref_name = r.get("compare_to")
        if r["name"] == baseline_name and not ref_name:
            continue
        if ref_name and ref_name in by_name:
            comparisons.append(evaluate_go_no_go(by_name[ref_name], r))
        elif r["name"] != baseline_name and not any(e.get("baseline") and e["name"] == r["name"] for e in entries):
            comparisons.append(evaluate_go_no_go(baseline_res, r))

    controls = {
        "threads": int(threads),
        "performance_hint": performance_hint,
        "warmup": int(warmup),
        "iters": int(iters),
        "max_eval_images": int(max_eval_images),
        "bench_conf": float(bench_conf),
        "bench_iou": float(bench_iou),
        "eval_conf": float(eval_conf),
        "eval_iou": float(eval_iou),
    }

    report_payload = {
        "baseline": baseline_name,
        "controls": controls,
        "results": results,
        "comparisons": comparisons,
    }
    md_text = render_markdown_report(results, comparisons, baseline_name, controls)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md_text, encoding="utf-8")
    output_json.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    return report_payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Automated OpenVINO IR A/B benchmark & Go/No-Go gate.")
    parser.add_argument("--manifest", type=Path, help="YAML or JSON manifest listing baseline & candidate models.")
    parser.add_argument("--models", nargs="+", help="Inline model specs: name=path.xml ...")
    parser.add_argument("--val-images", type=Path, default=None, help="Validation images directory.")
    parser.add_argument("--val-ann", type=Path, default=None, help="COCO validation annotation JSON.")
    parser.add_argument("--max-eval-images", type=int, default=0, help="Max images for COCO eval (0 = all).")
    parser.add_argument("--threads", type=int, default=4, help="Locked INFERENCE_NUM_THREADS (default: 4).")
    parser.add_argument("--performance-hint", default="LATENCY", help="Locked PERFORMANCE_HINT (default: LATENCY).")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations (default: 50).")
    parser.add_argument("--iters", type=int, default=300, help="Timed iterations (default: 300).")
    parser.add_argument("--bench-conf", type=float, default=0.25, help="Inference timing confidence threshold.")
    parser.add_argument("--bench-iou", type=float, default=0.45, help="Inference timing NMS IoU threshold.")
    parser.add_argument("--eval-conf", type=float, default=0.001, help="COCO eval confidence threshold.")
    parser.add_argument("--eval-iou", type=float, default=0.65, help="COCO eval NMS IoU threshold.")
    parser.add_argument("--eval-max-det", type=int, default=300, help="COCO eval max detections per image.")
    parser.add_argument("--output-md", type=Path, default=Path("benchmark_report.md"), help="Output Markdown path.")
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_report.json"), help="Output JSON path.")
    parser.add_argument("--no-isolate", action="store_true", help="Run all models in-process instead of isolated subprocesses.")
    parser.add_argument("--worker-cfg", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker_cfg and args.worker_out:
        cfg = json.loads(args.worker_cfg.read_text(encoding="utf-8"))
        res = benchmark_single_model(cfg)
        args.worker_out.write_text(json.dumps(res), encoding="utf-8")
        return 0

    report = run_benchmark_suite(
        manifest_path=args.manifest,
        model_specs=args.models,
        val_images=args.val_images,
        val_ann=args.val_ann,
        max_eval_images=args.max_eval_images,
        threads=args.threads,
        performance_hint=args.performance_hint,
        warmup=args.warmup,
        iters=args.iters,
        bench_conf=args.bench_conf,
        bench_iou=args.bench_iou,
        eval_conf=args.eval_conf,
        eval_iou=args.eval_iou,
        eval_max_det=args.eval_max_det,
        output_md=args.output_md,
        output_json=args.output_json,
        isolated=not args.no_isolate,
    )
    print(args.output_md.read_text(encoding="utf-8"))
    print(f"[OK] Saved benchmark reports -> {args.output_md}, {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
