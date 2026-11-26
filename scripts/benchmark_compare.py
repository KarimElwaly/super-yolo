"""Automated A/B benchmark suite for OpenVINO IR models with locked execution controls and Go/No-Go verdict."""

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


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _get_peak_rss_mb(proc: psutil.Process) -> float:
    mem = proc.memory_info()
    peak_bytes = getattr(mem, "peak_wset", None)
    if peak_bytes is None:
        try:
            import resource

            peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except Exception:
            peak_bytes = mem.rss
    return float(peak_bytes) / (1024.0 * 1024.0)


def _model_disk_size_mb(xml_path: Path) -> float:
    total = xml_path.stat().st_size if xml_path.is_file() else 0
    bin_path = xml_path.with_suffix(".bin")
    if bin_path.is_file():
        total += bin_path.stat().st_size
    return round(total / (1024.0 * 1024.0), 3)


def _load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    raw_text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        data = json.loads(raw_text)
    else:
        data = yaml.safe_load(raw_text)

    entries = data["models"] if isinstance(data, dict) and "models" in data else data
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest {manifest_path} must contain a non-empty list of model entries.")

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(entries):
        xml_raw = item.get("xml_path") or item.get("model_xml") or item.get("path")
        if not xml_raw:
            raise ValueError(f"Manifest entry #{idx} missing 'xml_path': {item}")
        xml_p = Path(xml_raw)
        if not xml_p.is_absolute() and not xml_p.is_file():
            rel_cand = (manifest_path.parent / xml_p).resolve()
            if rel_cand.is_file():
                xml_p = rel_cand
        normalized.append(
            {
                "name": str(item.get("name") or xml_p.stem),
                "xml_path": str(xml_p),
                "baseline": bool(item.get("baseline", False)),
            }
        )
    return normalized


def _evaluate_coco_accuracy(
    detector: OpenVINODetector,
    val_images_dir: Path,
    val_ann_path: Path,
    max_eval_images: int,
    eval_conf: float,
    eval_iou: float,
) -> Dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(val_ann_path))

    cat_ids = sorted(c["id"] for c in coco_gt.dataset.get("categories", []))
    all_img_ids = sorted(coco_gt.getImgIds())
    if max_eval_images > 0:
        eval_img_ids = all_img_ids[:max_eval_images]
    else:
        eval_img_ids = all_img_ids

    coco_results: List[Dict[str, Any]] = []
    evaluated_ids: List[int] = []

    for img_id in eval_img_ids:
        img_info = coco_gt.loadImgs([img_id])[0]
        img_path = val_images_dir / img_info["file_name"]
        if not img_path.is_file():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        evaluated_ids.append(img_id)

        dets = detector.detect(frame, conf_threshold=eval_conf, iou_threshold=eval_iou)
        if len(dets) == 0:
            continue
        xyxy = dets.xyxy
        confs = dets.confidence
        cls_ids = dets.class_id
        for box, score, cidx in zip(xyxy, confs, cls_ids):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cat_id = int(cat_ids[int(cidx)]) if int(cidx) < len(cat_ids) else int(cidx)
            coco_results.append(
                {
                    "image_id": int(img_id),
                    "category_id": cat_id,
                    "bbox": [round(x1, 3), round(y1, 3), round(w, 3), round(h, 3)],
                    "score": round(float(score), 6),
                }
            )

    if not evaluated_ids:
        raise RuntimeError(f"Zero validation images matched between {val_ann_path} and {val_images_dir}")

    if not coco_results:
        return {
            "evaluated_images": len(evaluated_ids),
            "detections_count": 0,
            "mAP_50_95": 0.0,
            "mAP_50": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
        }

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = evaluated_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = [max(0.0, float(s)) * 100.0 for s in coco_eval.stats]
    return {
        "evaluated_images": len(evaluated_ids),
        "detections_count": len(coco_results),
        "mAP_50_95": round(stats[0], 3),
        "mAP_50": round(stats[1], 3),
        "AP_small": round(stats[3], 3),
        "AP_medium": round(stats[4], 3),
        "AP_large": round(stats[5], 3),
    }


def _benchmark_single_model_in_process(
    *,
    name: str,
    xml_path: Path,
    num_threads: int,
    performance_hint: str,
    warmup: int,
    iters: int,
    bench_image_path: Optional[Path],
    val_images_dir: Optional[Path],
    val_ann_path: Optional[Path],
    max_eval_images: int,
    eval_conf: float,
    eval_iou: float,
) -> Dict[str, Any]:
    detector = OpenVINODetector(
        model_xml_path=xml_path,
        num_threads=num_threads,
        performance_hint=performance_hint,
        conf_threshold=eval_conf,
        iou_threshold=eval_iou,
    )

    if bench_image_path and bench_image_path.is_file():
        sample_frame = cv2.imread(str(bench_image_path))
    elif val_images_dir and val_images_dir.is_dir():
        first_img = next((p for p in sorted(val_images_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), None)
        sample_frame = cv2.imread(str(first_img)) if first_img else None
    else:
        sample_frame = None

    if sample_frame is None:
        w, h = detector.input_size
        rng = np.random.default_rng(42)
        sample_frame = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

    # Warmup (excluded from timing & CPU counters)
    for _ in range(max(0, warmup)):
        tensor, _ = detector.prepare_input(sample_frame)
        detector.infer_raw(tensor)

    proc = psutil.Process(os.getpid())
    latencies_ms: List[float] = []

    cpu_before = proc.cpu_times()
    wall_start = time.perf_counter()
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        tensor, _ = detector.prepare_input(sample_frame)
        detector.infer_raw(tensor)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
    wall_elapsed = max(time.perf_counter() - wall_start, 1e-9)
    cpu_after = proc.cpu_times()

    cpu_seconds = max(0.0, (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system))
    avg_cpu_pct = (cpu_seconds / wall_elapsed) * 100.0
    cpu_sec_per_100 = cpu_seconds * (100.0 / max(1, iters))

    lat_arr = np.asarray(latencies_ms, dtype=np.float64)
    lat_mean = float(np.mean(lat_arr))
    lat_p50 = float(np.percentile(lat_arr, 50))
    lat_p95 = float(np.percentile(lat_arr, 95))
    fps = 1000.0 / max(lat_mean, 1e-6)

    accuracy = None
    if val_images_dir and val_ann_path:
        accuracy = _evaluate_coco_accuracy(
            detector=detector,
            val_images_dir=val_images_dir,
            val_ann_path=val_ann_path,
            max_eval_images=max_eval_images,
            eval_conf=eval_conf,
            eval_iou=eval_iou,
        )

    peak_rss_mb = _get_peak_rss_mb(proc)
    disk_size_mb = _model_disk_size_mb(xml_path)

    return {
        "name": name,
        "xml_path": str(xml_path),
        "model_variant": detector.meta.get("model_variant", xml_path.stem),
        "precision": detector.meta.get("precision", "unknown"),
        "input_size": list(detector.input_size),
        "embedded_preprocess": detector.embedded_preprocess,
        "accuracy": accuracy,
        "latency_mean_ms": round(lat_mean, 3),
        "latency_p50_ms": round(lat_p50, 3),
        "latency_p95_ms": round(lat_p95, 3),
        "throughput_fps": round(fps, 2),
        "avg_process_cpu_pct": round(avg_cpu_pct, 2),
        "cpu_sec_per_100_frames": round(cpu_sec_per_100, 4),
        "peak_rss_mb": round(peak_rss_mb, 2),
        "disk_size_mb": disk_size_mb,
    }


def _run_isolated_worker(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp_out:
        out_json = Path(tmp_out.name)
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-json",
            json.dumps({**args_dict, "worker_output": str(out_json)}),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Benchmark worker failed for {args_dict['name']} (exit {proc.returncode}):\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return json.loads(out_json.read_text(encoding="utf-8"))
    finally:
        out_json.unlink(missing_ok=True)


def _evaluate_go_no_go(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []

    base_p95 = max(baseline["latency_p95_ms"], 1e-9)
    cand_p95 = candidate["latency_p95_ms"]
    p95_speedup_pct = (base_p95 - cand_p95) / base_p95 * 100.0

    base_cpu = max(baseline["cpu_sec_per_100_frames"], 1e-9)
    cand_cpu = candidate["cpu_sec_per_100_frames"]
    cpu_reduction_pct = (base_cpu - cand_cpu) / base_cpu * 100.0

    base_rss = max(baseline["peak_rss_mb"], 1e-9)
    cand_rss = candidate["peak_rss_mb"]
    rss_ratio_pct = (cand_rss / base_rss) * 100.0

    deltas: Dict[str, Any] = {
        "latency_p95_speedup_pct": round(p95_speedup_pct, 2),
        "cpu_sec_reduction_pct": round(cpu_reduction_pct, 2),
        "peak_rss_pct_of_baseline": round(rss_ratio_pct, 2),
        "disk_size_reduction_pct": round((1.0 - candidate["disk_size_mb"] / max(baseline["disk_size_mb"], 1e-9)) * 100.0, 2),
    }

    acc_base = baseline.get("accuracy")
    acc_cand = candidate.get("accuracy")
    if acc_base is not None and acc_cand is not None:
        map_delta = round(acc_cand["mAP_50_95"] - acc_base["mAP_50_95"], 3)
        map50_delta = round(acc_cand["mAP_50"] - acc_base["mAP_50"], 3)
        aps_delta = round(acc_cand["AP_small"] - acc_base["AP_small"], 3)
        deltas["mAP_50_95_delta"] = map_delta
        deltas["mAP_50_delta"] = map50_delta
        deltas["AP_small_delta"] = aps_delta
        if map_delta < -0.5:
            reasons.append(f"mAP@0.5:0.95 dropped {map_delta:+.3f}% (< -0.5% gate)")
        if aps_delta < -1.0:
            reasons.append(f"AP_small dropped {aps_delta:+.3f}% (< -1.0% gate)")

    if p95_speedup_pct < 25.0 and cpu_reduction_pct < 25.0:
        reasons.append(
            f"Neither P95 latency speedup ({p95_speedup_pct:+.2f}%) nor CPU-sec reduction ({cpu_reduction_pct:+.2f}%) reached >= 25.0%"
        )
    if rss_ratio_pct > 115.0:
        reasons.append(f"Peak RSS {rss_ratio_pct:.2f}% exceeds 115.0% baseline limit")

    verdict = "PASS (Adopt Upgraded)" if not reasons else "ROLLBACK (Keep Vanilla)"
    return {"verdict": verdict, "reasons": reasons, "deltas": deltas}


def _render_markdown_report(report: Dict[str, Any]) -> str:
    ctrl = report["controls"]
    lines = [
        "# OpenVINO A/B Benchmark Report",
        "",
        "## Locked Execution Controls",
        f"- **Device:** `{ctrl['device']}` | **INFERENCE_NUM_THREADS:** `{ctrl['num_threads']}` | **PERFORMANCE_HINT:** `{ctrl['performance_hint']}`",
        f"- **Warmup Iters:** `{ctrl['warmup']}` | **Timed Iters:** `{ctrl['iters']}` | **Eval Images:** `{ctrl['eval_images']}` | **Eval Conf/IoU:** `{ctrl['eval_conf']}/{ctrl['eval_iou']}`",
        f"- **Baseline Model:** `{report['baseline']}`",
        "",
        "## Model Metrics",
        "",
        "| Model | Precision | PPP | mAP@0.5:0.95 | mAP@0.5 | AP_small | Latency Mean (ms) | Latency P50 (ms) | Latency P95 (ms) | FPS | Avg CPU % | CPU-s / 100f | Peak RSS (MB) | Disk (MB) | Verdict |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for m in report["models"]:
        acc = m.get("accuracy")
        map_str = f"{acc['mAP_50_95']:.2f}%" if acc else "N/A"
        map50_str = f"{acc['mAP_50']:.2f}%" if acc else "N/A"
        aps_str = f"{acc['AP_small']:.2f}%" if acc else "N/A"
        verdict = m.get("gate", {}).get("verdict", "BASELINE")
        lines.append(
            f"| {m['name']} | {m['precision']} | {m['embedded_preprocess']} | "
            f"{map_str} | {map50_str} | {aps_str} | "
            f"{m['latency_mean_ms']:.2f} | {m['latency_p50_ms']:.2f} | {m['latency_p95_ms']:.2f} | "
            f"{m['throughput_fps']:.2f} | {m['avg_process_cpu_pct']:.1f}% | {m['cpu_sec_per_100_frames']:.4f} | "
            f"{m['peak_rss_mb']:.1f} | {m['disk_size_mb']:.2f} | {verdict} |"
        )

    lines.extend(["", "## Candidate Deltas & Automated Gate Decisions", ""])
    for m in report["models"]:
        if "gate" not in m:
            continue
        gate = m["gate"]
        d = gate["deltas"]
        lines.append(f"### {m['name']} -> **{gate['verdict']}**")
        if "mAP_50_95_delta" in d:
            lines.append(
                f"- Accuracy Delta: `mAP@0.5:0.95 {d['mAP_50_95_delta']:+.3f}%`, `mAP@0.5 {d['mAP_50_delta']:+.3f}%`, `AP_small {d['AP_small_delta']:+.3f}%`"
            )
        lines.append(
            f"- Efficiency Delta: `P95 Latency Speedup {d['latency_p95_speedup_pct']:+.2f}%`, "
            f"`CPU-s/100f Reduction {d['cpu_sec_reduction_pct']:+.2f}%`, "
            f"`Peak RSS {d['peak_rss_pct_of_baseline']:.2f}% of baseline`, "
            f"`Disk Size Reduction {d['disk_size_reduction_pct']:+.2f}%`"
        )
        if gate["reasons"]:
            lines.append("- Failed Gate Criteria:")
            for r in gate["reasons"]:
                lines.append(f"  - {r}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_benchmark_suite(
    *,
    manifest_path: Path,
    output_dir: Path,
    report_prefix: str = "benchmark_report",
    baseline_name: Optional[str] = None,
    num_threads: int = 4,
    performance_hint: str = "LATENCY",
    warmup: int = 50,
    iters: int = 300,
    val_images: Optional[Path] = None,
    val_ann: Optional[Path] = None,
    max_eval_images: int = 0,
    eval_conf: float = 0.001,
    eval_iou: float = 0.65,
    bench_image: Optional[Path] = None,
) -> Dict[str, Any]:
    entries = _load_manifest(manifest_path)

    if baseline_name:
        for e in entries:
            e["baseline"] = (e["name"] == baseline_name)
    if not any(e["baseline"] for e in entries):
        entries[0]["baseline"] = True

    results: List[Dict[str, Any]] = []
    for entry in entries:
        worker_args = {
            "name": entry["name"],
            "xml_path": str(Path(entry["xml_path"]).resolve()),
            "num_threads": num_threads,
            "performance_hint": performance_hint,
            "warmup": warmup,
            "iters": iters,
            "bench_image_path": str(bench_image.resolve()) if bench_image else None,
            "val_images_dir": str(val_images.resolve()) if val_images else None,
            "val_ann_path": str(val_ann.resolve()) if val_ann else None,
            "max_eval_images": max_eval_images,
            "eval_conf": eval_conf,
            "eval_iou": eval_iou,
        }
        res = _run_isolated_worker(worker_args)
        res["is_baseline"] = entry["baseline"]
        results.append(res)

    baseline_res = next(r for r in results if r["is_baseline"])
    for res in results:
        if not res["is_baseline"]:
            res["gate"] = _evaluate_go_no_go(baseline_res, res)

    eval_img_count = baseline_res["accuracy"]["evaluated_images"] if baseline_res.get("accuracy") else 0
    report = {
        "controls": {
            "device": "CPU",
            "num_threads": num_threads,
            "performance_hint": performance_hint,
            "warmup": warmup,
            "iters": iters,
            "max_eval_images": max_eval_images,
            "eval_images": eval_img_count,
            "eval_conf": eval_conf,
            "eval_iou": eval_iou,
        },
        "baseline": baseline_res["name"],
        "models": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{report_prefix}.json"
    md_path = output_dir / f"{report_prefix}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown_report(report), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Automated OpenVINO A/B benchmark suite with Go/No-Go gate.")
    parser.add_argument("--manifest", type=Path, help="JSON/YAML manifest listing exported OpenVINO models.")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for benchmark_report.md/.json.")
    parser.add_argument("--report-prefix", default="benchmark_report", help="Filename stem for .md/.json reports.")
    parser.add_argument("--baseline", default=None, help="Explicit baseline model name from manifest.")
    parser.add_argument("--num-threads", type=int, default=4, help="Locked INFERENCE_NUM_THREADS (default: 4).")
    parser.add_argument("--performance-hint", default="LATENCY", help="Locked PERFORMANCE_HINT (default: LATENCY).")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations (default: 50).")
    parser.add_argument("--iters", type=int, default=300, help="Timed iterations (default: 300).")
    parser.add_argument("--val-images", type=Path, default=None, help="Validation images directory for COCO mAP.")
    parser.add_argument("--val-ann", type=Path, default=None, help="Validation COCO annotation JSON for mAP.")
    parser.add_argument("--max-eval-images", type=int, default=0, help="Cap evaluation images (0 = all).")
    parser.add_argument("--eval-conf", type=float, default=0.001, help="Confidence threshold for COCO mAP (default: 0.001).")
    parser.add_argument("--eval-iou", type=float, default=0.65, help="NMS IoU threshold for COCO mAP (default: 0.65).")
    parser.add_argument("--bench-image", type=Path, default=None, help="Optional fixed frame for latency timing.")
    parser.add_argument("--worker-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker_json:
        w = json.loads(args.worker_json)
        res = _benchmark_single_model_in_process(
            name=w["name"],
            xml_path=Path(w["xml_path"]),
            num_threads=int(w["num_threads"]),
            performance_hint=w["performance_hint"],
            warmup=int(w["warmup"]),
            iters=int(w["iters"]),
            bench_image_path=Path(w["bench_image_path"]) if w.get("bench_image_path") else None,
            val_images_dir=Path(w["val_images_dir"]) if w.get("val_images_dir") else None,
            val_ann_path=Path(w["val_ann_path"]) if w.get("val_ann_path") else None,
            max_eval_images=int(w["max_eval_images"]),
            eval_conf=float(w["eval_conf"]),
            eval_iou=float(w["eval_iou"]),
        )
        Path(w["worker_output"]).write_text(json.dumps(res), encoding="utf-8")
        return 0

    if not args.manifest:
        parser.error("--manifest is required.")

    report = run_benchmark_suite(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        report_prefix=args.report_prefix,
        baseline_name=args.baseline,
        num_threads=args.num_threads,
        performance_hint=args.performance_hint,
        warmup=args.warmup,
        iters=args.iters,
        val_images=args.val_images,
        val_ann=args.val_ann,
        max_eval_images=args.max_eval_images,
        eval_conf=args.eval_conf,
        eval_iou=args.eval_iou,
        bench_image=args.bench_image,
    )
    print(f"[OK] Benchmark reports written:\n  - {report['md_path']}\n  - {report['json_path']}")
    for m in report["models"]:
        if "gate" in m:
            print(f"  Verdict [{m['name']}]: {m['gate']['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
