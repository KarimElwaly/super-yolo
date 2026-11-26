# YOLOv9 CPU-Optimized & High-Accuracy Upgrade — Handoff Guide

Branch: `feat/yolov9-cpu-upgrade`
Isolated Environment: `.venv` (Python 3.11.9, MIT/Apache-2.0/BSD-clean)

---

## 1. Summary of Added & Modified Files

### Dataset Tooling & License Audit (Phase 1)
- [scripts/check_licenses.py](scripts/check_licenses.py) — Audits every package in `.venv` via `importlib.metadata` + `.dist-info` license text fallback; explicitly bans `ultralytics`, `boxmot`, and strong `GPL`/`AGPL` dependencies.
- [scripts/convert_yolo_to_coco.py](scripts/convert_yolo_to_coco.py) — Converts Ultralytics YOLO `.txt` datasets (`images/`, `labels/`, `data.yaml`) into COCO JSON (`annotations/instances_<split>.json`), with `--verify` box/area/class validation and sample visualization.
- [scripts/filter_coco_classes.py](scripts/filter_coco_classes.py) — Filters COCO JSON datasets to target classes, remaps category IDs contiguously (`0..K-1`), transfers images, and emits a matching Hydra dataset YAML.
- [yolo/config/dataset/custom_template.yaml](yolo/config/dataset/custom_template.yaml) — Documented Hydra template for custom domain datasets.

### Additive CPU-Optimized & Small-Object Architectures (Phase 2)
- [yolo/config/model/v9-t-opt.yaml](yolo/config/model/v9-t-opt.yaml), [yolo/config/model/v9-s-opt.yaml](yolo/config/model/v9-s-opt.yaml) — CPU/INT8-friendly `ReLU` variants with explicit `strides: [8, 16, 32]`.
- [yolo/config/model/v9-m-opt.yaml](yolo/config/model/v9-m-opt.yaml) — `HardSwish` medium variant with explicit `strides: [8, 16, 32]`.
- [yolo/config/model/v9-t-opt-p2.yaml](yolo/config/model/v9-t-opt-p2.yaml), [yolo/config/model/v9-s-opt-p2.yaml](yolo/config/model/v9-s-opt-p2.yaml) — Stride-4 `P2` small-object variants (`strides: [4, 8, 16, 32]`) with `ReLU` and trimmed `P5` channels.
- [yolo/model/module.py](yolo/model/module.py), [yolo/utils/module_utils.py](yolo/utils/module_utils.py) — Config-driven `activation` propagation (`SiLU` default preserved for all vanilla configs; `ReLU`/`HardSwish` supported across backbone, neck, and heads) plus `Detection.class_neck` parameter.
- [yolo/model/yolo.py](yolo/model/yolo.py) — Added `strip_auxiliary_branches()` (PGI removal at export), `load_partial_weights()` (`strict=False` shape-matched transfer), opt-in `partial_load` in `create_model()`, and `_maybe_adapt_head_class_neck()` for `v1.0-alpha/v9-t.pt` (`class_neck=80`) compatibility.
- [yolo/config/config.py](yolo/config/config.py), [yolo/config/general.yaml](yolo/config/general.yaml), [yolo/utils/bounding_box_utils.py](yolo/utils/bounding_box_utils.py) — Opt-in `partial_load` / `partial_weights` config fields and device-aware `Vec2Box.create_auto_anchor`.

### Training-Time Accuracy Boosters & Pruning (Phase 3)
- [yolo/tools/distillation.py](yolo/tools/distillation.py) — Opt-in Teacher-Student Knowledge Distillation (`KnowledgeDistillationLoss` with hook-based multi-scale neck CWD + normalized MSE via 1x1 adapters and spatial-matched head logit/DFL KL; frozen teacher and adapters kept outside `self.model`).
- [yolo/config/task/train.yaml](yolo/config/task/train.yaml), [yolo/tools/solver.py](yolo/tools/solver.py), [yolo/tools/loss_functions.py](yolo/tools/loss_functions.py), [yolo/utils/model_utils.py](yolo/utils/model_utils.py) — Wired opt-in `distill`, `close_mosaic`, `multi_scale`, `multi_scale_range`, standalone pruned checkpoint loading, checkpoint KD exclusion (`on_save_checkpoint`), and dynamic `DFLoss` anchor grid syncing.
- [scripts/prune_model.py](scripts/prune_model.py) — Standalone structured channel pruning CLI using MIT-licensed `torch-pruning` (`l1` / `taylor`, head & chunk boundary protection, MACs/params reporting, standalone pruned checkpoint export).

### OpenVINO Export, Unified Detector, Benchmark & Tests (Phases 4–7)
- [scripts/export_openvino.py](scripts/export_openvino.py) — Strips PGI auxiliary branches, exports opset-17 ONNX (`dynamo=False` + `onnxsim`), verifies zero aux nodes + PyTorch/ORT/OpenVINO numerical parity, converts to `FP16` or NNCF `INT8` (`--protect-heads` `IgnoredScope` + `--embed-preprocess` `PrePostProcessor`), and writes `.meta.json` / `export_meta.json`.
- [inference/detector_ov.py](inference/detector_ov.py), [inference/__init__.py](inference/__init__.py) — `OpenVINODetector` supporting both manual letterbox and embedded `uint8` `NHWC` BGR models, OpenCV `cv2.dnn.NMSBoxes`, and `supervision.Detections` output.
- [inference/demo_stream_count.py](inference/demo_stream_count.py) — Threaded RTSP/video frame grabber + `OpenVINODetector` + MIT-clean `SimpleByteTrack` fallback + `sv.LineZone` / `sv.PolygonZone` + `--detect-every-n`.
- [scripts/benchmark_compare.py](scripts/benchmark_compare.py) — Isolated worker A/B benchmark suite (`pycocotools` mAP, latency Mean/P50/P95/FPS, `psutil` CPU % & CPU-seconds/100f, Peak RSS, disk size, and `PASS (Adopt Upgraded)` / `ROLLBACK (Keep Vanilla)` gate).
- [tests/test_upgrade_smoke.py](tests/test_upgrade_smoke.py) — End-to-end smoke test suite covering all 10 verification criteria (14 parametrized test cases across 8 test functions, including license policy enforcement, `YoloDataset` load, partial & pruned `BaseModel` reloads, `SimpleByteTrack` + `run_stream_pipeline`, and Go/No-Go gates).
- [.gitignore](.gitignore), [PROGRESS.md](PROGRESS.md) — Excludes `.venv/`, `data/`, `weights/`, `exports/`, `tmp/`, `runs/`, and checkpoints; tracks phase-by-phase progress and commit hashes.

---

## 2. Verification Results

- **License Audit:** `.venv\Scripts\python.exe scripts/check_licenses.py` -> exited `0` (`Scanned 109 installed packages. PASSED: All 109 packages are compliant (0 GPL/AGPL, 0 banned packages).`).
- **End-to-End Smoke Suite:** `.venv\Scripts\python.exe -m pytest tests/test_upgrade_smoke.py -v -p no:cacheprovider` -> **`14 passed`** (8 test functions / 14 parametrized cases, 0 failed).
- **Upstream Baseline Unit Tests:** `.venv\Scripts\python.exe -m pytest tests/test_model tests/test_utils tests/test_tools/test_data_augmentation.py tests/test_tools/test_loss_functions.py -v -p no:cacheprovider` -> **`43 passed`** (0 failed).

---

## 3. Real OpenVINO Benchmark on COCO `val2017` (500 Images, CPU)

Source report: [benchmark_report_coco_pretrained.md](benchmark_report_coco_pretrained.md) ([JSON](benchmark_report_coco_pretrained.json))

### Weight Provenance & License Notes
- Official pretrained weights downloaded into ignored `weights/`:
  - `v9-t.pt` (`12,244,990` bytes): `https://github.com/MultimediaTechLab/YOLO/releases/download/v1.0-alpha/v9-t.pt`
  - `v9-s.pt` (`39,930,046` bytes): `https://github.com/MultimediaTechLab/YOLO/releases/download/v1.0-alpha/v9-s.pt`
  - Repository license: **MIT License** (`LICENSE`).
- Evaluation data: COCO `val2017` (`http://images.cocodataset.org/zips/val2017.zip`, `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`) in ignored `data/coco_val2017/`.
- Locked controls: `device=CPU`, `INFERENCE_NUM_THREADS=4`, `PERFORMANCE_HINT=LATENCY`, `input_size=640x640`, `warmup=30`, `iters=200`, `eval_images=500`, NNCF calibration on `250` real `val2017` images.

### Measured OpenVINO IR Metrics

| Model | Precision | PPP | mAP@0.5:0.95 | mAP@0.5 | AP_small | Mean (ms) | P50 (ms) | P95 (ms) | FPS | CPU % | CPU-s/100f | Peak RSS (MB) | Disk (MB) | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v9-t-fp16` | FP16 | No | 41.39% | 56.13% | 22.92% | 57.34 | 57.45 | 68.99 | 17.44 | 254.5% | 14.594 | 533.9 | 4.92 | BASELINE |
| `v9-t-int8-ppp` | INT8 | Yes | 38.57% | 53.58% | 20.38% | 19.77 | 19.63 | 21.37 | 50.59 | 338.2% | 6.688 | 537.9 | 3.86 | ROLLBACK (`-2.823 pp`) |
| `v9-t-int8-ppp-protect-heads` | INT8 | Yes | **41.11%** | **56.09%** | **21.99%** | **26.70** | **26.95** | **33.84** | **37.46** | 271.2% | **7.242** | 559.9 | **3.89** | **PASS (Adopt Upgraded)** |
| `v9-s-fp16` | FP16 | No | 50.92% | 67.24% | 32.59% | 102.61 | 100.34 | 117.08 | 9.75 | 360.0% | 36.938 | 540.4 | 14.81 | BASELINE |
| `v9-s-int8-ppp` | INT8 | Yes | 48.57% | 65.82% | 30.43% | 41.44 | 38.50 | 54.30 | 24.13 | 346.8% | 14.375 | 553.1 | 8.85 | ROLLBACK (`-2.354 pp`) |
| `v9-s-int8-ppp-protect-heads` | INT8 | Yes | **50.70%** | **67.39%** | **32.35%** | **48.07** | **48.22** | **57.52** | **20.80** | 302.8% | **14.555** | 545.2 | **8.92** | **PASS (Adopt Upgraded)** |

Key takeaways:
- Unprotected INT8 triggered the automated `ROLLBACK` gate (`-2.82 pp` on `v9-t`, `-2.35 pp` on `v9-s`), automatically triggering `--protect-heads` fallback.
- With `--protect-heads`, `v9-t-int8-ppp-protect-heads` (`-0.284 pp` mAP, `+50.96%` P95 speedup, `+50.37%` CPU reduction, `104.88%` RSS) and `v9-s-int8-ppp-protect-heads` (`-0.223 pp` mAP, `+50.87%` P95 speedup, `+60.60%` CPU reduction, `100.89%` RSS) both **PASS** every acceptance gate.

Command to run the **full 5,000-image** COCO `val2017` evaluation:
```powershell
.venv\Scripts\python.exe scripts/benchmark_compare.py `
  --manifest exports/coco_bench/manifest_all.yaml `
  --val-images data/coco_val2017/images/val2017 `
  --val-ann data/coco_val2017/annotations/instances_val2017.json `
  --max-eval-images 0 --threads 4 --warmup 50 --iters 300 `
  --output-md benchmark_report_coco_pretrained_full5k.md `
  --output-json benchmark_report_coco_pretrained_full5k.json
```

---

## 4. Production Workflow Commands (`.venv`)

> Note: Examples use Windows PowerShell (`.venv\Scripts\python.exe`). On Linux/macOS replace with `.venv/bin/python`.

### 1) Convert Custom Ultralytics Dataset (e.g., Potholes) -> COCO JSON & Verify
```powershell
.venv\Scripts\python.exe scripts/convert_yolo_to_coco.py `
  --data-yaml data/potholes_raw/data.yaml `
  --output-dir data/potholes_coco `
  --hydra-yaml yolo/config/dataset/potholes.yaml `
  --verify --verify-samples 5 --verify-dir tmp/potholes_verify
```

### 2) Filter COCO JSON to Target Domain Classes
```powershell
.venv\Scripts\python.exe scripts/filter_coco_classes.py `
  --source-dir data/coco `
  --splits train2017 val2017 `
  --classes person,car,bus,truck,motorcycle `
  --output-dir data/coco_traffic5 `
  --hydra-yaml yolo/config/dataset/coco_traffic5.yaml
```

### 3) Train Vanilla Baseline (`v9-t` / `v9-s` / `v9-m`)
```powershell
.venv\Scripts\python.exe yolo/lazy.py `
  task=train model=v9-s dataset=coco_traffic5 name=v9s_baseline `
  image_size=[640,640] task.epoch=100 task.data.batch_size=16 weight=weights/v9-s.pt
```

### 4) Train Upgraded `-opt` / `-p2` Model (Partial Weights, Native Res, Close-Mosaic, & KD from `v9-m`)
**4a. Standard `-opt` with partial weight transfer + close_mosaic:**
```powershell
.venv\Scripts\python.exe yolo/lazy.py `
  task=train model=v9-t-opt dataset=coco_traffic5 name=v9t_opt `
  partial_weights=weights/v9-t.pt image_size=[512,512] `
  task.epoch=100 task.close_mosaic=15 task.multi_scale=True
```
**4b. Small-object `-p2` variant (e.g., potholes/cracks at `640x640`):**
```powershell
.venv\Scripts\python.exe yolo/lazy.py `
  task=train model=v9-t-opt-p2 dataset=potholes name=v9t_opt_p2 `
  partial_weights=weights/v9-t.pt image_size=[640,640] `
  task.epoch=100 task.close_mosaic=15
```
**4c. Teacher-Student Knowledge Distillation (`v9-m` teacher -> `v9-t-opt` student):**
```powershell
.venv\Scripts\python.exe yolo/lazy.py `
  task=train model=v9-t-opt dataset=coco_traffic5 name=v9t_opt_kd `
  partial_weights=weights/v9-t.pt image_size=[480,480] `
  task.epoch=100 task.close_mosaic=15 `
  task.distill.enabled=True task.distill.teacher_model=v9-m `
  task.distill.teacher_weights=weights/v9-m.pt `
  task.distill.lambda_kd=1.0 task.distill.feature_weight=0.5 task.distill.logit_weight=0.5
```

### 5) Optional Structured Channel Pruning + Recovery Fine-Tuning
```powershell
.venv\Scripts\python.exe scripts/prune_model.py `
  --model v9-t-opt --weights runs/train/v9t_opt_kd/checkpoints/last.ckpt `
  --class-num 5 --pruning-ratio 0.20 --importance l1 --image-size 480 `
  --output exports/v9t_opt_pruned20.pt

# Recovery fine-tuning (using the pruned checkpoint printed by prune_model.py):
.venv\Scripts\python.exe yolo/lazy.py `
  task=train model=v9-t-opt dataset=coco_traffic5 name=v9t_opt_pruned_ft `
  weight=exports/v9t_opt_pruned20.pt image_size=[480,480] `
  task.epoch=50 task.optimizer.args.lr=0.002 task.close_mosaic=10
```

### 6) Export Vanilla & Upgraded Models to OpenVINO FP16 & INT8 + PrePostProcessor
```powershell
# Vanilla FP16 baseline:
.venv\Scripts\python.exe scripts/export_openvino.py `
  --model v9-t --weights runs/train/v9t_baseline/checkpoints/best.ckpt `
  --dataset coco_traffic5 --imgsz 512 --output-dir exports/ab/v9t_fp16 --name v9t_fp16

# Upgraded INT8 + head protection + embedded PrePostProcessor:
.venv\Scripts\python.exe scripts/export_openvino.py `
  --model v9-t-opt --weights runs/train/v9t_opt_kd/checkpoints/best.ckpt `
  --dataset coco_traffic5 --imgsz 512 --int8 `
  --calib-dir data/coco_traffic5/images/val2017 --calib-samples 250 `
  --protect-heads --embed-preprocess `
  --output-dir exports/ab/v9t_opt_int8_ppp --name v9t_opt_int8_ppp
```

### 7) Run Automated A/B Benchmark (`PASS` / `ROLLBACK` Gate)
```powershell
# Fast 500-image validation check:
.venv\Scripts\python.exe scripts/benchmark_compare.py `
  --models v9t-fp16=exports/ab/v9t_fp16/v9t_fp16.xml v9t-opt-int8=exports/ab/v9t_opt_int8_ppp/v9t_opt_int8_ppp.xml `
  --val-images data/coco_traffic5/images/val2017 `
  --val-ann data/coco_traffic5/annotations/instances_val2017.json `
  --max-eval-images 500 --threads 4 --warmup 30 --iters 200 `
  --output-md exports/ab/bench_500.md --output-json exports/ab/bench_500.json

# Full validation set:
.venv\Scripts\python.exe scripts/benchmark_compare.py `
  --models v9t-fp16=exports/ab/v9t_fp16/v9t_fp16.xml v9t-opt-int8=exports/ab/v9t_opt_int8_ppp/v9t_opt_int8_ppp.xml `
  --val-images data/coco_traffic5/images/val2017 `
  --val-ann data/coco_traffic5/annotations/instances_val2017.json `
  --max-eval-images 0 --threads 4 --warmup 50 --iters 300 `
  --output-md exports/ab/bench_full.md --output-json exports/ab/bench_full.json
```

### 8) 1-Command Rollback Instructions (If Vanilla Baseline Wins)
- **Runtime model rollback (swap deployed OpenVINO XML back to vanilla FP16):**
  ```powershell
  .venv\Scripts\python.exe inference/demo_stream_count.py --model-xml exports/ab/v9t_fp16/v9t_fp16.xml --source path/to/video.mp4
  ```
- **Repository code rollback (discard upgrade branch and return cleanly to upstream `main`):**
  ```powershell
  git checkout main
  ```
