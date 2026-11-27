# Autonomous Master Plan: CPU-Optimized & High-Accuracy YOLOv9 Upgrade (MIT-Clean)

You are acting as a Senior Computer Vision & MLOps Engineer working autonomously inside a freshly cloned `MultimediaTechLab/YOLO` repository.

**Execution Mode:** Fully autonomous end-to-end. Do NOT pause for human confirmation between phases. Execute Step 0 through Phase 8 sequentially, verifying each phase with concrete tests before moving to the next. If you hit an error or repo-specific quirk, diagnose it from the actual codebase, fix it cleanly, and continue.

---

## Non-Negotiable Constraints
1. **Strict License Compliance (Commercial MIT-Clean):**
   - Only install and use MIT, Apache-2.0, or BSD-licensed packages.
   - NEVER install, import, or copy code from `ultralytics`, `boxmot`, or any GPL/AGPL package.
2. **Zero-Destructive & Rollback-Safe:**
   - All work must happen on a new Git branch: `feat/yolov9-cpu-upgrade`.
   - NEVER overwrite or break original vanilla model configs (`v9-t.yaml`, `v9-s.yaml`, `v9-m.yaml`) or default upstream training/export workflows when upgrade flags are disabled.
   - Every architectural upgrade must be additive (`v9-t-opt.yaml`, `v9-s-opt.yaml`, `v9-m-opt.yaml`, etc.) and every training enhancement must be opt-in via config/CLI flags.
3. **Numerical Parity & Real OpenVINO Measurement:**
   - Every ONNX and OpenVINO export must verify numerical parity against PyTorch outputs.
   - Accuracy (`mAP`), latency, and CPU usage benchmarks must run directly on the exported OpenVINO IR model — never extrapolate PyTorch metrics to INT8 OpenVINO.
4. **No Multi-Hour Training Runs During Autonomous Execution:**
   - Do NOT launch full multi-epoch training on COCO `train2017` or large datasets. Use tiny synthetic batches (1–2 steps) to verify training/distillation/pruning code paths, AND use official pretrained `v9-t`/`v9-s` weights on COCO `val2017` to produce a real OpenVINO FP16 vs INT8+PrePostProcessor benchmark report.

---

## Step 0: Automated Git & Isolated Environment Setup
1. Inspect the OS/shell environment (Windows PowerShell or Linux/macOS bash) and verify `git status` is clean.
2. Create and checkout branch `feat/yolov9-cpu-upgrade`.
3. Create an isolated virtual environment `.venv` inside the repository root:
   - Prefer `uv venv .venv` + `uv pip install` if `uv` is available; otherwise use `python -m venv .venv` + `.venv` pip.
   - Never modify global Python environments. Run all subsequent Python commands and tests using this `.venv` interpreter.
4. Inspect `requirements.txt` / `pyproject.toml` and install the repo's dependencies inside `.venv`, plus the MIT/Apache/BSD tooling required for this upgrade:
   - `openvino`, `nncf`, `onnx`, `onnxsim`, `onnxruntime`, `supervision`, `torch-pruning`, `pycocotools`, `psutil`, `pip-licenses`, `pytest`
   - Ensure PyTorch matches CUDA if an NVIDIA GPU is detected (otherwise CPU build).
5. **Inspect Codebase Before Editing:** Read `yolo/config/`, `yolo/model/`, `yolo/tools/`, `yolo/utils/`, and `yolo/lazy.py`. Never guess Hydra config keys, module registry names, or PyTorch Lightning hooks — ground every edit in the actual repository code.

---

## Phase 1: Dataset Pipeline, COCO Class Filter & License Audit
1. **`scripts/check_licenses.py`:**
   - Scans all packages installed in `.venv` (via `pip-licenses` or `importlib.metadata`).
   - Explicitly bans `ultralytics`, `boxmot`, and flags any `AGPL` or strong `GPL` runtime dependencies (allowing MIT, Apache-2.0, BSD, PSF, ISC, LGPL).
   - Exits with code `0` if clean, non-zero otherwise. Run it immediately to verify `.venv` is clean.
2. **`scripts/convert_yolo_to_coco.py`:**
   - Converts Ultralytics YOLO `.txt` datasets (`images/`, `labels/`, `data.yaml`) into COCO JSON (`annotations/instances_train.json`, `annotations/instances_val.json`) to avoid `.txt` loader mAP issues.
   - Includes `--verify` mode that validates bounding box bounds `[0, W/H]`, positive box areas, class ID alignment, and optionally saves sample annotated verification images.
3. **`scripts/filter_coco_classes.py`:**
   - Filters any COCO JSON dataset down to a user-specified list of classes (e.g., `--classes person,car,bus,truck,motorcycle`), remapping category IDs cleanly to contiguous indices, copying or symlinking matching images, and auto-generating the matching Hydra dataset YAML config.
   - Purpose: enables fast (hours instead of days) specialized `-opt` + distillation training on domain-specific COCO subsets instead of all 80 classes.
4. **`yolo/config/dataset/custom_template.yaml`:**
   - Clean, documented Hydra dataset config template for plugging in any custom domain dataset (e.g., potholes, road defects, PPE) by editing only `path`, `class_num`, and `class_list`.

---

## Phase 2: CPU-Optimized & High-Accuracy Architecture Upgrades (Additive)
Create new additive model configs and minimal supporting code in `yolo/model/` without altering vanilla behavior:
1. **CPU & INT8-Friendly Activations:**
   - Add clean config-driven activation selection supporting `ReLU` and `HardSwish` alongside default `SiLU` (vanilla configs continue defaulting to `SiLU`).
   - Create upgraded configs:
     - `yolo/config/model/v9-t-opt.yaml` (Tiny / Nano-class)
     - `yolo/config/model/v9-s-opt.yaml` (Small)
     - `yolo/config/model/v9-m-opt.yaml` (Medium)
   - Use `ReLU` / `HardSwish` in `-opt` configs to accelerate x86 AVX2/AVX-VNNI CPU execution and minimize INT8 quantization drop.
2. **Small-Object Domain Variants (P2 Head):**
   - Create `yolo/config/model/v9-t-opt-p2.yaml` and `yolo/config/model/v9-s-opt-p2.yaml` adding a stride-4 P2 detection head (and trimming heavy P5 channels where appropriate) for small/distant objects such as road cracks and potholes.
3. **Strict Auxiliary (PGI) Branch Stripping at Export:**
   - Verify how `MultimediaTechLab/YOLO` handles auxiliary PGI branches during `eval()` and export.
   - Implement/enforce a clean `strip_auxiliary_branches()` / export-mode path so zero training-only auxiliary ops enter the exported ONNX/OpenVINO graph.
4. **Partial Pretrained Weight Loader (`load_partial_weights`):**
   - Implement a helper (`strict=False`) wired into training initialization via an opt-in config flag that loads all shape-matched layers from vanilla `v9-t`/`v9-s`/`v9-m` checkpoints into `-opt` architectures and logs a concise summary of loaded vs re-initialized layers.

---

## Phase 3: Zero-Inference-Cost Accuracy Boosters (Training-Time Only)
All features below MUST be disabled by default so vanilla training runs identically to upstream:
1. **Teacher-Student Knowledge Distillation:**
   - Add opt-in Hydra config support (e.g., `distill.enabled=true`, `distill.teacher_model=v9-m`, `distill.teacher_weights=<path>`, `distill.lambda_kd=1.0`, `distill.feature_weight=0.5`, `distill.logit_weight=0.5`).
   - Load and freeze the teacher model in `eval()` mode during student training (`v9-t-opt` or `v9-s-opt`).
   - Compute:
     - Feature-level distillation loss on multi-scale neck features (P3/P4/P5, with lightweight 1x1 projection adapters if student/teacher channel counts differ, using Channel-Wise Distillation or normalized MSE).
     - Detection-head logit distillation loss.
   - Ensure the teacher and projection adapters are never saved into the final exported student model.
2. **Training Resolution & Augmentation Controls:**
   - Verify and document clean Hydra overrides for native `480x480`, `512x512`, and `640x640` training, multi-scale training, EMA, and `close_mosaic` in the final 15 epochs.
3. **Optional Structured Channel Pruning (`scripts/prune_model.py`):**
   - Build a standalone CLI script using MIT-licensed `torch-pruning` (`DependencyGraph` + L1/Taylor importance) supporting `--pruning-ratio` (default `0.20`).
   - Protect final detection head output layers from pruning, save a standalone pruned checkpoint (never overwriting the input checkpoint), report exact parameter and MACs reduction %, and print the exact recovery fine-tuning command.

---

## Phase 4: OpenVINO Export, NNCF INT8, PrePostProcessor & Unified Detector
1. **`scripts/export_openvino.py`:**
   - Supports both vanilla (`v9-t/s/m`) and upgraded (`v9-t/s/m-opt`, `-p2`, pruned) checkpoints.
   - Pipeline steps:
     1. Strip auxiliary PGI branches and export static-shape ONNX (opset 17, simplified with `onnxsim`).
     2. Inspect the ONNX graph to verify zero auxiliary branches remain and compare ONNX Runtime vs PyTorch outputs (assert numerical parity).
     3. Convert ONNX -> OpenVINO IR (`FP16`) using `openvino.convert_model`.
     4. Optional `--int8 --calib-dir <dir> --calib-samples 250`: quantize to INT8 via `nncf.quantize` with real calibration images. Support `--protect-heads` (`nncf.IgnoredScope`) to keep final box-regression/classification output layers in higher precision if INT8 accuracy drops more than 0.5% mAP.
     5. Optional `--embed-preprocess`: fuse `ov.preprocess.PrePostProcessor` into the saved IR model so it directly accepts raw `uint8` `NHWC` BGR frames (fusing resize, BGR->RGB, normalization, and layout transpose inside OpenVINO).
     6. Save a companion `export_meta.json` next to the `.xml`/`.bin` recording `input_size`, `precision`, `embedded_preprocess`, `class_names`, `model_variant`, and source weight provenance.
2. **`inference/detector_ov.py` (Model-Agnostic Production Adapter):**
   - Exposes `OpenVINODetector(model_xml_path, num_threads=4, performance_hint="LATENCY", conf_threshold=0.25, iou_threshold=0.45, device="CPU")`.
   - Reads `export_meta.json` automatically to handle both embedded-preprocess (`uint8` direct input) and manual letterbox input, accurately mapping predicted boxes back to original frame pixel coordinates.
   - Applies OpenCV `cv2.dnn.NMSBoxes` and returns standard `supervision.Detections` (`xyxy`, `confidence`, `class_id`).
3. **`inference/demo_stream_count.py` (Reference Pipeline Integration):**
   - Clean reference script demonstrating threaded RTSP/video frame capture (dropping stale frames), `OpenVINODetector`, `sv.ByteTrack`, `sv.LineZone` / `sv.PolygonZone`, and optional `--detect-every-n` frame skipping.

---

## Phase 5: Automated A/B Benchmark Suite (`scripts/benchmark_compare.py`)
Build a reproducible benchmark runner taking a JSON/YAML manifest of exported OpenVINO models:
- **Locked Execution Controls:**
  - Identical `INFERENCE_NUM_THREADS` (configurable, default `4`), identical `PERFORMANCE_HINT`, identical input resolution, `50` warmup iterations, and `300` timed iterations (CLI-overridable).
  - Support `--max-eval-images <N>` (default `0` = full validation set) so users can run either fast stratified evaluations or full 5,000-image COCO `val2017` evaluations.
- **Metrics Captured Per Model (Directly on OpenVINO IR):**
  1. Accuracy (when `--val-images` and `--val-ann` COCO JSON are provided): `mAP@0.5`, `mAP@0.5:0.95`, `AP_small`, `AP_medium`, `AP_large` via `pycocotools`.
  2. Speed: `Latency Mean (ms)`, `Latency P50 (ms)`, `Latency P95 (ms)`, `Throughput (FPS)`.
  3. CPU Efficiency (via `psutil`): `Avg Process CPU %` and `Total CPU-seconds per 100 frames` (`cpu_times().user + system`).
  4. Footprint: `Peak RSS RAM (MB)` and `Model Disk Size (MB)`.
- **Automated Go / No-Go Verdict:**
  - Compare every candidate against the designated vanilla baseline and output both `benchmark_report.md` and `benchmark_report.json` with exact `%` deltas and a clear verdict:
    - **`PASS (Adopt Upgraded)`:** `mAP@0.5:0.95` delta >= -0.5% AND `AP_small` delta >= -1.0% AND (`Latency P95` faster by >= 25% OR `CPU-seconds / 100 frames` reduced by >= 25%) AND Peak RSS <= 115% of baseline.
    - **`ROLLBACK (Keep Vanilla)`:** Any failure of the acceptance gate, clearly listing which criterion failed.

---

## Phase 6: Automated Real Baseline Benchmark on COCO `val2017`
Before finishing, produce a **real** OpenVINO benchmark comparing vanilla FP16 vs deployment-optimized INT8 + PrePostProcessor on real images:
1. Download official `MultimediaTechLab/YOLO` pretrained weights for `v9-t` and `v9-s` (log the exact release URL and license note in `HANDOFF.md`) and download COCO `val2017` images + `instances_val2017.json` into a local ignored `data/coco_val2017/` folder (if network download is unavailable, log why cleanly and fall back to synthetic smoke benchmark).
2. Export `v9-t` and `v9-s` into:
   - Vanilla baseline: OpenVINO `FP16` (`640x640`, standard preprocessing)
   - Optimized candidate: OpenVINO `INT8` (calibrated on 250 real `val2017` images, with `--protect-heads` auto-fallback if mAP drops > 0.5%) + embedded `PrePostProcessor`.
3. Run `scripts/benchmark_compare.py` with `--max-eval-images 500 --warmup 30 --iters 200` (and document the exact command for full 5,000-image evaluation) to generate a real `benchmark_report_coco_pretrained.md` and `.json`.

---

## Phase 7: End-to-End Verification & Smoke Tests (`tests/test_upgrade_smoke.py`)
Write and run `pytest tests/test_upgrade_smoke.py` inside `.venv` covering ALL of the following:
1. `scripts/check_licenses.py` exits `0` with zero GPL/AGPL packages.
2. `scripts/convert_yolo_to_coco.py` converts a tiny synthetic YOLO dataset and passes `--verify`.
3. `scripts/filter_coco_classes.py` filters a tiny multi-class COCO JSON to a 2-class subset and produces a valid Hydra dataset YAML.
4. Vanilla `v9-t`, `v9-s`, `v9-m` load cleanly through Hydra and pass forward inference without regression.
5. Upgraded `v9-t-opt`, `v9-s-opt`, `v9-m-opt`, `v9-t-opt-p2`, `v9-s-opt-p2` build cleanly, load partial weights from vanilla state dicts with expected layer-match logs, and pass forward inference.
6. Knowledge Distillation computes finite loss and valid gradients on a 2-step dummy batch (`v9-t-opt` student + frozen `v9-m` teacher).
7. `scripts/prune_model.py` prunes a dummy `v9-t-opt` checkpoint, verifies parameter/MACs reduction, and passes forward inference.
8. `scripts/export_openvino.py` exports both vanilla `v9-t` and upgraded `v9-t-opt` -> ONNX (verifying zero auxiliary PGI ops + PyTorch numerical parity) -> OpenVINO FP16 -> OpenVINO INT8 + embedded `PrePostProcessor`.
9. `OpenVINODetector` runs on frames and returns valid `sv.Detections`.
10. `scripts/benchmark_compare.py` passes end-to-end and generates valid Markdown + JSON reports.

Fix any failures until 100% of smoke tests pass.

---

## Phase 8: Clean Git Commits & `HANDOFF.md`
1. Update `.gitignore` so `.venv/`, `data/`, downloaded weights, and temporary artifacts are never committed.
2. Make clean, logical Git commits on `feat/yolov9-cpu-upgrade`.
3. Create `HANDOFF.md` in the repository root containing:
   - Concise summary of all added/modified files and test results.
   - The actual real benchmark table from `benchmark_report_coco_pretrained.md` (with provenance/license notes on the downloaded pretrained weights).
   - Copy-pasteable CLI commands (using `.venv`) for the user's real workflow:
     1. Convert custom Ultralytics dataset (e.g., potholes) -> COCO JSON & verify.
     2. Filter COCO JSON to target classes (e.g., `person,car,bus,truck,motorcycle`) for fast specialized training.
     3. Train vanilla baseline (`v9-t` / `v9-s` / `v9-m`).
     4. Train upgraded `-opt` model (standard, `-p2` small-object variant, and Teacher-Student Knowledge Distillation from `v9-m`).
     5. Optional structured pruning + recovery fine-tuning.
     6. Export both vanilla and upgraded models to OpenVINO FP16 & INT8 (with `PrePostProcessor`).
     7. Run `scripts/benchmark_compare.py` (both 500-image fast check and full validation set) to get the `PASS / ROLLBACK` decision.
     8. Exact 1-command rollback instructions if the vanilla baseline wins.

Start immediately at Step 0 and execute autonomously through Phase 8.