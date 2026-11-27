# Improvement Plan: CPU-Only 15-Class YOLOv9-t

Branch: `feat/yolov9-cpu-upgrade` · Status: **draft for review, nothing implemented yet**

## 1. Goal

Reduce CPU consumption of the v9-t OpenVINO pipeline on **CPU-only Intel devices (various models, no GPU)** for a
fixed set of 15 classes, without reducing:

- accuracy on those classes (mAP, AP_small, per-class AP),
- input image size (640×640),
- processed frames (every frame is detected; no frame skipping).

Only open-source (MIT/Apache/BSD) components are used.

## 2. Target classes (15)

| Group | Classes |
|---|---|
| People | person |
| Vehicles | bicycle, car, motorcycle, airplane, bus, train, truck, boat |
| Street objects | traffic light, fire hydrant, stop sign, parking meter, bench |
| Electronics | cell phone |

| New ID | Class | Current model output index | COCO category ID | Approx. instances train / val* |
|---:|---|---:|---:|---|
| 0 | person | 0 | 1 | 262k / 11.0k |
| 1 | bicycle | 1 | 2 | 7.1k / 316 |
| 2 | car | 2 | 3 | 43.9k / 1.9k |
| 3 | motorcycle | 3 | 4 | 8.7k / 371 |
| 4 | airplane | 4 | 5 | 5.1k / 143 |
| 5 | bus | 5 | 6 | 6.1k / 285 |
| 6 | train | 6 | 7 | 4.6k / 190 |
| 7 | truck | 7 | 8 | 9.9k / 415 |
| 8 | boat | 8 | 9 | 10.8k / 430 |
| 9 | traffic light | 9 | 10 | 12.9k / 637 |
| 10 | fire hydrant | 10 | 11 | 1.9k / 101 |
| 11 | stop sign | 11 | 13 | 2.0k / 75 |
| 12 | parking meter | 12 | 14 | 1.3k / 60 |
| 13 | bench | 13 | 15 | 9.8k / 413 |
| 14 | cell phone | 67 | 77 | 6.4k / 262 |

\* Approximate public COCO statistics; exact counts are printed by the data step (Phase 2).

## 3. Current state (measured)

All numbers are for the current deployed model `exports/coco_bench/v9t_fused_int8_ppp_prot` on an
i7-13650HX unless stated otherwise.

| Metric | Value |
|---|---|
| mAP@0.5:0.95 / mAP@0.5 / AP_small (80 classes, 500 val images) | 41.00% / 55.78% / 22.28% |
| Model compute (deployed graph, 640×640) | **3.90 GMACs** (class branch 0.98 = 25%, box branch 0.44 = 11%, backbone + neck 2.48 = 64%) |
| Inference only, 4 requests × 1 thread | 77.5 FPS, **5.0 CPU-s/100 frames** |
| Full FIFA video pipeline (decode + infer + track + draw + MP4 write) | **7.27 CPU-s/100 frames** |
| Pipeline stages (CPU-s/100 frames) | decode 0.63, letterbox 0.10, annotate 0.13, MP4 write 0.5–0.8 |

**Already ruled out by measurement:**

- **OpenVINO CPU options** made no meaningful difference: all within ±1% of 5.0 CPU-s/100f. Tested:
  - `SCHEDULING_CORE_TYPE=PCORE_ONLY`
  - `ENABLE_HYPER_THREADING=NO`
  - `CPU_DENORMALS_OPTIMIZATION`
  - `TBB_PARTITIONER=STATIC`
- **Single-request latency mode** costs more CPU per frame: 6.3 vs 5.0 CPU-s/100f.
- **Removing INT8 head protection** brings no speed gain (75.5 vs 77.5 FPS) and loses about 2.8 pp mAP.
- **OpenCV hardware decode** uses more CPU: 0.97 vs 0.63 CPU-s/100f.
- **Filtering classes in the demo (`--classes`)** does not reduce model compute.

> **Correction:** an earlier estimate of "−27% compute" for a reduced-class head was measured on the default config
> (class branch 128 channels wide). The official `v9-t.pt` weights use an **80-channel** class branch, so the deployed model is 3.90 GMACs and
> the real class-head saving is 8–18% (see Phase 3).

## 4. Expected outcome

| Step | Model compute | Change | Accuracy risk |
|---|---:|---:|---|
| Current deployed v9-t (80 classes) | 3.90 GMACs | — | — |
| 15-class head, `class_neck=64` | 3.59 GMACs | −7.9% | low |
| 15-class head, `class_neck=48` | 3.36 GMACs | −13.7% | low–medium |
| 15-class head, `class_neck=32` | 3.18 GMACs | −18.5% | medium |
| + structured pruning of backbone/neck 20–30% (estimate) | ~2.6–2.9 GMACs | ~−25 to −35% total | medium, gated |

Compute was measured with forward hooks on the deployed graph (auxiliary branch removed). Pruned figures are
estimates. CPU time does not scale exactly with compute, so every step is re-benchmarked.

Target (estimate): inference about **3.3–3.8 CPU-s/100f**, down from 5.0. The full pipeline is about **4.5–5.5 CPU-s/100f** headless, down from 7.27.

## 5. Phases

### Phase 0 — 15-class baseline (no changes to the model)

1. Add `--eval-classes` to `scripts/benchmark_compare.py`. It keeps only the 15 classes in both the ground truth and the
   predictions. Overlapping-box removal (NMS) is done per class, so dropping other classes afterwards does not change the remaining results.
2. Measure the current fused INT8 (and FP16) model on the **full 5,000 val2017 images**, restricted to the 15 classes:
   - overall mAP@0.5:0.95, mAP@0.5 and AP_small;
   - per-class AP;
   - CPU-s/100f.
3. This becomes the **reference for every gate below**.

### Phase 1 — No-retrain CPU savings (exact accuracy)

| Item | Change | Expected saving | Accuracy |
|---|---|---|---|
| 1.1 Headless mode | Skip `frame.copy()` and annotation when neither `--output` nor `--display` is set | ~0.13+ CPU-s/100f | unchanged |
| 1.2 Optional writing | Production runs without `--output` | 0.5–0.8 CPU-s/100f | unchanged |
| 1.3 Class-output slice | Cut the last class conv to 15 output rows (indices 0–13, 67) at export | ~1–2% compute + lighter post-processing | bit-identical scores |
| 1.4 Early class filter | Drop other classes before NMS in `postprocess` | small | unchanged |
| 1.5 Auto streams per device | Default the THROUGHPUT hint with streams = min(4, physical P-cores); keep the `--threads` / `--async-requests` overrides | avoids oversubscription on 2–4 core CPUs | unchanged |

**Deliverable:** a 15-output fused INT8 export plus headless pipeline; expected about −13% full-pipeline CPU. Gate: identical
accuracy to Phase 0 (within float noise).

### Phase 2 — Data

1. Download COCO `train2017` images (~18 GB) and the 2017 annotations to `D:\projects\YOLO\data\coco`.
   - Download the zip to `C:` and extract it to `D:`; free space measured: `D:` 29 GB, `C:` 19 GB.
2. Build the 15-class split with the existing script:

   ```powershell
   .\.venv\Scripts\python.exe scripts\filter_coco_classes.py `
     --source-dir data\coco --splits train2017 val2017 `
     --classes "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,cell phone" `
     --output-dir data\coco15 --keep-empty-images --link-mode hardlink `
     --hydra-yaml yolo\config\dataset\coco15.yaml
   ```

   - `--keep-empty-images` keeps images with none of the 15 classes as negatives, which reduces false positives.
   - Objects of the other 65 classes become background.
   - `--link-mode hardlink` avoids a second 18 GB copy.
3. Print the per-class instance counts for train and val.

### Phase 3 — Code changes

1. Replace the hardcoded 80-class `IDX_TO_ID` in `yolo/utils/model_utils.py` with a mapping built from the dataset config.
   This is needed for correct mAP with 15 classes.
2. Add a model variant `yolo/config/model/v9-t-c15.yaml`, identical to `v9-t` except for
   `MultiheadDetection.args.class_neck` (64 / 48 / 32 under evaluation).
3. Propagate the 15 class names through export metadata, `OpenVINODetector` and the demo `--classes` option.
4. Add a warm-start script `scripts/init_subset_head.py` (Phase 4).
5. Add tests:
   - the subset ID mapping;
   - the warm-start head shapes;
   - an equivalence test for the Phase 1.3 slice (sliced scores equal the original rows);
   - `--eval-classes` on a tiny dataset.

### Phase 4 — 15-class model: warm start + fine-tune

1. **Warm start (no training):**
   - Copy the backbone, neck and box branch from `weights/v9-t.pt` exactly.
   - Class branch 80 → `class_neck` channels: keep the most important channels. Importance is the BN gamma × L1 norm of the outgoing
     weights to the 15 kept classes.
   - Last 1×1 conv: keep rows 0–13 and 67.
   - Auxiliary (training-only) head: keep the same 15 rows. Its width does not matter because it is stripped at export.
   - Sanity check: evaluate the warm-started model on val **before** training; its mAP must be well above zero.
2. **Fine-tune on the our gpu:**
   - Pilot: 1 epoch to measure time per epoch and memory. Batch 16 (8 if it runs out of memory), effective batch 64, AMP, 640×640, current
     augmentations.
   - Main run: 30–60 epochs.
     - SGD, learning rate ~1e-3 with warmup.
     - EMA weights.
     - Mosaic off for the last ~10 epochs.
     - Validate on the 15 classes every few epochs.
   - Rough estimate 10–20 h total; the pilot gives the real number.
3. **Choosing `class_neck`:** train 64 first (lowest risk). If it passes the gates with margin, train 48, then 32.
   Ship the narrowest one that passes.
4. **Distillation fallback** (if a gate fails):
   - Teacher: the original 80-class v9-t, or v9-s (50.9% mAP), with its scores restricted to the 15 classes.
   - Loss: the normal loss plus BCE between student and teacher class probabilities.
   - Optional box-distribution loss.


### Phase 5 — Export

Same pipeline as the current production model:

- RepConv fusion.
- Protected INT8 heads.
- `fast_bias_correction=False`.
- Embedded preprocessing (u8 NHWC BGR).
- 500 calibration images that **exclude** the evaluation images.

Output goes to `exports/coco15/v9t15_<variant>_fused_int8_ppp_prot`.

### Phase 6 — Acceptance gates

These are measured on the **full 5,000-image val2017** set, 15 classes only, against the Phase 0 reference.

| Gate | Threshold |
|---|---|
| mAP@0.5:0.95 (15 classes) | ≥ −0.5 pp |
| mAP@0.5 | ≥ −0.5 pp |
| AP_small (traffic light, cell phone, distant people/cars) | ≥ −1.0 pp |
| Per-class AP | ≥ −1.0 pp. The rare classes (stop sign, parking meter, fire hydrant, airplane, train; <200 val instances) are reported and reviewed, not auto-failed |
| Inference CPU-s/100f (same harness, interleaved repeated runs, median) | ≥ 10% lower than the Phase 1 model |
| Peak RSS | not higher than current |
| FIFA video | side-by-side visual check: no missing people or phones, no new false positives |

If all gates pass, the new model becomes the default and the docs and demo commands are updated.

If a gate fails, apply distillation. If it still fails, keep the Phase 1 model.

### Phase 7 — Cross-CPU validation (different Intel CPUs)

1. Add a CPU probe to the benchmark report:
   - CPU name;
   - physical cores (performance and efficiency);
   - OpenVINO `OPTIMIZATION_CAPABILITIES`, which shows INT8 support;
   - VNNI support.
2. On this machine, emulate small devices by limiting to 2 and 4 cores (process affinity plus `INFERENCE_NUM_THREADS`).
3. Run the same benchmark command on at least one **older Intel CPU without VNNI** (e.g. 6th–9th gen Core, Celeron or Pentium).
   This is the only real check of INT8 speed-up there.
4. Report per CPU: FPS, CPU-s/100f, and whether the 25 FPS video keeps up in real time.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Rare classes (parking meter, stop sign, fire hydrant) lose AP | Warm-start head, distillation, oversample rare classes if needed |
| Small objects (traffic light, cell phone) lose AP | Separate AP_small gate, per-class review |
| Car / truck / bus confusion | Per-class AP and confusion matrix review |
| False positives on removed classes (e.g. dog → person) | Keep negative images; removed classes trained as background |
| INT8 adds error after retraining | Gate the INT8 export, not the FP32 model |
| Pruning too aggressive | Step-wise ratios, stop at the last passing step |
| INT8 gain smaller on CPUs without VNNI | Reduce compute instead (Phases 4–5 help on every CPU); measure in Phase 8 |
| 6 GB GPU memory / training time | 1-epoch pilot first; batch 8 with gradient accumulation if needed |
| Benchmark noise | Full 5,000 images, interleaved repeated runs, medians |

## 7. Effort

| Phase | Effort |
|---|---|
| 0. 15-class baseline | ~2 h (incl. full 5,000-image evaluation runs) |
| 1. No-retrain savings | ~3–4 h |
| 2. Data | 1–2 h (mostly download) |
| 3. Code changes + tests | ~4 h |
| 4. 15-class warm start + fine-tune | ~1 h setup + 10–20 h GPU per `class_neck` variant (unattended) |
| 5. Pruning + distillation | ~2 h setup + 1–2 days GPU (unattended, optional) |
| 6. Export | ~0.5 h per candidate |
| 7. Gates | ~1 h per candidate |
| 8. Cross-CPU validation | ~1 h here + time on other devices |

## 8. Rollback

The current production model, `exports/coco_bench/v9t_fused_int8_ppp_prot` (commit `f47ec4a`), stays untouched. Every
new model is a separate export directory, so rollback means pointing `--model-xml` back to it.

## 9. Decisions needed

1. **Negative images:** keep all images without the 15 classes (recommended, safest against false positives) or subsample them to cut
   training time?
2. **Distillation:** use from the start, or only if a gate fails (recommended)?
3. **Rare-class gate:** accept "report and review" for stop sign, parking meter, fire hydrant, airplane and train?
4. **Pruning (Phase 5):** include it now or decide after Phase 4 results?
5. **Older test CPU:** is there an older Intel device (without VNNI) available for Phase 8?
