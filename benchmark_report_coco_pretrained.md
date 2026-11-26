# OpenVINO A/B Benchmark Report

## Locked Execution Controls
- **Baseline:** `v9-t-fp16`
- **INFERENCE_NUM_THREADS:** `4` | **PERFORMANCE_HINT:** `LATENCY`
- **Warmup / Timed Iterations:** `30` / `200`
- **Max Eval Images:** `500`

## Model Metrics
| Model | Variant | Precision | PPP | mAP@0.5:0.95 | mAP@0.5 | AP_small | AP_med | AP_large | Mean (ms) | P50 (ms) | P95 (ms) | FPS | CPU % | CPU-s/100f | Peak RSS (MB) | Disk (MB) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v9-t-fp16 | v9-t | FP16 | No | 41.39% | 56.13% | 22.92% | 46.16% | 57.25% | 57.34 | 57.45 | 68.99 | 17.44 | 254.5% | 14.594 | 533.9 | 4.92 |
| v9-t-int8-ppp | v9-t | INT8 | Yes | 38.57% | 53.58% | 20.38% | 39.97% | 56.03% | 19.77 | 19.63 | 21.37 | 50.59 | 338.2% | 6.688 | 537.9 | 3.86 |
| v9-t-int8-ppp-protect-heads | v9-t | INT8 | Yes | 41.11% | 56.09% | 21.99% | 44.78% | 57.18% | 26.70 | 26.95 | 33.84 | 37.46 | 271.2% | 7.242 | 559.9 | 3.89 |
| v9-s-fp16 | v9-s | FP16 | No | 50.92% | 67.24% | 32.59% | 56.75% | 66.00% | 102.61 | 100.34 | 117.08 | 9.75 | 360.0% | 36.938 | 540.4 | 14.81 |
| v9-s-int8-ppp | v9-s | INT8 | Yes | 48.57% | 65.82% | 30.43% | 51.72% | 64.42% | 41.44 | 38.50 | 54.30 | 24.13 | 346.8% | 14.375 | 553.1 | 8.85 |
| v9-s-int8-ppp-protect-heads | v9-s | INT8 | Yes | 50.70% | 67.39% | 32.35% | 56.51% | 65.45% | 48.07 | 48.22 | 57.52 | 20.80 | 302.8% | 14.555 | 545.2 | 8.92 |

## Automated Go / No-Go Verdict
### `v9-t-int8-ppp` vs `v9-t-fp16` -> **ROLLBACK (Keep Vanilla)**
- mAP@0.5:0.95 delta: `-2.823 pp` | AP_small delta: `-2.546 pp` | P95 latency speedup: `+69.03%` | CPU-s/100f reduction: `+54.18%` | Peak RSS vs baseline: `100.76%`
- Failed criteria:
  - mAP@0.5:0.95 delta -2.823% < -0.500%
  - AP_small delta -2.546% < -1.000%
### `v9-t-int8-ppp-protect-heads` vs `v9-t-fp16` -> **PASS (Adopt Upgraded)**
- mAP@0.5:0.95 delta: `-0.284 pp` | AP_small delta: `-0.933 pp` | P95 latency speedup: `+50.96%` | CPU-s/100f reduction: `+50.37%` | Peak RSS vs baseline: `104.88%`
### `v9-s-int8-ppp` vs `v9-s-fp16` -> **ROLLBACK (Keep Vanilla)**
- mAP@0.5:0.95 delta: `-2.354 pp` | AP_small delta: `-2.165 pp` | P95 latency speedup: `+53.63%` | CPU-s/100f reduction: `+61.08%` | Peak RSS vs baseline: `102.35%`
- Failed criteria:
  - mAP@0.5:0.95 delta -2.354% < -0.500%
  - AP_small delta -2.165% < -1.000%
### `v9-s-int8-ppp-protect-heads` vs `v9-s-fp16` -> **PASS (Adopt Upgraded)**
- mAP@0.5:0.95 delta: `-0.223 pp` | AP_small delta: `-0.244 pp` | P95 latency speedup: `+50.87%` | CPU-s/100f reduction: `+60.60%` | Peak RSS vs baseline: `100.89%`
