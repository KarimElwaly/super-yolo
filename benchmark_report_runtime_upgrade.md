# OpenVINO A/B Benchmark Report

## Locked Execution Controls
- **Baseline:** `v9t-fp16`
- **INFERENCE_NUM_THREADS:** `4` | **PERFORMANCE_HINT:** `THROUGHPUT`
- **Async Requests / Streams:** `4` / `4`
- **Warmup / Timed Iterations:** `20` / `150`
- **Max Eval Images:** `500`

## Model Metrics
Latency is end-to-end per submitted frame (including queue wait in throughput mode); FPS is total timed frames divided by timed wall time.

| Model | Variant | Precision | PPP | mAP@0.5:0.95 | mAP@0.5 | AP_small | AP_med | AP_large | Mean (ms) | P50 (ms) | P95 (ms) | FPS | CPU % | CPU-s/100f | Peak RSS (MB) | Disk (MB) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v9t-fp16 | v9-t | FP16 | No | 41.39% | 56.13% | 22.92% | 46.16% | 57.25% | 163.56 | 149.32 | 218.27 | 29.87 | 419.8% | 14.052 | 671.7 | 4.92 |
| v9t-fused-fp16-ppp | v9-t | FP16 | Yes | 41.38% | 56.11% | 22.93% | 46.16% | 57.23% | 159.40 | 137.23 | 251.24 | 30.84 | 392.0% | 12.708 | 645.7 | 4.71 |
| v9t-fused-int8-protected | v9-t | INT8 | Yes | 41.00% | 55.78% | 22.27% | 44.92% | 57.77% | 68.87 | 59.66 | 103.72 | 71.15 | 389.8% | 5.479 | 669.5 | 3.49 |
| v9s-fp16 | v9-s | FP16 | No | 50.92% | 67.24% | 32.59% | 56.75% | 66.00% | 474.42 | 411.64 | 703.01 | 10.34 | 400.1% | 38.688 | 716.7 | 14.81 |
| v9s-fused-fp16-ppp | v9-s | FP16 | Yes | 50.92% | 67.25% | 32.60% | 56.73% | 65.99% | 463.54 | 389.30 | 721.84 | 10.61 | 390.9% | 36.833 | 689.6 | 14.44 |
| v9s-fused-int8-protected | v9-s | INT8 | Yes | 50.66% | 67.12% | 31.56% | 56.09% | 66.49% | 164.68 | 138.98 | 252.93 | 29.81 | 393.2% | 13.188 | 670.7 | 8.43 |

## Automated Go / No-Go Verdict
### `v9t-fused-fp16-ppp` vs `v9t-fp16` -> **ROLLBACK (Keep Vanilla)**
- mAP@0.5:0.95 delta: `-0.009 pp` | AP_small delta: `+0.008 pp` | P95 latency speedup: `-15.11%` | CPU-s/100f reduction: `+9.56%` | Peak RSS vs baseline: `96.13%`
- Failed criteria:
  - Neither Latency P95 speedup (-15.11%) nor CPU-sec/100f reduction (+9.56%) reached >= +25.0%
### `v9t-fused-int8-protected` vs `v9t-fp16` -> **PASS (Adopt Upgraded)**
- mAP@0.5:0.95 delta: `-0.394 pp` | AP_small delta: `-0.649 pp` | P95 latency speedup: `+52.48%` | CPU-s/100f reduction: `+61.01%` | Peak RSS vs baseline: `99.69%`
### `v9s-fused-fp16-ppp` vs `v9s-fp16` -> **ROLLBACK (Keep Vanilla)**
- mAP@0.5:0.95 delta: `-0.007 pp` | AP_small delta: `+0.004 pp` | P95 latency speedup: `-2.68%` | CPU-s/100f reduction: `+4.79%` | Peak RSS vs baseline: `96.21%`
- Failed criteria:
  - Neither Latency P95 speedup (-2.68%) nor CPU-sec/100f reduction (+4.79%) reached >= +25.0%
### `v9s-fused-int8-protected` vs `v9s-fp16` -> **ROLLBACK (Keep Vanilla)**
- mAP@0.5:0.95 delta: `-0.269 pp` | AP_small delta: `-1.029 pp` | P95 latency speedup: `+64.02%` | CPU-s/100f reduction: `+65.91%` | Peak RSS vs baseline: `93.58%`
- Failed criteria:
  - AP_small delta -1.029% < -1.000%
