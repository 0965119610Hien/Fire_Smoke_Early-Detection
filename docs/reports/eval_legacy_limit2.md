# Evaluation — run `legacy`

- Generated: 2026-10-19T10:46:50 | eval git: `33d58d8-dirty` | limit clips: 2
- Stage 1: `E:\NCKH\Code\Fire_Smoke_Early-Detection\checkpoints\segformer_spatial_best.pth` (run legacy, epoch 35)
- Stage 2: `E:\NCKH\Code\Fire_Smoke_Early-Detection\checkpoints\checkpoints\timesformer_spatiotemporal_best.pth` (run legacy, epoch 13, gamma -0.0201, legacy=True)
- Protocol: each (video, frame) counted once; Stage 2 prediction from the clip where the frame is at its latest position; area(prob_c > tau) >= max(A_min, 1) pixels at 512x512 model resolution, pad excluded; thresholds on val: recall >= 0.95 with lowest FAR, else max F1.

## Pixel level (test, %)

| Config | Frames | Videos | IoU BG | IoU Fire | IoU Smoke | Dice Fire | Dice Smoke | Fg-mIoU [95% CI] | All-mIoU |
|---|---|---|---|---|---|---|---|---|---|
| stage1_framewise | 12 | 1 | 93.17 | 40.96 [40.96, 40.96] | n/a [n/a, n/a] | 58.11 | n/a | 40.96 [40.96, 40.96] | 67.06 |
| stage2_gamma0 | 12 | 1 | 93.42 | 36.17 [36.17, 36.17] | 0.00 [0.00, 0.00] | 53.13 | 0.00 | 18.09 [18.09, 18.09] | 43.20 |
| stage2_full | 12 | 1 | 92.59 | 43.47 [43.47, 43.47] | 0.00 [0.00, 0.00] | 60.60 | 0.00 | 21.73 [21.73, 21.73] | 45.35 |

## Frame level (test; thresholds fixed from val)

| Config | Class | tau | A_min | Base rate | TP | FP | TN | FN | Recall [95% CI] | FAR [95% CI] | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stage1_framewise | fire | 0.3 | 0 | 100.00 | 12 | 0 | 0 | 0 | 100.00 [100.00, 100.00] | n/a [n/a, n/a] | 100.00 | 100.00 | 0.000 |
| stage1_framewise | smoke | 0.3 | 0 | 0.00 | 0 | 0 | 12 | 0 | n/a [n/a, n/a] | 0.00 [0.00, 0.00] | n/a | 0.00 | 0.000 |
| stage1_framewise | any | - | - | 100.00 | 12 | 0 | 0 | 0 | 100.00  | n/a  | 100.00 | 100.00 | 0.000 |
| stage2_gamma0 | fire | 0.3 | 0 | 100.00 | 12 | 0 | 0 | 0 | 100.00 [100.00, 100.00] | n/a [n/a, n/a] | 100.00 | 100.00 | 0.000 |
| stage2_gamma0 | smoke | 0.3 | 0 | 0.00 | 0 | 1 | 11 | 0 | n/a [n/a, n/a] | 8.33 [8.33, 8.33] | 0.00 | 0.00 | 0.000 |
| stage2_gamma0 | any | - | - | 100.00 | 12 | 0 | 0 | 0 | 100.00  | n/a  | 100.00 | 100.00 | 0.000 |
| stage2_full | fire | 0.3 | 0 | 100.00 | 12 | 0 | 0 | 0 | 100.00 [100.00, 100.00] | n/a [n/a, n/a] | 100.00 | 100.00 | 0.000 |
| stage2_full | smoke | 0.6 | 2000 | 0.00 | 0 | 0 | 12 | 0 | n/a [n/a, n/a] | 0.00 [0.00, 0.00] | n/a | 0.00 | 0.000 |
| stage2_full | any | - | - | 100.00 | 12 | 0 | 0 | 0 | 100.00  | n/a  | 100.00 | 100.00 | 0.000 |

## Video level (test)

| Config | Class | k | TP | FP | TN | FN | Recall | FAR |
|---|---|---|---|---|---|---|---|---|
| stage1_framewise | fire | 1 | 1 | 0 | 0 | 0 | 100.00 | n/a |
| stage1_framewise | smoke | 3 | 0 | 0 | 1 | 0 | n/a | 0.00 |
| stage2_gamma0 | fire | 1 | 1 | 0 | 0 | 0 | 100.00 | n/a |
| stage2_gamma0 | smoke | 3 | 0 | 0 | 1 | 0 | n/a | 0.00 |
| stage2_full | fire | 1 | 1 | 0 | 0 | 0 | 100.00 | n/a |
| stage2_full | smoke | 1 | 0 | 0 | 1 | 0 | n/a | 0.00 |

## Object level (test)

| Config | Class | LTPR | LPPV | S | GT regions | Pred regions |
|---|---|---|---|---|---|---|
| stage1_framewise | fire | 28.07 | 46.27 | 66.87 | 114 | 67 |
| stage1_framewise | smoke | n/a | n/a | n/a | 0 | 0 |
| stage2_gamma0 | fire | 26.61 | 37.35 | 71.73 | 124 | 83 |
| stage2_gamma0 | smoke | n/a | 0.00 | n/a | 0 | 2 |
| stage2_full | fire | 22.58 | 25.23 | 59.83 | 124 | 107 |
| stage2_full | smoke | n/a | n/a | n/a | 0 | 0 |

## Speed

- Device: cpu | batch 1 | T 8 | AMP fp16: False | runs 3
- (a) CPU preprocessing: 172.6 ms/frame (source [1920, 1080])
- (b) Model per clip: Stage 1 5617.9 ms | Stage 2 6651.3 ms
- (c) Streaming stride 1: Stage 1 1429.8 ms/frame (0.7 FPS) | Stage 2 with feature cache 2731.3 ms/frame (0.4 FPS)

## Per video (stage2_full, test)

| Video | Frames | IoU BG | IoU Fire | IoU Smoke | Fire FP fr. | Fire FN fr. | Smoke FP fr. | Smoke FN fr. |
|---|---|---|---|---|---|---|---|---|
| 12192259_1080_1920_30fps_pure | 12 | 92.59 | 43.47 | 0.00 | 0 | 0 | 0 | 0 |
