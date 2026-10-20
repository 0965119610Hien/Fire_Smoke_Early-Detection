# So sánh đánh giá trên tập test

Cùng 208 frame test có nhãn (mỗi frame tính 1 lần); ngưỡng chọn trên val cho từng cấu hình.
`[sparse]` = clip 1–2 FPS gồm các frame đã có nhãn; `[dense10fps]` = 7 frame ngữ cảnh 10 FPS trước mỗi frame có nhãn.

## Pixel level (%)

| Cấu hình | Run | Epoch | Frames | IoU Fire [95% CI] | IoU Smoke [95% CI] | Fg-mIoU [95% CI] | All-mIoU |
|---|---|---|---|---|---|---|---|
| stage1_framewise [sparse] | 20261019-060041 | 33 | 208 | 48.93 [17.40, 64.20] | 71.51 [17.51, 87.76] | 60.22 [33.48, 71.02] | 72.46 |
| stage2_gamma0 [sparse] | 20261019-060041 | 2 | 208 | 49.04 [19.33, 62.03] | 72.33 [18.67, 87.40] | 60.68 [33.78, 70.14] | 72.77 |
| stage2_full [sparse] | 20261019-060041 | 2 | 208 | 44.68 [23.67, 53.00] | 69.63 [18.21, 82.82] | 57.16 [30.84, 64.21] | 70.13 |
| stage1_framewise [dense10fps] | 20261019-094419 | 33 | 208 | 49.34 [22.35, 64.43] | 71.11 [15.89, 87.56] | 60.23 [32.62, 71.12] | 72.46 |
| stage2_gamma0 [dense10fps] | 20261019-094419 | 18 | 208 | 52.84 [41.62, 68.15] | 71.38 [16.06, 86.02] | 62.11 [34.89, 73.25] | 73.75 |
| stage2_full [dense10fps] | 20261019-094419 | 18 | 208 | 47.99 [40.08, 64.52] | 61.66 [21.48, 69.60] | 54.82 [36.17, 63.19] | 68.51 |

## Frame level — báo động (lửa hoặc khói)

| Cấu hình | Recall | FAR | Precision | F1 | MCC | Fire Recall / FAR | Smoke Recall / FAR |
|---|---|---|---|---|---|---|---|
| stage1_framewise [sparse] | 95.15 | 32.38 | 74.24 | 83.40 | 0.652 | 67.09 / 16.28 | 89.66 / 14.00 |
| stage2_gamma0 [sparse] | 95.15 | 37.14 | 71.53 | 81.67 | 0.612 | 67.09 / 20.93 | 89.66 / 17.33 |
| stage2_full [sparse] | 100.00 | 60.95 | 61.68 | 76.30 | 0.491 | 75.95 / 32.56 | 94.83 / 39.33 |
| stage1_framewise [dense10fps] | 98.06 | 56.19 | 63.12 | 76.81 | 0.497 | 81.01 / 38.76 | 89.66 / 20.67 |
| stage2_gamma0 [dense10fps] | 99.03 | 32.38 | 75.00 | 85.36 | 0.700 | 70.89 / 13.95 | 91.38 / 19.33 |
| stage2_full [dense10fps] | 100.00 | 51.43 | 65.61 | 79.23 | 0.564 | 75.95 / 12.40 | 98.28 / 46.67 |

## Object level

| Cấu hình | Fire LTPR / LPPV | Smoke LTPR / LPPV | Pred regions fire / smoke |
|---|---|---|---|
| stage1_framewise [sparse] | 30.04 / 26.55 | 62.16 / 8.68 | 275 / 530 |
| stage2_gamma0 [sparse] | 31.80 / 23.48 | 64.86 / 8.38 | 328 / 573 |
| stage2_full [sparse] | 34.28 / 16.04 | 68.92 / 7.63 | 530 / 668 |
| stage1_framewise [dense10fps] | 35.66 / 15.38 | 65.75 / 10.34 | 585 / 464 |
| stage2_gamma0 [dense10fps] | 32.17 / 54.05 | 67.12 / 10.40 | 148 / 471 |
| stage2_full [dense10fps] | 27.62 / 18.95 | 72.60 / 6.54 | 380 / 811 |
