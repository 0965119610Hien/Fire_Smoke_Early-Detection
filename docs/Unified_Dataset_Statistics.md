# Thống kê bộ dữ liệu hợp nhất (unified)

Sinh tự động bởi `scripts/report_unified_dataset.py` lúc 2026-10-23 21:24, từ bản build `20261023-204135` của `scripts/build_unified_dataset.py`.

Quy ước: **frame có nhãn** = frame được giám sát khi train/đánh giá. **Rỗng** = frame có nhãn nhưng không có lửa/khói. Một frame có thể có cả lửa và khói (`both`), nên cột *Có lửa* + *Có khói* lớn hơn số frame.

## 1. Tổng quan

| Split | Video | Frame video có nhãn | Clip sparse | Clip dense | Ảnh tĩnh |
|---|---|---|---|---|---|
| train | 74 (66.7%) | 1266 (67.9%) | 435 | 1266 | 842 (68.0%) |
| val | 19 (17.1%) | 316 (17.0%) | 69 | 316 | 210 (17.0%) |
| test | 18 (16.2%) | 282 (15.1%) | 57 | 282 | 186 (15.0%) |
| Tổng | 111 | 1864 | 561 | 1864 | 1238 |

- Thư mục `data/dataset_video_unified/`: 16,335 file, 1.57 GB (1864 frame có nhãn + 12,607 frame ngữ cảnh 10 FPS, mask đi kèm frame có nhãn).
- Chia: test 15% + 5 fold (xem 1b); bố cục mặc định val = fold 0. Video cùng nguồn luôn nằm cùng split (nhóm nhiều video: [['2016 April - Fire Investigation 1B - Milpitas CA_pure_chunk_02', '2016 April - Fire Investigation 1B - Milpitas CA_pure_chunk_03']]).
- Frame `auto_bg` cũ bị loại khỏi nhãn (chỉ còn là ngữ cảnh): {'0bcf05c5_pexels_cigarette_smoke_8425862_pure': 2, '0d48a1ea_pexels_heavy_smoke_8339804_pure': 2, '14477514_1080_1920_30fps_pure': 1, '36f3eac1_pexels_cigarette_smoke_5075501_pure': 1}.

## 1b. Chia K-fold (MILP)

MILP minimax (scipy/HiGHS): test + K folds by source group; video kinds as hard count bounds. Test cố định 15% + 5 fold (mỗi fold ≈ 17%). Fold k làm val trong lần train thứ k, các fold còn lại làm train; test giống nhau cho mọi fold. File: `data/data_splits/folds/fold<k>/`; `data/data_splits/` = bố cục fold 0.

Video: lệch tránh được tối đa 3.55 điểm % | Time limit reached. (HiGHS Status 13: Time limit reached). Ảnh tĩnh: lệch tránh được tối đa 0.44 điểm % | Time limit reached. (HiGHS Status 13: Time limit reached).

### Số video mỗi phần

| Video | test | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|---|
| neg_hard | 5 | 5 | 4 | 5 | 4 | 5 |
| neg_normal | 1 | 2 | 2 | 1 | 1 | 1 |
| positive | 12 | 12 | 12 | 13 | 13 | 13 |

### Video: số frame và tỉ lệ của mỗi đại lượng (mục tiêu: test 15%, fold0 17%)

| Đại lượng | test | fold0 | fold1 | fold2 | fold3 | fold4 | Lệch tối đa |
|---|---|---|---|---|---|---|---|
| Frame / ảnh | 282 (15.1%) | 316 (17.0%) | 319 (17.1%) | 307 (16.5%) | 325 (17.4%) | 315 (16.9%) | 0.5 |
| Có lửa | 150 (14.9%) | 173 (17.2%) | 174 (17.3%) | 170 (16.9%) | 166 (16.5%) | 171 (17.0%) | 0.5 |
| Có khói | 94 (14.8%) | 108 (17.1%) | 101 (16.0%) | 104 (16.4%) | 124 (19.6%) | 102 (16.1%) | 2.6 |
| fire_only | 107 (15.5%) | 118 (17.1%) | 122 (17.7%) | 111 (16.1%) | 107 (15.5%) | 125 (18.1%) | 1.5 |
| smoke_only | 51 (16.0%) | 53 (16.6%) | 49 (15.4%) | 45 (14.1%) | 65 (20.4%) | 56 (17.6%) | 3.4 |
| both | 43 (13.7%) | 55 (17.5%) | 52 (16.6%) | 59 (18.8%) | 59 (18.8%) | 46 (14.6%) | 2.4 |
| Rỗng | 81 (15.0%) | 90 (16.6%) | 96 (17.7%) | 92 (17.0%) | 94 (17.4%) | 88 (16.3%) | 0.7 |
| Tổng diện tích lửa | 14.9% | 14.6% | 17.6% | 19.0% | 17.6% | 16.2% | 2.4 |
| Tổng diện tích khói | 14.4% | 15.2% | 14.6% | 20.5% | 15.1% | 20.1% | 3.5 |
| Lửa nhỏ (<0.3%) | 30 (14.5%) | 34 (16.4%) | 38 (18.4%) | 35 (16.9%) | 35 (16.9%) | 35 (16.9%) | 1.4 |
| Lửa vừa (0.3–2%) | 64 (15.2%) | 65 (15.4%) | 71 (16.8%) | 83 (19.7%) | 68 (16.1%) | 71 (16.8%) | 2.7 |
| Lửa lớn (≥2%) | 56 (14.9%) | 74 (19.7%) | 65 (17.3%) | 52 (13.9%) | 63 (16.8%) | 65 (17.3%) | 3.1 |
| Khói nhỏ (<3%) | 11 (6.5%) | 22 (13.1%) | 24 (14.3%) | 40 (23.8%) | 51 (30.4%) | 20 (11.9%) | 13.4 |
| Khói vừa (3–15%) | 44 (18.4%) | 47 (19.7%) | 38 (15.9%) | 33 (13.8%) | 34 (14.2%) | 43 (18.0%) | 3.4 |
| Khói lớn (≥15%) | 39 (17.3%) | 39 (17.3%) | 39 (17.3%) | 31 (13.7%) | 39 (17.3%) | 39 (17.3%) | 3.3 |

### Ảnh tĩnh

| Đại lượng | test | fold0 | fold1 | fold2 | fold3 | fold4 | Lệch tối đa |
|---|---|---|---|---|---|---|---|
| Frame / ảnh | 186 (15.0%) | 210 (17.0%) | 211 (17.0%) | 211 (17.0%) | 210 (17.0%) | 210 (17.0%) | 0.0 |
| Có lửa | 110 (15.0%) | 124 (17.0%) | 124 (17.0%) | 125 (17.1%) | 124 (17.0%) | 124 (17.0%) | 0.1 |
| Có khói | 72 (15.0%) | 82 (17.0%) | 82 (17.0%) | 82 (17.0%) | 81 (16.8%) | 82 (17.0%) | 0.2 |
| fire_only | 83 (15.1%) | 93 (16.9%) | 93 (16.9%) | 94 (17.1%) | 94 (17.1%) | 93 (16.9%) | 0.1 |
| smoke_only | 45 (15.0%) | 51 (17.0%) | 51 (17.0%) | 51 (17.0%) | 51 (17.0%) | 51 (17.0%) | 0.0 |
| both | 27 (14.9%) | 31 (17.1%) | 31 (17.1%) | 31 (17.1%) | 30 (16.6%) | 31 (17.1%) | 0.4 |
| Rỗng | 31 (15.0%) | 35 (16.9%) | 36 (17.4%) | 35 (16.9%) | 35 (16.9%) | 35 (16.9%) | 0.4 |
| Tổng diện tích lửa | 15.1% | 17.1% | 17.0% | 17.0% | 16.9% | 16.9% | 0.1 |
| Tổng diện tích khói | 15.0% | 17.0% | 17.0% | 17.0% | 16.9% | 17.0% | 0.1 |
| Lửa nhỏ (<0.3%) | 35 (15.0%) | 39 (16.7%) | 39 (16.7%) | 40 (17.2%) | 40 (17.2%) | 40 (17.2%) | 0.3 |
| Lửa vừa (0.3–2%) | 49 (15.0%) | 56 (17.2%) | 56 (17.2%) | 55 (16.9%) | 55 (16.9%) | 55 (16.9%) | 0.2 |
| Lửa lớn (≥2%) | 26 (15.1%) | 29 (16.9%) | 29 (16.9%) | 30 (17.4%) | 29 (16.9%) | 29 (16.9%) | 0.4 |
| Khói nhỏ (<3%) | 21 (15.2%) | 24 (17.4%) | 24 (17.4%) | 23 (16.7%) | 23 (16.7%) | 23 (16.7%) | 0.4 |
| Khói vừa (3–15%) | 31 (14.7%) | 36 (17.1%) | 36 (17.1%) | 36 (17.1%) | 36 (17.1%) | 36 (17.1%) | 0.3 |
| Khói lớn (≥15%) | 20 (15.2%) | 22 (16.7%) | 22 (16.7%) | 23 (17.4%) | 22 (16.7%) | 23 (17.4%) | 0.4 |

- *Lệch tối đa* tính bằng điểm %. Một phần độ lệch là không tránh được, ví dụ `bucket11` một mình chứa ~30% số frame khói nhỏ; ảnh tĩnh rỗng là frame của video âm tính nên bắt buộc đi theo fold của video đó.

## 2. Dữ liệu video (bố cục mặc định: val = fold 0)

### 2.1 Theo loại frame (category)

| Split | fire_only | smoke_only | both | empty | Tổng frame | Có lửa | Có khói |
|---|---|---|---|---|---|---|---|
| train | 465 | 215 | 216 | 370 | 1266 | 681 | 431 |
| val | 118 | 53 | 55 | 90 | 316 | 173 | 108 |
| test | 107 | 51 | 43 | 81 | 282 | 150 | 94 |
| Tổng | 690 | 319 | 314 | 541 | 1864 | 1004 | 633 |

Tỉ lệ trong từng split:

| Split | fire_only | smoke_only | both | empty |
|---|---|---|---|---|
| train | 36.7% | 17.0% | 17.1% | 29.2% |
| val | 37.3% | 16.8% | 17.4% | 28.5% |
| test | 37.9% | 18.1% | 15.2% | 28.7% |
| Tổng | 37.0% | 17.1% | 16.8% | 29.0% |

Tỉ lệ mỗi loại được chia vào train / val / test (mục tiêu ≈ 68 / 17 / 15 với 5 fold):

| Đại lượng | train | val | test | Tổng |
|---|---|---|---|---|
| Frame | 1266 (67.9%) | 316 (17.0%) | 282 (15.1%) | 1864 |
| Có lửa | 681 (67.8%) | 173 (17.2%) | 150 (14.9%) | 1004 |
| Có khói | 431 (68.1%) | 108 (17.1%) | 94 (14.8%) | 633 |
| Rỗng | 370 (68.4%) | 90 (16.6%) | 81 (15.0%) | 541 |
| fire_only | 465 (67.4%) | 118 (17.1%) | 107 (15.5%) | 690 |
| smoke_only | 215 (67.4%) | 53 (16.6%) | 51 (16.0%) | 319 |
| both | 216 (68.8%) | 55 (17.5%) | 43 (13.7%) | 314 |
| empty | 370 (68.4%) | 90 (16.6%) | 81 (15.0%) | 541 |
| Video dương tính | 51 (68.0%) | 12 (16.0%) | 12 (16.0%) | 75 |
| Video âm tính khó | 18 (64.3%) | 5 (17.9%) | 5 (17.9%) | 28 |
| Video âm tính thường | 5 (62.5%) | 2 (25.0%) | 1 (12.5%) | 8 |

### 2.2 Theo nguồn gán nhãn (round)

| Nguồn | train | val | test | Video | Frame | fire_only | smoke_only | both | empty |
|---|---|---|---|---|---|---|---|---|---|
| Âm tính | 368 | 64 | 80 | 32 | 512 | 0 | 22 | 6 | 484 |
| Round 1 | 418 | 167 | 75 | 40 | 660 | 451 | 71 | 128 | 10 |
| Round 2 | 230 | 85 | 73 | 28 | 388 | 125 | 168 | 49 | 46 |
| Round 3 (mới) | 250 | 0 | 54 | 11 | 304 | 114 | 58 | 131 | 1 |
| Tổng | 1266 | 316 | 282 | 111 | 1864 | 690 | 319 | 314 | 541 |

- Round 1: `fire_smoke_video.v3i` (frame gốc). Round 2: `Frames_video.v1i` (28 video). Round 3: 304 ảnh mới trong `Frames_video.v2i` (11 video). Âm tính: 32 video không có lửa/khói (mask toàn 0).

### 2.3 Theo loại video và chủ đề

| Loại video | train | val | test | Tổng video | Frame |
|---|---|---|---|---|---|
| neg_hard | 18 | 5 | 5 | 28 | 390 |
| neg_normal | 5 | 2 | 1 | 8 | 128 |
| positive | 51 | 12 | 12 | 75 | 1346 |
| Tổng | 74 | 19 | 18 | 111 | 1864 |

| Chủ đề (theo tên file) | Video | Frame | Lửa | Khói | train | val | test |
|---|---|---|---|---|---|---|---|
| pexels (không ghi chủ đề) | 29 | 387 | 386 | 91 | 18 | 5 | 6 |
| âm tính khó (hard negative) | 24 | 384 | 6 | 28 | 18 | 2 | 4 |
| heavy smoke | 14 | 196 | 110 | 93 | 8 | 2 | 4 |
| khác | 4 | 161 | 155 | 50 | 3 | 1 | 0 |
| electrical ignition | 6 | 139 | 109 | 85 | 4 | 0 | 2 |
| âm tính thường (CCTV/cảnh thường) | 8 | 128 | 0 | 0 | 5 | 2 | 1 |
| battery thermal runaway | 4 | 111 | 65 | 95 | 4 | 0 | 0 |
| cigarette smoke | 7 | 90 | 31 | 74 | 5 | 2 | 0 |
| diffused smoke | 5 | 87 | 6 | 87 | 4 | 1 | 0 |
| cooking smoke | 3 | 70 | 69 | 0 | 1 | 1 | 1 |
| visible flame | 2 | 65 | 64 | 19 | 1 | 1 | 0 |
| welding sparks | 3 | 33 | 0 | 0 | 2 | 1 | 0 |
| early smoke | 2 | 13 | 3 | 11 | 1 | 1 | 0 |

### 2.4 Số frame có nhãn mỗi video

| Split | Video | Min | Trung vị | TB | Max | Video < 8 frame (clip sparse bị pad) |
|---|---|---|---|---|---|---|
| train | 74 | 3 | 16 | 17.1 | 53 | 8 |
| val | 19 | 1 | 13 | 16.6 | 50 | 3 |
| test | 18 | 1 | 16 | 15.7 | 31 | 1 |
| Tổng | 111 | 1 | 16 | 16.8 | 53 | 12 |

Video nhiều frame nhất: `e731c09a_pexels_visible_flame_14071614` (53, train), `bucket11` (50, train), `00e18d86-3389d8dc_pure` (50, val), `battery_thermal_runaway_krHU3jlR4Pc` (49, train), `electrical_ignition_5NAkyEmC0IU_continuous_trimmed` (47, train).

### 2.5 Độ phân giải và FPS

| Kích thước lưu (W×H) | train | val | test | Tổng |
|---|---|---|---|---|
| 960x540 | 832 | 117 | 143 | 1092 |
| 540x960 | 288 | 133 | 95 | 516 |
| 640x360 | 0 | 50 | 16 | 66 |
| 640x480 | 32 | 0 | 0 | 32 |
| 650x480 | 0 | 0 | 28 | 28 |
| 854x480 | 26 | 0 | 0 | 26 |
| 960x506 | 22 | 0 | 0 | 22 |
| 960x720 | 0 | 16 | 0 | 16 |
| 960x544 | 15 | 0 | 0 | 15 |
| 600x480 | 14 | 0 | 0 | 14 |
| 240x426 | 13 | 0 | 0 | 13 |
| 960x432 | 13 | 0 | 0 | 13 |
| 960x444 | 11 | 0 | 0 | 11 |

- Hướng khung hình (theo video): {'ngang': 74, 'dọc': 37}.
- Độ phân giải video gốc phổ biến: {'1280x720': 35, '1920x1080': 22, '540x960': 13, '2160x3840': 13, '1080x1920': 8, '2048x1080': 3}.
- FPS video gốc: {6: 1, 20: 1, 24: 14, 25: 32, 30: 50, 32: 1, 50: 3, 60: 9}. Bước lấy frame ngữ cảnh = round(FPS/10).

### 2.6 Diện tích lửa / khói trên frame (% diện tích frame, chỉ tính frame có lớp đó)

| Split — lớp | Số frame | TB % | Trung vị % | P10 % | P90 % | Frame <1% | Frame <0.1% |
|---|---|---|---|---|---|---|---|
| train — Lửa | 681 | 2.99 | 0.89 | 0.18 | 9.09 | 53.0% | 5.3% |
| train — Khói | 431 | 16.88 | 8.57 | 0.66 | 50.34 | 16.7% | 0.9% |
| val — Lửa | 173 | 2.44 | 0.68 | 0.12 | 6.83 | 53.2% | 9.8% |
| val — Khói | 108 | 14.49 | 7.89 | 2.09 | 31.04 | 6.5% | 0.0% |
| test — Lửa | 150 | 2.86 | 1.45 | 0.03 | 9.77 | 39.3% | 10.7% |
| test — Khói | 94 | 15.83 | 13.08 | 1.24 | 28.89 | 9.6% | 0.0% |
| Tổng — Lửa | 1004 | 2.87 | 0.95 | 0.16 | 8.96 | 51.0% | 6.9% |
| Tổng — Khói | 633 | 16.32 | 10.06 | 0.71 | 36.53 | 13.9% | 0.6% |

### 2.7 Số vùng (connected component ≥ 16 px) mỗi frame

| Split | Vùng lửa | TB vùng lửa / frame có lửa | Frame ≥3 vùng lửa | Vùng khói | TB vùng khói / frame có khói |
|---|---|---|---|---|---|
| train | 1433 | 2.1 | 142 | 557 | 1.29 |
| val | 388 | 2.24 | 37 | 172 | 1.59 |
| test | 319 | 2.13 | 36 | 129 | 1.37 |
| Tổng | 2140 | 2.13 | 215 | 858 | 1.36 |

### 2.8 Pixel theo lớp (toàn bộ frame có nhãn)

| Split | Nền % | Lửa % | Khói % | Tỉ lệ nền : lửa : khói |
|---|---|---|---|---|
| train | 92.61 | 1.55 | 5.84 | 60 : 1 : 3.8 |
| val | 93.27 | 1.39 | 5.33 | 67 : 1 : 3.8 |
| test | 93.21 | 1.64 | 5.15 | 57 : 1 : 3.2 |
| Tổng | 92.81 | 1.54 | 5.65 | 60 : 1 : 3.7 |

### 2.9 Clip

| Split | Sparse: clip | Sparse: clip pad (video ngắn) | Dense: clip | Dense: clip chạm đầu video |
|---|---|---|---|---|
| train | 435 | 12 | 1266 | 38 |
| val | 69 | 3 | 316 | 14 |
| test | 57 | 1 | 282 | 11 |

- **Sparse**: T = 8 frame có nhãn liên tiếp (1–2 FPS), stride 2 (train) / 4 (val, test); khoảng trống > 2× nhịp thường của video thì tách đoạn; mọi frame có nhãn nằm trong ≥ 1 clip.
- **Dense**: mỗi frame có nhãn là frame cuối của một clip T = 8; 7 frame trước lấy ở 10 FPS từ video gốc (không có nhãn → mask 255). *Chạm đầu video* = frame có nhãn quá sát đầu video nên các frame đầu clip bị lặp.

## 3. Ảnh tĩnh (Stage 1, `Fire_Smoke.v4i`)

| Split | fire_only | smoke_only | both | empty | Tổng ảnh |
|---|---|---|---|---|---|
| train | 374 | 204 | 123 | 141 | 842 |
| val | 93 | 51 | 31 | 35 | 210 |
| test | 83 | 45 | 27 | 31 | 186 |
| Tổng | 550 | 300 | 181 | 207 | 1238 |

| Split | fire_only | smoke_only | both | empty |
|---|---|---|---|---|
| train | 44.4% | 24.2% | 14.6% | 16.7% |
| val | 44.3% | 24.3% | 14.8% | 16.7% |
| test | 44.6% | 24.2% | 14.5% | 16.7% |
| Tổng | 44.4% | 24.2% | 14.6% | 16.7% |

- Kích thước ảnh phổ biến: {'1280x720': 417, '640x640': 196, '720x405': 158, '416x416': 82, '1920x1080': 80, '960x540': 60, '720x1280': 35, '640x360': 29}. Ảnh vuông 640×640 / 416×416 nhiều khả năng đã bị Roboflow kéo giãn (không còn ảnh gốc).
- Ảnh tĩnh được chia theo nhóm ảnh gần trùng (phash Hamming ≤ 4 + cùng ảnh nguồn Roboflow); nhóm giống frame video val/test bị ép vào đúng split đó.

Diện tích lửa / khói (% diện tích ảnh, chỉ tính ảnh có lớp đó):

| Split — lớp | Số frame | TB % | Trung vị % | P10 % | P90 % | Frame <1% | Frame <0.1% |
|---|---|---|---|---|---|---|---|
| train — Lửa | 497 | 1.73 | 0.77 | 0.09 | 3.86 | 57.1% | 11.7% |
| train — Khói | 327 | 11.57 | 6.82 | 0.49 | 27.73 | 15.0% | 0.3% |
| val — Lửa | 124 | 1.75 | 0.62 | 0.07 | 4.50 | 62.9% | 13.7% |
| val — Khói | 82 | 11.54 | 7.92 | 0.40 | 29.04 | 17.1% | 0.0% |
| test — Lửa | 110 | 1.74 | 0.72 | 0.10 | 5.19 | 60.9% | 12.7% |
| test — Khói | 72 | 11.60 | 8.13 | 0.43 | 27.81 | 20.8% | 0.0% |
| Tổng — Lửa | 731 | 1.74 | 0.71 | 0.08 | 4.28 | 58.7% | 12.2% |
| Tổng — Khói | 481 | 11.57 | 6.98 | 0.46 | 27.96 | 16.2% | 0.2% |

## 4. Danh sách video

### train — 74 video, 1266 frame

| Video | Nguồn | Chủ đề | Frame | Lửa | Khói | Rỗng | Gốc | FPS |
|---|---|---|---|---|---|---|---|---|
| `neg_hard_negatives_BvTBuLESrJ0` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 9 | 7 | 1280x720 | 30 |
| `neg_hard_negatives_erqtaaYXWzY` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 9 | 7 | 1280x720 | 60 |
| `neg_hard_negatives_iaZ3E1mD96Q` | Âm tính | âm tính khó (hard negative) | 16 | 6 | 10 | 6 | 1280x720 | 25 |
| `01c73785_pexels_heavy_smoke_37765564_pure` | Round 1 | heavy smoke | 13 | 13 | 0 | 0 | 2400x1080 | 30 |
| `034f5d53_pexels_cigarette_smoke_8139853_pure` | Round 1 | cigarette smoke | 12 | 0 | 12 | 0 | 1920x1080 | 30 |
| `0bcf05c5_pexels_cigarette_smoke_8425862_pure` | Round 1 | cigarette smoke | 13 | 6 | 10 | 1 | 1920x1080 | 25 |
| `0d48a1ea_pexels_heavy_smoke_8339804_pure` | Round 1 | heavy smoke | 47 | 47 | 0 | 0 | 1920x1080 | 24 |
| `0f39007e_pexels_cooking_smoke_18270789_pure` | Round 1 | cooking smoke | 24 | 24 | 0 | 0 | 540x960 | 30 |
| `12172035_2160_3840_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 13 | 13 | 0 | 0 | 2160x3840 | 30 |
| `12192259_1080_1920_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 19 | 19 | 0 | 0 | 1080x1920 | 30 |
| `12218199_3114_1440_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 11 | 11 | 4 | 0 | 3114x1440 | 30 |
| `12393269_1440_2560_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 11 | 11 | 0 | 0 | 1440x2560 | 30 |
| `12935692_1080_1920_60fps_pure` | Round 1 | pexels (không ghi chủ đề) | 5 | 5 | 0 | 0 | 1080x1920 | 60 |
| `13097120_2160_3840_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 16 | 16 | 0 | 0 | 2160x3840 | 30 |
| `13256420_1080_1920_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 43 | 43 | 0 | 0 | 1080x1920 | 30 |
| `13280461_3840_2160_24fps_pure` | Round 1 | pexels (không ghi chủ đề) | 13 | 13 | 0 | 0 | 3840x2160 | 24 |
| `13589189_2160_3840_60fps_pure` | Round 1 | pexels (không ghi chủ đề) | 11 | 11 | 11 | 0 | 2160x3840 | 60 |
| `13890243_2160_3840_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 11 | 11 | 0 | 0 | 2160x3840 | 30 |
| `13903817_3840_2160_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 23 | 23 | 23 | 0 | 3840x2160 | 30 |
| `14003137_2160_3840_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 6 | 6 | 0 | 0 | 2160x3840 | 30 |
| `14056668_2160_3840_60fps_pure` | Round 1 | pexels (không ghi chủ đề) | 8 | 8 | 6 | 0 | 2160x3840 | 60 |
| `14477514_1080_1920_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 5 | 5 | 0 | 0 | 1080x1920 | 30 |
| `2016 April - Fire Investigation 1B - Milpitas CA_pure_chunk_02` | Round 1 | khác | 31 | 31 | 0 | 0 | 1920x1080 | 30 |
| `2016 April - Fire Investigation 1B - Milpitas CA_pure_chunk_03` | Round 1 | khác | 30 | 30 | 0 | 0 | 1920x1080 | 30 |
| `28f4ec5d_pexels_cigarette_smoke_8139860_pure` | Round 1 | cigarette smoke | 11 | 7 | 11 | 0 | 1920x1080 | 30 |
| `316af409_pexels_heavy_smoke_18105251_pure` | Round 1 | heavy smoke | 13 | 12 | 0 | 1 | 240x426 | 30 |
| `36f3eac1_pexels_cigarette_smoke_5075501_pure` | Round 1 | cigarette smoke | 7 | 5 | 4 | 0 | 1280x720 | 25 |
| `4e0040e9_pexels_heavy_smoke_34992285_pure` | Round 1 | heavy smoke | 14 | 0 | 14 | 0 | 1920x1080 | 60 |
| `13106016_2160_3840_60fps` | Round 2 | pexels (không ghi chủ đề) | 16 | 16 | 0 | 0 | 2160x3840 | 60 |
| `13106082_2160_3840_60fps` | Round 2 | pexels (không ghi chủ đề) | 8 | 8 | 0 | 0 | 2160x3840 | 60 |
| `13270890_2160_3840_60fps` | Round 2 | pexels (không ghi chủ đề) | 14 | 14 | 0 | 0 | 1080x1920 | 30 |
| `13323137_1920_1080_60fps` | Round 2 | pexels (không ghi chủ đề) | 9 | 9 | 0 | 0 | 1920x1080 | 60 |
| `312fe283_pexels_cigarette_smoke_6143909` | Round 2 | cigarette smoke | 11 | 1 | 11 | 0 | 2048x1080 | 25 |
| `38c6467b_pexels_heavy_smoke_35865134` | Round 2 | heavy smoke | 8 | 0 | 8 | 0 | 540x960 | 30 |
| `407b06ea_pexels_early_smoke_8516427` | Round 2 | early smoke | 4 | 0 | 4 | 0 | 2048x1080 | 24 |
| `66d00a95_pexels_diffused_smoke_35582792` | Round 2 | diffused smoke | 15 | 6 | 15 | 0 | 1920x1080 | 25 |
| `6a95cab6_pexels_heavy_smoke_10666165` | Round 2 | heavy smoke | 11 | 11 | 0 | 0 | 540x960 | 30 |
| `6bc652d9_pexels_diffused_smoke_9589423` | Round 2 | diffused smoke | 7 | 0 | 7 | 0 | 2048x1080 | 25 |
| `774e8545_pexels_diffused_smoke_36252846` | Round 2 | diffused smoke | 22 | 0 | 22 | 0 | 540x960 | 25 |
| `7cc52846_pexels_heavy_smoke_35241029` | Round 2 | heavy smoke | 6 | 0 | 6 | 0 | 540x960 | 30 |
| `a7f3eb27_pexels_diffused_smoke_6731483` | Round 2 | diffused smoke | 31 | 0 | 31 | 0 | 540x960 | 25 |
| `battery_thermal_runaway_krHU3jlR4Pc` | Round 2 | battery thermal runaway | 49 | 25 | 48 | 0 | 1280x720 | 25 |
| `battery_thermal_runaway_ld00r-tEEJ4_continuous_trimmed` | Round 3 (mới) | battery thermal runaway | 33 | 18 | 30 | 0 | 1280x720 | 25 |
| `battery_thermal_runaway_o7szQvxW_KM` | Round 3 (mới) | battery thermal runaway | 10 | 7 | 10 | 0 | 854x480 | 30 |
| `battery_thermal_runaway_oieH2wwDGzo` | Round 3 (mới) | battery thermal runaway | 19 | 15 | 7 | 0 | 1280x720 | 30 |
| `bucket11` | Round 3 (mới) | khác | 50 | 44 | 50 | 0 | 1280x720 | 25 |
| `e731c09a_pexels_visible_flame_14071614` | Round 3 (mới) | visible flame | 53 | 52 | 7 | 0 | 1920x1080 | 50 |
| `electrical_ignition_5NAkyEmC0IU_continuous_trimmed` | Round 3 (mới) | electrical ignition | 47 | 47 | 38 | 0 | 1280x720 | 25 |
| `electrical_ignition_Kzv1Qn1pPpc` | Round 3 (mới) | electrical ignition | 14 | 11 | 6 | 1 | 600x480 | 25 |
| `electrical_ignition_Roltnt5J_e0` | Round 3 (mới) | electrical ignition | 15 | 12 | 3 | 0 | 1270x720 | 30 |
| `electrical_ignition_g6T6G_-htVw_continuous_trimmed` | Round 3 (mới) | electrical ignition | 9 | 9 | 5 | 0 | 1920x1080 | 24 |
| `neg_normal_G8z74loJO7U_ClearPix_Camera_Warehouse_` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 6 |
| `neg_normal_Od6EeCWytZo_The_famous_Shibuya_crossin` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 25 |
| `neg_normal_YqkiOmfMQjk_Garmin_20_Dash_Cam___Sunny` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_normal_pc8eQZW9E8Q_YI_Indoor_Camera___Sample_` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 20 |
| `neg_normal_rBZk3mR6VM0_HD_Convenience_Store_CCTV_` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_1_OXM4mr_i0` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 640x480 | 25 |
| `neg_hard_negatives_6rCG4oF2qrA` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_Afa6qOVfn7M` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 25 |
| `neg_hard_negatives_CIvTKFlBTaw` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 25 |
| `neg_hard_negatives_Gc-70RMuyBI` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 854x480 | 25 |
| `neg_hard_negatives_OGFxYoQHGco` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_P2QxCFyusSA` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_aM2GDiFoy8Q` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_fq3l2_IRW_0` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 24 |
| `neg_hard_negatives_k1leKhvOtqg` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_m7QxBFe8wZw` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 25 |
| `neg_hard_negatives_r-9qPsm_pvc` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_tMUbJs2iweQ` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 24 |
| `neg_hard_negatives_zFDB6XlHWVU` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 24 |
| `neg_hard_negatives_zzclsZ5v0SM` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 640x480 | 30 |
| `0aefc692_pexels_welding_sparks_4941457_pure` | Round 1 | welding sparks | 8 | 0 | 0 | 8 | 1920x1080 | 25 |
| `0fd09443_pexels_welding_sparks_14314814` | Round 2 | welding sparks | 3 | 0 | 0 | 3 | 1920x1080 | 50 |
| `6c3dee00_pexels_heavy_smoke_6618036` | Round 2 | heavy smoke | 16 | 0 | 0 | 16 | 1920x1080 | 24 |

### val — 19 video, 316 frame

| Video | Nguồn | Chủ đề | Frame | Lửa | Khói | Rỗng | Gốc | FPS |
|---|---|---|---|---|---|---|---|---|
| `00e18d86-3389d8dc_pure` | Round 1 | khác | 50 | 50 | 0 | 0 | 640x360 | 30 |
| `02af2970_pexels_cooking_smoke_3472823_pure` | Round 1 | cooking smoke | 45 | 45 | 0 | 0 | 1920x1080 | 24 |
| `04d83ee6_pexels_early_smoke_7926437_pure` | Round 1 | early smoke | 9 | 3 | 7 | 0 | 1920x1080 | 25 |
| `12150804_1080_1920_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 6 | 6 | 6 | 0 | 1080x1920 | 30 |
| `13570717_2160_3840_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 13 | 13 | 13 | 0 | 2160x3840 | 30 |
| `14112316_1440_2560_60fps_pure` | Round 1 | pexels (không ghi chủ đề) | 11 | 11 | 0 | 0 | 1440x2560 | 60 |
| `14283156_1080_1920_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 21 | 21 | 21 | 0 | 1080x1920 | 30 |
| `394a2a56_pexels_visible_flame_6985324_pure` | Round 1 | visible flame | 12 | 12 | 12 | 0 | 540x960 | 24 |
| `012f0550_pexels_heavy_smoke_36164483` | Round 2 | heavy smoke | 11 | 0 | 11 | 0 | 540x960 | 24 |
| `3aa2be24_pexels_cigarette_smoke_5320212` | Round 2 | cigarette smoke | 25 | 12 | 15 | 0 | 540x960 | 25 |
| `47edc2fb_pexels_cigarette_smoke_9644281` | Round 2 | cigarette smoke | 11 | 0 | 11 | 0 | 1280x720 | 25 |
| `adc26840_pexels_diffused_smoke_6956078` | Round 2 | diffused smoke | 12 | 0 | 12 | 0 | 540x960 | 25 |
| `neg_normal_bpxhdKKRD5c_Cottonwood_tree_and_windst` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_normal_cmEmAzR0ybA_How_to_Protect_Parking_Lot` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_9bas4RV3btM` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 960x720 | 30 |
| `neg_hard_negatives_fHGR3iYfX4w` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 25 |
| `05ca4b9b_pexels_welding_sparks_34771762` | Round 2 | welding sparks | 22 | 0 | 0 | 22 | 540x960 | 32 |
| `14379036_1920_1080_50fps` | Round 2 | pexels (không ghi chủ đề) | 1 | 0 | 0 | 1 | 1920x1080 | 50 |
| `9a06358f_pexels_heavy_smoke_3568453` | Round 2 | heavy smoke | 3 | 0 | 0 | 3 | 1920x1080 | 30 |

### test — 18 video, 282 frame

| Video | Nguồn | Chủ đề | Frame | Lửa | Khói | Rỗng | Gốc | FPS |
|---|---|---|---|---|---|---|---|---|
| `13676716_2160_3840_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 11 | 11 | 7 | 0 | 2160x3840 | 30 |
| `13976903_1080_1920_25fps_pure` | Round 1 | pexels (không ghi chủ đề) | 16 | 16 | 0 | 0 | 1080x1920 | 25 |
| `17116108-hd_1920_1080_25fps_pure` | Round 1 | pexels (không ghi chủ đề) | 10 | 10 | 0 | 0 | 1920x1080 | 25 |
| `4a37a9df_pexels_heavy_smoke_17202470_pure` | Round 1 | heavy smoke | 11 | 0 | 11 | 0 | 1920x1080 | 30 |
| `4cf31d6f_pexels_heavy_smoke_36068162_pure` | Round 1 | heavy smoke | 16 | 0 | 16 | 0 | 1920x1080 | 25 |
| `6770d887_pexels_heavy_smoke_28478389_pure` | Round 1 | heavy smoke | 11 | 11 | 11 | 0 | 540x960 | 30 |
| `13269425_2160_3840_30fps` | Round 2 | pexels (không ghi chủ đề) | 31 | 31 | 0 | 0 | 2160x3840 | 30 |
| `13845809_2160_3840_25fps` | Round 2 | pexels (không ghi chủ đề) | 13 | 13 | 0 | 0 | 2160x3840 | 25 |
| `13862697_2160_3840_25fps` | Round 2 | pexels (không ghi chủ đề) | 12 | 12 | 0 | 0 | 2160x3840 | 25 |
| `6eeb70d4_pexels_heavy_smoke_34004202` | Round 2 | heavy smoke | 16 | 16 | 16 | 0 | 1920x1080 | 24 |
| `electrical_ignition_VimYV-zgYo0_continuous_trimmed` | Round 3 (mới) | electrical ignition | 28 | 6 | 28 | 0 | 650x480 | 30 |
| `electrical_ignition_qRoT41qla9I_continuous_trimmed` | Round 3 (mới) | electrical ignition | 26 | 24 | 5 | 0 | 1280x720 | 30 |
| `neg_normal_1gi5qn1khVk_4MP_PTZ_Camera_with_Night_` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 25 |
| `neg_hard_negatives_jKMNSvpB9dY` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 640x360 | 25 |
| `neg_hard_negatives_lqNZtZFUgcE` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_nJe-Y9QM0cA` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_wb6iirzZKlk` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 24 |
| `816fd10a_pexels_cooking_smoke_37848046` | Round 2 | cooking smoke | 1 | 0 | 0 | 1 | 540x960 | 24 |

## 5. Lưu ý khi dùng

- Val (19 video) và test (18 video) vẫn nhỏ: kết quả đánh giá có khoảng tin cậy rộng, nên báo cáo kèm bootstrap CI theo video.
- Split hiện tại khác split ngày 2026-10-19, nên số liệu test cũ không so sánh trực tiếp được.
- Với K-fold: chọn siêu tham số theo trung bình val của các fold; Stage 2 fold k phải dùng Stage 1 fold k (`segformer_spatial_best_fold<k>.pth`) vì ảnh tĩnh giống frame video đã được xếp cùng fold.
- Lửa chiếm rất ít pixel (xem 2.6, 2.8): phần lớn frame có lửa < 1% diện tích → cần class weight / Dice và metric object-level.
- 3 video `welding sparks` (tia hàn) được gán nhãn lửa/khói (0 frame lửa, 0 frame khói). Nếu muốn dùng chúng làm mẫu âm tính khó thì cần xem lại nhãn.
- Frame chưa được export từ Roboflow chỉ làm ngữ cảnh, nên video round 2/3 gần như không có frame rỗng được giám sát (47/692 frame). Frame rỗng chủ yếu đến từ 32 video âm tính.
- *Chủ đề* lấy từ từ khóa lúc crawl (tên file), không phải từ nhãn: ví dụ một số video `heavy smoke` / `cooking smoke` chỉ có nhãn lửa.
