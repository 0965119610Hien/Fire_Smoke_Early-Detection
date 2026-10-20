# Thống kê bộ dữ liệu hợp nhất (unified)

Sinh tự động bởi `scripts/report_unified_dataset.py` lúc 2026-10-20 22:03, từ bản build `20261020-214229` của `scripts/build_unified_dataset.py`.

Quy ước: **frame có nhãn** = frame được giám sát khi train/đánh giá. **Rỗng** = frame có nhãn nhưng không có lửa/khói. Một frame có thể có cả lửa và khói (`both`), nên cột *Có lửa* + *Có khói* lớn hơn số frame.

## 1. Tổng quan

| Split | Video | Frame video có nhãn | Clip sparse | Clip dense | Ảnh tĩnh |
|---|---|---|---|---|---|
| train | 75 (67.6%) | 1270 (68.1%) | 439 | 1270 | 845 (68.3%) |
| val | 19 (17.1%) | 315 (16.9%) | 67 | 315 | 211 (17.0%) |
| test | 17 (15.3%) | 279 (15.0%) | 57 | 279 | 182 (14.7%) |
| Tổng | 111 | 1864 | 563 | 1864 | 1238 |

- Thư mục `data/dataset_video_unified/`: 16,335 file, 1.57 GB (1864 frame có nhãn + 12,607 frame ngữ cảnh 10 FPS, mask đi kèm frame có nhãn).
- Chia: test 15% + 5 fold (xem 1b); bố cục mặc định val = fold 0. Video cùng nguồn luôn nằm cùng split (nhóm nhiều video: [['2016 April - Fire Investigation 1B - Milpitas CA_pure_chunk_02', '2016 April - Fire Investigation 1B - Milpitas CA_pure_chunk_03']]).
- Frame `auto_bg` cũ bị loại khỏi nhãn (chỉ còn là ngữ cảnh): {'0bcf05c5_pexels_cigarette_smoke_8425862_pure': 2, '0d48a1ea_pexels_heavy_smoke_8339804_pure': 2, '14477514_1080_1920_30fps_pure': 1, '36f3eac1_pexels_cigarette_smoke_5075501_pure': 1}.

## 1b. Chia K-fold (MILP)

MILP minimax (scipy/HiGHS): test + K folds by source group; video kinds as hard count bounds. Test cố định 15% + 5 fold (mỗi fold ≈ 17%). Fold k làm val trong lần train thứ k, các fold còn lại làm train; test giống nhau cho mọi fold. File: `data/data_splits/folds/fold<k>/`; `data/data_splits/` = bố cục fold 0.

Video: lệch tránh được tối đa 2.78 điểm % | Time limit reached. (HiGHS Status 13: Time limit reached). Ảnh tĩnh: lệch tránh được tối đa 2.29 điểm % | Time limit reached. (HiGHS Status 13: Time limit reached).

### Số video mỗi phần

| Video | test | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|---|
| neg_hard | 3 | 4 | 5 | 4 | 4 | 4 |
| neg_normal | 2 | 1 | 1 | 2 | 1 | 1 |
| positive | 12 | 14 | 13 | 14 | 13 | 13 |

### Video: số frame và tỉ lệ của mỗi đại lượng (mục tiêu: test 15%, fold0 17%)

| Đại lượng | test | fold0 | fold1 | fold2 | fold3 | fold4 | Lệch tối đa |
|---|---|---|---|---|---|---|---|
| Frame / ảnh | 279 (15.0%) | 315 (16.9%) | 324 (17.4%) | 324 (17.4%) | 308 (16.5%) | 314 (16.8%) | 0.5 |
| Có lửa | 150 (14.7%) | 172 (16.9%) | 173 (17.0%) | 175 (17.2%) | 173 (17.0%) | 174 (17.1%) | 0.3 |
| Có khói | 99 (14.8%) | 125 (18.7%) | 113 (16.9%) | 112 (16.8%) | 107 (16.0%) | 112 (16.8%) | 1.7 |
| fire_only | 100 (14.8%) | 107 (15.8%) | 115 (17.0%) | 115 (17.0%) | 119 (17.6%) | 120 (17.8%) | 1.2 |
| smoke_only | 49 (15.0%) | 60 (18.3%) | 55 (16.8%) | 52 (15.9%) | 53 (16.2%) | 58 (17.7%) | 1.3 |
| both | 50 (14.7%) | 65 (19.1%) | 58 (17.0%) | 60 (17.6%) | 54 (15.8%) | 54 (15.8%) | 2.1 |
| Rỗng | 80 (15.4%) | 83 (16.0%) | 96 (18.5%) | 97 (18.7%) | 82 (15.8%) | 82 (15.8%) | 1.7 |
| Tổng diện tích lửa | 14.5% | 16.9% | 17.6% | 17.1% | 17.1% | 16.8% | 0.6 |
| Tổng diện tích khói | 14.5% | 17.0% | 17.2% | 16.9% | 19.1% | 15.2% | 2.1 |
| Lửa nhỏ (<0.3%) | 31 (14.6%) | 36 (16.9%) | 36 (16.9%) | 36 (16.9%) | 37 (17.4%) | 37 (17.4%) | 0.4 |
| Lửa vừa (0.3–2%) | 64 (14.9%) | 72 (16.7%) | 73 (17.0%) | 71 (16.5%) | 74 (17.2%) | 76 (17.7%) | 0.7 |
| Lửa lớn (≥2%) | 55 (14.7%) | 64 (17.1%) | 64 (17.1%) | 68 (18.2%) | 62 (16.6%) | 61 (16.3%) | 1.2 |
| Khói nhỏ (<3%) | 23 (13.8%) | 50 (29.9%) | 23 (13.8%) | 21 (12.6%) | 28 (16.8%) | 22 (13.2%) | 12.9 |
| Khói vừa (3–15%) | 38 (14.9%) | 38 (14.9%) | 45 (17.6%) | 47 (18.4%) | 42 (16.5%) | 45 (17.6%) | 2.1 |
| Khói lớn (≥15%) | 38 (15.4%) | 37 (15.0%) | 45 (18.3%) | 44 (17.9%) | 37 (15.0%) | 45 (18.3%) | 2.0 |

### Ảnh tĩnh

| Đại lượng | test | fold0 | fold1 | fold2 | fold3 | fold4 | Lệch tối đa |
|---|---|---|---|---|---|---|---|
| Frame / ảnh | 182 (14.7%) | 211 (17.0%) | 211 (17.0%) | 212 (17.1%) | 211 (17.0%) | 211 (17.0%) | 0.3 |
| Có lửa | 110 (15.0%) | 124 (17.0%) | 124 (17.0%) | 124 (17.0%) | 124 (17.0%) | 125 (17.1%) | 0.1 |
| Có khói | 76 (15.0%) | 86 (17.0%) | 86 (17.0%) | 86 (17.0%) | 86 (17.0%) | 87 (17.2%) | 0.2 |
| fire_only | 83 (15.1%) | 93 (16.9%) | 93 (16.9%) | 94 (17.1%) | 93 (16.9%) | 94 (17.1%) | 0.1 |
| smoke_only | 49 (15.0%) | 55 (16.9%) | 55 (16.9%) | 56 (17.2%) | 55 (16.9%) | 56 (17.2%) | 0.2 |
| both | 27 (14.9%) | 31 (17.1%) | 31 (17.1%) | 30 (16.6%) | 31 (17.1%) | 31 (17.1%) | 0.4 |
| Rỗng | 23 (12.7%) | 32 (17.7%) | 32 (17.7%) | 32 (17.7%) | 32 (17.7%) | 30 (16.6%) | 2.3 |
| Tổng diện tích lửa | 15.0% | 17.0% | 17.0% | 17.0% | 17.0% | 17.0% | 0.0 |
| Tổng diện tích khói | 15.0% | 16.9% | 17.0% | 17.0% | 17.0% | 17.0% | 0.1 |
| Lửa nhỏ (<0.3%) | 35 (15.0%) | 40 (17.2%) | 40 (17.2%) | 39 (16.7%) | 40 (17.2%) | 39 (16.7%) | 0.3 |
| Lửa vừa (0.3–2%) | 49 (15.0%) | 55 (16.9%) | 55 (16.9%) | 56 (17.2%) | 55 (16.9%) | 56 (17.2%) | 0.2 |
| Lửa lớn (≥2%) | 26 (15.1%) | 29 (16.9%) | 29 (16.9%) | 29 (16.9%) | 29 (16.9%) | 30 (17.4%) | 0.4 |
| Khói nhỏ (<3%) | 23 (15.1%) | 26 (17.1%) | 26 (17.1%) | 26 (17.1%) | 25 (16.4%) | 26 (17.1%) | 0.6 |
| Khói vừa (3–15%) | 33 (15.0%) | 37 (16.8%) | 37 (16.8%) | 37 (16.8%) | 38 (17.3%) | 38 (17.3%) | 0.3 |
| Khói lớn (≥15%) | 20 (14.8%) | 23 (17.0%) | 23 (17.0%) | 23 (17.0%) | 23 (17.0%) | 23 (17.0%) | 0.2 |

- *Lệch tối đa* tính bằng điểm %. Một phần độ lệch là không tránh được, ví dụ `bucket11` một mình chứa ~30% số frame khói nhỏ; ảnh tĩnh rỗng là frame của video âm tính nên bắt buộc đi theo fold của video đó.

## 2. Dữ liệu video (bố cục mặc định: val = fold 0)

### 2.1 Theo loại frame (category)

| Split | fire_only | smoke_only | both | empty | Tổng frame | Có lửa | Có khói |
|---|---|---|---|---|---|---|---|
| train | 469 | 218 | 226 | 357 | 1270 | 695 | 444 |
| val | 107 | 60 | 65 | 83 | 315 | 172 | 125 |
| test | 100 | 49 | 50 | 80 | 279 | 150 | 99 |
| Tổng | 676 | 327 | 341 | 520 | 1864 | 1017 | 668 |

Tỉ lệ trong từng split:

| Split | fire_only | smoke_only | both | empty |
|---|---|---|---|---|
| train | 36.9% | 17.2% | 17.8% | 28.1% |
| val | 34.0% | 19.0% | 20.6% | 26.3% |
| test | 35.8% | 17.6% | 17.9% | 28.7% |
| Tổng | 36.3% | 17.5% | 18.3% | 27.9% |

Tỉ lệ mỗi loại được chia vào train / val / test (mục tiêu ≈ 68 / 17 / 15 với 5 fold):

| Đại lượng | train | val | test | Tổng |
|---|---|---|---|---|
| Frame | 1270 (68.1%) | 315 (16.9%) | 279 (15.0%) | 1864 |
| Có lửa | 695 (68.3%) | 172 (16.9%) | 150 (14.7%) | 1017 |
| Có khói | 444 (66.5%) | 125 (18.7%) | 99 (14.8%) | 668 |
| Rỗng | 357 (68.7%) | 83 (16.0%) | 80 (15.4%) | 520 |
| fire_only | 469 (69.4%) | 107 (15.8%) | 100 (14.8%) | 676 |
| smoke_only | 218 (66.7%) | 60 (18.3%) | 49 (15.0%) | 327 |
| both | 226 (66.3%) | 65 (19.1%) | 50 (14.7%) | 341 |
| empty | 357 (68.7%) | 83 (16.0%) | 80 (15.4%) | 520 |
| Video dương tính | 53 (67.1%) | 14 (17.7%) | 12 (15.2%) | 79 |
| Video âm tính khó | 17 (70.8%) | 4 (16.7%) | 3 (12.5%) | 24 |
| Video âm tính thường | 5 (62.5%) | 1 (12.5%) | 2 (25.0%) | 8 |

### 2.2 Theo nguồn gán nhãn (round)

| Nguồn | train | val | test | Video | Frame | fire_only | smoke_only | both | empty |
|---|---|---|---|---|---|---|---|---|---|
| Âm tính | 352 | 80 | 80 | 32 | 512 | 0 | 0 | 0 | 512 |
| Round 1 | 510 | 86 | 64 | 40 | 660 | 420 | 80 | 158 | 2 |
| Round 2 | 328 | 40 | 20 | 28 | 388 | 128 | 189 | 66 | 5 |
| Round 3 (mới) | 80 | 109 | 115 | 11 | 304 | 128 | 58 | 117 | 1 |
| Tổng | 1270 | 315 | 279 | 111 | 1864 | 676 | 327 | 341 | 520 |

- Round 1: `fire_smoke_video.v3i` (frame gốc). Round 2: `Frames_video.v1i` (28 video). Round 3: 304 ảnh mới trong `Frames_video.v2i` (11 video). Âm tính: 32 video không có lửa/khói (mask toàn 0).

### 2.3 Theo loại video và chủ đề

| Loại video | train | val | test | Tổng video | Frame |
|---|---|---|---|---|---|
| neg_hard | 17 | 4 | 3 | 24 | 384 |
| neg_normal | 5 | 1 | 2 | 8 | 128 |
| positive | 53 | 14 | 12 | 79 | 1352 |
| Tổng | 75 | 19 | 17 | 111 | 1864 |

| Chủ đề (theo tên file) | Video | Frame | Lửa | Khói | train | val | test |
|---|---|---|---|---|---|---|---|
| pexels (không ghi chủ đề) | 29 | 387 | 385 | 113 | 22 | 4 | 3 |
| âm tính khó (hard negative) | 24 | 384 | 0 | 0 | 17 | 4 | 3 |
| heavy smoke | 14 | 196 | 110 | 117 | 8 | 3 | 3 |
| khác | 4 | 161 | 155 | 50 | 2 | 2 | 0 |
| electrical ignition | 6 | 139 | 109 | 85 | 2 | 1 | 3 |
| âm tính thường (CCTV/cảnh thường) | 8 | 128 | 0 | 0 | 5 | 1 | 2 |
| battery thermal runaway | 4 | 111 | 65 | 81 | 2 | 1 | 1 |
| cigarette smoke | 7 | 90 | 31 | 75 | 4 | 2 | 1 |
| diffused smoke | 5 | 87 | 6 | 87 | 5 | 0 | 0 |
| cooking smoke | 3 | 70 | 69 | 0 | 3 | 0 | 0 |
| visible flame | 2 | 65 | 64 | 19 | 1 | 0 | 1 |
| welding sparks | 3 | 33 | 20 | 30 | 2 | 1 | 0 |
| early smoke | 2 | 13 | 3 | 11 | 2 | 0 | 0 |

### 2.4 Số frame có nhãn mỗi video

| Split | Video | Min | Trung vị | TB | Max | Video < 8 frame (clip sparse bị pad) |
|---|---|---|---|---|---|---|
| train | 75 | 1 | 16 | 16.9 | 49 | 8 |
| val | 19 | 3 | 12 | 16.6 | 50 | 4 |
| test | 17 | 8 | 15 | 16.4 | 53 | 0 |
| Tổng | 111 | 1 | 16 | 16.8 | 53 | 12 |

Video nhiều frame nhất: `e731c09a_pexels_visible_flame_14071614` (53, test), `bucket11` (50, val), `00e18d86-3389d8dc_pure` (50, val), `battery_thermal_runaway_krHU3jlR4Pc` (49, train), `electrical_ignition_5NAkyEmC0IU_continuous_trimmed` (47, train).

### 2.5 Độ phân giải và FPS

| Kích thước lưu (W×H) | train | val | test | Tổng |
|---|---|---|---|---|
| 960x540 | 755 | 184 | 153 | 1092 |
| 540x960 | 447 | 38 | 31 | 516 |
| 640x360 | 0 | 50 | 16 | 66 |
| 640x480 | 32 | 0 | 0 | 32 |
| 650x480 | 0 | 0 | 28 | 28 |
| 854x480 | 0 | 16 | 10 | 26 |
| 960x506 | 11 | 11 | 0 | 22 |
| 960x720 | 0 | 16 | 0 | 16 |
| 960x544 | 0 | 0 | 15 | 15 |
| 600x480 | 14 | 0 | 0 | 14 |
| 240x426 | 0 | 0 | 13 | 13 |
| 960x432 | 0 | 0 | 13 | 13 |
| 960x444 | 11 | 0 | 0 | 11 |

- Hướng khung hình (theo video): {'ngang': 74, 'dọc': 37}.
- Độ phân giải video gốc phổ biến: {'1280x720': 35, '1920x1080': 22, '540x960': 13, '2160x3840': 13, '1080x1920': 8, '2048x1080': 3}.
- FPS video gốc: {6: 1, 20: 1, 24: 14, 25: 32, 30: 50, 32: 1, 50: 3, 60: 9}. Bước lấy frame ngữ cảnh = round(FPS/10).

### 2.6 Diện tích lửa / khói trên frame (% diện tích frame, chỉ tính frame có lớp đó)

| Split — lớp | Số frame | TB % | Trung vị % | P10 % | P90 % | Frame <1% | Frame <0.1% |
|---|---|---|---|---|---|---|---|
| train — Lửa | 695 | 2.86 | 1.07 | 0.13 | 8.58 | 48.9% | 8.8% |
| train — Khói | 444 | 16.16 | 10.75 | 1.34 | 35.26 | 7.9% | 0.5% |
| val — Lửa | 172 | 2.84 | 0.57 | 0.24 | 8.00 | 58.1% | 1.7% |
| val — Khói | 125 | 14.30 | 7.44 | 0.62 | 48.09 | 31.2% | 0.0% |
| test — Lửa | 150 | 2.81 | 0.82 | 0.19 | 9.37 | 53.3% | 4.7% |
| test — Khói | 99 | 15.37 | 12.70 | 0.58 | 29.78 | 13.1% | 2.0% |
| Tổng — Lửa | 1017 | 2.85 | 0.95 | 0.15 | 8.70 | 51.1% | 7.0% |
| Tổng — Khói | 668 | 15.69 | 10.95 | 0.77 | 35.55 | 13.0% | 0.6% |

### 2.7 Số vùng (connected component ≥ 16 px) mỗi frame

| Split | Vùng lửa | TB vùng lửa / frame có lửa | Frame ≥3 vùng lửa | Vùng khói | TB vùng khói / frame có khói |
|---|---|---|---|---|---|
| train | 1592 | 2.29 | 163 | 625 | 1.41 |
| val | 267 | 1.55 | 24 | 168 | 1.34 |
| test | 278 | 1.85 | 32 | 142 | 1.43 |
| Tổng | 2137 | 2.1 | 219 | 935 | 1.4 |

### 2.8 Pixel theo lớp (toàn bộ frame có nhãn)

| Split | Nền % | Lửa % | Khói % | Tỉ lệ nền : lửa : khói |
|---|---|---|---|---|
| train | 92.75 | 1.52 | 5.73 | 61 : 1 : 3.8 |
| val | 92.21 | 1.64 | 6.14 | 56 : 1 : 3.7 |
| test | 93.57 | 1.51 | 4.92 | 62 : 1 : 3.3 |
| Tổng | 92.77 | 1.54 | 5.69 | 60 : 1 : 3.7 |

### 2.9 Clip

| Split | Sparse: clip | Sparse: clip pad (video ngắn) | Dense: clip | Dense: clip chạm đầu video |
|---|---|---|---|---|
| train | 439 | 10 | 1270 | 44 |
| val | 67 | 4 | 315 | 12 |
| test | 57 | 2 | 279 | 7 |

- **Sparse**: T = 8 frame có nhãn liên tiếp (1–2 FPS), stride 2 (train) / 4 (val, test); khoảng trống > 2× nhịp thường của video thì tách đoạn; mọi frame có nhãn nằm trong ≥ 1 clip.
- **Dense**: mỗi frame có nhãn là frame cuối của một clip T = 8; 7 frame trước lấy ở 10 FPS từ video gốc (không có nhãn → mask 255). *Chạm đầu video* = frame có nhãn quá sát đầu video nên các frame đầu clip bị lặp.

## 3. Ảnh tĩnh (Stage 1, `Fire_Smoke.v4i`)

| Split | fire_only | smoke_only | both | empty | Tổng ảnh |
|---|---|---|---|---|---|
| train | 374 | 222 | 123 | 126 | 845 |
| val | 93 | 55 | 31 | 32 | 211 |
| test | 83 | 49 | 27 | 23 | 182 |
| Tổng | 550 | 326 | 181 | 181 | 1238 |

| Split | fire_only | smoke_only | both | empty |
|---|---|---|---|---|
| train | 44.3% | 26.3% | 14.6% | 14.9% |
| val | 44.1% | 26.1% | 14.7% | 15.2% |
| test | 45.6% | 26.9% | 14.8% | 12.6% |
| Tổng | 44.4% | 26.3% | 14.6% | 14.6% |

- Kích thước ảnh phổ biến: {'1280x720': 417, '640x640': 196, '720x405': 158, '416x416': 82, '1920x1080': 80, '960x540': 60, '720x1280': 35, '640x360': 29}. Ảnh vuông 640×640 / 416×416 nhiều khả năng đã bị Roboflow kéo giãn (không còn ảnh gốc).
- Ảnh tĩnh được chia theo nhóm ảnh gần trùng (phash Hamming ≤ 4 + cùng ảnh nguồn Roboflow); nhóm giống frame video val/test bị ép vào đúng split đó.

Diện tích lửa / khói (% diện tích ảnh, chỉ tính ảnh có lớp đó):

| Split — lớp | Số frame | TB % | Trung vị % | P10 % | P90 % | Frame <1% | Frame <0.1% |
|---|---|---|---|---|---|---|---|
| train — Lửa | 497 | 1.74 | 0.76 | 0.08 | 4.43 | 57.5% | 12.3% |
| train — Khói | 345 | 11.24 | 6.87 | 0.42 | 26.69 | 18.3% | 0.3% |
| val — Lửa | 124 | 1.74 | 0.78 | 0.09 | 4.30 | 54.8% | 13.7% |
| val — Khói | 86 | 11.20 | 6.09 | 0.96 | 28.61 | 10.5% | 0.0% |
| test — Lửa | 110 | 1.73 | 0.45 | 0.10 | 3.77 | 68.2% | 10.0% |
| test — Khói | 76 | 11.26 | 7.83 | 0.47 | 29.40 | 14.5% | 0.0% |
| Tổng — Lửa | 731 | 1.74 | 0.71 | 0.08 | 4.28 | 58.7% | 12.2% |
| Tổng — Khói | 507 | 11.24 | 6.76 | 0.45 | 27.33 | 16.4% | 0.2% |

## 4. Danh sách video

### train — 75 video, 1270 frame

| Video | Nguồn | Chủ đề | Frame | Lửa | Khói | Rỗng | Gốc | FPS |
|---|---|---|---|---|---|---|---|---|
| `02af2970_pexels_cooking_smoke_3472823_pure` | Round 1 | cooking smoke | 45 | 45 | 0 | 0 | 1920x1080 | 24 |
| `04d83ee6_pexels_early_smoke_7926437_pure` | Round 1 | early smoke | 9 | 3 | 7 | 0 | 1920x1080 | 25 |
| `0aefc692_pexels_welding_sparks_4941457_pure` | Round 1 | welding sparks | 8 | 0 | 8 | 0 | 1920x1080 | 25 |
| `0bcf05c5_pexels_cigarette_smoke_8425862_pure` | Round 1 | cigarette smoke | 13 | 6 | 10 | 1 | 1920x1080 | 25 |
| `0d48a1ea_pexels_heavy_smoke_8339804_pure` | Round 1 | heavy smoke | 47 | 47 | 0 | 0 | 1920x1080 | 24 |
| `0f39007e_pexels_cooking_smoke_18270789_pure` | Round 1 | cooking smoke | 24 | 24 | 0 | 0 | 540x960 | 30 |
| `12150804_1080_1920_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 6 | 6 | 6 | 0 | 1080x1920 | 30 |
| `12172035_2160_3840_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 13 | 13 | 0 | 0 | 2160x3840 | 30 |
| `12192259_1080_1920_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 19 | 19 | 0 | 0 | 1080x1920 | 30 |
| `12218199_3114_1440_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 11 | 10 | 4 | 1 | 3114x1440 | 30 |
| `13097120_2160_3840_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 16 | 16 | 0 | 0 | 2160x3840 | 30 |
| `13256420_1080_1920_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 43 | 43 | 0 | 0 | 1080x1920 | 30 |
| `13280461_3840_2160_24fps_pure` | Round 1 | pexels (không ghi chủ đề) | 13 | 13 | 0 | 0 | 3840x2160 | 24 |
| `13570717_2160_3840_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 13 | 13 | 13 | 0 | 2160x3840 | 30 |
| `13589189_2160_3840_60fps_pure` | Round 1 | pexels (không ghi chủ đề) | 11 | 11 | 11 | 0 | 2160x3840 | 60 |
| `13676716_2160_3840_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 11 | 11 | 7 | 0 | 2160x3840 | 30 |
| `13890243_2160_3840_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 11 | 11 | 0 | 0 | 2160x3840 | 30 |
| `13903817_3840_2160_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 23 | 23 | 23 | 0 | 3840x2160 | 30 |
| `13976903_1080_1920_25fps_pure` | Round 1 | pexels (không ghi chủ đề) | 16 | 16 | 0 | 0 | 1080x1920 | 25 |
| `14003137_2160_3840_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 6 | 6 | 0 | 0 | 2160x3840 | 30 |
| `14283156_1080_1920_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 21 | 21 | 21 | 0 | 1080x1920 | 30 |
| `14477514_1080_1920_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 5 | 5 | 0 | 0 | 1080x1920 | 30 |
| `17116108-hd_1920_1080_25fps_pure` | Round 1 | pexels (không ghi chủ đề) | 10 | 10 | 0 | 0 | 1920x1080 | 25 |
| `2016 April - Fire Investigation 1B - Milpitas CA_pure_chunk_02` | Round 1 | khác | 31 | 31 | 0 | 0 | 1920x1080 | 30 |
| `2016 April - Fire Investigation 1B - Milpitas CA_pure_chunk_03` | Round 1 | khác | 30 | 30 | 0 | 0 | 1920x1080 | 30 |
| `36f3eac1_pexels_cigarette_smoke_5075501_pure` | Round 1 | cigarette smoke | 7 | 5 | 5 | 0 | 1280x720 | 25 |
| `394a2a56_pexels_visible_flame_6985324_pure` | Round 1 | visible flame | 12 | 12 | 12 | 0 | 540x960 | 24 |
| `4a37a9df_pexels_heavy_smoke_17202470_pure` | Round 1 | heavy smoke | 11 | 0 | 11 | 0 | 1920x1080 | 30 |
| `4e0040e9_pexels_heavy_smoke_34992285_pure` | Round 1 | heavy smoke | 14 | 0 | 14 | 0 | 1920x1080 | 60 |
| `6770d887_pexels_heavy_smoke_28478389_pure` | Round 1 | heavy smoke | 11 | 11 | 11 | 0 | 540x960 | 30 |
| `012f0550_pexels_heavy_smoke_36164483` | Round 2 | heavy smoke | 11 | 0 | 11 | 0 | 540x960 | 24 |
| `05ca4b9b_pexels_welding_sparks_34771762` | Round 2 | welding sparks | 22 | 17 | 22 | 0 | 540x960 | 32 |
| `13106016_2160_3840_60fps` | Round 2 | pexels (không ghi chủ đề) | 16 | 16 | 0 | 0 | 2160x3840 | 60 |
| `13269425_2160_3840_30fps` | Round 2 | pexels (không ghi chủ đề) | 31 | 31 | 0 | 0 | 2160x3840 | 30 |
| `13270890_2160_3840_60fps` | Round 2 | pexels (không ghi chủ đề) | 14 | 14 | 0 | 0 | 1080x1920 | 30 |
| `13845809_2160_3840_25fps` | Round 2 | pexels (không ghi chủ đề) | 13 | 13 | 0 | 0 | 2160x3840 | 25 |
| `14379036_1920_1080_50fps` | Round 2 | pexels (không ghi chủ đề) | 1 | 0 | 0 | 1 | 1920x1080 | 50 |
| `3aa2be24_pexels_cigarette_smoke_5320212` | Round 2 | cigarette smoke | 25 | 12 | 15 | 0 | 540x960 | 25 |
| `407b06ea_pexels_early_smoke_8516427` | Round 2 | early smoke | 4 | 0 | 4 | 0 | 2048x1080 | 24 |
| `47edc2fb_pexels_cigarette_smoke_9644281` | Round 2 | cigarette smoke | 11 | 0 | 11 | 0 | 1280x720 | 25 |
| `66d00a95_pexels_diffused_smoke_35582792` | Round 2 | diffused smoke | 15 | 6 | 15 | 0 | 1920x1080 | 25 |
| `6a95cab6_pexels_heavy_smoke_10666165` | Round 2 | heavy smoke | 11 | 11 | 0 | 0 | 540x960 | 30 |
| `6bc652d9_pexels_diffused_smoke_9589423` | Round 2 | diffused smoke | 7 | 0 | 7 | 0 | 2048x1080 | 25 |
| `6c3dee00_pexels_heavy_smoke_6618036` | Round 2 | heavy smoke | 16 | 0 | 16 | 0 | 1920x1080 | 24 |
| `6eeb70d4_pexels_heavy_smoke_34004202` | Round 2 | heavy smoke | 16 | 16 | 16 | 0 | 1920x1080 | 24 |
| `774e8545_pexels_diffused_smoke_36252846` | Round 2 | diffused smoke | 22 | 0 | 22 | 0 | 540x960 | 25 |
| `816fd10a_pexels_cooking_smoke_37848046` | Round 2 | cooking smoke | 1 | 0 | 0 | 1 | 540x960 | 24 |
| `a7f3eb27_pexels_diffused_smoke_6731483` | Round 2 | diffused smoke | 31 | 0 | 31 | 0 | 540x960 | 25 |
| `adc26840_pexels_diffused_smoke_6956078` | Round 2 | diffused smoke | 12 | 0 | 12 | 0 | 540x960 | 25 |
| `battery_thermal_runaway_krHU3jlR4Pc` | Round 2 | battery thermal runaway | 49 | 25 | 48 | 0 | 1280x720 | 25 |
| `battery_thermal_runaway_oieH2wwDGzo` | Round 3 (mới) | battery thermal runaway | 19 | 15 | 7 | 0 | 1280x720 | 30 |
| `electrical_ignition_5NAkyEmC0IU_continuous_trimmed` | Round 3 (mới) | electrical ignition | 47 | 47 | 38 | 0 | 1280x720 | 25 |
| `electrical_ignition_Kzv1Qn1pPpc` | Round 3 (mới) | electrical ignition | 14 | 11 | 6 | 1 | 600x480 | 25 |
| `neg_normal_1gi5qn1khVk_4MP_PTZ_Camera_with_Night_` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 25 |
| `neg_normal_G8z74loJO7U_ClearPix_Camera_Warehouse_` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 6 |
| `neg_normal_Od6EeCWytZo_The_famous_Shibuya_crossin` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 25 |
| `neg_normal_bpxhdKKRD5c_Cottonwood_tree_and_windst` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_normal_cmEmAzR0ybA_How_to_Protect_Parking_Lot` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_1_OXM4mr_i0` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 640x480 | 25 |
| `neg_hard_negatives_6rCG4oF2qrA` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_Afa6qOVfn7M` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 25 |
| `neg_hard_negatives_BvTBuLESrJ0` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_OGFxYoQHGco` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_P2QxCFyusSA` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_aM2GDiFoy8Q` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_erqtaaYXWzY` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 60 |
| `neg_hard_negatives_fHGR3iYfX4w` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 25 |
| `neg_hard_negatives_iaZ3E1mD96Q` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 25 |
| `neg_hard_negatives_k1leKhvOtqg` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_lqNZtZFUgcE` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_m7QxBFe8wZw` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 25 |
| `neg_hard_negatives_r-9qPsm_pvc` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_tMUbJs2iweQ` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 24 |
| `neg_hard_negatives_wb6iirzZKlk` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 24 |
| `neg_hard_negatives_zzclsZ5v0SM` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 640x480 | 30 |

### val — 19 video, 315 frame

| Video | Nguồn | Chủ đề | Frame | Lửa | Khói | Rỗng | Gốc | FPS |
|---|---|---|---|---|---|---|---|---|
| `00e18d86-3389d8dc_pure` | Round 1 | khác | 50 | 50 | 0 | 0 | 640x360 | 30 |
| `034f5d53_pexels_cigarette_smoke_8139853_pure` | Round 1 | cigarette smoke | 12 | 0 | 12 | 0 | 1920x1080 | 30 |
| `12935692_1080_1920_60fps_pure` | Round 1 | pexels (không ghi chủ đề) | 5 | 5 | 0 | 0 | 1080x1920 | 60 |
| `14056668_2160_3840_60fps_pure` | Round 1 | pexels (không ghi chủ đề) | 8 | 8 | 6 | 0 | 2160x3840 | 60 |
| `14112316_1440_2560_60fps_pure` | Round 1 | pexels (không ghi chủ đề) | 11 | 11 | 11 | 0 | 1440x2560 | 60 |
| `0fd09443_pexels_welding_sparks_14314814` | Round 2 | welding sparks | 3 | 3 | 0 | 0 | 1920x1080 | 50 |
| `13323137_1920_1080_60fps` | Round 2 | pexels (không ghi chủ đề) | 9 | 9 | 0 | 0 | 1920x1080 | 60 |
| `312fe283_pexels_cigarette_smoke_6143909` | Round 2 | cigarette smoke | 11 | 1 | 11 | 0 | 2048x1080 | 25 |
| `38c6467b_pexels_heavy_smoke_35865134` | Round 2 | heavy smoke | 8 | 0 | 8 | 0 | 540x960 | 30 |
| `7cc52846_pexels_heavy_smoke_35241029` | Round 2 | heavy smoke | 6 | 0 | 6 | 0 | 540x960 | 30 |
| `9a06358f_pexels_heavy_smoke_3568453` | Round 2 | heavy smoke | 3 | 0 | 0 | 3 | 1920x1080 | 30 |
| `battery_thermal_runaway_ld00r-tEEJ4_continuous_trimmed` | Round 3 (mới) | battery thermal runaway | 33 | 18 | 16 | 0 | 1280x720 | 25 |
| `bucket11` | Round 3 (mới) | khác | 50 | 44 | 50 | 0 | 1280x720 | 25 |
| `electrical_ignition_qRoT41qla9I_continuous_trimmed` | Round 3 (mới) | electrical ignition | 26 | 23 | 5 | 0 | 1280x720 | 30 |
| `neg_normal_YqkiOmfMQjk_Garmin_20_Dash_Cam___Sunny` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_9bas4RV3btM` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 960x720 | 30 |
| `neg_hard_negatives_CIvTKFlBTaw` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 25 |
| `neg_hard_negatives_Gc-70RMuyBI` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 854x480 | 25 |
| `neg_hard_negatives_fq3l2_IRW_0` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 24 |

### test — 17 video, 279 frame

| Video | Nguồn | Chủ đề | Frame | Lửa | Khói | Rỗng | Gốc | FPS |
|---|---|---|---|---|---|---|---|---|
| `01c73785_pexels_heavy_smoke_37765564_pure` | Round 1 | heavy smoke | 13 | 13 | 0 | 0 | 2400x1080 | 30 |
| `12393269_1440_2560_30fps_pure` | Round 1 | pexels (không ghi chủ đề) | 11 | 11 | 11 | 0 | 1440x2560 | 30 |
| `28f4ec5d_pexels_cigarette_smoke_8139860_pure` | Round 1 | cigarette smoke | 11 | 7 | 11 | 0 | 1920x1080 | 30 |
| `316af409_pexels_heavy_smoke_18105251_pure` | Round 1 | heavy smoke | 13 | 12 | 8 | 0 | 240x426 | 30 |
| `4cf31d6f_pexels_heavy_smoke_36068162_pure` | Round 1 | heavy smoke | 16 | 0 | 16 | 0 | 1920x1080 | 25 |
| `13106082_2160_3840_60fps` | Round 2 | pexels (không ghi chủ đề) | 8 | 8 | 0 | 0 | 2160x3840 | 60 |
| `13862697_2160_3840_25fps` | Round 2 | pexels (không ghi chủ đề) | 12 | 12 | 0 | 0 | 2160x3840 | 25 |
| `battery_thermal_runaway_o7szQvxW_KM` | Round 3 (mới) | battery thermal runaway | 10 | 7 | 10 | 0 | 854x480 | 30 |
| `e731c09a_pexels_visible_flame_14071614` | Round 3 (mới) | visible flame | 53 | 52 | 7 | 0 | 1920x1080 | 50 |
| `electrical_ignition_Roltnt5J_e0` | Round 3 (mới) | electrical ignition | 15 | 12 | 3 | 0 | 1270x720 | 30 |
| `electrical_ignition_VimYV-zgYo0_continuous_trimmed` | Round 3 (mới) | electrical ignition | 28 | 7 | 28 | 0 | 650x480 | 30 |
| `electrical_ignition_g6T6G_-htVw_continuous_trimmed` | Round 3 (mới) | electrical ignition | 9 | 9 | 5 | 0 | 1920x1080 | 24 |
| `neg_normal_pc8eQZW9E8Q_YI_Indoor_Camera___Sample_` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 20 |
| `neg_normal_rBZk3mR6VM0_HD_Convenience_Store_CCTV_` | Âm tính | âm tính thường (CCTV/cảnh thường) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_jKMNSvpB9dY` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 640x360 | 25 |
| `neg_hard_negatives_nJe-Y9QM0cA` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 30 |
| `neg_hard_negatives_zFDB6XlHWVU` | Âm tính | âm tính khó (hard negative) | 16 | 0 | 0 | 16 | 1280x720 | 24 |

## 5. Lưu ý khi dùng

- Val (19 video) và test (17 video) vẫn nhỏ: kết quả đánh giá có khoảng tin cậy rộng, nên báo cáo kèm bootstrap CI theo video.
- Split hiện tại khác split ngày 2026-10-19, nên số liệu test cũ không so sánh trực tiếp được.
- Với K-fold: chọn siêu tham số theo trung bình val của các fold; Stage 2 fold k phải dùng Stage 1 fold k (`segformer_spatial_best_fold<k>.pth`) vì ảnh tĩnh giống frame video đã được xếp cùng fold.
- Lửa chiếm rất ít pixel (xem 2.6, 2.8): phần lớn frame có lửa < 1% diện tích → cần class weight / Dice và metric object-level.
- 3 video `welding sparks` (tia hàn) được gán nhãn lửa/khói (20 frame lửa, 30 frame khói). Nếu muốn dùng chúng làm mẫu âm tính khó thì cần xem lại nhãn.
- Frame chưa được export từ Roboflow chỉ làm ngữ cảnh, nên video round 2/3 gần như không có frame rỗng được giám sát (6/692 frame). Frame rỗng chủ yếu đến từ 32 video âm tính.
- 3 video dương tính không có frame lửa/khói nào được gán nhãn (chỉ có frame rỗng): `14379036_1920_1080_50fps` (1 frame, train), `816fd10a_pexels_cooking_smoke_37848046` (1 frame, train), `9a06358f_pexels_heavy_smoke_3568453` (3 frame, val). Nên kiểm tra lại nhãn trên Roboflow.
- *Chủ đề* lấy từ từ khóa lúc crawl (tên file), không phải từ nhãn: ví dụ một số video `heavy smoke` / `cooking smoke` chỉ có nhãn lửa.
