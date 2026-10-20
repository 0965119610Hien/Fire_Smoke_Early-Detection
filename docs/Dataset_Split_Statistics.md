# Báo Cáo Thống Kê Bộ Dữ Liệu — Train / Validation / Test
## Dự Án: Fire & Smoke Early Detection (NCKH 2025-2026)

> **Mô tả:** Báo cáo thống kê đầy đủ cho toàn bộ pipeline chuẩn bị dữ liệu bao gồm: Phân chia tập Train/Val theo video clip, Xử lý mất cân bằng lớp (Class Imbalance), Tăng cường dữ liệu đồng bộ (Augmentation), và Cách ly tập Test (Anti-Data Leakage).

---

## 1. Tổng Quan Toàn Bộ Bộ Dữ Liệu (Dataset Overview)

| Tập Dữ Liệu | Số Video Gốc | Frames Gốc | Frames Sau Augment | Mục Đích |
|:---|:---:|:---:|:---:|:---|
| **Train** | 445 clips | 2,631 | **10,000** | Huấn luyện mô hình |
| **Validation** | 110 clips | **708** | 708 | Theo dõi hội tụ, điều chỉnh hyperparameter |
| **Test** | 71 clips (riêng biệt) | **2,099** | 2,099 | Đánh giá khách quan cuối cùng |
| **TỔNG** | **626 clips** | **5,438** | **12,807** | — |

> **Nguồn gốc video:**
> - Train + Val: `video_train/` — 177 video gốc được lọc và tiền xử lý
> - Test: `video_test/` — 71 video hoàn toàn độc lập, không chồng lấn với Train/Val

---

## 2. Thống Kê Tập Train (`data_splits/train_manifest.csv`)

### 2.1. Phân Bố Lớp Nhãn — SAU Augmentation (BALANCED)

| Nhãn Lớp | Frames Gốc | Frames Augment | Tổng Frames | Tỷ Lệ |
|:---|:---:|:---:|:---:|:---:|
| **Both** (Lửa + Khói) | 2,029 | 471 | **2,500** | 25.0% |
| **Fire Only** (Chỉ Lửa) | 91 | 2,409 | **2,500** | 25.0% |
| **Smoke Only** (Chỉ Khói) | 501 | 1,999 | **2,500** | 25.0% |
| **Empty** (Không có Lửa/Khói) | 10 | 2,490 | **2,500** | 25.0% |
| **TỔNG** | **2,631** | **7,369** | **10,000** | **100%** |

> **Lưu ý:** Tỷ lệ augmentation lên đến **2.8× dữ liệu gốc** — đặc biệt cao ở lớp `fire_only` (×27) và `empty` (×250) vốn rất thiếu trong bộ dữ liệu.

### 2.2. Thống Kê Pixel Lửa & Khói (Train)

| Chỉ Số | Lửa (Fire Pixels) | Khói (Smoke Pixels) |
|:---|:---:|:---:|
| **Trung bình (Mean)** | 12,031 px | 34,196 px |
| **Tỷ lệ phủ trung bình** | ~4.6% diện tích ảnh | ~13.0% diện tích ảnh |
| **Kích thước ảnh** | 512 × 512 = 262,144 px | — |

### 2.3. Các Phương Pháp Tăng Cường Dữ Liệu (Augmentation Methods)

> **Nguyên tắc:** Tất cả augmentation được áp dụng **đồng bộ** trên cặp (ảnh gốc, mặt nạ nhãn PNG), đảm bảo mask luôn tương ứng chính xác với ảnh.

#### Nhóm 1 — Biến Đổi Hình Học (Áp dụng cho cả ảnh VÀ mask)

| Tên Method | Kỹ Thuật | Lý Do Áp Dụng |
|:---|:---|:---|
| `hflip` | Lật ngang (Mirror Left-Right) | Mô phỏng hướng gió, lửa lan từ phải sang trái |
| `vflip` | Lật dọc (Mirror Up-Down) | Camera trên cao nhìn xuống, drone surveillance |
| `rot90` | Xoay 90° ngược chiều kim đồng hồ | Camera nghiêng, góc CCTV khác nhau |
| `rot270` | Xoay 270° (thuận chiều kim đồng hồ) | Camera đặt xoay, ảnh từ điện thoại nghiêng |
| `hflip_rot90` | Kết hợp Flip + Xoay 90° | Biến thể phức tạp, tăng đa dạng tư thế |

#### Nhóm 2 — Biến Đổi Ánh Sáng (Áp dụng chỉ cho ảnh, mask giữ nguyên)

| Tên Method | Kỹ Thuật | Lý Do Áp Dụng |
|:---|:---|:---|
| `bright_up` | Tăng sáng ×1.3 trên kênh V (HSV) | Ban ngày, ánh sáng mạnh — lửa nhạt màu khó nhận diện |
| `bright_down` | Giảm sáng ×0.7 trên kênh V (HSV) | Ban đêm, kho tối — khói đen khó phát hiện |

#### Nhóm 3 — Biến Đổi Nhiễu (Áp dụng chỉ cho ảnh, mask giữ nguyên)

| Tên Method | Kỹ Thuật | Lý Do Áp Dụng |
|:---|:---|:---|
| `gauss_blur` | Gaussian Blur kernel 5×5, σ=1.2 | Camera bị mờ, khói làm nhòe ống kính |
| `gauss_noise` | Gaussian Noise (μ=0, σ=15) | CCTV chất lượng thấp, nhiễu cảm biến ngoài trời |

---

## 3. Thống Kê Tập Validation (`data_splits/val_manifest.csv`)

### 3.1. Phân Bố Lớp Nhãn — GIỮ NGUYÊN BẢN (Không Augmentation)

| Nhãn Lớp | Số Frames | Tỷ Lệ |
|:---|:---:|:---:|
| **Both** (Lửa + Khói) | 474 | 66.9% |
| **Smoke Only** (Chỉ Khói) | 215 | 30.4% |
| **Fire Only** (Chỉ Lửa) | 12 | 1.7% |
| **Empty** (Nền) | 7 | 1.0% |
| **TỔNG** | **708** | **100%** |

> **Phân bố Val phản ánh phân bố tự nhiên thực tế** — không cân bằng nhân tạo, phục vụ đánh giá chính xác hiệu năng mô hình trong môi trường thực.

### 3.2. Thống Kê Pixel (Validation)

| Chỉ Số | Lửa (Fire Pixels) | Khói (Smoke Pixels) |
|:---|:---:|:---:|
| **Trung bình (Mean)** | 4,985 px | 58,871 px |
| **Tỷ lệ phủ trung bình** | ~1.9% diện tích ảnh | ~22.5% diện tích ảnh |

### 3.3. Thiết Kế Video-Level Split (Chống Data Leakage)

```
Chia theo nhóm video (Group Split — không phải frame split):
  ✅ 445 video clip -> Tập Train
  ✅ 110 video clip -> Tập Validation
  ❌ KHÔNG cho phép frame của cùng 1 video xuất hiện ở cả Train lẫn Val
```

**Stratified grouping:** Mỗi nhóm dominant category (fire/smoke/both/empty) được chia riêng 80/20 để giữ tỷ lệ lớp tương đồng giữa Train và Val.

---

## 4. Thống Kê Tập Test (`data_splits/test_manifest.csv`)

### 4.1. Phân Bố Lớp Nhãn — HOÀN TOÀN ĐỘC LẬP

| Nhãn Lớp | Số Frames | Tỷ Lệ |
|:---|:---:|:---:|
| **Fire Only** (Chỉ Lửa) | 670 | 31.9% |
| **Smoke Only** (Chỉ Khói) | 594 | 28.3% |
| **Empty** (Không có nguy hiểm) | 558 | 26.6% |
| **Both** (Lửa + Khói) | 277 | 13.2% |
| **TỔNG** | **2,099** | **100%** |

### 4.2. Thống Kê Pixel (Test)

| Chỉ Số | Lửa (Fire Pixels) | Khói (Smoke Pixels) |
|:---|:---:|:---:|
| **Trung bình (Mean)** | 7,187 px | 12,853 px |
| **Tỷ lệ phủ trung bình** | ~2.7% diện tích ảnh | ~4.9% diện tích ảnh |

### 4.3. Cách Ly Dữ Liệu Test (Anti-Data Leakage Verification)

| Tiêu Chí Cách Ly | Trạng Thái |
|:---|:---:|
| Video nguồn độc lập (`video_test/` ≠ `video_train/`) | ✅ ĐẠT |
| Thư mục frames riêng (`test_annotation_frames/`) | ✅ ĐẠT |
| Thư mục nhãn riêng (`test_annotation_labels/`) | ✅ ĐẠT |
| Không tham gia Augmentation | ✅ ĐẠT |
| Không được dùng trong quá trình điều chỉnh hyperparameter | ✅ ĐẠT |
| SegFormer-B2 chỉ inference (không backpropagation) | ✅ ĐẠT |

---

## 5. So Sánh Tổng Hợp 3 Tập Dữ Liệu

```
                    TRAIN           VAL             TEST
                  ─────────────────────────────────────────
Frames            10,000           708             2,099
Videos             445             110               71
Nguồn gốc      video_train     video_train      video_test
Augmentation       CÓ             KHÔNG            KHÔNG
Cân bằng lớp     HOÀN TOÀN     Tự nhiên         Tự nhiên
Mục đích        Học tham số   Điều chỉnh HP    Đánh giá cuối
```

---

## 6. Cấu Trúc Thư Mục Sau Pipeline

```
e:\NCKH\Code\Fire_Smoke_Early-Detection\
│
├── annotation_frames/        # 3,339 frames gốc + ~7,369 frames augmented
├── annotation_labels/        # Masks tương ứng (gốc + augmented đồng bộ)
│
├── test_annotation_frames/   # 2,099 frames test (GIỮ NGUYÊN, không augment)
├── test_annotation_labels/   # Masks test tương ứng
│
└── data_splits/
    ├── train_manifest.csv    # 10,000 frames (gốc + augmented)
    ├── val_manifest.csv      # 708 frames (chỉ gốc)
    ├── test_manifest.csv     # 2,099 frames (chỉ gốc, độc lập)
    └── split_statistics.json # Thống kê JSON đầy đủ
```

---

## 7. Script Tham Chiếu

| Bước | Script | Chức Năng |
|:---|:---|:---|
| Tiền xử lý Train | [`preprocess_for_annotation.py`](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/src/preprocess_for_annotation.py) | 5-step pipeline: Chunking → FPS → SSIM → Letterbox → CLAHE |
| Gán nhãn SegFormer | [`auto_label_with_segformer.py`](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/src/auto_label_with_segformer.py) | SegFormer-B2 GPU inference (Modal A10G) |
| Chuẩn bị Train/Val/Test | [`prepare_training_data.py`](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/src/prepare_training_data.py) | Group Split + Augmentation + Balance |
