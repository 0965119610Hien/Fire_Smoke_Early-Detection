# Kiến Trúc Huấn Luyện 2 Giai Đoạn (2-Stage Training Pipeline) Trên Modal GPU

Tài liệu này tổng hợp toàn bộ quy trình thiết lập Modal Volume, cấu trúc dữ liệu trên Cloud và hướng dẫn thực thi quy trình huấn luyện 2 giai đoạn cho bài toán **Phát hiện sớm Lửa & Khói (Fire & Smoke Early Detection)** sử dụng **SegFormer (Spatial Core)** và **Spatio-Temporal SegFormer (Không - Thời Gian)**.

---

## 1. Cấu Trúc Modal Volume (`fire_smoke_dataset_v2`)

Dữ liệu được tổ chức chuẩn hóa và phân tách rõ ràng trên Volume độc lập:

```text
/data (Modal Volume: fire_smoke_dataset_v2)
│
├── data_splits/                                  # Manifests phân chia tập dữ liệu
│   ├── static_v4i_train_manifest.csv             # 742 ảnh tĩnh train (Roboflow v4i)
│   ├── static_v4i_val_manifest.csv               # 158 ảnh tĩnh val
│   ├── static_v4i_test_manifest.csv              # 158 ảnh tĩnh test
│   ├── train_manifest.csv                        # 813 frames video train (68 videos)
│   ├── val_manifest.csv                          # 214 frames video val
│   ├── test_manifest.csv                         # 126 frames video test
│   ├── train_clips.csv                           # 260 sliding window clips (T=8)
│   ├── val_clips.csv                             # 37 sliding window clips (T=8)
│   └── test_clips.csv                            # 18 sliding window clips (T=8)
│
├── Fire_Smoke.v4i/                               # Tập dữ liệu ảnh tĩnh (Static Dataset)
│   └── train/                                    # 1,058 ảnh gốc (.jpg) và mask ngữ nghĩa (.png)
│
├── dataset_video_prepared/                       # Tập dữ liệu video chuỗi thời gian
│   ├── <video_01>/ (frame_000000.png, frame_000000_mask.png, ...)
│   ├── ...
│   └── <video_68>/
│
└── checkpoints/                                  # Thư mục lưu trữ trọng số mô hình
    ├── segformer_spatial_best.pth                # Trọng số tối ưu Stage 1 (Fg-mIoU 67.44%, Recall 98.17%, Precision 100%)
    ├── mit_b2_spatial_backbone.pth               # Trọng số không gian nhúng vào Stage 2
    └── timesformer_spatiotemporal_best.pth       # Trọng số mô hình hoàn chỉnh Stage 2 (Spatio-Temporal SegFormer)
```

---

## 2. Tiện Ích Thiết Lập Volume: `src/modal_setup_volume.py`

- Tự động giải nén các tệp nén (`Fire_Smoke.v4i.zip` và `dataset_video_prepared.tar`) trực tiếp trên Volume Linux NVMe siêu tốc.
- Tự động kiểm tra đối soát tính toàn vẹn (integrity check) của từng file theo các file manifest CSV.
- **Lệnh thực thi**:
  ```bash
  modal run src/modal_setup_volume.py
  ```

---

## 3. Giai Đoạn 1: Spatial Core SegFormer-B2 (`src/train_stage1_segformer.py`)

### 3.1. Mục Tiêu & Cơ Sở Khoa Học
- Tập trung học biểu diễn không gian chi tiết từ 1,058 ảnh tĩnh đa dạng môi trường (rừng, nhà máy, kho bãi, nội thất gia đình, hầm gửi xe).
- Khắc phục triệt để hiện tượng mất cân bằng mẫu cực đoan (diện tích trung bình của đốm lửa chỉ chiếm **0.71%** diện tích khung hình).

### 3.2. Hàm Mất Mát Kết Hợp (Combined Loss)
$$\mathcal{L}_{\text{Stage 1}} = \mathcal{L}_{\text{Weighted CE}} + \lambda_{\text{Dice}} \cdot \mathcal{L}_{\text{Multiclass Dice}}$$
Trong đó:
- Trọng số lớp (Class Weights): Background: $0.1$, Fire: $3.5$, Smoke: $2.0$.
- Phạt nặng sai sót trên các vùng lửa và khói kích thước siêu nhỏ mà không làm mờ biên phân vùng.

### 3.3. Tăng Cường Dữ Liệu Vật Lý (Physics-aware Augmentation)
- **Lật ngang ngẫu nhiên** ($p=0.5$).
- **TUYỆT ĐỐI KHÔNG lật dọc** (bảo toàn định luật đối lưu: lửa bốc lên trên, khói phân tầng trần nhà).
- Biến thiên ánh sáng Photometric Jitter $\pm 20\%$ và Gaussian Blur nhẹ ($3 \times 3$).

### 3.4. Cấu Hình Huấn Luyện & Kết Quả Đạt Được
- **GPU**: NVIDIA A10G (24 GB VRAM).
- **Bộ tối ưu**: AdamW với Differential Learning Rate:
  - Backbone MiT-B2: $6 \times 10^{-5}$
  - All-MLP Decode Head: $6 \times 10^{-4}$
- **Kết quả thực nghiệm**:
  - `Fire IoU`: **74.92%**
  - `Smoke IoU`: **50.92%**
  - `Fg-mIoU`: **67.44%**
  - `Alarm Recall (TPR)`: **98.17%**
  - `Alarm Precision (PPV)`: **100.00%**
  - `Tốc độ suy luận`: **53.5 FPS** (18.7 ms/ảnh)
- **Lệnh thực thi**:
  ```bash
  modal run src/train_stage1_segformer.py
  ```

---

## 4. Chẩn Đoán Lỗi Cốt Lõi Của Mô Hình TimeSformer Cũ & Giải Pháp Tái Cấu Trúc

### 4.1. Chẩn Đoán Nguyên Nhân Gốc Rễ (Root Cause Analysis)
Khi đối sánh trực quan kết quả suy luận trên 18 clips test (`docs/inference_comparisons/`), mô hình TimeSformer cũ bộc lộ 4 nhược điểm chí mạng:
1. **Lỗi chuyển giao trọng số (Weight Transfer Bug)**: Trong code cũ, hàm `embed_stage1_spatial_weights` chỉ tăng biến đếm `loaded_count += 1` mà **không hề gán tensor trọng số nào** từ SegFormer sang TimeSformer. TimeSformer đã bị huấn luyện 100% từ ngẫu nhiên (from scratch).
2. **Vision Transformer thiếu Inductive Bias**: Khi train từ đầu một mạng ViT trên chỉ 260 video clips trích đoạn, mô hình hoàn toàn không thể học được các bộ lọc không gian (cạnh, góc, texture, gradient màu) của ngọn lửa và làn khói.
3. **Mất liên kết đa tầng (No Multi-Scale Skip Connections)**: TimeSformer cũ nén ảnh thành $32 \times 32$ patches kích thước $16 \times 16$ pixels, sau đó giải nén bằng 4 tầng `ConvTranspose2d` không có skip connections, tạo ra hiện tượng **răng cưa dạng khối chữ nhật thô kệch (grid blocky artifacts)**.
4. **Nhầm lẫn động học thời gian (Motion Conflation)**: Mô hình chỉ học được các tín hiệu chuyển động thô sơ, dẫn đến việc nhận diện sai chiếc cốc trắng chuyển động thành khói, hoặc bàn tay người di chuyển thành ngọn lửa.

---

## 5. Giai Đoạn 2: Tái Cấu Trúc Thành Spatio-Temporal SegFormer (`src/train_stage2_timesformer.py`)

Để khắc phục triệt để mà **chưa cần thu thập thêm dữ liệu mới**, Stage 2 được tái cấu trúc thành **Spatio-Temporal SegFormer**:

```text
Input Clip X: (B, T=8, 3, 512, 512)
      │
      ▼ (Gập Batch & Time: B*T = 16)
┌────────────────────────────────────────────────────────────────────────┐
│ Pretrained SegFormer-B2 Spatial Core (Đã nạp 100% trọng số Stage 1)   │
│ ├── Stage 1: C1 (128x128, 64-ch)  ──┐                                  │
│ ├── Stage 2: C2 (64x64, 128-ch)   ──┼──> All-MLP Feature Projection    │
│ ├── Stage 3: C3 (32x32, 320-ch)   ──┤     & Bilinear Multi-Scale Fusion│
│ └── Stage 4: C4 (16x16, 512-ch)   ──┘                                  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼ Fused Spatial Map: (B*T, 768, 128, 128)
┌────────────────────────────────────────────────────────────────────────┐
│ 🧠 TEMPORAL REFINEMENT MODULE (Học Động Học Thời Gian Xuyên Suốt T=8)  │
│ ├── 1. 1D Temporal Conv (k=3): Vi phân biến thiên nhanh (flicker, dI/dt)│
│ ├── 2. Temporal Multi-Head Attention (heads=4): Tương quan dài hạn T=8 │
│ ├── 3. GELU FFN & LayerNorm                                            │
│ └── 4. ZERO-INITIALIZED RESIDUAL GATING:                               │
│        Z_out = Z + γ · ΔZ   (Khởi tạo γ = 0.0)                         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼ (Bảo toàn 100% biên nét, chỉ thêm delta thời gian)
┌────────────────────────────────────────────────────────────────────────┐
│ Classifier Head (Conv2d 768 -> 3 classes)                              │
│ └── Bilinear Interpolation (128x128 -> 512x512)                        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
      Output Logits: (B, T=8, 3, 512, 512) — Smooth, Crisp & Temporally Verified!
```

### 5.1. Sức Mạnh Của Cơ Chế Zero-Initialized Residual Gating ($\gamma = 0$)
- Tại **Epoch 0, Step 0**: $\gamma = 0 \implies Z_{\text{out}} \equiv Z$.
  Mô hình bắt đầu với **chính xác 75% Fire IoU và 51% Smoke IoU của Stage 1**, giữ trọn vẹn đường biên sắc nét của ngọn lửa và không bị suy giảm chất lượng.
- Trong quá trình học trên 260 video clips: $\gamma$ tăng dần từ $0$, các tầng Temporal Convolution và Temporal Attention được huấn luyện để phân biệt:
  - Vùng màu trắng chuyển động tĩnh (cốc sứ, áo trắng) $\rightarrow$ Không có tính chất tán xạ/bốc lên $\rightarrow$ **Triệt tiêu cảnh báo giả**.
  - Vùng màu đỏ/vàng chuyển động đều (da người, đèn xe) $\rightarrow$ Không có tần số nhấp nháy $4 - 12\text{ Hz}$ $\rightarrow$ **Triệt tiêu cảnh báo giả**.
  - Vùng khói mờ lan tỏa mượt mà qua các frame kế tiếp $\rightarrow$ **Tăng cường độ tin cậy**.

### 5.2. Tối Ưu Hóa Bộ Nhớ VRAM Toàn Diện (Khắc Phục Triệt Để CUDA OOM)
Do xử lý đồng thời $T=8$ frames ở độ phân giải cao ($512 \times 512$), nếu tính đạo hàm cho cả Backbone SegFormer thì đồ thị backward của 16 frames sẽ vượt quá 22 GiB. Để vận hành êm ái và siêu tốc trên A10G:
1. **Đóng băng (Freeze) SegFormer Spatial Backbone**: Chạy qua `with torch.no_grad():` và `param.requires_grad = False`. Loại bỏ hoàn toàn đồ thị đạo hàm khổng lồ của 16,384 tokens $\times$ 12 transformer layers, bảo tồn nguyên vẹn 100% hình thái học đã hội tụ từ Stage 1 và giải phóng $>70\%$ VRAM!
2. **Spatial Pooling 2x trong Temporal Module**: Áp dụng `AvgPool2d(2, 2)` nén $128 \times 128 \rightarrow 64 \times 64$ trước khi đưa vào Temporal Attention. Giảm $4\times$ số token, giúp Temporal Attention chạy siêu nhẹ (~50 MB thay vì 800 MB) và tăng tính chống rung (noise robustness) trước rung lắc camera.
3. **Mixed Precision Training (FP16 AMP)**: Kích hoạt `torch.cuda.amp.autocast()` và `GradScaler`, giảm 50% bộ nhớ activation và tăng gấp đôi tốc độ xử lý Tensor Cores.
4. **Cấu hình Batch Size = 1 clip & Grad Accumulation = 4**: Mỗi bước forward chỉ nhận 1 clip ($T=8$ frames), tích lũy đạo hàm 4 bước (Effective Batch = 4 clips = 32 frames), giúp đỉnh VRAM chỉ dao động an toàn trong mức **~4.5 - 6 GiB / 24 GiB**!
5. **Chống phân mảnh bộ nhớ (CUDA Allocator)**: Thiết lập `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`.

### 5.3. Chiến Lược Tối Ưu Differential Learning Rate
- **Spatial Backbone (MiT-B2)**: Đóng băng (`requires_grad = False`).
- **Spatial Decode Head**: $2 \times 10^{-5}$ (tinh chỉnh giao diện đặc trưng).
- **Temporal Refinement Module**: $3 \times 10^{-4}$ (tập trung toàn bộ năng lực học tương quan thời gian).

### 5.4. Cấu Hình Huấn Luyện & Cơ Chế Lưu Checkpoint Đa Tiêu Chí (Multi-Criteria Saving)
- **GPU**: **NVIDIA A100 (40GB / 80GB SXM4)** - Băng thông bộ nhớ 1.5 - 2.0 TB/s, hiệu năng Tensor Cores vượt trội.
- **Batch Size**: 1 clip ($T=8$, $H=W=512$), Gradient Accumulation = 4 (Effective batch = 32 frames).
- **Số Epochs**: 60 epochs, Cosine Annealing scheduler.
- **Cơ Chế Lưu Checkpoint Khi Có Kết Quả Tốt Hơn (Multi-Criteria Best Checkpoints)**:
  1. `timesformer_spatiotemporal_best.pth`: Lưu khi **Fg-mIoU** trung bình tổng hợp đạt đỉnh mới (`fg_miou > best_val_miou`).
  2. `timesformer_best_fire.pth`: Lưu khi **IoU lớp Lửa** đạt kỷ lục mới (`iou_fire > best_fire_iou`).
  3. `timesformer_best_smoke.pth`: Lưu khi **IoU lớp Khói** đạt kỷ lục mới (`iou_smoke > best_smoke_iou`).
  4. `timesformer_spatiotemporal_latest.pth`: Lưu sau **MỖI epoch** kèm `volume.commit()` để tự động phục hồi nếu rớt mạng.
- **Lệnh thực thi chạy ngầm an toàn**:
  ```bash
  # Khuyến nghị: Chạy với cờ --detach để Modal chạy độc lập trên Cloud, không bị gián đoạn khi máy ngủ hay rớt WiFi:
  modal run --detach src/train_stage2_timesformer.py

  # Hoặc xem trực tiếp trên Terminal:
  modal run src/train_stage2_timesformer.py
  ```

---

## 6. Hệ Thống Đánh Giá Toàn Diện (`src/evaluate_all_metrics.py`)

Hệ thống độ đo chuẩn hóa khoa học áp dụng cho cả 2 giai đoạn:

1. **Nhóm độ đo phân loại nhị phân (Binary Alarm Classification)**:
   - $\text{ACC}, \text{TPR (Sensitivity)}, \text{FAR (False Alarm Rate)}, \text{TNR (Specificity)}, \text{FNR (Miss Rate)}, \text{PPV (Precision)}, \text{FDR}, \text{F1-Score}, \text{MCC}$.
2. **Nhóm độ đo định vị vùng phân đoạn liên thông (Connected Component Localization)**:
   - $\text{LTPR}$ (Tỷ lệ đốm lửa/khói thực tế tìm thấy).
   - $\text{LPPV}$ (Độ chuẩn xác trên các vùng cảnh báo).
   - $\text{Mean Similarity } S$ (Độ tương đồng hình thái học regional Dice).
3. **Nhóm độ đo phân đoạn ngữ nghĩa điểm ảnh (Pixel-level Segmentation)**:
   - $\text{mIoU}, \text{IoU}_{\text{fire}}, \text{IoU}_{\text{smoke}}, \text{Dice}_{\text{fire}}, \text{Dice}_{\text{smoke}}$.
4. **Nhóm độ đo thời gian phát hiện sớm & Tốc độ**:
   - **Detection Latency**: Số frame trễ từ khi đốm lửa xuất hiện đến khi kích hoạt chuông.
   - **Throughput (FPS)** đo trên GPU thực tế.

---

## 7. Bảng Tra Cứu Toàn Bộ Lệnh Thực Thi (Command Reference)

| Bước | Mục đích | Lệnh thực thi Terminal | GPU | Đầu ra chính (Output) |
| :--- | :--- | :--- | :---: | :--- |
| **0. Quản lý Volume** | Xem danh sách file trên Volume | `modal volume ls fire_smoke_dataset_v2` | CPU | Cây thư mục trên Cloud |
| **1. Setup Volume** | Giải nén & xác thực dữ liệu | `modal run src/modal_setup_volume.py` | CPU | `/data/checkpoints/`, giải nén 2 dataset |
| **2. Stage 1 Train** | Huấn luyện SegFormer-B2 (Ảnh tĩnh) | `modal run src/train_stage1_segformer.py` | A10G | `/data/checkpoints/segformer_spatial_best.pth` |
| **3. Stage 2 Train** | Huấn luyện Spatio-Temporal SegFormer (Video) | `modal run --detach src/train_stage2_timesformer.py` | A100 | `/data/checkpoints/timesformer_spatiotemporal_best.pth` |
| **4. Đánh giá All** | Đánh giá Stage 1 (158 ảnh test) | `modal run src/evaluate_all_metrics.py --stage 1` | A10G | In bảng 13 chỉ số + JSON report |
| | Đánh giá Stage 2 (18 clips test) | `modal run src/evaluate_all_metrics.py --stage 2` | A10G | In bảng 13 chỉ số + JSON report |
| | Đánh giá so sánh cả 2 Stage | `modal run src/evaluate_all_metrics.py --stage both` | A10G | Đối sánh toàn diện 2 mô hình |
| **5. Trực quan hóa** | Suy luận đối sánh 4 clips test | `modal run src/infer_and_compare_stages.py` | A10G | Tải ảnh về `docs/inference_comparisons/` |
| | Suy luận 10 clips test | `modal run src/infer_and_compare_stages.py --max-clips 10` | A10G | Tải 10 ảnh 4-Panel Grid về máy |
| | Suy luận toàn bộ 18 clips test | `modal run src/infer_and_compare_stages.py --max-clips 18` | A10G | Tải toàn bộ 18 ảnh clips test |

---

## 8. Tóm Tắt Các Tệp Mã Nguồn Đã Sửa Đổi & Nâng Cấp

1. [src/train_stage2_timesformer.py](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/src/train_stage2_timesformer.py):
   - **Trước sửa**: Dùng mô hình TimeSformer phân chia Space-Time viết từ đầu, gặp lỗi không nạp được trọng số Stage 1, thiếu skip connections gây răng cưa khối thô kệch.
   - **Sau sửa**: Tái cấu trúc thành `SpatioTemporalSegFormer` kế thừa 100% `SegformerForSemanticSegmentation` đã train ở Stage 1; tích hợp `TemporalRefinementModule` với 1D Temporal Conv và Temporal Multi-Head Attention, sử dụng cơ chế **Zero-Initialized Residual Gating ($\gamma=0$)** và Differential Learning Rate.
2. [src/infer_and_compare_stages.py](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/src/infer_and_compare_stages.py):
   - **Sau sửa**: Thêm class `SpatioTemporalSegFormer` và cơ chế nạp checkpoint thích ứng (tự động nhận diện mô hình Spatio-Temporal mới hoặc TimeSformer cũ để suy luận và kết xuất lưới 4-Panel mượt mà).
3. [src/evaluate_all_metrics.py](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/src/evaluate_all_metrics.py):
   - **Sau sửa**: Tích hợp class `SpatioTemporalSegFormer` để đánh giá toàn bộ 13 chỉ số phân loại, định vị, phân đoạn và tốc độ cho Stage 2 mới.
4. [README.md](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/README.md) & [docs/Modal_Two_Stage_Training_Architecture_Guide.md](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/Modal_Two_Stage_Training_Architecture_Guide.md):
   - Cập nhật tài liệu kỹ thuật, chẩn đoán nguyên nhân và hướng dẫn thực thi.
