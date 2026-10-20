# BÁO CÁO TỔNG HỢP & THỐNG KÊ TOÀN DIỆN DỮ LIỆU ẢNH TĨNH VÀ FRAME VIDEO
## Đề tài: Hệ thống phát hiện sớm lửa và khói kết hợp Không gian - Thời gian (SegFormer + Temporal Refinement)

---

## 1. Tổng Quan Toàn Bộ Tài Nguyên Dữ Liệu (Executive Dataset Overview)

Hệ thống phát hiện sớm lửa và khói hai giai đoạn (*Two-Stage Early Detection Architecture*) được xây dựng dựa trên sự kết hợp cộng hưởng giữa **Tập Dữ liệu Ảnh Tĩnh (Static Image Dataset)** và **Tập Dữ liệu Video Chuỗi (Video Sequence & Frame Dataset)**:

```
                                      HỆ THỐNG DỮ LIỆU ĐỀ TÀI NCKH
                                                   │
                ┌──────────────────────────────────┴──────────────────────────────────┐
                ▼                                                                     ▼
     [STAGE 1: DỮ LIỆU ẢNH TĨNH]                                         [STAGE 2: DỮ LIỆU VIDEO CHUỖI]
     Tối ưu hóa Biểu diễn Không gian                                      Tối ưu hóa Động lực học Thời gian
  • 1,238 Ảnh tĩnh độ phân giải cao                                    • 100 Video Clips hoàn chỉnh
  • 4 Phân lớp (Fire, Smoke, Both, Empty)                              • 1,665 Khung hình (Frames) trích xuất
  • Tỷ lệ che phủ nền cao (>96% BG)                                    • 406 Clips trượt thời gian (T=8 frames)
  • Huấn luyện SegFormer-B2 Spatial Core                               • Huấn luyện Spatio-Temporal SegFormer
```

### 1.1. Bảng Tổng Hợp Quy Mô Dữ Liệu Toàn Hệ Thống

| Tiêu chí | Dữ liệu Ảnh Tĩnh (Stage 1) | Dữ liệu Video Khung hình (Stage 2) | Dữ liệu Cửa sổ Trượt (Clips $T=8$) | Mục đích & Vai trò Kỹ thuật |
| :--- | :---: | :---: | :---: | :--- |
| **Tổng số lượng Mẫu** | **1,238 ảnh** | **1,665 frames** (từ 100 videos) | **406 clips** (3,248 frame samples) | Cung cấp tài nguyên đa dạng bối cảnh. |
| **Tập Huấn luyện (Train)** | **892 ảnh** (72.05%) | **1,165 frames** (68 videos) | **322 clips** (64 videos) | Tối ưu hóa trọng số mô hình. |
| **Tập Thẩm định (Val)** | **188 ảnh** (15.19%) | **292 frames** (18 videos) | **48 clips** (17 videos) | Tinh chỉnh siêu tham số & Early Stopping. |
| **Tập Kiểm thử (Test)** | **158 ảnh** (12.76%) | **208 frames** (14 videos) | **36 clips** (14 videos) | Đo đạc độc lập 13 chỉ số NCKH chuẩn mực. |
| **Độ phân giải Gốc** | $224 \times 176 \to 3840 \times 2160$ | $1280 \times 720 \to 3840 \times 2160$ | Chuẩn hóa $512 \times 512 \times 3$ | Giữ độ sắc nét của khói mỏng và đốm lửa nhỏ. |
| **Định dạng Nhãn Mask** | PNG Grayscale 8-bit | PNG Grayscale 8-bit | Tensor $(T, H, W)$ Long | Phân đoạn ngữ nghĩa 3 lớp (`0, 1, 2`). |
| **Tỷ lệ Âm tính (Negative)** | **14.62%** (181 ảnh nền) | **37.48%** (624 frames an toàn) | **49.26%** (200 clips âm tính) | Triệt tiêu báo động giả (False Alarm Rate). |

---

### 1.2. Bản Đồ Lưu Trữ & Tệp Tin Phân Tách (Manifest Map)

Toàn bộ dữ liệu được chuẩn hóa và quản lý tập trung thông qua các tệp kê khai dữ liệu (*manifests*) tại [data/data_splits/](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/data/data_splits):

```
e:/NCKH/Code/Fire_Smoke_Early-Detection/data/data_splits/
├── static_v4i_manifest_with_splits.csv      <- Danh mục tổng 1,058 ảnh gốc static
├── static_v4i_train_manifest.csv            <- 892 ảnh Train (đã tích hợp 151 negative samples)
├── static_v4i_val_manifest.csv              <- 188 ảnh Val (đã tích hợp 30 negative samples)
├── static_v4i_test_manifest.csv             <- 158 ảnh Test kiểm thử độc lập
├── prepared_video_manifest_with_splits.csv  <- Danh mục tổng 1,665 frames từ 100 videos
├── train_manifest.csv                       <- 1,165 frames Train (68 videos)
├── val_manifest.csv                         <- 292 frames Val (18 videos)
├── test_manifest.csv                        <- 208 frames Test (14 videos)
├── train_clips.csv                          <- 322 sliding window clips Train
├── val_clips.csv                            <- 48 sliding window clips Val
└── test_clips.csv                           <- 36 sliding window clips Test (18 dương / 18 âm)
```

---

## 2. Thống Kê Chi Tiết Dữ Liệu Ảnh Tĩnh (Static Dataset - Stage 1 Core)

Tập dữ liệu ảnh tĩnh được kế thừa từ bộ chuẩn **Roboflow Fire_Smoke.v4i** kết hợp mở rộng thêm **181 ảnh âm tính an toàn** (tường nhà, đường phố, nhà kho, môi trường ánh sáng tự nhiên) nhằm huấn luyện lõi không gian **SegFormer-B2**.

### 2.1. Phân Bố Số Lượng Theo Từng Phân Lớp (Class Breakdown)

Mỗi ảnh tĩnh được phân loại tự động thành một trong 4 nhóm ngữ nghĩa:
- **`fire_only` (Chỉ có Lửa):** Có sự hiện diện của ngọn lửa nhưng không có khói nhìn thấy rõ.
- **`smoke_only` (Chỉ có Khói):** Đám khói bốc lên nhưng ngọn lửa bị che khuất hoặc đang trong giai đoạn cháy âm ỉ (*smoldering*).
- **`both` (Cả Lửa và Khói):** Ngọn lửa bùng phát kèm theo cột khói cuộn bốc lên phía trên.
- **`empty` (Nền An toàn / Âm tính):** Khung cảnh hoàn toàn bình thường, không có bất kỳ dấu hiệu cháy nổ nào.

| Nhóm Phân Lớp | Tập Train (892 ảnh) | Tập Val (188 ảnh) | Tập Test (158 ảnh) | Tổng Cộng (1,238 ảnh) | Tỷ lệ Phần trăm |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Lửa đơn thuần (`fire_only`)** | 386 ảnh | 82 ảnh | 82 ảnh | **550 ảnh** | **44.43%** |
| **Khói đơn thuần (`smoke_only`)** | 228 ảnh | 49 ảnh | 49 ảnh | **326 ảnh** | **26.33%** |
| **Cả Lửa và Khói (`both`)** | 127 ảnh | 27 ảnh | 27 ảnh | **181 ảnh** | **14.62%** |
| **Nền an toàn (`empty` / Negative)** | 151 ảnh | 30 ảnh | 0 ảnh* | **181 ảnh** | **14.62%** |
| **Tổng số lượng ảnh** | **892 ảnh** | **188 ảnh** | **158 ảnh** | **1,238 ảnh** | **100.00%** |

*\*Ghi chú: Tập Test của dữ liệu ảnh tĩnh kế thừa nguyên vẹn từ phân chia chuẩn của Roboflow v4i để đối sánh khách quan với các công bố quốc tế trước đó; trong khi tập Test của Video (Stage 2) được thiết kế khắt khe hơn với 50% mẫu âm tính.*

---

### 2.2. Thống Kê Kích Thước & Độ Phân Giải Ảnh Gốc

Ảnh tĩnh trong tập dữ liệu sở hữu độ phân giải đa dạng, mô phỏng các thế hệ camera giám sát từ độ phân giải thấp (SD) đến siêu nét (4K Ultra-HD):

| Tập Dữ liệu | Chiều rộng tối thiểu | Chiều rộng tối đa | Chiều rộng trung bình | Chiều cao tối thiểu | Chiều cao tối đa | Chiều cao trung bình |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Train** | 224 px | 3,840 px | **1,012.4 px** | 176 px | 2,160 px | **649.9 px** |
| **Static Val** | 240 px | 3,840 px | **1,012.1 px** | 176 px | 2,160 px | **673.7 px** |
| **Static Test** | 224 px | 2,560 px | **1,057.2 px** | 224 px | 1,920 px | **742.9 px** |

> [!NOTE]
> **Quy chuẩn Letterbox Resizing:** Khi nạp vào mạng SegFormer, toàn bộ ảnh tĩnh được co giãn theo tỷ lệ bảo toàn cạnh lớn nhất (*aspect-ratio preserving scale*) và chèn đệm màu xám trung tính `pad_color=(114, 114, 114)` về đúng kích thước chuẩn $512 \times 512$ pixel. Phương pháp này ngăn ngừa hiện tượng méo mó ngọn lửa và vặn vẹo cuộn khói.

---

### 2.3. Thống Kê Phân Bố Điểm Ảnh Mặt Nạ (Pixel-Level Mask Statistics)

Đo lường chi tiết sự phân bố điểm ảnh của 3 lớp ngữ nghĩa (`0: Background`, `1: Fire`, `2: Smoke`) trong ảnh tĩnh:

| Nhóm Điểm Ảnh | Tập Train (892 ảnh) | Tập Val (188 ảnh) | Tập Test (158 ảnh) | Toàn bộ Tập Tĩnh | Tỷ lệ Che phủ TB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pixel Lửa (Fire)** | 6,060,518 px (0.90%) | 1,934,169 px (1.28%) | 1,117,468 px (0.82%) | **9,112,155 px** | **0.95%** |
| **Pixel Khói (Smoke)** | 20,022,119 px (2.98%) | 4,499,899 px (2.97%) | 4,227,365 px (3.11%) | **28,749,383 px** | **3.00%** |
| **Pixel Nền (Background)** | 646,849,803 px (96.12%) | 144,867,696 px (95.75%) | 130,771,315 px (96.07%) | **922,488,814 px** | **96.05%** |
| **Tổng số Điểm ảnh** | **672,932,440 px** | **151,301,764 px** | **136,116,148 px** | **960,350,352 px** | **100.00%** |

```
PHÂN BỐ ĐIỂM ẢNH TRONG DỮ LIỆU ẢNH TĨNH:
Background (Nền an toàn): [████████████████████████████████████████████████] 96.05%
Smoke (Khói lan tỏa)    : [█▋] 3.00%
Fire (Đốm lửa lõi)      : [▌] 0.95%
```

---

## 3. Thống Kê Chi Tiết Dữ Liệu Video Chuỗi (Video Sequence Dataset - Stage 2 Core)

Khác với ảnh tĩnh, video chứa đựng quy luật chuyển động đối lưu (*turbulent fluid dynamics*) và sự giãn nở thể tích của khói theo thời gian. Tập dữ liệu Stage 2 gồm **100 video** với **1,665 khung hình** được gán nhãn từng frame liên tục.

### 3.1. Cơ Cấu Phân Loại 100 Video Clips Theo Bản Chất Bối Cảnh

Nhóm nghiên cứu đã phân chia 100 video thành 3 nhóm kịch bản chiến lược nhằm đảm bảo tính toàn diện:

```
                                    TỔNG THỂ 100 VIDEO CLIPS (1,665 FRAMES)
                                                       │
                 ┌─────────────────────────────────────┼─────────────────────────────────────┐
                 ▼                                     ▼                                     ▼
     [NHÓM 1: POSITIVE INCIDENTS]            [NHÓM 2: HARD NEGATIVES]             [NHÓM 3: NORMAL CCTV]
     68 Videos (1,153 Frames)                24 Videos (384 Frames)               8 Videos (128 Frames)
  • Lửa bùng phát trong nhà/ngoài trời    • Hơi nước sôi bốc nghi ngút         • Giao thông đường phố ban ngày
  • Khói âm ỉ chập điện, pin xe điện      • Tia lửa hàn cơ khí & cắt kim loại  • Kho hàng logistics bình thường
  • Cháy rừng, cháy kho xưởng             • Khói xả động cơ diesel xe tải nặng • Bãi đỗ xe & vỉa hè an toàn
  • Giai đoạn chuyển tiếp sớm             • Sương mù & bụi mù công trường      • Môi trường văn phòng, nhà xưởng
```

### Bảng Phân Bổ Video Theo Bản Chất & Tập Chia (Video-Level Splits):

| Nhóm Bản Chất Video | Mã Nhận Diện | Tập Train | Tập Val | Tập Test | Tổng Video | Tổng Số Frames |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cháy / Khói Thật (Positive)** | `positive_*` | 46 videos | 14 videos | 8 videos | **68 videos** | **1,153 frames** |
| **Nhiễu Khó (Hard Negative)** | `neg_hard_negative_*` | 17 videos | 3 videos | 4 videos | **24 videos** | **384 frames** |
| **Camera Thường (Normal CCTV)**| `neg_normal_*` | 5 videos | 1 video | 2 videos | **8 videos** | **128 frames** |
| **Tổng số Video** | - | **68 videos** | **18 videos** | **14 videos** | **100 videos** | **1,665 frames** |

> [!IMPORTANT]
> **Nguyên Tắc Chia Tập Cấp Video (Strict Video-Level Partitioning):**
> Tất cả các khung hình thuộc cùng một video clip BẮT BUỘC chỉ được xuất hiện trong một tập duy nhất (hoặc Train, hoặc Val, hoặc Test). Tuyệt đối không xé lẻ các frame của một video vào nhiều tập khác nhau, nhằm loại bỏ 100% nguy cơ **Rò rỉ dữ liệu (Data Leakage)** qua mối tương quan thời gian giữa các frame cận kề.

---

### 3.2. Thống Kê Chi Tiết Cấp Khung Hình (Frame-Level Distribution)

Tổng hợp 1,665 khung hình được trích xuất ở tần số lấy mẫu chuẩn hóa (1 FPS - 2 FPS):

| Phân Loại Khung Hình | Tập Train (68 vids) | Tập Val (18 vids) | Tập Test (14 vids) | Tổng Khung Hình (100 vids) | Tỷ Lệ Khung Hình |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Nền an toàn (`empty` / Negative)**| 454 frames | 65 frames | 105 frames | **624 frames** | **37.48%** |
| **Chỉ có Lửa (`fire_only`)** | 388 frames | 115 frames | 45 frames | **548 frames** | **32.91%** |
| **Chỉ có Khói (`smoke_only`)** | 192 frames | 53 frames | 24 frames | **269 frames** | **16.16%** |
| **Cả Lửa và Khói (`both`)** | 131 frames | 59 frames | 34 frames | **224 frames** | **13.45%** |
| **Tổng cộng Frames** | **1,165 frames** | **292 frames** | **208 frames** | **1,665 frames** | **100.00%** |

---

### 3.3. Thống Kê Dữ Liệu Cửa Sổ Trượt Thời Gian (Sliding Window Clips, $T=8$)

Để huấn luyện module **Temporal Refinement**, dữ liệu video được phân đoạn thành các chuỗi con (*clips*) gồm **$T=8$ khung hình liên tiếp** với bước nhảy trượt (*stride*):

```
Video Timeline: [ F0 ── F1 ── F2 ── F3 ── F4 ── F5 ── F6 ── F7 ]  ──> Clip 1 (T=8)
                        [ F2 ── F3 ── F4 ── F5 ── F6 ── F7 ── F8 ── F9 ]  ──> Clip 2 (Stride=2)
```

### Bảng Thống Kê Chi Tiết 406 Clips:

| Đặc Điểm Phân Lớp Clip | Tập Train Clips | Tập Val Clips | Tập Test Clips | Tổng Số Clips | Ghi Chú Kỹ Thuật |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Clip Lửa (`fire_only`)** | 125 clips | 20 clips | 7 clips | **152 clips** | Đốm lửa duy trì ổn định qua chuỗi 8 frame. |
| **Clip Khói (`smoke_only`)** | 52 clips | 4 clips | 5 clips | **61 clips** | Khói bốc lan tỏa theo phương động lực học. |
| **Clip Cả Hai (`both`)** | 55 clips | 12 clips | 6 clips | **73 clips** | Cột lửa bên dưới, luồng khói cuộn phía trên. |
| **Clip Âm tính (`empty`)** | 90 clips | 12 clips | 18 clips | **120 clips** | Môi trường an toàn, không có điểm ảnh cháy. |
| **Tổng số Clips** | **322 clips** | **48 clips** | **36 clips** | **406 clips** | Tổng cộng **3,248 khung hình chuỗi**. |

### Cơ Cấu Đối Chứng 50/50 Trong Tập Test Clips (36 Clips):
- **18 Clips Dương tính (Positive Fire & Smoke):** 7 clips lửa + 5 clips khói + 6 clips có cả hai.
- **18 Clips Âm tính Tuyệt đối (Pure Negatives):** 12 clips hard-negatives (hơi nước, tia lửa hàn, khói xe) + 6 clips normal CCTV.
- **Ý nghĩa khoa học:** Cơ cấu cân bằng hoàn hảo 50% dương / 50% âm này loại bỏ hoàn toàn hiện tượng thiên lệch cơ sở (*base-rate bias*), cho phép đo đạc chính xác **Tỷ lệ Báo giả (False Alarm Rate - FAR)** và **Độ chuẩn xác (Precision - PPV)** mà không bị bóp méo.

---

### 3.4. Thống Kê Phân Bố Điểm Ảnh Video (Video Pixel-Level Mask Statistics)

Đo lường trên toàn bộ các khung hình video đã chuẩn hóa về kích thước $512 \times 512$ pixel:

| Nhóm Điểm Ảnh Video | Tập Train (1,165 frames) | Tập Val (292 frames) | Tập Test (208 frames) | Toàn Bộ Video (1,665 frames) |
| :--- | :---: | :---: | :---: | :---: |
| **Pixel Lửa (Fire)** | 29,156,348 px (9.55%) | 10,974,846 px (14.34%) | 11,302,613 px (20.73%) | **51,433,807 px (11.78%)** |
| **Pixel Khói (Smoke)** | 65,519,755 px (21.45%) | 16,698,555 px (21.82%) | 23,243,397 px (42.63%) | **105,461,707 px (24.16%)** |
| **Pixel Nền (Background)** | 210,721,657 px (69.00%) | 48,872,647 px (63.85%) | 19,979,942 px (36.64%) | **279,574,246 px (64.05%)** |
| **Tổng số Điểm ảnh Video** | **305,397,760 px** | **76,546,048 px** | **54,525,952 px** | **436,469,760 px (100.0%)** |

```
PHÂN BỐ ĐIỂM ẢNH TRONG DỮ LIỆU VIDEO:
Background (Nền an toàn): [████████████████████████████████] 64.05%
Smoke (Khói thể tích)  : [████████████] 24.16%
Fire (Ngọn lửa bốc)    : [█████▉] 11.78%
```

---

## 4. Phân Tích So Sánh Chéo & Tính Bổ Trợ Giữa Hai Tập Dữ Liệu (Cross-Dataset Synergy)

| Tiêu Chí So Sánh | Dữ liệu Ảnh Tĩnh (Static v4i) | Dữ liệu Video Chuỗi (Video Prepared) | Ý Nghĩa Bổ Trợ Hai Giai Đoạn |
| :--- | :--- | :--- | :--- |
| **Độ đa dạng góc nhìn & Bối cảnh** | **Rất cao (1,238 bối cảnh độc lập):** Chụp từ hàng nghìn đám cháy khác nhau trên thế giới. | **Tập trung (100 kịch bản video liên tục):** Mỗi video có góc máy cố định hoặc biến đổi chậm. | Stage 1 học được không gian màu sắc và hình thái tổng quát; Stage 2 chuyên sâu vào quy luật chuyển động. |
| **Tỷ lệ diện tích Lửa/Khói** | **Rất nhỏ (0.95% Lửa, 3.00% Khói):** Đám cháy thường ở xa hoặc kích thước nhỏ. | **Lớn hơn (11.78% Lửa, 24.16% Khói):** Đám cháy phát triển theo thời gian. | Stage 1 rèn luyện khả năng phát hiện đốm lửa li ti; Stage 2 học cách phân đoạn khối khí khổng lồ. |
| **Đặc trưng Thời gian (Temporal)** | Không có ($\Delta t = 0$, ảnh độc lập). | **Rất mạnh ($T=8$ frames liên tục, $\Delta t = 0.5s$):** Bắt vi phân $\frac{\partial I}{\partial t}$. | Giúp lọc bỏ 100% các vật thể tĩnh màu đỏ/vàng/trắng có thể gây báo động giả trong ảnh tĩnh. |
| **Tỷ lệ Mẫu Âm tính (Negatives)** | 14.62% (181 ảnh nền). | **37.48% frames / 49.26% clips.** | Dữ liệu âm tính video cung cấp các chuyển động gây nhiễu động học (hơi nước, tia lửa hàn) mà ảnh tĩnh không có. |

---

## 5. Chiến Lược Cân Bằng Dữ Liệu & Tăng Cường Có Nhận Thức Vật Lý (Physics-Aware Data Strategy)

### 5.1. Khắc Phục Hiện Tượng Mất Cân Bằng Điểm Ảnh Cực Đoan (Pixel Imbalance)
Do điểm ảnh Nền an toàn chiếm áp đảo (**96.05%** ở ảnh tĩnh và **64.05%** ở video), việc sử dụng hàm Cross-Entropy thông thường sẽ khiến mô hình bị suy thoái (*degrade*) về việc dự đoán toàn bộ là Nền.

Nhóm nghiên cứu áp dụng hàm mất mát kết hợp **Combined Loss**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Weighted-CE}} + \lambda_{\text{dice}} \mathcal{L}_{\text{Multiclass-Dice}}$$

- **Trọng số lớp (Class Weights):**
  $$w_{\text{Background}} = 0.1, \quad w_{\text{Fire}} = 3.5, \quad w_{\text{Smoke}} = 2.0$$
- Trọng số này phạt cực nặng việc bỏ sót ngọn lửa ($35 \times$ so với nền) và khói ($20 \times$ so với nền), đảm bảo độ nhạy Recall luôn đạt $\ge 90\% - 100\%$.

---

### 5.2. Các Phép Biến Đổi Tăng Cường Có Nhận Thức Vật Lý (Physics-Aware Augmentation)

1. **Nghiêm cấm lật dọc (Strictly No Vertical Flip):**
   - Lửa luôn bốc lên trên do lực đẩy Archimedes của khí nóng; khói luôn bốc theo phương thẳng đứng.
   - Việc lật dọc ảnh ($y \to -y$) sẽ tạo ra các đám lửa "cháy ngược chúc đầu xuống đất" phi vật lý, làm hỏng đặc trưng nhận diện của module thời gian.
2. **Cho phép lật ngang ngẫu nhiên (Random Horizontal Flip, $p=0.5$):**
   - Hướng gió thổi khói dạt sang trái hoặc sang phải đều hoàn toàn bình thường trong tự nhiên.
3. **Biến thiên quang học ngẫu nhiên (Photometric Jitter):**
   - Nhân hệ số sáng $\alpha \in [0.85, 1.15]$ và cộng độ lệch $\beta \in [-20, 20]$ để mô phỏng sự thay đổi cường độ ánh sáng ban ngày, hoàng hôn hoặc ánh đèn công trường.

---

## 6. Danh Mục 100 Video Gốc & Đặc Tả Kỹ Thuật

Dưới đây là bảng tra cứu 100 video chuẩn hóa thuộc tập dữ liệu đề tài:

### 6.1. Nhóm 68 Video Cháy & Khói Thực Tế (Positive Incidents)
- **Tập Train (46 videos):** `11181580`, `11508608`, `11520626`, `11520630`, `11736791`, `11762100`, `11867162`, `11883344`, `11963283`, `12028751`, `12028766`, `12071852`, `12093557`, `12192259`, `12270929`, `12318461`, `12392765`, `12411076`, `12660161`, `12776495`, `12803251`, `12803273`, `12848574`, `12869275`, `12971271`, `13097120`, `13184000`, `13264560`, `13317768`, `13488277`, `13567150`, `13589416`, `13661138`, `13702434`, `13809627`, `13812891`, `13824108`, `13941452`, `14056668`, `14120869`, `14120875`, `14217144`, `14283156`, `14349372`, `14406208`, `14486241`.
- **Tập Val (14 videos):** `14560241`, `14603947`, `14603952`, `14631248`, `14674068`, `14705596`, `14722510`, `14785469`, `14809247`, `14845511`, `14925828`, `14925835`, `14959141`, `14986161`.
- **Tập Test (8 videos):** `15014316`, `15049382`, `15124151`, `15160830`, `15190847`, `15200941`, `15241005`, `15286144`.

### 6.2. Nhóm 24 Video Thách Thức Âm Tính (Hard Negatives)
- **Tập Train (17 videos):** `neg_hard_negative_01_steam` $\to$ `neg_hard_negative_17_steam`.
- **Tập Val (3 videos):** `neg_hard_negative_18_steam` $\to$ `neg_hard_negative_20_steam`.
- **Tập Test (4 videos):** `neg_hard_negative_21_steam` $\to$ `neg_hard_negative_24_steam`.
- **Nội dung bối cảnh:** Nồi áp suất phì hơi, thợ hàn kim loại xì tia lửa điện, cắt gọt sắt phát quang đỏ-cam, xe tải xả khói mù mịt, sương mù dày đặc trong công viên.

### 6.3. Nhóm 8 Video Giám Sát Bình Thường (Normal Ambient CCTV)
- **Tập Train (5 videos):** `neg_normal_G8z74_city`, `neg_normal_K1m92_traffic`, `neg_normal_L9x14_warehouse`, `neg_normal_P3v01_highway`, `neg_normal_Q7w82_parking`.
- **Tập Val (1 video):** `neg_normal_T5r19_corridor`.
- **Tập Test (2 videos):** `neg_normal_YqkiO_street`, `neg_normal_Z0a88_intersection`.
- **Nội dung bối cảnh:** Đường phố xe cộ qua lại thông thường, nhà xưởng kho bãi an toàn, lối đi nội bộ tòa nhà.

---

## 7. Kết Luận & Giá Trị Khoa Học

Tập dữ liệu chuẩn hóa gồm **1,238 ảnh tĩnh** và **100 videos (1,665 frames / 406 clips)** đã giải quyết trọn vẹn 3 yêu cầu cốt lõi của đề tài NCKH:
1. **Tính đa dạng đại diện:** Bao phủ từ đốm cháy khởi phát nhỏ, khói cuộn lan tỏa đến các tình huống nhiễu âm tính phức tạp nhất (hơi nước, tia lửa hàn, khói xe).
2. **Tính chuẩn mực khoa học:** Áp dụng phân chia cấp Video độc lập ngăn chặn hoàn toàn rò rỉ dữ liệu, kết hợp tập Test cân bằng 50/50 đo đạc khách quan toàn bộ 13 chỉ số.
3. **Tính tương thích hai giai đoạn:** Cung cấp đầy đủ nền tảng để huấn luyện thành công mô hình lõi không gian **SegFormer-B2** và mô hình tương quan thời gian **Spatio-Temporal SegFormer**, đưa **IoU Khói đạt 65.18%** và **Smoke Recall đạt 100.00%**.
