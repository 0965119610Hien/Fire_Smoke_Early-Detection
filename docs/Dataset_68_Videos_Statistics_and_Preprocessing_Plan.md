# Báo Cáo Thống Kê Toàn Diện 68 Video & Đề Xuất Pipeline Tiền Xử Lý, Phân Chia Tập Dữ Liệu

> **Dự án**: Nghiên cứu Khoa học - Hệ thống Phát hiện Sớm Lửa & Khói qua Video Giám Sát  
> **Tập dữ liệu**: `data/dataset_video_prepared/`  
> **File Manifest hợp nhất**: `data/data_splits/prepared_video_manifest.csv`  
> **Ngày cập nhật**: 01/10/2026  

---

## MỤC LỤC
1. [Tổng Quan Bộ Dữ Liệu Hoàn Chỉnh](#1-tổng-quan-bộ-dữ-liệu-hoàn-chỉnh)
2. [Thống Kê Chi Tiết Cấp Độ Khung Hình (Frame-level)](#2-thống-kê-chi-tiết-cấp-độ-khung-hình-frame-level)
3. [Thống Kê Đặc Tính Hình Học & Che Phủ Pixel (Pixel Coverage)](#3-thống-kê-đặc-tính-hình-học--che-phủ-pixel-pixel-coverage)
4. [Thống Kê Cấp Độ Video & Phân Loại Bối Cảnh (Subtypes)](#4-thống-kê-cấp-độ-video--phân-loại-bối-cảnh-subtypes)
5. [Đề Xuất Pipeline Tiền Xử Lý Dữ Liệu (Preprocessing Pipeline)](#5-đề-xuất-pipeline-tiền-xử-lý-dữ-liệu-preprocessing-pipeline)
6. [Chiến Lược Tăng Cường Dữ Liệu Thích Ứng (Physics-Aware Data Augmentation)](#6-chiến-lược-tăng-cường-dữ-liệu-thích-ứng-physics-aware-data-augmentation)
7. [Phương Án Phân Chia 3 Tập Train / Val / Test (Group Stratified Split)](#7-phương-án-phân-chia-3-tập-train--val--test-group-stratified-split)
8. [Danh Sách Phân Bổ 68 Video Vào 3 Tập](#8-danh-sách-phân-bổ-68-video-vào-3-tập)

---

## 1. TỔNG QUAN BỘ DỮ LIỆU HOÀN CHỈNH

Bộ dữ liệu video chuẩn hóa được tổng hợp qua 2 đợt gán nhãn thủ công (Roboflow Semantic Mask) và tiền xử lý đồng bộ tại tần số **1 FPS**, độ phân giải chuẩn hóa **640×640**.

| Chỉ số tổng quan | Giá trị | Ghi chú |
|---|:---:|---|
| **Tổng số Video** | **68 video** | 40 video (Đợt 1) + 28 video (Đợt 2) |
| **Tổng số Khung hình (Frames)** | **1,153 frames** | Mỗi frame đều có 1 mask phân vùng ngữ nghĩa (Semantic Mask) |
| - *Frame gán nhãn trực tiếp (Roboflow)* | **1,048 frames** (90.9%) | Chứa nhãn vẽ tay chi tiết |
| - *Frame âm tính nền (Auto Background)* | **105 frames** (9.1%) | Mask giá trị 0 nhằm giữ trọn chuỗi thời gian liên tục |
| **Độ phân giải chuẩn** | **640 × 640 px** | Tương thích chuẩn YOLO / ViT / Temporal Backbone |
| **Tần số lấy mẫu khung hình** | **1 FPS** | Đủ nắm bắt biến đổi hình thái ngọn lửa và cuộn khói |
| **Các lớp đối tượng (Classes)** | **3 lớp** | `0`: Background, `1`: Fire (Lửa), `2`: Smoke (Khói) |

---

## 2. THỐNG KÊ CHI TIẾT CẤP ĐỘ KHUNG HÌNH (FRAME-LEVEL)

Toàn bộ 1,153 frame được phân chia thành 4 nhóm trạng thái nhận diện:

| Nhóm trạng thái (Category) | Số Frame | Tỷ lệ (%) | Ý nghĩa trong bài toán phát hiện sớm |
|---|:---:|:---:|---|
| 🔥 **`fire_only`** | **548** | **47.5%** | Giai đoạn ngọn lửa bùng phát rõ nét |
| 💨 **`smoke_only`** | **269** | **23.3%** | Giai đoạn âm ỉ ban đầu (khói xuất hiện trước khi thấy lửa) |
| 🔥💨 **`both`** | **224** | **19.4%** | Giai đoạn đám cháy phát triển (vừa có lửa vừa có khói) |
| ⬛ **`empty`** *(Background)* | **112** | **9.7%** | Giai đoạn bình thường trước cháy & các nguồn gây nhiễu |
| **Tổng cộng** | **1,153** | **100.0%** | |

- **Tổng số frame chứa Lửa**: **772 frames** (67.0%)
- **Tổng số frame chứa Khói**: **493 frames** (42.8%)
- **Số frame chứa đồng thời Lửa & Khói**: **224 frames** (19.4%)

---

## 3. THỐNG KÊ ĐẶC TÍNH HÌNH HỌC & CHE PHỦ PIXEL (PIXEL COVERAGE)

Kích thước khung hình chuẩn: $640 \times 640 = 409,600$ pixels. Thống kê diện tích vùng nhãn thể hiện tính chất đối lập giữa Lửa và Khói:

| Đại lượng thống kê | Vùng Lửa (Fire) | Vùng Khói (Smoke) | Nhận xét chuyên môn NCKH |
|---|:---:|:---:|---|
| **Diện tích trung bình (Mean)** | **66,624 px** *(16.3%)* | **213,918 px** *(52.2%)* | Khói lan tỏa diện rộng gấp 3.2 lần ngọn lửa |
| **Trung vị (Median)** | **14,265 px** *(**3.48%**)* | **112,608 px** *(**27.5%**)* | **Lửa là mục tiêu nhỏ (Small Target)**, hơn 50% số khung hình đốm lửa chiếm dưới 3.5% diện tích |
| **Diện tích nhỏ nhất (Min)** | **26 px** *(0.006%)* | **994 px** *(0.24%)* | Đốm lửa mồi cực nhỏ ở khoảng cách xa |
| **Diện tích lớn nhất (Max)** | **894,159 px** | **2,485,371 px** | Cháy cận cảnh bao trùm toàn bộ góc quay |

> [!IMPORTANT]
> **Hệ quả cho thiết kế mô hình**: Vì **Trung vị diện tích Lửa chỉ chiếm 3.48% khung hình**, kiến trúc mạng bắt buộc phải sử dụng **Feature Pyramid Network (FPN)** hoặc các tầng trích xuất độ phân giải cao ($P_2, P_3$) để không làm tiêu biến đặc trưng của đốm lửa nhỏ. Ngược lại, Khói cần trường tiếp nhận lớn (Large Receptive Field / Dilated Convolutions) để bao quát đường bao khuếch tán.

---

## 4. THỐNG KÊ CẤP ĐỘ VIDEO & PHÂN LOẠI BỐI CẢNH (SUBTYPES)

### 4.1. Phân loại Video theo nội dung nhãn
- **Video có Lửa thuần túy (`fire_only`)**: **27 video** (39.7%)
- **Video có cả Lửa & Khói (`both`)**: **23 video** (33.8%)
- **Video Khói thuần túy (`smoke_only`)**: **15 video** (22.1%)
- **Video âm tính / nhiễu nền (`empty`)**: **3 video** (4.4%)

### 4.2. Độ dài video (Temporal Duration)
- **Độ dài trung bình**: **17.0 giây** (~17 frames @ 1 FPS)
- **Độ dài trung vị**: **13.0 giây** (~13 frames @ 1 FPS)
- **Ngắn nhất**: **5 giây**
- **Dài nhất**: **91 giây** (Video sự cố nổ pin Lithium `battery_thermal_runaway`)

### 4.3. Phân bố theo Bối cảnh Thực tế (8 Subtypes)

| Bối cảnh (Subtype) | Số Video | Tỷ lệ | Đặc điểm quang học & Thử thách nhận diện |
|---|:---:|:---:|---|
| **General Fire & Smoke** | 31 | 45.6% | Đám cháy ngoài trời, bãi rác, nhà xưởng quy mô vừa và lớn |
| **Heavy Industrial Smoke** | 14 | 20.6% | Khói đen/xám dày đặc, cuộn trào từ ống khói, sự cố công nghiệp |
| **Cigarette / Vape Smoke** | 7 | 10.3% | Nguồn khói nhỏ, mỏng, trong nhà — kiểm tra độ nhạy phát hiện sớm |
| **Diffused Early Smoke** | 7 | 10.3% | Khói mờ ảo giai đoạn bốc khói sớm, độ tương phản cực kỳ thấp |
| **Cooking / Kitchen Smoke** | 3 | 4.4% | Khói dầu mỡ nhà bếp, hơi nước — thách thức phân biệt khói nguy hiểm |
| **Welding Sparks (Nhiễu)** | 3 | 4.4% | **Tia lửa hàn**: Ánh sáng chói và khói hàn, dùng kiểm tra chống báo động giả |
| **Structural Room Fire** | 2 | 2.9% | Cháy trong phòng kín, ánh sáng yếu, khói tích tụ trên trần nhà |
| **Battery Thermal Runaway** | 1 | 1.5% | Quá trình phát nhiệt nổ pin xe điện: khói trắng phụt mạnh trước khi nổ lửa |

---

## 5. ĐỀ XUẤT PIPELINE TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING PIPELINE)

Để khai thác tối đa tính năng không gian - thời gian, pipeline tiền xử lý được chuẩn hóa như sau:

```mermaid
graph LR
    Raw[Raw Video Stream] --> Sample[1 FPS Sampling]
    Sample --> Letterbox[Letterbox Resize 640x640]
    Letterbox --> CLAHE[CLAHE LAB Enhancement]
    CLAHE --> Norm[ImageNet Normalization]
    Norm --> SlideWin[Temporal Sliding Window T=8, S=2]
    SlideWin --> Model[Spatio-Temporal Model]
```

### 5.1. Tiền xử lý Không gian (Spatial Preprocessing)
1. **Letterbox Resize (640×640 px)**:
   - Bảo toàn tỷ lệ khung hình gốc (Aspect Ratio), bù màu viền trung tính `(114, 114, 114)` vào hai cạnh thừa.
2. **Tăng cường tương phản thích ứng CLAHE trên không gian màu LAB**:
   - Chuyển ảnh sang không gian màu **LAB**.
   - Áp dụng CLAHE riêng biệt trên kênh **L** (Lightness) với tham số tối ưu:
     $$\text{clipLimit} = 2.0, \quad \text{tileGridSize} = (8, 8)$$
   - Gộp lại với kênh A, B và chuyển về RGB/BGR.
   - *Lợi ích*: Làm nổi bật các dải khói mờ trong môi trường thiếu sáng hoặc hậu cảnh phức tạp mà không làm đổi sắc độ màu ngọn lửa.
3. **Chuẩn hóa Tensor**:
   - Chuyển đổi giá trị pixel về dải $[0.0, 1.0]$.
   - Chuẩn hóa theo thống kê ImageNet:
     $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

### 5.2. Tiền xử lý Thời gian (Temporal Sliding Window cho Video Sequence)
Đối với các kiến trúc mô hình tuần tự (TimeSformer, ConvLSTM, SlowFast, Transformer Spatio-Temporal):
1. **Độ dài cửa sổ thời gian ($T$)**:
   - Cố định mỗi mẫu huấn luyện là một clip con dài **$T = 8$ frames** (tương đương 8 giây thực tế).
2. **Bước trượt lấy mẫu ($S$)**:
   - Tập **Train**: Bước trượt $S = 2$ frames (tạo độ chồng lấn 75% giữa các clip kề nhau, làm giàu dữ liệu huấn luyện).
   - Tập **Val/Test**: Bước trượt $S = 4$ hoặc $8$ frames (đánh giá tuần tự không trùng lặp).
3. **Bảo toàn chuỗi chuyển tiếp (Transition Dynamics)**:
   - Giữ lại các frame `empty` (nền) ở đầu video để mô hình học quy luật động học: $\text{Bình thường} \rightarrow \text{Bốc khói sớm} \rightarrow \text{Bùng lửa}$.

---

## 6. CHIẾN LƯỢC TĂNG CƯỜNG DỮ LIỆU THÍCH ỨNG (PHYSICS-AWARE DATA AUGMENTATION)

Dữ liệu lửa và khói tuân theo các định luật vật lý về đối lưu nhiệt và quang học, vì vậy các phép biến đổi phải đảm bảo tính chân thực:

### 6.1. Quy tắc biến đổi hình học
- ✅ **Random Horizontal Flip ($p = 0.5$)**: Đảo chiều ngang được phép hoàn toàn vì ngọn lửa và luồng gió có thể hướng sang trái hoặc phải.
- ❌ **TUYỆT ĐỐI KHÔNG lật dọc (No Vertical Flip)**: Khói nóng và ngọn lửa luôn bốc lên trên theo trọng lực và nhiệt độ. Lật ngược khiến khói chúc xuống đất — một hiện tượng phi vật lý làm sai lệch vector chuyển động của mô hình.
- ✅ **Random Scale & Shift ($0.8 \le \text{scale} \le 1.2$)**: Mô phỏng thay đổi khoảng cách từ camera tới nguồn cháy.

### 6.2. Tính nhất quán Không gian - Thời gian (Sequence-Level Consistency)
- Khi áp dụng Augmentation trên một chuỗi clip $T = 8$ frames:
  $$\mathbf{M}_{\text{aug}}^{(t_1)} = \mathbf{M}_{\text{aug}}^{(t_2)} = \dots = \mathbf{M}_{\text{aug}}^{(t_T)}$$
- **Mọi frame trong cùng 1 clip phải dùng chung một ma trận biến đổi** (cùng góc xoay, cùng hệ số lật, cùng độ sáng). Nếu biến đổi ngẫu nhiên từng frame riêng rẽ, dòng chuyển động quang học (Optical Flow) sẽ bị đứt gãy hoàn toàn.

### 6.3. Mô phỏng điều kiện môi trường giám sát
- **Photometric Jitter**: Thay đổi nhẹ độ sáng ($\pm 15\%$) và tương phản ($\pm 15\%$) để mô phỏng sự thay đổi mây che, bóng râm mặt trời.
- **Motion Blur nhẹ ($\text{kernel} = 3 \times 3$)**: Mô phỏng camera giám sát bị rung do gió hoặc khói cuộn di chuyển với tốc độ cao.

---

## 7. PHƯƠNG ÁN PHÂN CHIA 3 TẬP TRAIN / VAL / TEST (GROUP STRATIFIED SPLIT)

### 7.1. Nguyên tắc chống rò rỉ dữ liệu (No Data Leakage)
> [!CRITICAL]
> **Phân chia bắt buộc phải ở cấp độ VIDEO (Grouped by Video ID)**. 
> Toàn bộ các khung hình thuộc cùng một video chỉ được phép xuất hiện trong DUY NHẤT một tập (hoặc Train, hoặc Val, hoặc Test). Việc chia ngẫu nhiên từng frame sẽ khiến frame $t$ nằm trong Train còn frame $t+1$ nằm trong Test, gây ra hiện tượng học vẹt bối cảnh (Background Memorization) và độ chính xác kiểm thử cao ảo.

### 7.2. Tỷ lệ phân chia tối ưu: 70% / 18% / 12%

| Tập dữ liệu | Số Video | Tỷ lệ Video | Số Frame | Tỷ lệ Frame | Mục đích sử dụng |
|---|:---:|:---:|:---:|:---:|---|
| **TRAIN** | **46 video** | **67.6%** | **813 frames** | **70.5%** | Huấn luyện trọng số mô hình |
| **VAL** | **14 video** | **20.6%** | **228 frames** | **19.8%** | Tinh chỉnh siêu tham số & Early Stopping |
| **TEST** | **8 video** | **11.8%** | **112 frames** | **9.7%** | Đánh giá độc lập khả năng phát hiện sớm |
| **Tổng cộng** | **68 video** | **100.0%** | **1,153 frames** | **100.0%** | |

### 7.3. Ma trận kiểm chứng tính cân bằng nhãn giữa các tập

| Tập dữ liệu | Tổng số Frame | 🔥 `fire_only` | 💨 `smoke_only` | 🔥💨 `both` | ⬛ `empty` |
|---|:---:|:---:|:---:|:---:|:---:|
| **TRAIN** | **813** | 388 (47.7%) | 192 (23.6%) | 131 (16.1%) | 102 (12.5%) |
| **VAL** | **228** | 115 (50.4%) | 53 (23.2%) | 59 (25.9%) | 1 (0.4%) |
| **TEST** | **112** | 45 (40.2%) | 24 (21.4%) | 34 (30.4%) | 9 (8.0%) |

---

## 8. DANH SÁCH PHÂN BỔ 68 VIDEO VÀO 3 TẬP

### 8.1. Tập Huấn Luyện (TRAIN - 46 Video, 813 Frames)
1. `00e18d86-3389d8dc_pure` (50 frames - general)
2. `012f0550_pexels_heavy_smoke_36164483` (11 frames - heavy smoke)
3. `01c73785_pexels_heavy_smoke_37765564_pure` (11 frames - heavy smoke)
4. `02af2970_pexels_cooking_smoke_3472823_pure` (45 frames - cooking)
5. `034f5d53_pexels_cigarette_smoke_8139853_pure` (11 frames - cigarette)
6. `04d83ee6_pexels_early_smoke_7926437_pure` (13 frames - early smoke)
7. `05ca4b9b_pexels_welding_sparks_34771762` (22 frames - welding)
8. `0aefc692_pexels_welding_sparks_4941457_pure` (21 frames - welding)
9. `0bcf05c5_pexels_cigarette_smoke_8425862_pure` (11 frames - cigarette)
10. `0d48a1ea_pexels_heavy_smoke_8339804_pure` (49 frames - heavy smoke)
11. `0f39007e_pexels_cooking_smoke_18270789_pure` (11 frames - cooking)
12. `0fd09443_pexels_welding_sparks_14314814` (12 frames - welding)
13. `12150804_1080_1920_30fps_pure` (11 frames - general)
14. `12172035_2160_3840_30fps_pure` (11 frames - general)
15. `12192259_1080_1920_30fps_pure` (11 frames - general)
16. `12218199_3114_1440_30fps_pure` (11 frames - general)
17. `12393269_1440_2560_30fps_pure` (11 frames - general)
18. `12935692_1080_1920_60fps_pure` (11 frames - general)
19. `13097120_2160_3840_60fps_pure` (11 frames - general)
20. `13106016_2160_3840_60fps` (16 frames - general)
21. `13106082_2160_3840_60fps` (8 frames - general)
22. `13256420_1080_1920_30fps_pure` (43 frames - general)
23. `13269425_2160_3840_30fps` (31 frames - general)
24. `13270890_2160_3840_60fps` (14 frames - general)
25. `13280461_3840_2160_24fps_pure` (13 frames - general)
26. `13323137_1920_1080_60fps` (9 frames - general)
27. `13570717_2160_3840_30fps_pure` (13 frames - general)
28. `13589189_2160_3840_60fps_pure` (11 frames - general)
29. `13676716_2160_3840_30fps_pure` (11 frames - general)
30. `13845809_2160_3840_25fps` (13 frames - general)
31. `13862697_2160_3840_25fps` (12 frames - general)
32. `13890243_2160_3840_30fps_pure` (11 frames - general)
33. `13903817_3840_2160_30fps_pure` (23 frames - general)
34. `13976903_1080_1920_25fps_pure` (16 frames - general)
35. `14003137_2160_3840_30fps_pure` (6 frames - general)
36. `14056668_2160_3840_60fps_pure` (8 frames - general)
37. `14112316_1440_2560_60fps_pure` (11 frames - general)
38. `14283156_1080_1920_30fps_pure` (21 frames - general)
39. `14379036_1920_1080_50fps` (11 frames - general)
40. `14477514_1080_1920_30fps_pure` (6 frames - general)
41. `17116108-hd_1920_1080_25fps_pure` (10 frames - general)
42. `28f4ec5d_pexels_cigarette_smoke_8139860_pure` (11 frames - cigarette)
43. `312fe283_pexels_cigarette_smoke_6143909` (11 frames - cigarette)
44. `316af409_pexels_heavy_smoke_18105251_pure` (13 frames - heavy smoke)
45. `36f3eac1_pexels_cigarette_smoke_5075501_pure` (8 frames - cigarette)
46. `battery_thermal_runaway_krHU3jlR4Pc` (91 frames - battery thermal runaway)

### 8.2. Tập Kiểm Định (VAL - 14 Video, 228 Frames)
1. `38c6467b_pexels_heavy_smoke_35865134` (8 frames - heavy smoke)
2. `394a2a56_pexels_visible_flame_6985324_pure` (12 frames - general)
3. `3aa2be24_pexels_cigarette_smoke_5320212` (25 frames - cigarette)
4. `407b06ea_pexels_early_smoke_8516427` (13 frames - early smoke)
5. `47edc2fb_pexels_cigarette_smoke_9644281` (11 frames - cigarette)
6. `4a37a9df_pexels_heavy_smoke_17202470_pure` (11 frames - heavy smoke)
7. `4cf31d6f_pexels_heavy_smoke_36068162_pure` (16 frames - heavy smoke)
8. `4e0040e9_pexels_heavy_smoke_34992285_pure` (14 frames - heavy smoke)
9. `66d00a95_pexels_diffused_smoke_35582792` (15 frames - diffused smoke)
10. `6770d887_pexels_heavy_smoke_28478389_pure` (11 frames - heavy smoke)
11. `6a95cab6_pexels_heavy_smoke_10666165` (11 frames - heavy smoke)
12. `6bc652d9_pexels_diffused_smoke_9589423` (11 frames - diffused smoke)
13. `6c3dee00_pexels_heavy_smoke_6618036` (16 frames - heavy smoke)
14. `a7f3eb27_pexels_diffused_smoke_6731483` (31 frames - diffused smoke)

### 8.3. Tập Kiểm Thử Độc Lập (TEST - 8 Video, 112 Frames)
1. `2016 April - Fire Investigation 1B - Milpitas CA_pure_chunk_02` (31 frames - structural room fire)
2. `2016 April - Fire Investigation 1B - Milpitas CA_pure_chunk_03` (30 frames - structural room fire)
3. `6eeb70d4_pexels_heavy_smoke_34004202` (16 frames - heavy smoke)
4. `774e8545_pexels_diffused_smoke_36252846` (22 frames - diffused early smoke)
5. `7cc52846_pexels_heavy_smoke_35241029` (6 frames - heavy smoke)
6. `816fd10a_pexels_cooking_smoke_37848046` (16 frames - cooking smoke)
7. `9a06358f_pexels_heavy_smoke_3568453` (13 frames - heavy smoke)
8. `adc26840_pexels_diffused_smoke_6956078` (12 frames - diffused early smoke)

---

## 9. VỊ TRÍ CÁC FILE LIÊN QUAN TRONG HỆ THỐNG

- **Báo cáo này**: [Dataset_68_Videos_Statistics_and_Preprocessing_Plan.md](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/Dataset_68_Videos_Statistics_and_Preprocessing_Plan.md)
- **Thư mục dữ liệu chuẩn**: [data/dataset_video_prepared/](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/data/dataset_video_prepared)
- **Manifest đầy đủ có cột split**: [data/data_splits/prepared_video_manifest_with_splits.csv](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/data/data_splits/prepared_video_manifest_with_splits.csv)
- **File gán nhãn từng video vào split**: [data/data_splits/video_split_assignment_68vids.csv](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/data/data_splits/video_split_assignment_68vids.csv)
