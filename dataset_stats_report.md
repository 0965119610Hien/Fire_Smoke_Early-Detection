# Báo Cáo Phân Tích Dữ Liệu Chi Tiết & Hướng Tiền Xử Lý Mô Hình
*(Đánh giá chuyên sâu cho mô hình phát hiện cháy sớm V-JEPA / Spatiotemporal)*

> **Cập nhật:** Cấu trúc dữ liệu mới nhất
> **Train path:** `e:\NCKH\files\Data\Video\dataset_train`
> **Test path:** `e:\NCKH\files\Data\Video\dataset_test`
> Ngày tạo báo cáo: 2026-07-21 12:03:06

---

## 1. Thống Kê Số Liệu Toàn Diện

Dữ liệu đã được phân loại thành 3 nhãn chính (3-class classification). Dưới đây là bức tranh tổng thể về số lượng và tính chất video thực tế:

### Tổng quan
| Metric | Train (Học) | Test (Kiểm thử) |
|---|---|---|
| **Tổng số Video** | **978 video** | **942 video** |
| **Tổng thời lượng** | 275.3 phút | 576.6 phút |
| **Tổng dung lượng** | 9.4 GB | 16.6 GB |
| **Thời lượng trung bình** | **16.89 giây** | **36.72 giây** |

### Phân bố theo nhãn (Class Distribution)
| Nhãn | Train | Tỉ lệ Train | Test | Tỉ lệ Test |
|---|---|---|---|---|
| **0_normal_noise** | 367 | 37.5% | 423 | 44.9% |
| **1_controlled_fire** | 285 | 29.1% | 255 | 27.1% |
| **2_uncontrolled_hazard** | 326 | 33.3% | 264 | 28.0% |

---

## 2. Thống kê chi tiết & Phân bổ Subtype

### A. Chi tiết tập TRAIN
| Label | Videos | Dur min | Dur max | Dur mean | Dur stdev | FPS mean |
|---|---|---|---|---|---|---|
| `0_normal_noise` | 367 | 5.04s | 221.23s | 18.42s | 19.94s | 29.54 |
| `1_controlled_fire` | 285 | 3.60s | 182.53s | 15.48s | 14.05s | 32.68 |
| `2_uncontrolled_hazard` | 326 | 5.33s | 277.97s | 16.41s | 15.23s | 30.74 |
| **TOTAL** | **978** | **3.60s** | **277.97s** | **16.89s** | — | 30.86 |

### B. Chi tiết tập TEST
| Label | Videos | Dur min | Dur max | Dur mean | Dur stdev | FPS mean |
|---|---|---|---|---|---|---|
| `0_normal_noise` | 423 | 19.52s | 108.04s | 33.02s | 14.15s | 27.66 |
| `1_controlled_fire` | 255 | 19.54s | 189.28s | 36.53s | 20.84s | 27.87 |
| `2_uncontrolled_hazard` | 264 | 19.52s | 240.74s | 42.85s | 28.43s | 30.43 |
| **TOTAL** | **942** | **19.52s** | **240.74s** | **36.72s** | — | 28.50 |

---

## 3. Mô Tả Chi Tiết Ngữ Nghĩa Của Từng Nhãn

Phần này phân tích ngữ nghĩa vật lý của các video trong từng nhãn dựa trên số liệu thực tế được quét:

### 🟢 Nhãn 0: Normal Noise (Không có cháy rủi ro)
Nhãn quan trọng nhất để hệ thống ép tỷ lệ báo động giả (False Positive) xuống mức thấp.
- **`cigarette_vape_smoke`:** Khói thuốc lá, vape bốc lên và tan nhanh vào không khí. 
- **`dust_particles` / `steam_vapor`:** Hạt bụi bay lơ lửng gần ống kính (phản xạ đèn hồng ngoại) và hơi nước (như từ ấm đun nước). Rất dễ nhầm với khói sương.
- **`cooking_smoke_normal`:** Khói sinh hoạt nhà bếp thông thường, mật độ mỏng.
- **`optical_noise` / `daily_activity_fp`:** Các hoạt động nhiễu quang học (đèn xe lướt qua), mờ ống kính, nhện giăng tơ...
- **`welding_sparks_noise`:** Tia lửa hàn cường độ thấp.


### 🟡 Nhãn 1: Controlled Fire (Cháy/Khói Có Kiểm Soát)
Phân biệt sự cháy có chủ đích, có sự giám sát của con người.
- **`welding_sparks_ctrl`:** Máy cắt/hàn kim loại công nghiệp phát ra vệt lửa dài liên tục.
- **`heavy_smoke_controlled`:** Khói đốt rác, cỏ ngoài đồng, khói nhà máy. Mật độ dày nhưng an toàn.
- **`small_visible_flame`:** Lửa bếp gas, nến, ngọn lửa củi trại nhỏ nhắn.
- **`cooking_smoke_threshold`:** Khói bếp đặc (đồ ăn khét), mấp mé ngưỡng báo động.

### 🔴 Nhãn 2: Uncontrolled Hazard (Nguy Hiểm/Ngoài Tầm Kiểm Soát)
Yêu cầu còi báo động kích hoạt tức thì.
- **`residential_indoor_fire` / `large_visible_flame`:** Ngọn lửa lớn bùng phát trong phòng khách, kho.
- **`heavy_smoke_hazard` / `toxic_material_smoke`:** Khói đặc cuồn cuộn che khuất tầm nhìn, khói từ nhựa/hóa chất (màu đen/đục).
- **`diffused_smoke_spreading`:** Khói tràn lan khắp các hành lang, trần nhà (thời điểm hệ thống PCCC truyền thống mới bắt đầu nhận diện).
- **`early_smoke_real_fire` / `daily_ctx_real_fire`:** Giai đoạn bắt đầu chập cháy của các vật liệu thực tế.

---

## 4. Phân Tích Kỹ Thuật (Lỗ Hổng Dữ Liệu) & Hướng Tiền Xử Lý

Dựa trên việc đọc metadata (FPS, Dimensions, Duration) của toàn bộ video, hệ thống vạch ra 3 vấn đề lớn và các bước tiền xử lý bắt buộc cho mô hình V-JEPA / TimeSformer:

### A. Độ Lệch Thời Lượng (Duration Shift)
- **Vấn đề:** 
  - Train có clip ngắn nhất là **{train_agg['dur_min']:.1f} giây**, trung bình **{train_agg['dur_mean']:.1f} giây**.
  - Test có clip ngắn nhất là **{test_agg['dur_min']:.1f} giây**, trung bình **{test_agg['dur_mean']:.1f} giây**.
- **Hệ quả:** Mô hình Spatiotemporal học theo chuỗi thời gian. Dạy trên clip ngắn rồi test trên clip dài 40s sẽ làm hỏng Attention Map.
- **Hướng giải quyết:**
  - **Train:** Bỏ qua (Filter) toàn bộ clip < 5 giây.
  - **Test (Inference):** Dùng Sliding Window. Cắt video test dài thành các đoạn 10s (stride = 5s). Đưa từng đoạn vào mô hình, lấy điểm Max hoặc Majority Vote.

### B. Xung Đột Tốc Độ Khung Hình (FPS Inconsistency)
- **Vấn đề:** FPS của dữ liệu không đồng nhất (biến động từ 24fps đến 60fps). Độ lệch chuẩn FPS (STDEV) ở Train có lúc lên tới **{train_agg['label_stats'].get('1_controlled_fire', {}).get('fps_std', 0):.2f}**.
- **Hệ quả:** Nếu lấy frame theo index (ví dụ bốc frame số 10, 20), thời gian thực tế thu được sẽ sai lệch hoàn toàn giữa video 24fps và 60fps.
- **Hướng giải quyết:** Bắt buộc áp dụng **Uniform Time Sampling (Lấy mẫu theo giây tuyệt đối)**. Nếu lấy 16 frames, phải cắt chính xác ở các mốc thời gian $t = 0, rac{T}{16}, rac{2T}{16}...$ thay vì đếm frame.

### C. Độ Lệch Khung Hình (Spatial Resolution Chaos)
- **Vấn đề:** 
  - Dữ liệu chứa rất nhiều độ phân giải khác nhau, bao gồm cả quay ngang (Landscape) và quay dọc (Portrait) như: 1920x1080, 540x960, 2048x1080, 1280x720...
- **Hệ quả:** Nếu dùng lệnh `cv2.resize(224, 224)` thông thường, các video quay dọc 9:16 sẽ bị bóp méo thành hình vuông 1:1, phá vỡ hình dáng ngọn lửa và luồng khói.
- **Hướng giải quyết:** 
  - **Short-side Resize + CenterCrop:** Thu nhỏ cạnh ngắn nhất về 256px, rồi cắt chính giữa khung hình 224x224px.
  - **Letterbox (Zero Padding):** Thêm viền đen 2 bên để giữ nguyên tỉ lệ video ban đầu, không làm mất bối cảnh.
