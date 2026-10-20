# Báo Cáo Thống Kê Chi Tiết Dataset: `pure_fire_smoke_clips` (HOÀN THÀNH 100%)

> **Thời gian cập nhật:** 2026-08-04  
> **Nguồn dữ liệu:** `trimmed_fire_incidents/` (177 video gốc)  
> **Thư mục lưu trữ:** `pure_fire_smoke_clips/`  
> **Pipeline:** `src/filter_pure_fire_smoke.py` (Cắt bỏ khoảng lặng không có khói/lửa + ghép nối tự động per video)

---

## 1. Kết Quả Tổng Quan

| Chỉ Số | Giá Trị Chi Tiết |
|:---|:---|
| **Tổng số video quét** | **177 / 177 video** (100% Hoàn thành) |
| **Số video lưu (Saved)** | **177 video** (`*_pure.mp4`) |
| **Số video bỏ qua (Skipped)** | **0 video** |
| **Lỗi (Errors)** | **0 video** |
| **Thời lượng khói/lửa thuần (Active)** | **271.3 phút (~4.52 giờ)** |
| **Tổng thời lượng video gốc (Raw)** | **276.2 phút (~4.60 giờ)** |
| **Độ tinh khiết trung bình (Average Purity)** | **98.5%** |
| **Tổng dung lượng bộ dữ liệu** | **3.52 GB** (3,604.6 MB) |

---

## 2. Phân Bố Theo Loại Nội Dung (177 Videos)

| Thể Loại / Phân Loại | Số Lượng Video | Mô TảChi Tiết |
|:---|:---:|:---|
| `other` (Real Fire, CCTV, Fire Investigation) | **54** | Video đám cháy nhà, cháy xe, dashcam & cctv cháy thực tế |
| `fire_welding_sparks` (Pexels) | **22** | Tia lửa hàn, lửa cắt kim loại (mặt phẳng gần) |
| `yt_warehouse_factory` (YouTube CCTV) | **20** | Khói & cháy tại nhà kho, xưởng sản xuất |
| `smoke_cigarette` (Pexels) | **19** | Khói thuốc lá (khói nhẹ ban đầu) |
| `smoke_heavy` (Pexels) | **17** | Khói bốc cao, khói đen/xám dày đặc |
| `yt_indoor_building` (YouTube CCTV) | **8** | Khói & cháy trong tòa nhà, văn phòng |
| `smoke_cooking` (Pexels) | **7** | Khói bếp nấu ăn, khói chảo |
| `smoke_diffused` (Pexels) | **7** | Khói loãng lan tỏa trong không gian |
| `yt_vehicle_transport` (YouTube CCTV) | **6** | Cháy phương tiện giao thông (ô tô, xe tải) |
| `smoke_early` (Pexels) | **4** | Khói bốc sớm lúc mới phát sinh |
| `yt_welding_sparks` (YouTube) | **4** | Tia lửa hàn môi trường công nghiệp |
| `yt_outdoor_industrial` (YouTube CCTV) | **3** | Cháy/khói khu vực công nghiệp ngoài trời |
| `fire_visible_flame` (Pexels) | **2** | Ngọn lửa bùng phát rõ ràng |
| `yt_early_smoke` (YouTube) | **2** | Khói sớm trên camera giám sát YT |
| `yt_diffused_smoke` (YouTube) | **1** | Khói loãng CCTV |
| `yt_residential_fire` (YouTube) | **1** | Cháy khu dân cư |

---

## 3. Phân Bố Thời Lượng Clip Thuần (`*_pure.mp4`)

| Khoảng Thời Lượng | Số Video | % Tỷ Lệ | Đánh Giá Cho Huấn Luyện (Training) |
|:---|:---:|:---:|:---|
| `< 5 giây` (rất ngắn) | 1 | 0.6% | Clip siêu ngắn (12935692 - 4.4s) |
| `5 – 10 giây` (ngắn) | 27 | 15.3% | Thích hợp cho clip test nhanh |
| `10 – 30 giây` *(Tối Ưu)* | **80** | **45.2%** | **Khung thời lượng chuẩn nhất cho SegFormer/TimeSformer** |
| `30 – 60 giây` | 26 | 14.7% | Rất tốt cho theo dõi biến thiên vận tốc ($v_{growth}$) |
| `1 – 5 phút` | 27 | 15.3% | Tốt cho sequence tracking dài |
| `> 5 phút` | 16 | 9.0% | Video diễn biến cháy dài (cần cắt chunk nếu train SegFormer) |

* **Thời lượng trung bình:** `92.0 giây / clip`
* **Clip ngắn nhất:** `4.4s` (`12935692_1080_1920_60fps_pure.mp4`)
* **Clip dài nhất:** `1170.2s` (~19.5 phút - `fire6_003_pure.mp4`)

---

## 4. Thống Kê Video Có Nhiều Đoạn Cắt Nhỏ (Multi-Segment Concat)

Có **6 video** chứa các khoảng thừa (người đi lại, cảnh quay không có khói/lửa) ở giữa đã được thuật toán phát hiện, tự động cắt bỏ và ghép liền mạch các đoạn khói lửa lại:

1. `yt_indoor_building_BJ9ng9L1CA0_trimmed` (4 đoạn): `0.0-164.5s` \| `191.5-199.7s` \| `201.2-271.3s` \| `281.4-299.3s`
2. `yt_warehouse_factory_BJ9ng9L1CA0_trimmed` (4 đoạn): `0.0-164.5s` \| `191.5-199.7s` \| `201.2-271.3s` \| `281.4-299.3s`
3. `yt_warehouse_factory_wFgBjbj-tb8_trimmed` (2 đoạn): `0.0-62.9s` \| `64.5-79.0s`
4. `yt_warehouse_factory_-ItB9JAzheM_trimmed` (2 đoạn): `0.0-8.0s` \| `11.3-66.9s`
5. `9d89d1e8_yt_welding_sparks_3IvkCKzC9KE` (2 đoạn): `0.0-3.0s` \| `8.2-25.9s`
6. `3aa2be24_pexels_cigarette_smoke_5320212` (2 đoạn): `3.9-7.1s` \| `9.1-12.1s`

---

## 5. Danh Sách Video Cần Lưu Ý Cho Các Bước Tiếp Theo

### 🔹 Top 5 Video Dài Nhất (Cần Chunking 30s trước khi train SegFormer):
1. `fire6_003_pure.mp4` — **1,170.2s** (~19.5 phút)
2. `fire6_002_pure.mp4` — **1,048.9s** (~17.5 phút)
3. `fire12_001_pure.mp4` — **859.2s** (~14.3 phút)
4. `fire12_002_pure.mp4` — **802.8s** (~13.4 phút)
5. `fire9_001_pure.mp4` — **732.0s** (~12.2 phút)

---

## 6. Cấu Trúc Thư Mục Kết Quả Cuối Cùng

```
pure_fire_smoke_clips/
├── *.mp4                        ← 177 file video đã lọc thuần khói & lửa
└── pure_fire_smoke_manifest.csv ← Bảng kê chi tiết 177 video (đã ghi đầy đủ)
```
