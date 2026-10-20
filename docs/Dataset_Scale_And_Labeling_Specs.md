# Quy Chuẩn Chi Tiết Về Quy Mô, Số Lượng Và Nhãn Dữ Liệu Trước Khi Train (Dự Án NCKH)

> **Mục tiêu:** Cung cấp bộ tiêu chuẩn định lượng (Quantitative Standards) và quy chuẩn gán nhãn (Labeling Specifications) cho tác vụ **Detection & Tracking Lửa/Khói**, đảm bảo tính khách quan khoa học, tránh hiện tượng rò rỉ dữ liệu (Data Leakage) và báo động giả (False Alarms).

---

## 1. Quy Mô Tổng Thể Của Bộ Dữ Liệu (Dataset Scale)

Để huấn luyện một mô hình học sâu (Deep Learning / Vision Transformer / Spatiotemporal Model) đạt độ tin cậy khoa học và có khả năng tổng quát hóa (Generalization) tốt trên camera CCTV thực tế, quy mô dataset được chia làm 2 giai đoạn:

```
──────────────────────────────────────────────────────────────────────────────────────────
GIAI ĐOẠN 1: SPATIAL DETECTION (Ảnh tĩnh)
  • Tổng số ảnh       : 10,000 – 15,000 ảnh  (Hiện tại bạn đã có 11,437 ảnh -> ĐẠT CHUẨN)
  • Số lượng Bounding Box: ~20,000 – 30,000 bboxes
──────────────────────────────────────────────────────────────────────────────────────────
GIAI ĐOẠN 2: SPATIOTEMPORAL DETECTION & TRACKING (Video Sequences)
  • Tổng số Video Clips: 800 – 1,200 video clips (Hiện tại bạn có 978 train + 942 test -> ĐẠT CHUẨN)
  • Tần số trích xuất  : 5 FPS (Mỗi giây lấy 5 khung hình)
  • Tổng số Frames nhãn: 30,000 – 50,000 frames được gán nhãn Detection + Track ID
──────────────────────────────────────────────────────────────────────────────────────────
```

---

## 2. Thống Kê Định Lượng & Phân Bổ Nhãn (Quantitative Distribution)

### 2.1 Cân bằng Nhãn (Class Balance)
Dữ liệu nghiêng lệch (Imbalanced Data) sẽ khiến mô hình bị bias (ví dụ: chỉ đoán ra Lửa mà bỏ sót Khói, hoặc báo động giả liên tục). Phân bổ target tối ưu như sau:

| Tên Nhãn | Tỷ lệ % Tối ưu | Mục đích Kỹ thuật |
| :--- | :--- | :--- |
| **Class 0: Fire (Ngọn lửa)** | **35% - 40%** | Nhận diện màu sắc (đỏ/cam/vàng), hình dạng ngọn lửa, phản chiếu. |
| **Class 1: Smoke (Vệt khói)** | **35% - 40%** | Nhận diện kết cấu cuộn (texture), màu xám/đen/trắng, độ mờ/đục. |
| **Class 2: Hard Negatives (Nhiễu)** | **20% - 25%** | **CỰC KỲ QUAN TRỌNG:** Hơi nước đun sôi, khói bếp, khói thuốc, đèn xe, tia lửa hàn kim loại. |

---

### 2.2 Tỷ Lệ Phân Chia Dataset (Data Split Ratio & Rules)

```
       ┌─────────────────────────────────────────────────────────────┐
       │ TỔNG THỂ DATASET                                            │
       └──────────────┬──────────────────────┬───────────────────────┘
                      │                      │
                      ▼ 70%                  ▼ 15%                  ▼ 15%
             ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
             │    TRAIN SET    │    │ VALIDATION SET  │    │    TEST SET     │
             │   (Huấn luyện)  │    │  (Chỉnh hyper-  │    │   (Đánh giá     │
             │                 │    │    parameter)   │    │    Benchmark)   │
             └─────────────────┘    └─────────────────┘    └─────────────────┘
```

> [!CAUTION]
> **QUY TẮC VÀNG TRÁNH DATA LEAKAGE TRONG VIDEO (Rất nhiều bài báo bị bắt lỗi này):**  
> Khi phân chia tập Train / Val / Test cho Video, **BẮT BUỘC phân chia theo VIDEO ID (Scene ID)**, KHÔNG ĐƯỢC phân chia ngẫu nhiên từng frame.  
> *Ví dụ:* Nếu Video `V001.mp4` có 300 frames, TOÀN BỘ 300 frames của `V001` phải nằm ở tập Train. Không được đưa frame 1-200 vào Train và frame 201-300 vào Test (vì các frame trong cùng 1 video có bối cảnh giống hệt nhau, sẽ gây ra kết quả chính xác ảo >99%).

---

## 3. Quy Chuẩn Gán Nhãn Chi Tiết (Labeling Specifications)

### 3.1 Quy chuẩn gán nhãn Detection (YOLO / COCO Bounding Box)

1. **Nguyên tắc Ôm Sát (Tight Bounding Box):**
   - Khung hình chữ nhật phải ôm sát viền phát sáng của ngọn lửa hoặc phần mật độ dày nhất của vệt khói.
   - Không kéo khung quá rộng chứa quá 30% background không liên quan.

2. **Quy tắc gán nhãn Khói (Smoke Annotation):**
   - Khói có đặc tính tán rộng nhạt dần. Chỉ gắn nhãn phần khói có độ đục (Opacity) từ mỏng đến đặc.
   - Phần khói đã tan biến hoàn toàn vào không khí (không còn thấy ranh giới) thì không khoanh box.

3. **Xử lý Che khuất (Occlusion & Small Objects):**
   - Bỏ qua các đốm lửa/khói có kích thước quá nhỏ ($< 8 \times 8$ pixels trên ảnh 640×640), ngoại trừ trường hợp **Early Smoke Detection** (khói sớm xuất hiện từ thiết bị điện).
   - Nếu ngọn lửa bị vật cản (ghế, cây, tường) che ở giữa chia làm 2 phần tách biệt $\rightarrow$ Gắn nhãn thành **2 Bounding Boxes riêng biệt**.

---

### 3.2 Quy chuẩn gán nhãn Tracking (Multi-Object Tracking - MOT Standard)

Một tệp nhãn Tracking chuẩn MOT (`gt.txt`) bắt buộc có các trường sau cho từng frame:

```
Format: frame_id, track_id, bb_left, bb_top, bb_width, bb_height, confidence, class_id, visibility
```

```csv
1, 1, 120, 80, 45, 60, 1, 0, 1.0   <-- Frame 1: Lửa (Class 0) mang Track ID #1
1, 2, 300, 50, 90, 110, 1, 1, 1.0  <-- Frame 1: Khói (Class 1) mang Track ID #2
2, 1, 122, 82, 48, 65, 1, 0, 1.0   <-- Frame 2: Lửa ID #1 lớn lên chút và di chuyển
2, 2, 305, 48, 98, 120, 1, 1, 1.0  <-- Frame 2: Khói ID #2 cuộn rộng ra
```

#### Quy tắc quản lý Track ID (ID Consistency Rules):
- **Tính Nhất Quán (Persistence):** Một ngọn lửa từ lúc bùng phát đến khi tắt phải giữ nguyên **duy nhất 1 Track ID**.
- **Tách Đám Cháy (Splitting):** Nếu ngọn lửa ID #1 bị cháy lan tách ra thành 2 đốm cháy riêng biệt $\rightarrow$ Đốm ban đầu giữ ID #1, đốm mới sinh ra được cấp **Track ID #3 mới**.
- **Gộp Đám Cháy (Merging):** Nếu 2 ngọn lửa ID #1 và ID #3 cháy lan rộng rồi nhập làm một $\rightarrow$ Đám cháy mới sẽ mang ID của ngọn lửa xuất hiện trước hoặc có diện tích lớn hơn.

---

## 4. Checklist Kiểm Định Dữ Liệu Trước Khi Bắt Đầu Train

Trước khi ấn nút bấm Train mô hình, bạn cần chạy script kiểm tra (Sanity Check) danh mục sau:

- [ ] **Kiểm tra File Nhãn Trống (Empty Labels):** Đảm bảo tập train chứa khoảng 10-15% ảnh background thuần (không có lửa/khói) để huấn luyện mô hình khả năng "không dự đoán bừa".
- [ ] **Kiểm tra Tọa độ Hợp lệ (Coordinate Bounding):** Đảm bảo tất cả $x, y, w, h$ nằm nghiêm ngặt trong khoảng $[0.0, 1.0]$. Không có giá trị âm hoặc $> 1.0$.
- [ ] **Kiểm tra Corrupted Files:** Quét qua toàn bộ video/ảnh xem có file nào bị lỗi decoders (OpenCV không đọc được frame) hay không.
- [ ] **Độc lập Scene giữa các Tập Split:** Xác nhận 100% không có video/ảnh nào ở tập Val/Test xuất hiện cùng bối cảnh (Background scene) với tập Train.
- [ ] **Khả năng hiển thị (Visual Inspection):** Vẽ ngẫu nhiên 100 frames chứa Bounding Box & Track ID ra màn hình để mắt người kiểm tra trực quan độ chính xác của nhãn.
