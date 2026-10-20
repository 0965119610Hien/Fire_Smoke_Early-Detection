# Phân Tích Chuẩn Dữ Liệu Video Detection/Tracking & Chiến Lược Sinh Dữ Liệu Bằng Mô Hình Diffusion (NCKH)

> **Mục tiêu:** Đánh giá tính sẵn sàng của bộ dữ liệu hiện tại, đưa ra các tiêu chuẩn kỹ thuật bắt buộc để phục vụ bài toán **Video Detection & Tracking**, đồng thời đề xuất giải pháp sinh dữ liệu tổng hợp (Synthetic Data Generation) bằng **Diffusion Models** kèm nhãn tự động cho NCKH.

---

## 1. Đánh giá Bộ Dữ Liệu Video Hiện Tại (`Data/Video`)

| Tiêu chí | Trạng thái hiện tại | Chuẩn cho Detection & Tracking | Đánh giá |
| :--- | :--- | :--- | :--- |
| **Cấp độ Nhãn (Granularity)** | Nhãn cấp độ Video (`0_normal_noise`, `1_controlled_fire`, `2_uncontrolled_hazard`) | Nhãn cấp độ Frame (`Bounding Box` hoặc `Polygon Mask` cho từng frame) | ❌ **CHƯA ĐẠT** (Không thể train Detection nếu không có tọa độ $x,y,w,h$ per frame) |
| **Định danh Đối tượng (Track ID)** | Không có | Có mã ID cố định cho từng ngọn lửa/vệt khói qua các frame (`Track_ID: 1, 2,...`) | ❌ **CHƯA ĐẠT** (Không thể train/eval Tracking nếu không có ID duy nhất qua thời gian) |
| **Phân bố Thời lượng** | Train ~16.9s, Test ~36.7s | Đoạn clip ngắn 5–10s chứa chuyển động rõ ràng của đối tượng | 🟡 **TRUNG BÌNH** (Chưa gọt giũa các khoảng thời gian trống không có lửa/khói) |
| **FPS & Resolution** | FPS không đồng nhất (24–60fps), mixed resolution | FPS chuẩn hóa (15 hoặc 30fps), resolution ngắn nhất 256px trở lên | 🟡 **TRUNG BÌNH** (Cần tiền xử lý) |

> **Kết luận:** Bộ dữ liệu video hiện tại **chỉ mới phù hợp cho bài toán Video Classification**. Nếu muốn làm **Detection và Tracking**, bạn **BẮT BUỘC** phải gắn nhãn lại (Re-annotate) theo chuẩn Frame-level hoặc áp dụng quy trình tự động sinh nhãn/sinh dữ liệu bên dưới.

---

## 2. Tiêu Chuẩn Chi Tiết Cho Dataset Detection & Tracking (Chuẩn NCKH)

Để phục vụ nghiên cứu khoa học đạt chuẩn đăng bài báo/báo cáo khoa học, dataset dành cho Detection & Tracking cần đáp ứng 3 nhóm tiêu chí sau:

### 2.1 Định dạng Dữ liệu & Annotation Format

1. **Dạng Detection (Frame-level Object Detection - YOLO / COCO Format):**
   - Mỗi video được trích xuất ra các khung hình (ví dụ sampling 5 FPS hoặc 10 FPS).
   - Tệp nhãn tương ứng cho mỗi frame: `class_id x_center y_center width height` (normalized 0–1).
   - Classes: `0: Fire`, `1: Smoke`.

2. **Dạng Tracking (Multi-Object Tracking - MOT Format):**
   - Cấu trúc thư mục chuẩn MOT Challenge:
     ```
     sequence_001/
     ├── img1/
     │   ├── 000001.jpg
     │   ├── 000002.jpg
     │   └── ...
     └── gt/
         └── gt.txt  # Frame_idx, Track_ID, X, Y, W, H, Conf, Class, Visibility
     ```
   - Định dạng mỗi dòng trong `gt.txt`:
     `frame_index, track_id, top_left_x, top_left_y, width, height, 1, class_id, 1.0`

### 2.2 Đa dạng Ngữ cảnh & Môi trường (Dataset Diversity)

- **Môi trường (Domain):** 
  - *Indoor:* Kho hàng, nhà xưởng, chung cư, phòng tối, khu vực có góc khuất.
  - *Outdoor:* Rừng, bãi rác, công trường, đường phố.
- **Đối tượng Nhiễu (Hard Negatives - Cực kỳ quan trọng để giảm Báo động giả):**
  - Hơi nước đun sôi, khói bếp, khói thuốc lá/vape (để mô hình phân biệt với khói cháy).
  - Ánh đèn hồng ngoại phản chiếu trên camera đêm, tia lửa hàn kim loại, bóng đèn xe lướt qua.
- **Kịch bản Biến đổi Động (Temporal Dynamics):**
  - Giai đoạn phát sinh (Early stage - khói mỏng/lửa nhỏ).
  - Giai đoạn bùng phát (Spreading stage - lửa lan rộng, khói cuộn dồn dập).

---

## 3. Chiến Lược Sử Dụng Mô Hình Diffusion Để Sinh Dữ Liệu (Synthetic Data Generation)

> **Điểm Mới Khoa Học (Novelty):** Trong NCKH, việc áp dụng **Diffusion-based Generative Augmentation** để giải quyết bài toán thiếu dữ liệu hoặc mất cân bằng dữ liệu cháy nổ là một đóng góp rất giá trị. Đặc biệt, bạn có thể thiết kế quy trình **TỰ ĐỘNG TẠO NHÃN BOUNDING BOX CHUẨN 100% KHÔNG CẦN DÁN NHÃN THỦ CÔNG**.

---

### Phương án 3.1: Stable Diffusion Inpainting + ControlNet (Tự Động Sinh Ảnh/Video kèm Bounding Box)

Thay vì vẽ lửa/khói rồi đi dán nhãn tay, ta làm ngược lại: **Định nghĩa vị trí Bounding Box trước, rồi ép mô hình Diffusion sinh Lửa/Khói đúng vị trí đó!**

```
Ảnh môi trường thực tế (Kho hàng / Nhà xưởng) 
        │
        ▼
Tự vẽ một Bounding Box Mask ngẫu nhiên ở vị trí mong muốn (Vd: x=100, y=150, w=80, h=120)
        │
        ▼
Đưa Mask + Text Prompt ("real visible fire flame", "heavy toxic smoke") vào SD Inpainting / ControlNet
        │
        ▼
Mô hình Diffusion sinh ra Lửa/Khói hòa trộn tự nhiên vào khung hình tại ĐÚNG VỊ TRÍ MASK
        │
        ▼
OUTPUT TỰ ĐỘNG: Ảnh tổng hợp + Bounding Box Label chuẩn 100%!
```

#### Ưu điểm:
- **Tự động 100% Bounding Box:** Tọa độ mask ban đầu chính là nhãn $x, y, w, h$ cho YOLO/DETR.
- **Đa dạng ngữ cảnh vô hạn:** Lấy 1 ảnh nhà xưởng trống có thể sinh ra 100 kịch bản lửa/khói khác nhau ở các vị trí khác nhau.
- **Giải quyết bài toán thiếu ảnh cháy hiểm hóc:** Có thể sinh cháy ở những góc kín, môi trường đêm, khói đặc...

---

### Phương án 3.2: Video Diffusion (AnimateDiff / Stable Video Diffusion - SVD)

Để sinh ra các đoạn **Video Clip ngắn (2–4 giây)** phục vụ bài toán Tracking:

1. **Bước 1:** Dùng SD Inpainting (Phương án 3.1) để sinh ra frame 0 chứa ngọn lửa/khói kèm Bounding Box $B_0$.
2. **Bước 2:** Đưa frame 0 vào mô hình **Stable Video Diffusion (SVD)** hoặc **AnimateDiff** với motion prompt (ví dụ: *"fire spreading, smoke rising"*).
3. **Bước 3:** Mô hình sẽ sinh ra 16–24 frames tiếp theo thể hiện sự cuộn mở rộng của lửa/khói.
4. **Bước 4 (Auto-Tracking Label):** Sử dụng thuật toán Optical Flow hoặc Segment Anything 2 (SAM 2) để tự động lan truyền Bounding Box $B_0$ qua các frame $B_1, B_2, ..., B_N$ kèm theo `Track_ID = 1`.

---

## 4. Pipeline Tự Động Hóa Dữ Liệu Đề Xuất (End-to-End Workflow)

```
                       ┌────────────────────────────────────────────────────────┐
                       │ 1. Raw Video Crawl (Pexels / Pixabay / YouTube)       │
                       └───────────────────┬────────────────────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────────────────────┐
                       │ 2. Automated Pseudo-Labeling Pipeline                  │
                       │    - Run Grounding DINO ("fire", "smoke")             │
                       │    - Run SAM 2 (Segment Anything in Video) for tracking│
                       └───────────────────┬────────────────────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────────────────────┐
                       │ 3. Data Quality & Imbalance Inspection                 │
                       │    - Check missing cases / rare scenarios              │
                       └───────────────────┬────────────────────────────────────┘
                                           │
                                  ┌────────┴────────┐
                                  │ Thiếu dữ liệu?  │
                                  └────────┬────────┘
                                           │ YES
                                           ▼
                       ┌────────────────────────────────────────────────────────┐
                       │ 4. Diffusion Synthetic Generation                      │
                       │    - SD Inpainting (Generate Fire/Smoke on Masks)      │
                       │    - SVD / AnimateDiff (Generate Video Sequences)     │
                       └───────────────────┬────────────────────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────────────────────┐
                       │ 5. Combined Dataset for Detection & Tracking           │
                       │    (Real Annotations + Diffusion Synthetic Data)       │
                       └────────────────────────────────────────────────────────┘
```

---

## 5. Lộ Trình Hành Động Cụ Thể

1. **Bước 1: Không cần hủy bỏ video đã crawl.** Sử dụng công cụ **Grounding DINO** (Open-vocabulary Object Detector) kết hợp **SAM 2 (Segment Anything Model 2)** để tự động sinh Bounding Box và Track ID cho toàn bộ tập video `dataset_train` và `dataset_test` hiện tại.
2. **Bước 2: Kiểm tra các tình huống thiếu (Hard Scenarios).** Lọc ra các môi trường chưa có (ví dụ: nhà xưởng ban đêm, khói đặc lan trong hành lang).
3. **Bước 3: Sử dụng Stable Diffusion Inpainting** để sinh bổ sung 2,000–3,000 ảnh/video tổng hợp kèm Bounding Box tự động cho các tình huống thiếu này.
4. **Bước 4: Đánh giá NCKH.** Trong bài báo/báo cáo NCKH, thực hiện ablation study so sánh:
   - Mô hình train chỉ bằng Dữ liệu thực (Real Data).
   - Mô hình train bằng Dữ liệu thực + Dữ liệu sinh bởi Diffusion (Real + Diffusion Synthetic Data).
   - *Kết quả này sẽ là điểm cộng rất lớn về mặt đóng góp khoa học!*
