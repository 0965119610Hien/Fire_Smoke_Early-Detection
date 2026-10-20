# Các Giải Pháp Nâng Cao Chất Lượng Phân Đoạn & Gán Nhãn (Fire & Smoke Segmentation)

> **Phân tích nguyên nhân:** Phương pháp lọc màu HSV cơ bản kết hợp Bounding Box thô có ưu điểm chạy nhanh, nhưng gặp hạn chế khi phân biệt khói trắng trên nền sáng, lửa mờ/tàn tro, hoặc vật thể màu cam (bàn gỗ, áo cam).
> 
> Dưới đây là **5 giải pháp chiến lược cấp độ NCKH** giúp nâng độ chính xác của mặt nạ (Mask Segmentation) lên **90% - 95%**.

---

## 1. Top 5 Giải Pháp Cải Thiện Chất Lượng Mặt Nạ (Mask Quality)

---

### 🚀 Giải Pháp 1: Tích Hợp Meta SAM 2 (Segment Anything 2 - Video Memory Tracking)

Meta vừa phát hành **SAM 2 (2024)** chuyên biệt cho **Video Segmentation**:
* **Cơ chế:** Thay vì xử lý từng frame ảnh tĩnh độc lập, SAM 2 sở hữu bộ nhớ thời gian (**Memory Bank**).
* **Cách hoạt động:** Bạn chỉ cần xác định vùng khói/lửa ở **frame đầu tiên**, SAM 2 sẽ **tự động theo dõi (Track) và trích xuất mặt nạ mượt mà qua toàn bộ các frame tiếp theo** của clip mà không bị mất dấu hay nhầm nền.
* **Độ chính xác:** Rất cao, biên giới khói/lửa chuyển động bám sát theo thời gian.

---

### 🚀 Giải Pháp 2: Dùng Pretrained SegFormer / YOLOv8-Seg Checkpoint Chuyên Dụng

Thay vì gán nhãn từ đầu bằng luật màu, ta sử dụng mô hình Phân đoạn đã được huấn luyện sẵn trên các bộ dữ liệu cháy lớn trên thế giới (**D-Fire Dataset**, **FLAME Dataset**, **WildfireDB**):

* **Cách hoạt động:**
  1. Tải checkpoint SegFormer (MiT-B2) hoặc YOLOv8x-Seg đã pre-train trên 20,000+ ảnh cháy/khói thực tế.
  2. Chạy mô hình này quét qua 3,339 frame ảnh của bạn để xuất ra **Mặt nạ nháp (Pseudo-labels)**.
* **Ưu điểm:** Mô hình đã học được hàng vạn dạng khói/lửa phức tạp (khói mỏng, khói dày, lửa nhỏ, lửa bếp, tia lửa) nên nhãn nháp xuất ra rất sát thực tế.

---

### 🚀 Giải Pháp 3: Kết Hợp Point Prompts + Box Prompts Cho SAM

Hiện tại script mới chỉ truyền Bounding Box vào SAM. Khi Bounding Box hơi rộng, SAM dễ cắt nhầm cả phần nền xung quanh.

* **Cải tiến:** Truyền đồng thời cả **Positive Points** (điểm chắc chắn là Lửa/Khói) và **Negative Points** (điểm chắc chắn là Nền):
  $$\text{Prompts} = \{\text{Box: } [x_1, y_1, x_2, y_2], \text{Positive Points: } (x_c, y_c), \text{Negative Points: } (x_{bg}, y_{bg})\}$$
* **Kết quả:** Ép SAM loại bỏ hoàn toàn vùng nền bị bao trùng trong Bounding Box, chỉ giữ lại phần nhân Lửa và thảm Khói.

---

### 🚀 Giải Pháp 4: Sử Dụng Vision-Language Detector (YOLO-World v2 / OwlViT) Với Prompt Tinh Chỉnh

Sử dụng mô hình Vision-Language tân tiến với tập Prompt tự nhiên phong phú hơn:

* **Prompt Lửa:** `"bright orange flame", "raging fire", "glowing embers", "welding spark"`
* **Prompt Khói:** `"gray smoke plume", "billowing smoke cloud", "thin white smoke"`
* **Cơ chế:** Loại bỏ hoàn toàn sự phụ thuộc vào không gian màu HSV, nhận diện vật thể dựa trên đặc trưng thị giác ngữ nghĩa tiên tiến.

---

### 🚀 Giải Pháp 5: Quy Trình Human-in-the-Loop Với CVAT (Tiêu Chuẩn NCKH Quốc Tế)

Trong tất cả các công trình NCKH về dữ liệu thị giác máy tính:
* Nhãn tự động (Pseudo-labeling) được coi là **Nhãn Nháp (Draft Labels)** đạt 80-85%.
* **Quy trình chuẩn:**
  1. Xuất file `annotations_coco.json` từ script tự động.
  2. Import file JSON và ảnh vào công cụ gán nhãn mã nguồn mở **CVAT (Computer Vision Annotation Tool)** hoặc **LabelMe**.
  3. Người dùng chỉ cần dùng chuột tinh chỉnh lại các điểm Polygon bị méo/lem trong vài giây (tiết kiệm 90% thời gian so với vẽ từ đầu).

---

## 📊 Bảng So Sánh Các Phương Án Cải Thiện

| Phương Án | Độ Chính Xác Mask | Thời Gian Cài Đặt | Yêu Cầu Phần Cứng |
|:---|:---:|:---:|:---|
| **Dùng Pretrained SegFormer/YOLO-Seg** | 🌟🌟🌟🌟🌟 (Rất cao) | ⚡ Rất nhanh (Chỉ tải weights) | GPU / CPU cơ bản |
| **Tích hợp SAM 2 (Video Tracking)** | 🌟🌟🌟🌟🌟 (Rất cao) | ⏳ Trung bình (Cài SAM 2) | GPU (Cần VRAM $\ge 6\text{GB}$) |
| **SAM Point + Box Prompts** | 🌟🌟🌟🌟 (Tốt) | ⚡ Nhanh (Sửa code Python) | CPU / GPU |
| **Human-in-the-Loop (CVAT)** | 🌟🌟🌟🌟🌟 (100% Chuẩn) | ⏳ Tinh chỉnh thủ công | Trình duyệt Web |

---

## 🎯 Khuyên Dùng Hướng Đi Tiếp Theo:

👉 **Bước 1:** Áp dụng **Giải pháp 2 (Dùng Pretrained Fire/Smoke Checkpoint)** hoặc **Giải pháp 3 (Thêm Point Prompts cho SAM)** để tạo bộ nhãn nháp tự động sắc nét hơn.  
👉 **Bước 2:** Đưa file `annotations_coco.json` vào CVAT để kiểm tra và tinh chỉnh nhanh.
