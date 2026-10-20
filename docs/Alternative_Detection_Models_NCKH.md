# Đề Xuất Các Mô Hình Thay Thế YOLO Cho Đề Tài NCKH Phát Hiện & Định Vị Lửa/Khói

> **Mục tiêu NCKH:** Không phụ thuộc vào các framework YOLO thương mại/có sẵn, hướng tới việc xây dựng hoặc tùy biến các kiến trúc học sâu tiên tiến (Vision Transformer, DETR, Attention-based Localization, Segmentation) để tăng **tính mới (Novelty)** và **giá trị đóng góp khoa học (Scientific Contribution)** cho đề tài.

---

## 1. So sánh tổng quan các phương án đề xuất

| Mô hình / Phương pháp | Loại bài toán | Bản chất kiến trúc | Điểm mới khoa học (Novelty) | Khả năng tích hợp với TimeSformer |
| :--- | :--- | :--- | :--- | :--- |
| **1. Attention Rollout / Spatial CAM** | Weakly-Supervised Localization | Trích xuất Heatmap trực tiếp từ Self-Attention Map | ⭐⭐⭐⭐⭐ (Không cần head phụ, dùng trực tiếp Attention sẵn có) | **100% Tích hợp sẵn** |
| **2. Deformable DETR (Custom)** | End-to-End Object Detection | Transformer Encoder-Decoder + Multi-scale Deformable Attention | ⭐⭐⭐⭐ (Thay thế hoàn toàn Anchor/NMS của CNN) | **Rất cao** (Cùng hệ sinh thái Transformer) |
| **3. Mask2Former / SegFormer** | Semantic / Panoptic Segmentation | ViT Backbone + Masked Attention Decoder | ⭐⭐⭐⭐⭐ (Định vị theo viền thực tế của khói/lửa thay vì hình chữ nhật) | **Cao** |
| **4. Grounding DINO / OwLV2 (Lightweight)** | Zero-Shot / Open-Vocabulary Detection | Cross-Modal Vision-Language Transformer | ⭐⭐⭐⭐ (Định vị qua Prompt text "fire", "smoke") | **Trung bình** |

---

## 2. Chi tiết từng phương án đề xuất

---

### Phương án 1: Trích xuất Bounding Box / Heatmap từ Attention Map của TimeSformer (Khuyên dùng cho NCKH)

> **Ý tưởng NCKH:** Vì bạn đã chọn **TimeSformer** làm mô hình cốt lõi, thay vì gắn thêm 1 mạng Detection bên ngoài (như YOLO/Faster R-CNN), bạn có thể khai thác trực tiếp **Spatial Attention Weights** của các Transformer Blocks để định vị ngọn lửa và khói (Weakly-Supervised Object Localization - WSOL).

```
Input Frame (224×224)
        │
        ▼
TimeSformer Spatial Attention Maps (Tầng cuối/Tất cả tầng)
        │
        ▼
Attention Rollout / Grad-CAM cho Vision Transformer
        │
        ▼
Bản đồ nhiệt (Heatmap) thể hiện vùng Lửa / Khói
        │
        ▼
Thresholding + Contour Detection (OpenCV) ──► Bounding Boxes (Tự động)
```

#### Ưu điểm khoa học:
1. **Tính mới cao (Novelty):** Đề xuất giải pháp *End-to-End Spatiotemporal Classification & Localization* dựa trên 1 kiến trúc Transformer duy nhất.
2. **Không cần Anchor Boxes hay NMS:** Loại bỏ các kỹ thuật thủ công của CNN cổ điển.
3. **Tối ưu tính toán:** Không tốn thêm GPU VRAM để chạy một mạng Detection riêng biệt.

#### Phương pháp cài đặt (Code Snippet minh họa):
```python
import torch
import numpy as np
import cv2

def extract_attention_heatmap(model, video_tensor, layer_idx=-1):
    """
    Trích xuất Attention Map từ Spatial Attention của TimeSformer Block cuối cùng
    video_tensor: [1, 3, T, H, W]
    """
    model.eval()
    # Lấy attention weights của Spatial Attention trong Block cuối
    # Shape of attn_weights: [B*T, num_heads, N_patches+1, N_patches+1]
    attn_weights = model.layers[layer_idx].spatial_attn_weights
    
    # Lấy trung bình qua các heads: [B*T, N+1, N+1]
    attn_mean = attn_weights.mean(dim=1)
    
    # Lấy attention từ CLS token tới các patch tokens (bỏ CLS token ở index 0)
    cls_attn = attn_mean[:, 0, 1:] # [B*T, 196]
    
    # Reshape về dạng 14x14 spatial grid
    T = video_tensor.shape[2]
    heatmaps = cls_attn.view(T, 14, 14).cpu().numpy()
    
    bboxes_per_frame = []
    for t in range(T):
        hp = heatmaps[t]
        # Normalize về 0-255
        hp_norm = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hp_resized = cv2.resize(hp_norm, (224, 224))
        
        # Ngưỡng hóa (Thresholding) lấy vùng chú ý cao nhất
        _, thresh = cv2.threshold(hp_resized, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        frame_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 100: # Lọc nhiễu nhỏ
                frame_boxes.append([x, y, x+w, y+h])
        bboxes_per_frame.append(frame_boxes)
        
    return bboxes_per_frame
```

---

### Phương án 2: Deformable DETR (Detection Transformer)

> **Ý tưởng NCKH:** DETR (DEtection TRansformer) của Facebook AI là mô hình thay thế hoàn toàn paradigm của YOLO. DETR coi bài toán Detection là bài toán **Direct Set Prediction** sử dụng Transformer Encoder-Decoder.

```
Input Image ──► CNN / ViT Backbone ──► Deformable Transformer Encoder ──► Transformer Decoder (Object Queries) ──► Bounding Boxes
```

#### Ưu điểm khoa học:
1. **Không có NMS, không có Anchor:** Sử dụng thuật toán **Bipartite Matching (Hungarian Loss)** để nối trực tiếp dự đoán với Ground Truth.
2. **Hệ sinh thái Transformer thống nhất:** Cả module Classification (TimeSformer) và Detection (DETR) đều dựa trên tự chú ý (Self-Attention).
3. **Hiệu quả cao với đám cháy nhỏ/khói mờ:** Deformable Attention tập trung vào các điểm mấu chốt (Key points) thay vì quét toàn bộ ảnh.

#### Cấu trúc Head tùy chỉnh cho bài toán Lửa/Khói:
- **Backbone:** ResNet-50 hoặc ViT-Tiny.
- **Decoder Queries:** Đặt $N=20$ Object Queries (vì trong 1 frame hiếm khi có quá 20 vùng cháy riêng biệt).
- **Class Head:** 2 classes (Lửa, Khói).
- **BBox Head:** MLP 3 tầng dự đoán thẳng `(x_center, y_center, width, height)`.

---

### Phương án 3: Segmentation-based Detection (Mask2Former / SegFormer)

> **Ý tưởng NCKH:** Lửa và khói là các đối tượng **không có hình dạng cố định (Amorphous Objects)**. Việc dùng khung hình chữ nhật (Bounding Box) thường bao bọc cả nhiều vùng nền (Background). Chuyển bài toán từ Detection sang **Semantic/Instance Segmentation** là một hướng tiếp cận có giá trị khoa học rất cao.

```
Khuyết điểm của Bounding Box:            Ưu điểm của Segmentation Mask:
┌─────────────────────────┐               ░░░░░░██████░░░░░░
│  Background   ░░░███░░  │               ░░░████████████░░░
│  ░░░████████████████░░  │  ───────►     ░████████████████░
│  ░░░░░░██████████░░░░░  │               ░░░░░██████████░░░
│  Background   ░░░░░░░░  │               ░░░░░░░░░░░░░░░░░░
└─────────────────────────┘               (Chỉ lấy đúng vùng lửa/khói)
```

#### Các mô hình khuyến nghị:
1. **SegFormer (Nvidia):** Kiến trúc ViT siêu nhẹ cho Semantic Segmentation, không dùng Positional Encoding (dùng Mix-FFN), rất thích hợp chạy Real-time.
2. **Mask2Former:** Kiến trúc SOTA về Mask Classification, sử dụng Masked Attention để trích xuất chính xác ranh giới cuộn của khói và ngọn lửa.

#### Chuyển đổi từ Mask sang Bounding Box khi cần Tracking:
Từ Mask phân đoạn, có thể dễ dàng tính toán bounding box bao quanh:
```python
# Từ Binary Mask của Lửa/Khói -> Bounding Box
y_indices, x_indices = np.where(mask > 0)
x_min, x_max = np.min(x_indices), np.max(x_indices)
y_min, y_max = np.min(y_indices), np.max(y_indices)
bbox = [x_min, y_min, x_max, y_max]
```

---

### Phương án 4: Self-Supervised / Open-Vocabulary Grounding (OwLV2 / Grounding DINO)

> **Ý tưởng NCKH:** Sử dụng mô hình Vision-Language (VLM) để định vị đối tượng thông qua câu lệnh văn bản (Text Prompt) mà không cần train lại anchor.

```
Input Image + Text Prompt ("fire flame", "smoke plume") ──► Swin/ViT Backbone + Text Encoder ──► Cross-Attention ──► Bounding Boxes
```

#### Ưu điểm khoa học:
- Khả năng **Zero-Shot / Few-Shot Generalization**: Nhận diện được cả các dạng khói đặc biệt (khói hóa chất, khói màu) mà mô hình truyền thống chưa từng học qua nhờ vào tri thức từ vựng của Text Encoder.

---

## 3. Đánh giá & Khuyên dùng cho Báo cáo/Luận văn NCKH

| Tiêu chí | Phương án 1 (Attention Rollout) | Phương án 2 (Deformable DETR) | Phương án 3 (SegFormer/Mask) |
| :--- | :--- | :--- | :--- |
| **Đóng góp khoa học (Novelty)** | 🟢 **Rất cao** (Tự thiết kế pipeline trích xuất từ ViT) | 🟢 **Cao** (Áp dụng Transformer Detection) | 🟢 **Rất cao** (Giải quyết bài toán ranh giới linh hoạt) |
| **Độ phức tạp cài đặt** | 🟢 **Thấp** (Chỉ viết thêm module post-processing) | 🟡 **Trung bình** (Train thêm 1 mạng DETR) | 🟡 **Trung bình** (Cần chuyển label bbox sang mask nếu muốn train mask) |
| **Tốc độ Inference (FPS)** | 🟢 **Siêu nhanh** (0ms extra cost) | 🟡 **Trung bình** (20-30 FPS) | 🟢 **Nhanh** (với SegFormer-B0) |
| **Tính ứng dụng NCKH** | Rất phù hợp đăng bài báo về *Efficient Transformer* | Phù hợp bài báo về *End-to-End Transformer Detection* | Phù hợp bài báo về *Segmentation & Early Warning* |

### 🎯 Đề xuất lộ trình triển khai cho Đề tài của bạn:

1. **Hướng tiếp cận 1 (Ưu tiên số 1 cho NCKH):**  
   Sử dụng **Attention Rollout / Grad-CAM trực tiếp trên TimeSformer**.  
   *Lý do:* Bạn không cần cài thêm mô hình bên ngoài như YOLO hay DETR. Bạn chứng minh được rằng **mô hình TimeSformer của bạn vừa có khả năng phân loại video (Classification) vừa có khả năng chỉ ra vị trí cháy (Localization)** thông qua cơ chế Attention Maps. Điều này tạo nên đóng góp khoa học rõ ràng cho đề tài.

2. **Hướng tiếp cận 2 (Nếu muốn có Bounding Box chuẩn xác hơn):**  
   Xây dựng mô hình **Deformable DETR (hoặc DETR nhỏ)** thuần Transformer để huấn luyện trên tập `Dataset_Balanced_NoEmpty`. Việc này khẳng định bài báo/đề tài của bạn hoàn toàn sử dụng **công nghệ Transformer tiên tiến**, loại bỏ hoàn toàn các framework CNN truyền thống như YOLO.
