# Kế Hoạch Triển Khai Toàn Bộ Mô Hình — 3 Baseline Spatial + Temporal (NCKH)

> **Phạm vi tài liệu này:** Mô tả chi tiết kiến trúc, chiến lược training và evaluation cho **3 baseline** tương ứng với 3 phương án khối Spatial, cộng với phần Temporal dùng chung TimeSformer. Dữ liệu video hiện chưa có nhãn tracking đầy đủ — tài liệu ghi rõ phần nào có thể thực hiện ngay và phần nào cần chờ.

---

## 0. Tổng Quan Hệ Thống 2 Giai Đoạn

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    HỆ THỐNG PHÁT HIỆN & TRACKING LỬA/KHÓI                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  STAGE 1: SPATIAL DETECTION (Ảnh tĩnh — CÓ THỂ TRAIN NGAY)                 ║
║  ┌──────────────────┐ ┌───────────────────┐ ┌────────────────────────────┐  ║
║  │  Baseline A      │ │  Baseline B        │ │  Baseline C                │  ║
║  │  TimeSformer-Det │ │  Deformable DETR   │ │  SegFormer-B2              │  ║
║  │  (Patch-level)   │ │  (Query-based Det) │ │  (Segmentation)            │  ║
║  └──────────────────┘ └───────────────────┘ └────────────────────────────┘  ║
║           ↓ Transfer spatial weights (Backbone shared hoặc independent)      ║
║                                                                              ║
║  STAGE 2: TEMPORAL DETECTION & TRACKING (Video — CẦN THÊM DATA+NHÃN)       ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │  TimeSformer Backbone (Divided Space-Time Attention)                  │    ║
║  │  + Detection Head (từ Baseline thắng Stage 1)                         │    ║
║  │  + SORT/ByteTrack Tracker                                              │    ║
║  │  + State Engine (Rule-based Decision: Normal/Controlled/Hazard)       │    ║
║  └──────────────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 1. Dữ Liệu Hiện Có — Mapping Tới Từng Stage

| Stage | Dữ liệu | Số lượng | Nhãn hiện có | Trạng thái |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1 (Ảnh)** | `Dataset_Balanced_NoEmpty` | 11,437 ảnh | YOLO BBox (`fire`=0, `smoke`=1) | ✅ **SẴN SÀNG TRAIN** |
| **Stage 2 (Video)** | `Data/Video` | ~978 clips | Scene label (3 class) — **CHƯA có BBox/Track ID** | ❌ **CẦN ANNOTATE THÊM** |

> [!IMPORTANT]
> Toàn bộ Stage 1 với 3 Baseline **có thể thực hiện ngay** trên dữ liệu ảnh hiện có.  
> Stage 2 sẽ sử dụng **pseudo-labeling** (dùng model Stage 1 tốt nhất tự sinh BBox) + **SAM 2** (tự động propagate Track ID qua frames) để giảm thiểu công annotate thủ công.

---

## 2. BASELINE A — TimeSformer Patch-level Detection (Hiện Tại)

> **Bản chất:** Mỗi trong số 196 patches (14×14 grid) dự đoán độc lập liệu patch đó có chứa lửa/khói không, kèm tọa độ bbox thô. Đây là kiến trúc **hiện đang có trong code** (`train_spatial.py`).

### 2.1 Kiến Trúc Chi Tiết

```
Input Ảnh: [B, 3, 224, 224]
     │
     ▼  Patch Embedding (Conv2D 16×16 → Linear)
[B, 1, 196, 256]  ← T=1 (ảnh tĩnh)
     │  + CLS token → [B, 1, 197, 256]
     │  + pos_embed
     ▼
DividedSpaceTimeBlock × 6
(Temporal attention vô nghĩa vì T=1 → chỉ Spatial Attention hoạt động)
     │
     ▼  Lấy 196 patch tokens (bỏ CLS)
[B, 196, 256]
     │
     ▼  Detection Head: LayerNorm → Linear(256→6)
[B, 196, 6]  →  Sigmoid  →  [B, 196, (P_fire, P_smoke, x, y, w, h)]
```

### 2.2 Hạn Chế Cần Cải Thiện

| Vấn đề | Hậu quả | Giải pháp (để ghi vào báo cáo) |
| :--- | :--- | :--- |
| Mỗi patch dự đoán BBox **tuyệt đối** chứ không tương đối so với tâm cell | Bbox không chính xác vì patch 16×16 không phải tâm tự nhiên | Dùng YOLO-style offset prediction: $(t_x, t_y)$ là offset từ tâm cell |
| Không có **Anchor** → mô hình khó học w, h vì range quá rộng | Loss bbox không ổn định khi train | Thêm 3 anchor boxes theo kích thước thống kê dataset |
| Không có **NMS** → sinh nhiều bbox trùng lấp nhau | Output messy, không thể dùng cho tracking | Thêm NMS post-processing |
| Không có **Confidence threshold** | False positive cao | Thêm objectness score riêng biệt |

### 2.3 Baseline A Cải Tiến (A+)

Đây là phiên bản nâng cấp Baseline A theo kiểu YOLO-on-patches, không thay đổi backbone:

```python
# Detection Head cải tiến — YOLO-style trên lưới 14×14
class YOLOPatchHead(nn.Module):
    """
    Giữ nguyên TimeSformer backbone.
    Thay detection head:  Linear(256→6)  →  ConvHead trên spatial grid
    A = 3 anchors, C = 2 classes (fire, smoke)
    Output per cell: A × (1_obj + 4_bbox + C_cls) = 3 × 7 = 21
    """
    def __init__(self, dim=256, num_anchors=3, num_classes=2):
        super().__init__()
        out_ch = num_anchors * (5 + num_classes)  # 3 × 7 = 21
        self.conv1 = nn.Conv2d(dim, 128, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(128)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, out_ch, 1)   # 1×1 conv per cell

    def forward(self, patch_tokens):
        # patch_tokens: [B, 196, 256]
        B, N, D = patch_tokens.shape
        x = patch_tokens.view(B, 14, 14, D).permute(0, 3, 1, 2)  # [B, 256, 14, 14]
        x = self.relu(self.bn1(self.conv1(x)))
        return self.conv2(x)  # [B, 21, 14, 14]

# Anchors thống kê từ K-means trên Dataset_Balanced_NoEmpty
ANCHORS = [(0.05, 0.10), (0.15, 0.25), (0.40, 0.55)]  # (w%, h%) tương đối

# Loss: YOLOv3-style multi-task loss
def yolo_loss(pred, targets, anchors):
    # pred: [B, 21, 14, 14]
    # targets: list of [class_id, cx, cy, w, h] per ảnh (YOLO format)
    # Returns: total_loss = obj_loss + noobj_loss + bbox_loss + cls_loss
    pass
```

### 2.4 Training Plan — Baseline A+

| Tham số | Giá trị | Lý do |
| :--- | :--- | :--- |
| **Optimizer** | AdamW, lr=1e-4 | Ổn định hơn Adam với weight decay |
| **LR Scheduler** | CosineAnnealingLR (T_max=80) | Tránh overshoot |
| **Batch Size** | 32 (A10G) | Fit VRAM ~24GB |
| **Epochs** | 100 + Early Stopping (patience=10) | Tiêu chuẩn cho Vision Transformer |
| **Loss** | YOLO multi-task loss | obj + bbox + cls |
| **Data Augment** | H-Flip, ColorJitter, RandomCrop(0.85-1.0), GaussianBlur | Không dùng V-Flip (lửa lên trên) |
| **Input Size** | 224×224 | Chuẩn ViT |
| **GPU** | Modal A10G | ~4h/run |

---

## 3. BASELINE B — Deformable DETR Detection

> **Bản chất:** Coi detection là bài toán **Direct Set Prediction** — Transformer Encoder mã hóa patch features → Transformer Decoder với N Object Queries dự đoán N bounding box trực tiếp. Loại bỏ hoàn toàn Anchor và NMS.

### 3.1 Kiến Trúc Chi Tiết

```
Input Ảnh: [B, 3, 224, 224]
     │
     ▼  ENCODER: TimeSformer Backbone (6 blocks)
[B, 196, 256]  ← Spatial patch features (dùng làm Memory cho Decoder)
     │
     │  ─────────── DECODER (DETR-style) ────────────
     │
     ▼  N Object Queries (N=20, learnable)
[B, N, 256]  →  Cross-Attention với Memory [B, 196, 256]
     │                                           ↑
     │  Self-Attention giữa các queries         Memory từ Encoder
     │
     ▼  FFN Decoder Heads (2 nhánh song song):
     │
     ├─► Class Head: Linear(256→3) → Softmax
     │   Output: [B, N, 3]  (fire / smoke / no-object)
     │
     └─► Box Head: MLP(256→256→256→4) → Sigmoid
         Output: [B, N, 4]  (cx, cy, w, h) normalized 0–1
```

### 3.2 Vì Sao Deformable Attention Tốt Hơn Standard DETR?

Standard DETR: Mỗi query attend tới **tất cả 196 patches** → độ phức tạp O(N×196) → chậm, khó hội tụ (cần 500+ epochs theo bài báo gốc).

**Deformable Attention:** Mỗi query chỉ attend tới **K=4 key points** được học tự động trên feature map → hội tụ nhanh hơn ~10×, phù hợp với vật thể nhỏ (đốm lửa sớm, vệt khói mỏng).

```python
class DeformableAttention(nn.Module):
    """
    Mỗi query dự đoán K reference points và sampling offsets.
    Chỉ compute attention tại K=4 điểm thay vì toàn bộ 196 patches.
    """
    def __init__(self, d_model=256, n_heads=8, n_points=4):
        super().__init__()
        self.n_heads   = n_heads
        self.n_points  = n_points
        # Dự đoán offset từ reference point
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj  = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, input_flatten):
        # query:           [B, N_q, 256]
        # reference_points:[B, N_q, 2]   (cx, cy — normalized 0-1)
        # input_flatten:   [B, 196, 256] (memory từ encoder)
        B, N_q, _ = query.shape
        offsets  = self.sampling_offsets(query).view(B, N_q, self.n_heads, self.n_points, 2)
        weights  = self.attention_weights(query).view(B, N_q, self.n_heads, self.n_points)
        weights  = weights.softmax(dim=-1)
        # Bilinear sampling tại reference_points + offsets
        # ...  (dùng grid_sample của PyTorch)
        return self.output_proj(attended)

class DETRDecoder(nn.Module):
    def __init__(self, num_queries=20, d_model=256, n_heads=8, n_layers=6):
        super().__init__()
        self.queries = nn.Embedding(num_queries, d_model)
        self.layers  = nn.ModuleList([
            nn.ModuleDict({
                'self_attn':  nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'cross_attn': DeformableAttention(d_model, n_heads),
                'ffn':        Mlp(d_model, d_model * 4),
            }) for _ in range(n_layers)
        ])
        # 2 nhánh output
        self.cls_head = nn.Linear(d_model, 3)         # fire / smoke / no-obj
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 4), nn.Sigmoid()        # cx, cy, w, h
        )
```

### 3.3 Loss Function — Hungarian Matching (Bipartite Matching)

```
Quy trình tính loss mỗi batch:
1. Model dự đoán N=20 boxes (bao gồm cả "no-object")
2. Ground truth có M box thực (M << N, thường M ≤ 5)
3. Dùng Hungarian Algorithm tìm cặp ghép tối ưu:
   - Cost = λ_cls × CrossEntropy + λ_bbox × L1 + λ_giou × GIoU
4. Chỉ tính loss cho N matched pairs (không phạt no-object quá mức)

Tham số: λ_cls = 2, λ_bbox = 5, λ_giou = 2
```

### 3.4 Training Plan — Baseline B

| Tham số | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| **Num Queries** | N=20 | Ảnh fire hiếm khi có >5 region, 20 đủ thừa |
| **Backbone LR** | 1e-5 | Nhỏ hơn — pretrained spatial features cần giữ |
| **Decoder LR** | 1e-4 | Decoder mới hoàn toàn — học nhanh hơn |
| **Loss** | Hungarian Matching (L1 + GIoU + CrossEntropy) | Chuẩn DETR |
| **Epochs** | 150 (DETR hội tụ chậm hơn YOLO) | Deformable variant hội tụ ~50 epochs |
| **Batch Size** | 16 | Decoder nặng hơn YOLO head |
| **GPU** | Modal A10G | ~6-8h/run |

---

## 4. BASELINE C — SegFormer Segmentation

> **Bản chất:** Thay vì dự đoán hình chữ nhật bao quanh (Bounding Box), bài toán được chuyển sang **Semantic Segmentation** — gán nhãn từng pixel thuộc lửa (class 0), khói (class 1), hoặc background (class 2). BBox sau đó được extract từ mask.

### 4.1 Lý Do Chọn SegFormer Cho Lửa/Khói

Ngọn lửa và đám khói là **Amorphous Objects** — không có hình dạng cố định, ranh giới không rõ ràng. BBox thường bao bọc 30-50% background không liên quan → BBox loss noise cao → detection accuracy thấp hơn mask-based methods.

### 4.2 Kiến Trúc SegFormer-B2

```
Input: [B, 3, 512, 512]  (SegFormer dùng 512px, không phải 224px)
     │
     ▼  Mix Transformer Encoder (MiT-B2) — 4 stages
     │
     ├─ Stage 1: [B, 64,  128, 128]   (stride 4,  patches 4×4)
     ├─ Stage 2: [B, 128,  64,  64]   (stride 8,  patches 2×2)
     ├─ Stage 3: [B, 320,  32,  32]   (stride 16)
     └─ Stage 4: [B, 512,  16,  16]   (stride 32)
               │
     ▼  ALL-MLP Decoder (nhẹ, không dùng FPN phức tạp)
     │  1. Linear(C_i → 256) cho mỗi stage
     │  2. Upsample về 128×128
     │  3. Concat → Linear(4×256 → 256)
     │  4. Dropout → Linear(256 → num_classes=3)
     │
     ▼  Output: [B, 3, 128, 128] → Upsample → [B, 3, 512, 512]
     │  Argmax → Segmentation Mask
     │
     ▼  Post-process: Mask → Bounding Box + Contour
     │  [Connected Component Analysis → BBox per fire/smoke region]
```

### 4.3 Chuyển Đổi Nhãn YOLO → Segmentation Mask

Đây là bước bắt buộc vì dataset hiện tại chỉ có BBox format, chưa có mask:

```python
def yolo_bbox_to_mask(image_h, image_w, yolo_labels):
    """
    Chuyển YOLO BBox sang Segmentation Mask đơn giản.
    Hạn chế: Mask là hình chữ nhật thẳng (không phải polygon thực sự).
    Giải pháp tốt hơn: Dùng SAM 2 để refine mask từ BBox prompt.
    
    Args:
        yolo_labels: list of [class_id, cx, cy, w, h] (normalized)
    Returns:
        mask: [H, W] với value 0=background, 1=fire, 2=smoke
    """
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    for cls_id, cx, cy, w, h in yolo_labels:
        x1 = int((cx - w/2) * image_w)
        y1 = int((cy - h/2) * image_h)
        x2 = int((cx + w/2) * image_w)
        y2 = int((cy + h/2) * image_h)
        mask[y1:y2, x1:x2] = int(cls_id) + 1  # 1=fire, 2=smoke
    return mask
```

> **Nâng cấp bằng SAM 2:** Thay vì dùng rectangle mask (trên), dùng SAM 2 với BBox prompt để tạo accurate polygon mask cho từng vùng lửa/khói → chất lượng segmentation cao hơn đáng kể.

### 4.4 Training Plan — Baseline C

| Tham số | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| **Model variant** | SegFormer-B2 (pretrained ImageNet) | B0 nhẹ nhất, B5 nặng nhất. B2 cân bằng tốt |
| **Input Size** | 512×512 | Segmentation cần resolution cao hơn |
| **Loss** | CrossEntropyLoss + DiceLoss (0.5 mỗi loại) | Dice giúp balance khi smoke area nhỏ |
| **Optimizer** | AdamW, backbone_lr=6e-5, decode_lr=6e-4 | Differential LR |
| **Epochs** | 80 | SegFormer hội tụ nhanh |
| **Batch Size** | 8 (vì 512px nặng hơn 224px) | |
| **GPU** | Modal A10G | ~5h/run |
| **Augment** | RandomFlip, PhotoMetricDistortion, RandomCrop(512) | Chuẩn segmentation |

---

## 5. STAGE 2 — Temporal Detection & Tracking (Dùng Chung Cho Cả 3 Baseline)

> Stage 2 sử dụng **backbone của baseline thắng Stage 1** làm spatial encoder, thêm Temporal Attention để học biến đổi không-thời gian, sau đó ghép với tracker và state engine.

### 5.1 Kiến Trúc Stage 2 Tổng Thể

```
Video Clip: [B, 3, T=8, 224, 224]
     │
     ▼  PatchEmbed (Linear 16×16) — DÙNG CHUNG VỚI STAGE 1
[B, T, 196, 256]
     │  + CLS token + pos_embed + time_embed
     ▼
DividedSpaceTimeBlock × 6
  ├─ Temporal Attention: học diễn tiến lửa qua 8 frames
  └─ Spatial Attention:  kế thừa từ Stage 1
     │
     ▼  Output: [B, T, 197, 256]
     │
     ├─── [BRANCH 1] Scene Classification
     │     CLS tokens → mean(T) → Linear(256→3) → Normal/Controlled/Hazard
     │     (Auxiliary branch, dùng cho ablation study)
     │
     └─── [BRANCH 2] Per-frame Detection
           Lấy frame t=T-1 (frame mới nhất để real-time):
           patch_tokens[-1]: [B, 196, 256]
           → Detection Head (từ baseline A/B/C đã train Stage 1)
           → BBox per frame: [[x1,y1,x2,y2, conf, cls_id], ...]
```

### 5.2 Transfer Learning Từ Stage 1 Sang Stage 2

```python
def load_stage1_into_stage2(stage1_weights_path, stage2_model):
    """
    Load spatial weights từ Stage 1 vào Stage 2.
    Temporal_fc được init zeros → temporal attention ban đầu không ảnh hưởng.
    Time_embedding được interpolate từ T=1 → T=8.
    """
    state_dict_1 = torch.load(stage1_weights_path, map_location='cpu')
    state_dict_2 = stage2_model.state_dict()

    transferred = {}
    for k, v in state_dict_1.items():
        if 'temporal_fc' in k:
            # Init zeros — temporal contribution bắt đầu từ 0
            transferred[k] = torch.zeros_like(state_dict_2[k])
        elif 'time_embedding' in k:
            # Interpolate T=1 → T=8
            transferred[k] = v.repeat(1, 8, 1, 1) / 8  # hoặc dùng interpolate
        elif k in state_dict_2 and v.shape == state_dict_2[k].shape:
            transferred[k] = v
    
    stage2_model.load_state_dict(transferred, strict=False)
    return stage2_model
```

### 5.3 Training Stage 2 — Kế Hoạch Khi Data Video Có Sẵn

| Tham số | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| **Frames per clip** | T=8 | Uniform sampling từ toàn clip |
| **Spatial LR** | 1e-5 | Giữ spatial features |
| **Temporal LR** | 1e-4 | Temporal attention học từ đầu |
| **Loss** | Detection loss + 0.3×Classification loss | Multi-task |
| **Clip length** | 16-32s | Đủ thể hiện diễn tiến cháy |
| **Data annotation** | Pseudo-label (Baseline A/B winner) + SAM 2 propagation | Tự động 80%, manual review 20% |
| **GPU** | Modal H100 | ~12-24h/run |

### 5.4 SORT Tracker Integration

```python
class SimpleSORT:
    """
    Lightweight tracker — không cần thư viện ngoài.
    Phù hợp cho real-time CCTV.
    """
    def __init__(self, max_age=15, min_hits=2, iou_threshold=0.3):
        self.tracks = []
        self.next_id = 0
        self.max_age = max_age        # Frames track sống khi không detect được
        self.min_hits = min_hits      # Số frames cần detect trước khi hiện Track ID
        self.iou_threshold = iou_threshold

    def update(self, detections, timestamp):
        # detections: [[x1,y1,x2,y2,conf,cls_id], ...]
        # Returns: active tracks với Track ID, metrics (lifetime, area_history)
        pass

class FireStateEngine:
    """
    Rule-based Decision Engine dựa trên Tracking metrics.
    """
    def decide(self, tracks) -> dict:
        for track in tracks:
            # Trạng thái 2: HAZARD
            if track.area_growth_rate > 0.5:   # Diện tích tăng >50% trong 3s
                return {'level': 2, 'label': '🚨 NGUY HIỂM', 'track_id': track.id}
            if track.velocity > 0.05:           # Lửa đang lan rộng
                return {'level': 2, 'label': '🚨 ĐANG LAN RỘNG', 'track_id': track.id}
            # Trạng thái 1: CONTROLLED
            if track.lifetime_s > 3.0:
                return {'level': 1, 'label': '⚠️ CÓ KIỂM SOÁT', 'track_id': track.id}
        return {'level': 0, 'label': '✅ BÌNH THƯỜNG'}
```

---

## 6. Evaluation Metrics (Chỉ Số Đánh Giá Khoa Học)

### 6.1 Stage 1 — Detection Metrics

| Metric | Công Thức | Mức Mục Tiêu (Tham Khảo) |
| :--- | :--- | :--- |
| **mAP@0.5** | Mean Average Precision tại IoU=0.5 | ≥ 0.75 |
| **mAP@0.5:0.95** | mAP trung bình tại IoU=0.5→0.95 | ≥ 0.55 |
| **Recall@50** | Tỷ lệ phát hiện đúng ở threshold thấp | ≥ 0.80 (ưu tiên không bỏ sót cháy) |
| **FPS** | Frames Per Second inference | ≥ 15 FPS (real-time) |
| **Params (M)** | Số lượng tham số mô hình | Càng nhỏ càng tốt (Edge deployment) |

### 6.2 Stage 2 — Tracking Metrics (MOT Challenge Standard)

| Metric | Ý Nghĩa | Mức Mục Tiêu |
| :--- | :--- | :--- |
| **MOTA** | Multiple Object Tracking Accuracy | ≥ 0.60 |
| **MOTP** | Multiple Object Tracking Precision | ≥ 0.75 |
| **IDF1** | ID F1 Score (track identity preservation) | ≥ 0.65 |
| **ID Sw.** | Số lần mô hình nhầm lẫn identity track | Càng thấp càng tốt |

---

## 7. Ablation Study Plan (Cho Bài Báo NCKH)

```
Thí nghiệm 1: So sánh 3 Baseline Spatial
  A: TimeSformer-Det (Patch-level YOLO) 
  B: Deformable DETR
  C: SegFormer-B2
  → So sánh: mAP@0.5, mAP@0.5:0.95, FPS, Params

Thí nghiệm 2: Stage 1 → Stage 2 Transfer
  2a: Train Stage 2 từ đầu (no pretrain)
  2b: Transfer từ Stage 1 tốt nhất
  → Kiểm chứng: 2-stage strategy có thực sự tốt hơn không?

Thí nghiệm 3: Real Data vs. Real + Diffusion Synthetic
  3a: Train chỉ bằng Real Data (11,437 ảnh)
  3b: Train với Real + Diffusion Synthetic (+3,000 ảnh sinh từ SD Inpainting)
  → So sánh mAP — đây là đóng góp khoa học của phần Diffusion

Thí nghiệm 4: State Engine Rules Evaluation
  → So sánh Alert Accuracy giữa:
     4a: Scene Classification (3 labels cứng)
     4b: State Engine dựa trên Tracking metrics (đề xuất mới)
```

---

## 8. Thứ Tự Thực Hiện & Timeline

```
TUẦN 1–2:  Chuẩn bị dữ liệu Stage 1
  - Audit Dataset_Balanced_NoEmpty
  - K-means clustering tìm Anchor Boxes tối ưu
  - Viết data pipeline (generator) cho A/B/C

TUẦN 3–4:  Train & Eval Stage 1
  - Train Baseline A+ (TimeSformer-YOLO Patch)
  - Train Baseline B  (Deformable DETR)
  - Train Baseline C  (SegFormer-B2)
  - So sánh mAP, chọn baseline tốt nhất

TUẦN 5–6:  Sinh Dữ Liệu Video (Song Song Với Trên)
  - Dùng baseline A/B/C tốt nhất pseudo-label 978 video clips
  - SAM 2 propagate Track ID
  - Manual review 10–20% kết quả

TUẦN 7–8:  Train Stage 2
  - Transfer spatial weights → TimeSformer Full
  - Train temporal trên video clips đã có nhãn
  - Tích hợp SORT Tracker + State Engine

TUẦN 9:   Ablation Study & Viết Báo Cáo
  - Chạy 4 thí nghiệm so sánh
  - Vẽ bảng kết quả, biểu đồ
  - Viết phần Experiments trong báo cáo

TUẦN 10:  Demo Real-time CCTV
  - Test inference_realtime.py với camera thực
  - Đo FPS, latency, alert accuracy
```

> [!NOTE]
> Tuần 5–6 (sinh dữ liệu video) được thực hiện **song song** với Tuần 3–4 (train Stage 1) để tối ưu thời gian. Khi Stage 1 train xong thì data video cũng gần sẵn sàng để train Stage 2.
