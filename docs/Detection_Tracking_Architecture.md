# Nâng cấp sang Detection + Tracking — Hướng dẫn Kỹ thuật Chi tiết

> **Mục tiêu:** Thêm khả năng **định vị (bounding box) và theo dõi (tracking) ngọn lửa/khói** qua các frame liên tiếp để hệ thống có thể đưa ra cảnh báo thông minh (nguy hiểm/an toàn) dựa vào hành vi của đám cháy theo thời gian — không chỉ dựa vào 1 frame đơn lẻ.

---

## Mục lục

1. [Tại sao cần Detection + Tracking?](#1-tại-sao-cần-detection--tracking)
2. [Kiến trúc đề xuất: TimeSformer + Detection Head + Tracker](#2-kiến-trúc-đề-xuất)
3. [Thay đổi Data Pipeline](#3-thay-đổi-data-pipeline)
4. [Logic cảnh báo dựa trên Tracking](#4-logic-cảnh-báo-dựa-trên-tracking)
5. [Thay đổi code cụ thể](#5-thay-đổi-code-cụ-thể)
6. [So sánh các phương án Tracker](#6-so-sánh-các-phương-án-tracker)
7. [Khó khăn kỹ thuật và giải pháp](#7-khó-khăn-kỹ-thuật-và-giải-pháp)
8. [Lộ trình thực hiện](#8-lộ-trình-thực-hiện)

---

## 1. Tại sao cần Detection + Tracking?

### Vấn đề của Classification thuần túy

```
Hệ thống hiện tại:
  Frame 1: "fire detected" (prob=0.72) → ⚠️ ALERT
  Frame 2: "normal" (prob=0.48) → ✅ OK
  Frame 3: "fire detected" (prob=0.61) → ⚠️ ALERT

Câu hỏi quan trọng mà classification KHÔNG TRẢ LỜI ĐƯỢC:
  ❓ Lửa ở đâu trong frame? → Cần bounding box
  ❓ Đây có phải cùng 1 đám lửa không? → Cần tracking
  ❓ Đám lửa đang lớn lên hay nhỏ đi? → Cần theo dõi diện tích
  ❓ Lửa đang di chuyển sang hướng nào? → Cần velocity tracking
  ❓ Đã cháy bao lâu rồi? → Cần thời gian tồn tại (persistence)
```

### Kịch bản cảnh báo thông minh cần Tracking

| Kịch bản | Classification | Detection+Tracking |
|----------|---------------|-------------------|
| Ngọn lửa nhỏ, đứng yên, < 3s | ⚠️ Alert ngay | ✅ Theo dõi thêm — có thể bình thường (bếp gas) |
| Ngọn lửa tăng diện tích 200% trong 5s | ✅ Alert ngay | 🔴 Alert cao nhất — đang lan rộng! |
| Khói xuất hiện rồi biến mất sau 2s | ⚠️ Alert | ✅ Giảm độ ưu tiên — nhiễu |
| Lửa lan từ vị trí A sang B trong 10s | ❌ Không biết | 🔴 Alert — đang lan rộng |
| Lửa trong vùng an toàn (bếp nấu ăn) | ⚠️ Alert nhầm | ✅ Có thể filter theo vùng |

---

## 2. Kiến trúc đề xuất

### 2.1 Tổng quan: TimeSformer với Dual Head

```
Input Video Clip:  [B, C, T, H, W]  (B=batch, C=3, T=8, H=W=224)
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│                  TimeSformer Backbone                       │
│                                                            │
│  PatchEmbed → [B, T, 196+1, dim]                          │
│       │                                                    │
│  × 6 DividedSpaceTimeBlock                                 │
│  (temporal attention + spatial attention)                  │
│       │                                                    │
│  Output: [B, T, 197, 256]  (197 = 1 CLS + 196 patches)   │
└────────────────────────────────────────────────────────────┘
         │
    ┌────┴───────────┐
    │                │
    ▼                ▼
┌──────────┐   ┌──────────────────────────────────┐
│ CLS HEAD │   │       DETECTION HEAD              │
│          │   │                                  │
│ mean(CLS │   │  Patch features: [B, T, 196, 256]│
│ over T)  │   │       │                          │
│    │     │   │  → mean over T: [B, 196, 256]   │
│    ▼     │   │       │                          │
│ Linear   │   │  → reshape: [B, 14, 14, 256]    │
│ (256→3)  │   │       │                          │
│    │     │   │  → Conv head: [B, A*(5+C), 14, 14]│
│    ▼     │   │       │                          │
│ 3-class  │   │  YOLO-style per-cell prediction: │
│ classify │   │  A anchors × (x,y,w,h,obj + 2cls)│
└──────────┘   └──────────────────────────────────┘
    │                │
    ▼                ▼
Temporal          Bounding Boxes
Classification    [fire/smoke, x, y, w, h, confidence]
(scene label)          │
                       ▼
                  ┌──────────┐
                  │  TRACKER │  (ByteTrack / SORT)
                  │          │
                  │ Track ID │
                  │ Lifetime │
                  │ Velocity │
                  │ Area hist│
                  └──────────┘
                       │
                       ▼
                ┌────────────┐
                │  DECISION  │
                │  ENGINE    │
                │            │
                │ Classify + │
                │ Tracking   │
                │ Metrics    │
                └────────────┘
                       │
                  ALERT LEVEL
                  (0/1/2/3)
```

### 2.2 Detection Head Chi tiết

TimeSformer không cần detector riêng — **patch features từ attention đã mang thông tin không gian** (mỗi patch tương ứng 1 ô 16×16 pixel). Ta có thể tái sử dụng chúng:

```
Transformer output: [B, T, 197, 256]
                          │
         Lấy patch tokens (bỏ CLS): [B, T, 196, 256]
                          │
         Average over T (temporal aggregation): [B, 196, 256]
                          │
         Reshape thành spatial grid: [B, 256, 14, 14]
         (196 = 14×14 patches)
                          │
    ┌─────────────────────┼──────────────────────┐
    ▼                     ▼                      ▼
Conv2d(256→128)    Conv2d(256→128)        Upsample→FPN
3×3, stride=1      3×3, stride=1          (nếu cần thêm)
    │                     │
    ▼                     ▼
Conv2d(128→A×7)    Conv2d(128→A×7)
                    A=3 anchors
                    7 = (x,y,w,h,obj,fire_cls,smoke_cls)
```

---

## 3. Thay đổi Data Pipeline

### 3.1 Bộ dữ liệu hiện có (Dataset_Balanced_NoEmpty)

Tin tốt: **Bộ dữ liệu ảnh YOLO đã có sẵn bounding box** — chính xác là format cần cho Stage 1 detection training!

```
Dataset_Balanced_NoEmpty/
├── train/
│   ├── images/  (11,437 ảnh)
│   └── labels/  (YOLO format: class x_center y_center w h)
│        class 0 = fire
│        class 1 = smoke
├── val/
└── test/
```

### 3.2 Thay đổi Stage 1: Detection thay vì Classification

**Trước (code hiện tại):** Detection đơn giản không có anchor, output 6 giá trị per patch
```python
# HIỆN TẠI — quá đơn giản, không có NMS, không có anchor
self.detection_head = nn.Linear(dim, 6)  # [P_fire, P_smoke, x, y, w, h]
```

**Sau (đề xuất):** YOLO-style head với anchor boxes
```python
class YOLOHead(nn.Module):
    """
    Detection head dạng YOLO trên lưới 14×14 từ patch features.
    Dự đoán 3 anchor boxes per cell.
    Mỗi box: [x, y, w, h, objectness, P_fire, P_smoke]
    """
    def __init__(self, in_channels=256, num_classes=2, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        out_channels = num_anchors * (5 + num_classes)  # = 3 * 7 = 21
        
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(128)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, out_channels, 1)  # 1×1 conv = linear per cell
    
    def forward(self, patch_features):
        # patch_features: [B, 256, 14, 14]
        x = self.relu(self.bn1(self.conv1(patch_features)))
        return self.conv2(x)  # [B, num_anchors*(5+num_cls), 14, 14]

# Anchor boxes được định nghĩa dựa trên clustering bounding boxes trong dataset
ANCHORS = [
    (0.05, 0.10),   # small fire/smoke: 5%×10% của ảnh
    (0.15, 0.25),   # medium
    (0.40, 0.50),   # large (đám cháy lớn)
]
```

### 3.3 Thay đổi Stage 2: Video Detection

**Vấn đề:** Stage 2 hiện tại chỉ output 1 class label cho cả video clip. Cần output **bounding box PER FRAME** để tracker có thể làm việc.

**Giải pháp:** Thêm một detection branch sử dụng patch features theo từng frame:

```python
class TimeSformerDetector(nn.Module):
    """
    Mô hình thống nhất: backbone TimeSformer + 2 head song song
    Head 1: Video Classification (scene level) — CLS token
    Head 2: Per-frame Detection (object level) — patch tokens
    """
    def __init__(self, image_size=224, patch_size=16, num_frames=8,
                 num_classes=3, num_det_classes=2,
                 dim=256, depth=6, heads=8, ...):
        super().__init__()
        # Backbone (giữ nguyên từ TimeSformerFull)
        self.backbone = TimeSformerBackbone(...)
        
        # Head 1: Video-level classification (normal/controlled/hazard)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        # Head 2: Per-frame detection (fire/smoke bounding boxes)
        self.det_head = YOLOHead(
            in_channels=dim,
            num_classes=num_det_classes,  # fire + smoke
            num_anchors=3
        )
    
    def forward(self, video):
        # video: [B, C, T, H, W]
        patch_tokens, cls_tokens = self.backbone(video)
        # patch_tokens: [B, T, 196, dim]
        # cls_tokens:   [B, T, dim]
        
        # --- HEAD 1: Video Classification ---
        cls_mean = cls_tokens.mean(dim=1)  # [B, dim]
        scene_logits = self.cls_head(cls_mean)  # [B, 3]
        
        # --- HEAD 2: Per-frame Detection ---
        # Aggregate temporal cho detection (dùng weighted sum thay vì mean)
        # hoặc output per-frame cho mỗi frame
        B, T, N, D = patch_tokens.shape
        
        # Lấy frame cuối cùng (most recent) cho real-time detection
        last_frame_patches = patch_tokens[:, -1, :, :]  # [B, 196, dim]
        
        # Reshape thành spatial grid
        spatial = last_frame_patches.reshape(B, 14, 14, D).permute(0, 3, 1, 2)
        # spatial: [B, dim, 14, 14]
        
        det_logits = self.det_head(spatial)  # [B, 21, 14, 14]
        
        return scene_logits, det_logits
```

### 3.4 Video Dataset: Thêm bbox annotations

**Vấn đề lớn nhất:** Video dataset hiện tại (978 video) chỉ có **class label** (normal/controlled/hazard) — KHÔNG có bounding box annotations!

**Giải pháp tiệm cận (không cần label lại toàn bộ):**

```
CHIẾN LƯỢC BÁN TỰ ĐỘNG:

Bước 1: Train detection model trên ảnh (Stage 1, có bbox sẵn)
         → Đạt accuracy ~80-85% detection trên ảnh

Bước 2: Dùng Stage 1 model chạy pseudo-labeling trên video frames
         → Mỗi frame trong video: chạy detector → sinh bbox tự động
         → Lọc confidence > 0.7 → lấy làm ground truth

Bước 3: Review + correct ~10% pseudo labels
         → Chỉ cần review ~100 video × vài frame = ~500 ảnh
         → Tiết kiệm 90% công annotate thủ công

Bước 4: Train Stage 2 với combined loss:
         Loss = cls_loss (video label) + λ × det_loss (bbox pseudo labels)
```

---

## 4. Logic cảnh báo dựa trên Tracking

### 4.1 FireTracker — Máy trạng thái đơn giản

```python
class FireTracker:
    """
    Theo dõi từng đối tượng lửa/khói riêng biệt.
    Dựa trên ByteTrack / SORT để match detection qua các frame.
    """
    
    def __init__(self):
        self.tracks = {}  # track_id → TrackState
    
    class TrackState:
        def __init__(self, track_id, cls, bbox, timestamp):
            self.id         = track_id
            self.cls        = cls          # 'fire' or 'smoke'
            self.bbox_hist  = [bbox]       # lịch sử bounding boxes
            self.area_hist  = [bbox[2]*bbox[3]]  # lịch sử diện tích
            self.center_hist = [(bbox[0], bbox[1])]  # lịch sử tâm
            self.first_seen = timestamp
            self.last_seen  = timestamp
            self.frames     = 1
        
        @property
        def lifetime_s(self):
            return self.last_seen - self.first_seen
        
        @property
        def area_growth_rate(self):
            """Tốc độ tăng diện tích: dương = đang lớn lên"""
            if len(self.area_hist) < 3:
                return 0.0
            recent = np.mean(self.area_hist[-3:])
            older  = np.mean(self.area_hist[:3])
            return (recent - older) / (older + 1e-6)
        
        @property
        def velocity(self):
            """Tốc độ di chuyển của tâm bbox"""
            if len(self.center_hist) < 2:
                return 0.0
            dx = self.center_hist[-1][0] - self.center_hist[-3][0]
            dy = self.center_hist[-1][1] - self.center_hist[-3][1]
            return np.sqrt(dx**2 + dy**2)
        
        @property
        def area_ratio(self):
            """Tỉ lệ diện tích hiện tại / ban đầu"""
            if len(self.area_hist) < 1:
                return 1.0
            return self.area_hist[-1] / (self.area_hist[0] + 1e-6)
```

### 4.2 Decision Engine — Logic quyết định cảnh báo

```python
class AlertDecisionEngine:
    """
    Kết hợp kết quả:
    1. Video classification (scene level): normal/controlled/hazard
    2. Fire tracker metrics: tốc độ lớn, persistence, vùng
    → Đưa ra mức cảnh báo cuối cùng
    """
    
    # Ngưỡng cảnh báo
    THRESHOLDS = {
        'persistence_warning_s':  3.0,   # Cháy liên tục > 3s → nghi ngờ
        'persistence_alert_s':   10.0,   # > 10s → alert
        'area_growth_alert':      0.5,   # Tăng diện tích > 50% → nguy hiểm
        'spread_alert':           0.05,  # Di chuyển > 5% khung hình/giây
        'large_fire_ratio':       0.15,  # Lửa chiếm > 15% diện tích frame
    }
    
    def decide(self, scene_label: str, tracks: list, frame_area: float) -> dict:
        """
        Args:
            scene_label: 'normal' | 'controlled' | 'hazard' từ classifier
            tracks: danh sách TrackState đang active
            frame_area: diện tích frame (H×W) để tính tỉ lệ
        
        Returns:
            dict: {
                'level': 0/1/2/3,  # 0=OK, 1=watch, 2=warning, 3=danger
                'reason': str,
                'tracks': list of annotated tracks
            }
        """
        fire_tracks = [t for t in tracks if t.cls == 'fire']
        smoke_tracks = [t for t in tracks if t.cls == 'smoke']
        
        # --- RULE 1: Scene classifier nói HAZARD → luôn alert ---
        if scene_label == 'hazard':
            return self._alert(3, "Scene classifier: UNCONTROLLED HAZARD")
        
        # --- RULE 2: Không có detection → OK ---
        if not fire_tracks and not smoke_tracks:
            return self._ok()
        
        # --- RULE 3: Fire đang lớn nhanh → DANGER ---
        for track in fire_tracks:
            if track.area_growth_rate > self.THRESHOLDS['area_growth_alert']:
                return self._alert(3,
                    f"Fire #{track.id}: SPREADING FAST! "
                    f"Area ×{track.area_ratio:.1f} in {track.lifetime_s:.0f}s")
        
        # --- RULE 4: Fire lan sang vị trí khác → DANGER ---
        for track in fire_tracks:
            if track.velocity > self.THRESHOLDS['spread_alert']:
                return self._alert(3,
                    f"Fire #{track.id}: SPREADING TO NEW AREA!")
        
        # --- RULE 5: Fire lớn chiếm nhiều diện tích → DANGER ---
        for track in fire_tracks:
            fire_area = track.area_hist[-1] if track.area_hist else 0
            if fire_area / frame_area > self.THRESHOLDS['large_fire_ratio']:
                return self._alert(3,
                    f"Fire #{track.id}: LARGE FIRE detected "
                    f"({fire_area/frame_area*100:.0f}% of frame)")
        
        # --- RULE 6: Fire tồn tại lâu → WARNING ---
        for track in fire_tracks:
            if track.lifetime_s > self.THRESHOLDS['persistence_alert_s']:
                return self._alert(2,
                    f"Fire #{track.id}: Persistent fire "
                    f"{track.lifetime_s:.0f}s (may be uncontrolled)")
        
        # --- RULE 7: Fire mới xuất hiện → WATCH ---
        if fire_tracks:
            return self._alert(1, f"Fire detected — monitoring {len(fire_tracks)} track(s)")
        
        # --- RULE 8: Chỉ có khói → WATCH ---
        if smoke_tracks:
            # Khói không nhất thiết nguy hiểm — cần xem thêm
            oldest_smoke = max(t.lifetime_s for t in smoke_tracks)
            if oldest_smoke > self.THRESHOLDS['persistence_alert_s']:
                return self._alert(2, f"Persistent smoke: {oldest_smoke:.0f}s")
            return self._alert(1, f"Smoke detected — monitoring")
        
        return self._ok()
    
    def _alert(self, level, reason):
        colors = {0: (0,200,80), 1: (0,165,255), 2: (0,100,255), 3: (0,0,255)}
        labels = {0: "✅ OK", 1: "👀 WATCH", 2: "⚠️ WARNING", 3: "🚨 DANGER"}
        return {'level': level, 'reason': reason,
                'color': colors[level], 'label': labels[level]}
    
    def _ok(self):
        return self._alert(0, "No fire detected")
```

---

## 5. Thay đổi code cụ thể

### 5.1 Sửa `inference_realtime.py` — Thêm Detection + Tracking

**Điểm thay đổi chính:**

```python
# inference_realtime.py — PHIÊN BẢN MỚI

class FireDetectionSystem:
    """
    Hệ thống hoàn chỉnh: TimeSformer + YOLO head + ByteTrack + Decision Engine
    """
    
    def __init__(self, weights_path, num_frames=8, threshold=0.5):
        # 1. Load model với dual head
        self.model = TimeSformerDetector(...)  # model mới
        self.model.load_state_dict(torch.load(weights_path))
        
        # 2. Tracker
        from ultralytics import YOLO  # dùng ByteTrack từ ultralytics
        # HOẶC implement SORT đơn giản (không cần cài thêm)
        self.tracker = SimpleFireTracker(max_age=10, min_hits=2)
        
        # 3. Decision Engine
        self.decision = AlertDecisionEngine()
        
        # 4. Frame buffer cho temporal
        self.frame_buffer = deque(maxlen=num_frames)
        
        # 5. Alert history (debounce)
        self.alert_history = deque(maxlen=30)  # 30 frames history
    
    def process_frame(self, frame_bgr):
        """Xử lý 1 frame, trả về frame có overlay"""
        
        # 1. Preprocess
        preprocessed = self.preprocess(frame_bgr)
        self.frame_buffer.append(preprocessed)
        
        if len(self.frame_buffer) < self.num_frames:
            return frame_bgr  # chưa đủ frames
        
        # 2. Inference
        clip = self.buffer_to_tensor()  # [1, C, T, H, W]
        with torch.no_grad():
            scene_logits, det_logits = self.model(clip)
        
        # 3. Decode scene label
        scene_label = ['normal', 'controlled', 'hazard'][scene_logits.argmax()]
        
        # 4. Decode bounding boxes (với NMS)
        detections = self.decode_detections(det_logits, frame_bgr.shape)
        # detections: list of [x1,y1,x2,y2,conf,cls_id]
        
        # 5. Update tracker
        tracked_objs = self.tracker.update(detections)
        # tracked_objs: list of TrackState
        
        # 6. Decision
        h, w = frame_bgr.shape[:2]
        alert = self.decision.decide(scene_label, tracked_objs, h * w)
        
        # 7. Draw
        return self.draw_result(frame_bgr, detections, tracked_objs, alert, scene_label)
    
    def decode_detections(self, det_logits, frame_shape):
        """
        Chuyển output YOLO head → bounding boxes thực tế
        """
        B, C, H, W = det_logits.shape  # C = 21 = 3*(5+2)
        h_frame, w_frame = frame_shape[:2]
        
        # Reshape: [B, 3, 7, 14, 14]
        pred = det_logits.view(B, 3, 7, 14, 14).permute(0, 1, 3, 4, 2)
        # pred: [B, 3, 14, 14, 7]  (7 = x,y,w,h,obj,fire,smoke)
        
        boxes = []
        for anchor_idx, (aw, ah) in enumerate(self.anchors):
            p = pred[0, anchor_idx]  # [14, 14, 7]
            
            obj_conf  = torch.sigmoid(p[..., 4])
            fire_conf = torch.sigmoid(p[..., 5]) * obj_conf
            smoke_conf = torch.sigmoid(p[..., 6]) * obj_conf
            
            # Tìm cells có confidence cao
            for row in range(14):
                for col in range(14):
                    fire_prob  = fire_conf[row, col].item()
                    smoke_prob = smoke_conf[row, col].item()
                    
                    best_conf = max(fire_prob, smoke_prob)
                    if best_conf < 0.4:  # ngưỡng
                        continue
                    
                    # Decode tọa độ YOLO → pixel coordinates
                    cx = (col + torch.sigmoid(p[row, col, 0]).item()) / 14
                    cy = (row + torch.sigmoid(p[row, col, 1]).item()) / 14
                    bw = aw * torch.exp(p[row, col, 2]).item()
                    bh = ah * torch.exp(p[row, col, 3]).item()
                    
                    # Convert to pixel
                    x1 = int((cx - bw/2) * w_frame)
                    y1 = int((cy - bh/2) * h_frame)
                    x2 = int((cx + bw/2) * w_frame)
                    y2 = int((cy + bh/2) * h_frame)
                    
                    cls_id = 0 if fire_prob > smoke_prob else 1
                    boxes.append([x1, y1, x2, y2, best_conf, cls_id])
        
        # Non-Maximum Suppression
        return self.nms(boxes, iou_threshold=0.4)
    
    def draw_result(self, frame, detections, tracks, alert, scene_label):
        """Vẽ bbox, track ID, alert level lên frame"""
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Vẽ bounding boxes
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            cls_name = ['FIRE', 'SMOKE'][cls_id]
            color = (0, 80, 255) if cls_id == 0 else (128, 128, 128)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result, f"{cls_name} {conf:.2f}",
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Vẽ Track ID và thông tin
        for track in tracks:
            if track.bbox_hist:
                bbox = track.bbox_hist[-1]
                cx = int(bbox[0] * w); cy = int(bbox[1] * h)
                track_color = (0, 255, 255)  # cyan cho track
                cv2.circle(result, (cx, cy), 5, track_color, -1)
                cv2.putText(result,
                    f"ID:{track.id} {track.lifetime_s:.1f}s "
                    f"×{track.area_ratio:.1f}",
                    (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, track_color, 1)
        
        # Alert banner
        banner_color = alert['color']
        cv2.rectangle(result, (0, 0), (w, 60), banner_color, -1)
        cv2.addWeighted(result, 0.75, frame, 0.25, 0, result)
        cv2.putText(result, alert['label'], (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(result, alert['reason'], (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(result, f"Scene: {scene_label.upper()}", (w - 200, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return result
```

### 5.2 Implement SORT Tracker đơn giản (không cần ultralytics)

```python
from scipy.optimize import linear_sum_assignment
import numpy as np

class SimpleFireTracker:
    """
    SORT (Simple Online and Realtime Tracking) — đơn giản, không cần thư viện ngoài
    IoU-based matching + Kalman filter optional
    """
    
    def __init__(self, max_age=15, min_hits=2, iou_threshold=0.3):
        self.max_age      = max_age       # Số frames track tồn tại khi không thấy
        self.min_hits     = min_hits      # Số lần phải detect trước khi hiện lên UI
        self.iou_threshold = iou_threshold
        self.tracks       = []
        self._next_id     = 0
        self._frame_count = 0
    
    def update(self, detections, timestamp):
        """
        detections: list of [x1,y1,x2,y2,conf,cls_id]
        Returns: list of active TrackState
        """
        self._frame_count += 1
        
        # Nếu không có detection
        if not detections:
            for t in self.tracks:
                t.last_seen_age += 1
            self.tracks = [t for t in self.tracks if t.last_seen_age <= self.max_age]
            return [t for t in self.tracks if t.frames >= self.min_hits]
        
        dets = np.array(detections)
        
        if not self.tracks:
            # Tạo tracks mới từ tất cả detections
            for det in dets:
                self._create_track(det, timestamp)
        else:
            # Match detections với tracks hiện có
            iou_matrix = self._compute_iou_matrix(dets, self.tracks)
            
            # Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            
            matched_dets = set()
            matched_tracks = set()
            
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    self._update_track(self.tracks[c], dets[r], timestamp)
                    matched_dets.add(r)
                    matched_tracks.add(c)
            
            # Unmatched detections → new tracks
            for i, det in enumerate(dets):
                if i not in matched_dets:
                    self._create_track(det, timestamp)
            
            # Unmatched tracks → age them out
            for j, track in enumerate(self.tracks):
                if j not in matched_tracks:
                    track.last_seen_age += 1
        
        # Xóa tracks quá già
        self.tracks = [t for t in self.tracks if t.last_seen_age <= self.max_age]
        
        return [t for t in self.tracks if t.frames >= self.min_hits]
    
    def _create_track(self, det, timestamp):
        x1,y1,x2,y2,conf,cls_id = det
        bbox = ((x1+x2)/2, (y1+y2)/2, (x2-x1), (y2-y1))  # cx,cy,w,h normalized
        track = FireTracker.TrackState(self._next_id, cls_id, bbox, timestamp)
        self.tracks.append(track)
        self._next_id += 1
    
    def _update_track(self, track, det, timestamp):
        x1,y1,x2,y2,conf,cls_id = det
        # Smooth bbox với exponential moving average
        alpha = 0.7
        cx = (x1+x2)/2; cy = (y1+y2)/2
        w = x2-x1; h = y2-y1
        bbox = (cx, cy, w, h)
        
        track.bbox_hist.append(bbox)
        track.area_hist.append(w * h)
        track.center_hist.append((cx, cy))
        track.last_seen = timestamp
        track.last_seen_age = 0
        track.frames += 1
    
    def _compute_iou_matrix(self, dets, tracks):
        """Tính ma trận IoU giữa mỗi cặp (detection, track)"""
        iou = np.zeros((len(dets), len(tracks)))
        for i, det in enumerate(dets):
            for j, track in enumerate(tracks):
                if track.bbox_hist:
                    last_bbox = track.bbox_hist[-1]  # cx,cy,w,h
                    iou[i,j] = self._bbox_iou(det[:4], last_bbox)
        return iou
    
    def _bbox_iou(self, det_xyxy, track_cxcywh):
        dx1 = track_cxcywh[0] - track_cxcywh[2]/2
        dy1 = track_cxcywh[1] - track_cxcywh[3]/2
        dx2 = track_cxcywh[0] + track_cxcywh[2]/2
        dy2 = track_cxcywh[1] + track_cxcywh[3]/2
        
        ix1 = max(det_xyxy[0], dx1); iy1 = max(det_xyxy[1], dy1)
        ix2 = min(det_xyxy[2], dx2); iy2 = min(det_xyxy[3], dy2)
        
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = ((det_xyxy[2]-det_xyxy[0])*(det_xyxy[3]-det_xyxy[1]) +
                 (dx2-dx1)*(dy2-dy1) - inter)
        return inter / (union + 1e-6)
```

---

## 6. So sánh các phương án Tracker

| Tracker | Độ phức tạp | Yêu cầu | Phù hợp | Ghi chú |
|---------|------------|---------|---------|---------|
| **SORT** (đề xuất) | ⭐⭐ | scipy | ✅ | Đơn giản, hiệu quả cho fire |
| **ByteTrack** | ⭐⭐⭐ | ultralytics | ✅✅ | Tốt hơn với occlusion |
| **DeepSORT** | ⭐⭐⭐⭐ | ReID model | ⚠️ | Cần train thêm ReID |
| **Kalman + IoU** | ⭐ | numpy | ✅ | Nhẹ nhất, phù hợp Edge |
| **YOLO track** | ⭐⭐ | ultralytics | ✅ | Nếu dùng YOLOv8 làm detector |

> [!TIP]
> Với đặc thù lửa/khói (chuyển động liên tục, không bị che khuất nhiều), **SORT + IoU** là đủ. Không cần DeepSORT vì appearance features của lửa thay đổi liên tục.

---

## 7. Khó khăn kỹ thuật và giải pháp

### K1. Thiếu bbox annotations trong video dataset

**Vấn đề:** 978 video chỉ có scene label, không có bbox per frame.

**Giải pháp:**
```
Option A (Khuyến nghị): Pseudo-labeling
  1. Train Stage 1 detector trên ảnh (11,437 ảnh có bbox)
  2. Chạy detector trên video frames → sinh bbox tự động
  3. Lọc: giữ bbox với confidence > 0.7
  4. Review thủ công ~500 frames (10%)
  5. Dùng làm training labels cho Stage 2

Option B: Train detector riêng song song
  1. Giữ TimeSformer cho classification
  2. Train YOLOv8-nano riêng trên Dataset_Balanced_NoEmpty
  3. Inference: TimeSformer cho scene label + YOLOv8 cho bbox
  4. Kết hợp trong Decision Engine
  → Không cần sửa TimeSformer architecture
```

### K2. Tracking khi lửa biến hình liên tục

**Vấn đề:** Ngọn lửa thay đổi hình dạng, màu sắc từng frame → IoU thấp → track thường xuyên bị mất.

**Giải pháp:**
```python
# Tăng max_age: track vẫn sống dù không detect trong 15 frames
self.tracker = SimpleFireTracker(max_age=15, min_hits=2)

# Giảm ngưỡng IoU match: chấp nhận overlap 25% vẫn là cùng object
self.tracker.iou_threshold = 0.25

# Thêm center distance metric: ngoài IoU, dùng thêm khoảng cách tâm
def match_score(det, track):
    iou = compute_iou(det, track)
    dist = compute_center_distance(det, track)
    return 0.7 * iou + 0.3 * (1 - dist)  # kết hợp
```

### K3. False Positive từ ánh sáng (đèn xe, phản chiếu)

**Vấn đề:** Detector có thể nhầm ánh đèn xe qua cửa sổ, phản chiếu kính là lửa.

**Giải pháp:**
```python
# Filter track ngắn: chỉ alert khi track tồn tại > 2 giây
MIN_TRACK_DURATION = 2.0  # giây

# Filter theo aspect ratio: lửa thường cao hơn rộng (tỉ lệ > 0.5)
def is_valid_fire_bbox(bbox):
    w, h = bbox[2], bbox[3]
    if h < w * 0.3:  # quá phẳng → không phải lửa
        return False
    return True

# Kết hợp với scene classifier:
# Nếu scene = 'normal' và track ngắn → giảm severity
```

### K4. Hiệu năng real-time (FPS)

**Vấn đề:** TimeSformer + YOLOHead + Tracker có thể chậm hơn yêu cầu CCTV (25 FPS).

**Benchmark kỳ vọng:**
| Hardware | FPS ước tính |
|---------|-------------|
| RTX 3060 (local) | 8–15 FPS |
| A10G (Modal) | 30–50 FPS |
| Raspberry Pi 5 (Edge) | 0.5–1 FPS (cần quantize) |
| Rock 5B (Edge) | 2–4 FPS (với INT8) |

**Giải pháp cho Edge:**
```python
# 1. Chỉ chạy TimeSformer mỗi 5 frames (temporal model không cần 25fps)
# 2. Chạy lightweight detector mỗi frame (hoặc dùng attention map trực tiếp)
# 3. Tracker luôn chạy mỗi frame (rất nhẹ)

# Cấu trúc hybrid:
# Frame 1,6,11,...: TimeSformer full inference
# Frame 2,3,4,5:   Chỉ chạy tracker (predict từ Kalman filter)
```

---

## 8. Lộ trình thực hiện

### Phase A — Thêm YOLOv8 song song (Nhanh nhất, 1–2 tuần)

> Không sửa TimeSformer, chỉ thêm tracker. **Khuyến nghị cho bước đầu.**

```bash
# Cài YOLOv8 (đã có sẵn dataset_balanced_nonempty)
pip install ultralytics

# Train YOLOv8-nano trên Dataset_Balanced_NoEmpty
yolo train model=yolov8n.pt data=fire_smoke.yaml epochs=100 imgsz=224

# Inference dual: TimeSformer (scene) + YOLOv8 (bbox) + SORT (tracking)
```

**Kết quả:** Hệ thống hoạt động ngay với tracker đơn giản, không cần sửa kiến trúc.

### Phase B — Tích hợp Detection Head vào TimeSformer (3–4 tuần)

1. Thêm `YOLOHead` vào `TimeSformerDetector`
2. Pseudo-label video frames bằng YOLOv8 đã train
3. Train Stage 2 với combined loss (classification + detection)
4. Test sliding window inference với tracker

### Phase C — Tinh chỉnh Decision Engine (1 tuần)

1. Calibrate threshold dựa trên test set thực tế
2. Thêm zone-based filtering (vùng bếp = ít sensitive hơn)
3. Debounce alerts (không spam notification mỗi frame)
4. Log tracking events để phân tích sau

### Tóm tắt thay đổi cần thiết

| File | Loại thay đổi | Mức độ | Phase |
|------|--------------|--------|-------|
| `inference_realtime.py` | Thêm tracker + decision engine | **Lớn** | A |
| `train_spatial.py` | Sửa detection head → YOLO-style | **Trung bình** | B |
| `train_temporal.py` | Thêm dual head + combined loss | **Lớn** | B |
| **[NEW]** `tracker.py` | Module SORT tracker | **Mới hoàn toàn** | A |
| **[NEW]** `decision_engine.py` | Logic quyết định cảnh báo | **Mới hoàn toàn** | A |
| **[NEW]** `pseudo_label_video.py` | Tạo bbox cho video frames | **Mới** | B |

> [!IMPORTANT]
> **Khuyến nghị thứ tự thực hiện:**
> 1. Bắt đầu với **Phase A** — Train YOLOv8 trên 11,437 ảnh (data đã có sẵn), thêm SORT tracker và Decision Engine vào `inference_realtime.py`. Có thể demo trong 1–2 tuần.
> 2. Sau khi validate tracking logic hoạt động đúng, mới chuyển sang **Phase B** để tích hợp detection head vào TimeSformer (unified model).

> [!WARNING]
> Nếu chỉ có 978 video và không có bbox annotation → **KHÔNG thể train dual-head TimeSformer** trực tiếp cho detection. Phải làm pseudo-labeling trước (Phase B prerequisite).
