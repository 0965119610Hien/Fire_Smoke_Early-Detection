# Phân Tích Pipeline & Vấn Đề Overlay — Fire/Smoke Early Detection

> **Ngày phân tích:** 2026-08-17 | **Người phân tích:** Antigravity AI  
> **Trạng thái:** 🔴 Nghiêm trọng — Cần xử lý trước khi dùng dataset Part2 để huấn luyện

---

## 1. Tổng Quan Pipeline

```mermaid
flowchart LR
    subgraph INPUT
        V[Video Clips\n681 clips] -->|extract frames| F[Dataset_Part2_Video_Frames\n~341,000 raw frames]
        F -->|augmentation\nhflip, rot, bright, noise| F2[Augmented Frames\n×2-3 / frame]
    end

    subgraph ANNOTATION [Auto-Annotation Pipeline]
        F2 -->|YOLO Inference\nconf=0.25, iou=0.45| YOLO[YOLOv8s-seg\nyolov8_fire_smoke_best.pt]
        YOLO -->|BBox xyxy| SAM[SAM ViT-B\nsegment_anything]
        SAM -->|Binary Mask| CONTOUR[cv2.findContours]
        CONTOUR -->|normalize / img_wh| POLY[YOLO Polygon .txt]
    end

    subgraph OUTPUT
        POLY --> LBL[Dataset_Part2_YOLO_SAM_Segmented\n/labels — 925 files]
        SAM -->|binary_mask PNG| MASK[/masks — 925 files]
        LBL --> VIS[/visualizations — 52 files]
    end
```

---

## 2. Các Vấn Đề Được Phát Hiện

### 🔴 Vấn Đề 1 — NAMING COLLISION: Frame trùng tên từ nhiều clip khác nhau

**Mức độ:** CRITICAL — Làm hỏng toàn bộ tính nhất quán giữa image và label

**Chi tiết:**
- 681 clip video đều tạo ra frame với tên dạng `frame_000000.png`, `frame_000001.png`...
- Khi pipeline ghi label vào `Dataset_Part2_YOLO_SAM_Segmented/labels/frame_000000.txt`, nếu có 2 clip cùng tạo ra `frame_000000.png` → file label cuối sẽ ghi đè file trước
- Ví dụ: `frame_000000` tồn tại trong **hàng chục clip khác nhau** như `fire1_001_pure_chunk_01`, `pexels_cigarette_controlled_5036384`, `pexels_welding_sparks_ctrl_855137`...
- **Hậu quả:** Label của clip X có thể gán nhầm cho image của clip Y, dẫn đến ground-truth sai

**Bằng chứng:**
```python
# frame_000000 comes from clip: fcbb7478_yt_residential_fire_pK7SUSZwwj4_pure_chunk_01
# frame_000000 comes from clip: pexels_decorative_fire_fp_6024944
# frame_000000 comes from clip: pexels_cigarette_controlled_5036384
# frame_000000 comes from clip: yt_warehouse_factory_wFgBjbj-tb8_trimmed_pure
# ... (hàng trăm clip tương tự)
```

**Root Cause trong code:**
```python
# modal_yolo_sam_pipeline.py line 180
(out_lbls / f"{stem}.txt").write_text(res["txt_content"], encoding="utf-8")
# → stem = "frame_000000" cho TẤT CẢ clips → ghi đè nhau
```

---

### 🔴 Vấn Đề 2 — VISUALIZATION: Render overlay không chính xác

**Mức độ:** HIGH — Khiến không thể đánh giá chất lượng gán nhãn

**Chi tiết:**
- Các ảnh trong `/visualizations` chỉ hiển thị **màu đỏ thô** (solid color paint) trên vùng lửa
- Không có overlay trong suốt (transparency), không vẽ polygon contour, không text label
- Hình ảnh `frame_000000_vis.jpg` (phòng khách với lò sưởi) cho thấy phần lửa trong lò bị tô đỏ thô thiếu chính xác về hình dạng

**Xem trực quan:**

| Biểu hiện | Kỳ vọng | Thực tế |
|-----------|---------|---------|
| SAM mask overlay | Polygon màu bán trong suốt | Màu đặc không trong suốt |
| Class label | Text "fire"/"smoke" + bounding box | Không có |
| Polygon boundary | Đường viền rõ nét | Không vẽ |
| Confidence score | Hiển thị confidence YOLO | Không có |

---

### 🟡 Vấn Đề 3 — DETECTION QUALITY: Tỷ lệ phát hiện thấp + class imbalance

**Mức độ:** MEDIUM-HIGH

**Số liệu:**
```
DS2 Labels: 925 tổng số
  - Có detection:   316 files (34.2%)
  - Không detection: 609 files (65.8%)

Class distribution:
  - Fire (class 0):  123 polygons (19.8%)
  - Smoke (class 1): 499 polygons (80.2%)
```

**Phân tích:**
- 65.8% file label rỗng → YOLO không detect gì trên phần lớn frames
- Có thể có 2 nguyên nhân:
  1. **False Negative**: YOLO bỏ sót khói nhẹ, khói sớm, khói từ xa (threshold `conf=0.25` vẫn còn bỏ sót)
  2. **Negative frames**: Frames từ welding sparks, steam, cigarette được đưa vào nhưng YOLO (đúng) không detect

- Smoke chiếm 80.2% → **nghiêm trọng mất cân bằng** với Fire 19.8%

---

### 🟡 Vấn Đề 4 — FALSE POSITIVES: YOLO detect nhầm trong một số clip

**Mức độ:** MEDIUM

**Chi tiết:**
- Clip `pexels_decorative_fire_fp_*` và `pexels_daily_activity_fire_ctx_*` chứa lửa trong lò sưởi trang trí / bếp nấu — đây có thể là **false positive** (lửa có kiểm soát, không phải sự cố)
- YOLO detect lửa trong lò sưởi phòng khách → SAM vẽ mask đúng về mặt kỹ thuật nhưng **sai về ngữ nghĩa** (đây không phải fire hazard)
- Dữ liệu Part2 ban đầu được thiết kế để phân biệt "khói sớm / lửa nguy hiểm" với "welding / cigarette / steam", nhưng một số clip lại là lửa trang trí trong nhà

---

### 🟠 Vấn Đề 5 — TINY MASKS: Nhiều polygon với quá ít điểm

**Mức độ:** MEDIUM

**Số liệu:** Nhiều file label có polygon chỉ với 8-25 điểm, một số chỉ 10-14 điểm

**Nguyên nhân:**
```python
# modal_yolo_sam_pipeline.py line 124
if cv2.contourArea(cnt) > 20:  # Threshold diện tích quá thấp (20 pixels²)
```
- Mask nhỏ (< 100px²) vẫn được chấp nhận → noise/artifact từ SAM
- Những polygon 10-14 điểm không đủ để mô tả hình dạng khói/lửa phức tạp

---

## 3. Sơ Đồ Root Cause

```
Kết quả Overlay Không Ổn
├── Frame trùng tên (Naming Collision)
│   └── Label sai image → nhãn vô nghĩa
├── Visualization code thiếu opacity/polygon render
│   └── Không thể đánh giá visual quality
├── Nhiều frame rỗng (65.8%)
│   ├── Negative clips (welding, steam) → OK
│   └── Fire/smoke clips bị bỏ sót → Mất dữ liệu
└── False Positive (lửa trang trí)
    └── Label noise làm giảm chất lượng model
```

---

## 4. Đề Xuất Phương Án Xử Lý

### ✅ Phương Án A — Fix Naming (URGENT — phải làm trước)

**Mục tiêu:** Đảm bảo mỗi frame có tên duy nhất kèm clip ID

```python
# Thay vì stem = p.stem ("frame_000000")
# Dùng: clip_id + frame_index
import hashlib

def make_unique_stem(clip_dir: Path, frame_path: Path) -> str:
    """Tạo stem duy nhất: <clip_id_short>_<frame_stem>"""
    clip_hash = hashlib.md5(clip_dir.name.encode()).hexdigest()[:8]
    return f"{clip_hash}_{frame_path.stem}"
    
# Ví dụ: "a3f9b21c_frame_000000" thay vì "frame_000000"
```

**Tác động:** Cần chạy lại toàn bộ pipeline với naming mới (hoặc post-process copy ảnh vào thư mục images kèm rename)

---

### ✅ Phương Án B — Fix Visualization Code

**Mục tiêu:** Render overlay thực sự với transparency + polygon + label

```python
def render_overlay(img_bgr, label_path, alpha=0.4):
    """Vẽ overlay SAM polygon lên ảnh gốc."""
    import cv2, numpy as np
    
    CLASS_COLORS = {0: (0, 0, 255), 1: (0, 165, 255)}  # Fire=đỏ, Smoke=cam
    CLASS_NAMES  = {0: "fire", 1: "smoke"}
    
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    
    lines = Path(label_path).read_text().strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        pts = np.array([(int(coords[i]*w), int(coords[i+1]*h))
                        for i in range(0, len(coords), 2)], dtype=np.int32)
        
        color = CLASS_COLORS.get(cls_id, (128, 128, 128))
        cv2.fillPoly(overlay, [pts], color)          # Tô màu filled
        cv2.polylines(img_bgr, [pts], True, color, 2) # Vẽ viền
        cv2.putText(img_bgr, CLASS_NAMES[cls_id], pts[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Blend: img_bgr + overlay với alpha
    vis = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    return vis
```

---

### ✅ Phương Án C — Lọc False Positive & Tiny Masks

**Mục tiêu:** Loại bỏ nhãn nhiễu trước khi đưa vào huấn luyện

```python
# Tăng area threshold để loại noise
MIN_AREA_PX = 200          # Tăng từ 20 → 200 pixels²

# Lọc clip decorative/false-positive
EXCLUDE_CLIP_PATTERNS = [
    "decorative_fire_fp",    # Lửa trang trí - non-hazardous
    "daily_activity_fp",     # Hoạt động thường ngày có lửa
    "cooking_smoke_normal",  # Khói nấu ăn - non-hazardous
]

# Thêm điều kiện vào pipeline:
if any(pat in clip_dir.name for pat in EXCLUDE_CLIP_PATTERNS):
    continue  # Skip clip này
```

---

### ✅ Phương Án D — Tái Cấu Trúc Dataset Part2 (Dài Hạn)

**Mục tiêu:** Tổ chức lại cấu trúc thư mục chuẩn YOLO với split train/val/test

```
Dataset_Part2_YOLO_SAM_Segmented/  (sau fix)
├── train/
│   ├── images/   # <clip_hash>_frame_XXXXXX.png (unique names)
│   └── labels/   # <clip_hash>_frame_XXXXXX.txt
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

Tỷ lệ phân chia đề xuất: **80/15/5** (train/val/test), phân theo clip-level để tránh data leakage giữa các frames của cùng một clip.

---

## 5. Kế Hoạch Thực Thi

| Ưu tiên | Nhiệm vụ | Ước tính | Trạng thái |
|---------|----------|----------|------------|
| 🔴 P0 | Fix naming collision (unique stem per clip) | 2h | Chưa làm |
| 🔴 P0 | Copy images vào `/images` với tên mới | 2h | Chưa làm |
| 🟠 P1 | Fix visualization render (transparency + polygon) | 1h | Chưa làm |
| 🟠 P1 | Tăng `MIN_AREA_PX` từ 20 → 200 | 30min | Chưa làm |
| 🟡 P2 | Lọc false positive clips (decorative_fire_fp) | 1h | Chưa làm |
| 🟡 P2 | Tổ chức cấu trúc train/val/test | 2h | Chưa làm |
| 🟢 P3 | Merge DS1 + DS2 → Dataset hợp nhất cuối cùng | 4h | Chưa làm |
| 🟢 P3 | Rerun visualization sau khi fix | 1h | Chưa làm |

---

## 6. Tổng Quan Dữ Liệu Hiện Tại

| Dataset | Nguồn | Polygons | Trạng thái |
|---------|-------|----------|------------|
| `Dataset_Segmentation_AutoConverted` | Static images gốc + SAM ViT-B | ~11,175 | ✅ Dùng được |
| `Dataset_Balanced_NoEmpty` | BBox YOLO cho detection | 10,756 imgs (train) | ✅ Dùng cho detection |
| `Dataset_Part2_YOLO_SAM_Segmented` | Video frames + YOLO+SAM | 622 polygons / 316 files | ❌ Có naming collision |

> **Khuyến nghị:** Chỉ dùng `Dataset_Segmentation_AutoConverted` làm nguồn huấn luyện segmentation cho đến khi DS2 được fix xong.

---

## 7. Thống Kê Nhanh DS2 Hiện Tại

```
Total label files:    925
With detections:      316 (34.2%)
Empty files:          609 (65.8%)

Fire polygons (cls 0): 123 (19.8%)
Smoke polygons (cls 1): 499 (80.2%)

Clip-level analysis:
  Positive clips (smoke/fire): 55
  Negative clips (welding/steam/cigarette): 13
  (nhiều clips không tìm thấy frame tương ứng do naming collision)

Visualizations: 52 files
  → Chỉ vẽ solid color, không có transparency/polygon overlay
```
