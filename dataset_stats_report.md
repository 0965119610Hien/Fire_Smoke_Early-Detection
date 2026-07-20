# Báo Cáo Thống Kê Dataset & Hướng Tiền Xử Lý

> Trạng thái: Sau relabel (110 files dời từ label 1 → label 0)  
> Ngày: 2026-07-20

---

## 1. Tổng quan số lượng

| | Train (`Video/Video/`) | Test (`dataset_test/`) |
|---|---|---|
| **Tổng video** | **1,001** | **1,066** |
| **Tổng thời lượng** | 281.3 phút | 652.4 phút |
| **Tổng dung lượng** | 9.6 GB | 19.3 GB |
| **Duration trung bình** | **16.86 giây** | **36.72 giây** |

### Phân bố theo label

| Label | Train | Test | Tỉ lệ Train | Tỉ lệ Test |
|---|---|---|---|---|
| `0_normal_noise` | 386 | 345 | 38.6% | 32.4% |
| `1_controlled_fire` | 292 | 380 | 29.2% | 35.7% |
| `2_uncontrolled_hazard` | 323 | 341 | 32.3% | 32.0% |

---

## 2. Thống kê chi tiết Train Set

| Label | Videos | Dur min | Dur max | Dur mean | Dur stdev | FPS mean | Size max |
|---|---|---|---|---|---|---|---|
| `0_normal_noise` | 386 | 5.05s | 221.23s | 18.35s | **19.48s** | 29.42 | 39 MB |
| `1_controlled_fire` | 292 | **3.6s** | 182.53s | 15.38s | **13.91s** | 32.97 | **400 MB** |
| `2_uncontrolled_hazard` | 323 | 5.33s | **277.97s** | 16.42s | 15.32s | 30.47 | **676 MB** |
| **TOTAL** | **1,001** | **3.6s** | **277.97s** | **16.86s** | — | 30.79 | — |

### Top resolutions train set

| Label | #1 | #2 | #3 | #4 |
|---|---|---|---|---|
| `0_normal_noise` | 1920×1080 (233) | 540×960 (50) | 2048×1080 (38) | 1280×720 (34) |
| `1_controlled_fire` | 1920×1080 (127) | 1080×1920 (37) | **2160×3840 (34)** | 540×960 (25) |
| `2_uncontrolled_hazard` | 1920×1080 (192) | 1280×720 (66) | **640×480 (46)** | 540×960 (7) |

---

## 3. Thống kê chi tiết Test Set

| Label | Videos | Dur min | Dur max | Dur mean | Dur stdev | FPS mean |
|---|---|---|---|---|---|---|
| `0_normal_noise` | 345 | 19.52s | 106.81s | 33.52s | 15.04s | 28.13 |
| `1_controlled_fire` | 380 | 19.56s | 189.29s | 35.29s | 18.70s | 27.45 |
| `2_uncontrolled_hazard` | 341 | 19.52s | 240.75s | 41.57s | **27.85s** | 30.71 |
| **TOTAL** | **1,066** | **19.52s** | **240.75s** | **36.72s** | — | 28.71 |

### Phân bố subtype test set (duration mean)

````carousel
**Label 0 — Normal Noise**

| Subtype | Videos | Dur mean | Dur range |
|---|---|---|---|
| optical_noise | 62 | 33.6s | 19.8–107s |
| steam_vapor | 49 | 34.6s | 20.3–88s |
| cooking_smoke_normal | 40 | 31.9s | 19.7–86s |
| dust_particles | 40 | 31.1s | 19.5–71s |
| decorative_fire_fp | 39 | 37.0s | 20.0–92s |
| welding_sparks_noise | 39 | 30.4s | 19.6–76s |
| cigarette_vape_smoke | 38 | 34.9s | 19.6–83s |
| daily_activity_fp | 38 | 34.6s | 21.2–86s |

<!-- slide -->
**Label 1 — Controlled Fire**

| Subtype | Videos | Dur mean | Dur range |
|---|---|---|---|
| heavy_smoke_controlled | 69 | 41.3s | 19.7–136s |
| cooking_smoke_threshold | 68 | 35.4s | 19.8–189s |
| early_smoke_onset | 66 | 32.7s | 19.6–108s |
| cigarette_controlled | 59 | 32.8s | 20.0–60s |
| welding_sparks_ctrl | 45 | 37.1s | 20.2–86s |
| small_visible_flame | 38 | 32.6s | 20.8–67s |
| daily_activity_fire_ctx | 35 | 32.9s | 19.6–80s |

<!-- slide -->
**Label 2 — Uncontrolled Hazard**

| Subtype | Videos | Dur mean | Dur range |
|---|---|---|---|
| daily_ctx_real_fire | 70 | 35.7s | 19.9–110s |
| heavy_smoke_hazard | 55 | **52.0s** | 20.0–241s |
| early_smoke_real_fire | 49 | 44.6s | 19.6–224s |
| large_visible_flame | 47 | 42.2s | 19.5–208s |
| residential_indoor_fire | 42 | 38.9s | 19.8–194s |
| toxic_material_smoke | 39 | 39.3s | 20.0–82s |
| diffused_smoke_spreading | 39 | 38.0s | 19.6–76s |
````

---

## 4. Phân tích vấn đề & Hướng tiền xử lý

---

### 🔴 VẤN ĐỀ 1 — Duration Gap nghiêm trọng giữa Train và Test

| | Train | Test |
|---|---|---|
| Duration min | **3.6s** | 19.5s |
| Duration mean | **16.9s** | 36.7s |
| Duration stdev | **~16s** (rất cao) | ~20s |

**Nhận xét:** Train có clips rất ngắn (3.6s, nhiều clip dưới 10s) trong khi test được thiết kế với minimum 20s. Model học temporal patterns trên clips ngắn sẽ không generaliz tốt sang test clips dài hơn.

**Hướng xử lý:**
> **A. Cắt sliding window cho test videos dài:**  
> Với video test dài ~36s, cắt thành các window 8–16s (overlap 50%) → mỗi video test tạo ra 4–8 clips → lấy majority vote hoặc max score
>
> **B. Lọc bỏ train clips quá ngắn:**  
> Loại bỏ clips < 5s (không đủ temporal information cho TimeSformer)
>
> **C. Temporal augmentation:**  
> Tăng tốc / giảm tốc video (speed jitter ±20%) để train thấy nhiều duration hơn

---

### 🔴 VẤN ĐỀ 2 — Resolution cực kỳ hỗn loạn

**Train có ít nhất 15+ resolution khác nhau:**
- Portrait 4K: `2160×3840` (34 clips trong label 1!)
- Landscape 4K: `3840×2160`
- Full HD: `1920×1080` (dominant ~50%)
- HD: `1280×720`
- Low: `640×480` (46 clips trong label 2!)
- Vertical mobile: `540×960`, `506×960`, `1080×1920`

**Nhận xét:** Model phải xử lý ảnh portrait (9:16) lẫn landscape (16:9) lẫn SD. Đặc biệt `640×480` trong label 2 (video cháy nhà cũ) tạo distribution shift.

**Hướng xử lý:**
> **Resize về 224×224 (TimeSformer) hoặc 256×256 rồi center-crop 224×224**  
> - KHÔNG scale theo chiều dài cạnh → thêm padding → distort thông tin spatial
> - Dùng `center_crop` với aspect-ratio aware resize: resize short-side → 256, crop 224
> - Cân nhắc `letterbox` cho video portrait để tránh mất thông tin

---

### 🟠 VẤN ĐỀ 3 — Class Imbalance trong Train (sau relabel)

| Label | Videos | Tỉ lệ |
|---|---|---|
| `0_normal_noise` | **386** | 38.6% |
| `1_controlled_fire` | 292 | 29.2% ← ít nhất |
| `2_uncontrolled_hazard` | 323 | 32.3% |

**Nhận xét:** Label 1 thiếu ~94 videos so với label 0 (chênh 32%). Khi huấn luyện, model có xu hướng thiên về label 0.

**Hướng xử lý:**
> - **WeightedRandomSampler:** `weights = [1/count_per_class]` cho DataLoader
> - **Hoặc Loss weighting:** `CrossEntropy(weight=[0.85, 1.12, 0.98])`  
>   (tỉ lệ nghịch với count: 1000/386, 1000/292, 1000/323)
> - **Augmentation tăng cường label 1:** horizontal flip + color jitter + temporal crop

---

### 🟠 VẤN ĐỀ 4 — FPS biến động cao

| Label | FPS mean | FPS stdev |
|---|---|---|
| Train label 0 | 29.42 | **9.10** |
| Train label 1 | 32.97 | **12.73** ← rất cao |
| Train label 2 | 30.47 | 8.53 |
| Test tổng | 28.71 | 7.74 |

**Nhận xét:** FPS stdev=12.73 ở label 1 nghĩa là có clips 24fps lẫn 60fps trong cùng label. Nếu lấy 8 frames đều nhau theo thời gian thì sampling interval khác nhau → temporal motion khác nhau.

**Hướng xử lý:**
> **Uniform temporal sampling theo giây, KHÔNG theo frame index:**
> - Ví dụ: lấy frame tại giây 0, 1, 2, ..., 7 (8 frames cố định)
> - Hoặc: lấy N frames uniformly distributed theo duration
> - Tuyệt đối KHÔNG dùng `frame[0], frame[30], frame[60]...` (sẽ bị lệch với video 60fps)

---

### 🟡 VẤN ĐỀ 5 — Train có clips cực ngắn và file size cực lớn

- **Clip ngắn nhất:** 3.6s (label 1) — không đủ cho TimeSformer 8-frame
- **File lớn nhất train:** 675.9 MB (label 2), 400 MB (label 1) — un-encoded 4K raw
- **File nhỏ nhất:** 0.07 MB (có thể corrupt hoặc sub-second)

**Hướng xử lý:**
> **Filter trước khi training:**
> ```python
> MIN_DURATION = 5.0   # giây
> MAX_DURATION = 120.0 # giây (tránh video dài bất thường)
> MIN_SIZE_MB  = 0.5   # loại file có thể corrupt
> ```
> → Loại khoảng 15–20 clips train (ước tính)

---

### 🟡 VẤN ĐỀ 6 — Train vs Test Distribution Shift (Duration × Label)

```
Train label 2: mean=16.4s, nhiều clip ngắn từ YouTube crawl
Test  label 2: mean=41.6s, STDEV=27.9s (rất phân tán, max 241s)
```

Video cháy nhà thực (residential fire) dài hơn và phức tạp hơn. Train model trên clip 16s rồi test với clip 40s sẽ bị drop performance.

**Hướng xử lý:**
> **Sliding window inference cho test:**
> ```python
> def predict_long_video(video, window=8, stride=4, fps=25):
>     # Cắt frames: [0:8], [4:12], [8:16]...
>     # Lấy max confidence hoặc majority vote
>     scores = [model(clip) for clip in windows]
>     return max(scores, key=lambda x: x.confidence)
> ```

---

## 5. Tóm tắt Pipeline Tiền Xử Lý đề xuất

```
Video thô
    │
    ├─ [FILTER]  Loại clip < 5s hoặc > 120s, size < 0.5 MB
    │
    ├─ [DECODE]  Decode với PyAV / decord (không cv2 — chậm hơn)
    │
    ├─ [SAMPLE]  Lấy N=8 frames uniform theo THỜI GIAN (không theo frame index)
    │            - Train: random temporal crop từ duration
    │            - Test:  sliding window, stride 50%
    │
    ├─ [RESIZE]  Short-side resize → 256px, rồi CenterCrop 224×224
    │            (giữ aspect ratio, tránh distort)
    │
    ├─ [AUGMENT] Chỉ khi training:
    │            - RandomHorizontalFlip (p=0.5)
    │            - ColorJitter (brightness=0.3, contrast=0.3, saturation=0.2)
    │            - RandomGrayscale (p=0.1) — train robustness với IR camera
    │            - Temporal speed jitter ×[0.8, 1.25]
    │
    ├─ [NORM]    Normalize với ImageNet mean/std
    │            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    │
    └─ [BALANCE] WeightedRandomSampler với weights [0.85, 1.12, 0.98]
                 (bù đắp label 1 bị thiếu)
```

---

## 6. Số liệu tham chiếu sau pipeline

| Metric | Trước | Sau pipeline (ước tính) |
|---|---|---|
| Videos train | 1,001 | ~960–975 (sau lọc clip ngắn) |
| Videos test | 1,066 | ~3,000–4,000 windows (sliding window) |
| Resolution | 15+ loại | Chuẩn hóa 224×224 |
| Duration | 3.6–278s | 5–120s (filter) |
| FPS | 24–120fps | Không quan trọng (sample theo giây) |
| Class ratio train | 38.6/29.2/32.3 | ~33/33/33 (sau WeightedSampler) |
