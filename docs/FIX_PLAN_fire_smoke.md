# FIX PLAN — Fire/Smoke Two-Stage SegFormer (Stage 1 + Stage 2)

> Tài liệu này là **đặc tả công việc cho coding agent** (Antigravity CLI). Đọc toàn bộ trước khi sửa.
> Repo: `Fire_Smoke_Early-Detection/` — file chính: `src/train_stage1_segformer.py`, `src/train_stage2_timesformer.py`.

---

## 0. Quy tắc bắt buộc cho agent

1. **KHÔNG** tự chạy `modal run` (train trên GPU tốn tiền). Chỉ chạy test cục bộ trên CPU. Khi cần chạy Modal, dừng lại và in lệnh cho người dùng.
2. **KHÔNG** xóa dữ liệu, checkpoint hay manifest. Chỉ di chuyển vào thư mục archive khi được đặc tả yêu cầu.
3. **KHÔNG** đổi siêu tham số (lr, epochs, class weights, batch size, T=8, kích thước 512) trừ khi task ghi rõ.
4. Mỗi task = 1 commit riêng, message dạng `fix(stage2): T1 resume ordering`.
5. Sau mỗi task, chạy `pytest tests/ -q` và đảm bảo pass trước khi sang task tiếp.
6. Nếu một task phụ thuộc thông tin không có trong repo (vd. cách tạo `dataset_video_prepared`), **ghi vào `FIX_NOTES.md`** và hỏi người dùng thay vì đoán.
7. Code và comment viết tiếng Anh; log `print` giữ tiếng Việt như hiện tại cũng được.

---

## 1. Thứ tự thực hiện

| Ưu tiên | Task | File | Ảnh hưởng kết quả cũ? |
|---|---|---|---|
| P0 | T0 Tách code dùng chung ra `src/fire_core.py` | mới + 2 file train | Không |
| P0 | T1 Sửa thứ tự lưu checkpoint / resume | stage 2 | Không (chỉ an toàn hơn) |
| P0 | T2 Archive checkpoint khi chạy mới, gắn `run_id` | stage 1 + 2 | Không |
| P0 | T3 Kiểm tra giá trị mask (không `clip` mù) | core | Có thể |
| P1 | T4 Sửa motion term trong tiền xử lý | core | Có (Stage 2) |
| P1 | T5 Thống nhất letterbox giữa 2 stage | core + stage 2 | Có (Stage 2) |
| P1 | T6 Đóng băng running stats BN của decode head ở Stage 2 | core | Có (Stage 2) |
| P1 | T7 Temporal module causal + positional embedding | core | Có — **checkpoint Stage 2 cũ không resume được** |
| P1 | T8 Dice theo batch, bỏ lớp nền, ignore pad | core | Có |
| P1 | T9 Temporal smoothness chỉ phạt nơi GT ổn định | core | Có (Stage 2) |
| P2 | T10 Grad accumulation tail, pin version, entrypoint, history log | 2 file train | Nhỏ |
| P0 | T11 Script audit dữ liệu | `scripts/audit_data.py` | Không |
| P0 | T12 Script đánh giá test chuẩn + ablation | `src/evaluate_test.py` | Không |
| P0 | T13 Unit tests | `tests/test_core.py` | Không |

Làm T0 → T1 → T2 → T13 (khung test) → T3 → T11 → các task P1 → T10 → T12.

---

## T0. Tách code dùng chung ra `src/fire_core.py`

**Vấn đề:** Toàn bộ model, dataset, loss, tiền xử lý đang được định nghĩa *bên trong* hàm Modal → không unit-test được, và hai file có bản sao `enhance_smoke_and_fire_v2` có thể lệch nhau.

**Yêu cầu:**
- Tạo `src/fire_core.py` chứa: `IMAGENET_MEAN/STD`, `letterbox_image_and_mask`, `enhance_base`, `apply_motion`, `validate_mask`, `StaticSegDataset`, `VideoClipDataset`, `TemporalRefinementModule`, `SpatioTemporalSegFormer`, `seg_ce_dice_loss`, `SpatioTemporalLoss`, `confusion_matrix_3c`, `iou_dice_from_cm`.
- Hai file train import từ `fire_core`. Đưa module vào image Modal:
  - Kiểm tra version: `python -c "import modal; print(modal.__version__)"`.
  - Nếu có `Image.add_local_python_source` → dùng `.add_local_python_source("fire_core")` (cần `src/` trong `sys.path` khi chạy, hoặc dùng `add_local_file("src/fire_core.py", "/root/fire_core.py")`).
  - Ghi cách đã chọn vào `FIX_NOTES.md`.
- Trong file train, import `fire_core` **bên trong** hàm Modal (hoặc dùng `with image.imports():`) để máy local không cần đủ dependency khi chỉ deploy.
- Hành vi phải **giữ nguyên** sau T0 (refactor thuần). Các task sau mới thay đổi logic.

**Nghiệm thu:** `python -c "import sys; sys.path.insert(0,'src'); import fire_core"` chạy được; diff logic = 0.

---

## T1. Sửa thứ tự lưu checkpoint / resume (Stage 2) — BUG

**Vấn đề:** `LATEST_CHECKPOINT` được lưu **trước** khi cập nhật `best_val_miou / best_fire_iou / best_smoke_iou`. Nếu epoch e vừa lập kỷ lục X rồi job bị ngắt, khi resume kỷ lục được khôi phục thành giá trị cũ Y < X → epoch sau với Y < Z < X sẽ **ghi đè file best bằng model kém hơn**.

**Sửa:** chuyển khối `torch.save(..., LATEST_CHECKPOINT)` xuống **cuối** vòng epoch, sau 3 khối `if` lưu best. Lưu thêm `run_id`, `config` (xem T2) và trạng thái RNG:

```python
# ... 3 khối cập nhật best_* và lưu BEST/FIRE/SMOKE ở trên ...
torch.save({
    "epoch": epoch,
    "run_id": RUN_ID,
    "config": CONFIG,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
    "best_val_miou": best_val_miou,
    "best_fire_iou": best_fire_iou,
    "best_smoke_iou": best_smoke_iou,
    "best_epoch": best_epoch,
    "fg_miou": fg_miou, "iou_fire": iou_fire, "iou_smoke": iou_smoke,
    "gamma": cur_gamma,
    "rng": {"py": random.getstate(), "np": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
}, LATEST_CHECKPOINT)
volume.commit()
```

Khi resume: khôi phục RNG nếu có; **từ chối resume** nếu `resume_data["config"]` khác `CONFIG` ở các khóa kiến trúc (`causal`, `temporal_hidden`, `num_heads`, `letterbox`, `preprocess_version`) — in lỗi rõ ràng.

**Nghiệm thu:** test giả lập (mock `torch.save`) cho thấy giá trị `best_val_miou` trong latest = giá trị sau cập nhật của epoch đó.

---

## T2. Archive checkpoint khi chạy mới + `run_id`

**Vấn đề:** Chạy không `--resume` reset kỷ lục về 0 nhưng chỉ backup 1 file → `best`, `best_fire`, `best_smoke`, `latest` có thể đến từ **các lần chạy khác nhau** (báo cáo hiện có mâu thuẫn: epoch 13 có Smoke IoU 59.28% > file `best_smoke` 54.91%).

**Sửa (cả Stage 1 và Stage 2):**
- `RUN_ID = datetime.utcnow().strftime("%Y%m%d-%H%M%S")`.
- `CONFIG = {...}` dict chứa toàn bộ siêu tham số + `preprocess_version` + `git_commit` (truyền từ local entrypoint bằng `git rev-parse --short HEAD`, fallback `"unknown"`).
- Khi **không** resume: nếu tồn tại bất kỳ file nào trong `{BEST, BEST_FIRE, BEST_SMOKE, LATEST}` → `shutil.move` tất cả vào `/data/checkpoints/archive/<old_run_id hoặc 'legacy'>_<timestamp>/`, rồi `volume.commit()`.
- Xóa logic `backup_before_enhanced_preprocess` cũ (thay bằng cơ chế trên).
- Mọi checkpoint lưu `run_id` và `config`.

---

## T3. Kiểm tra mask thay vì `np.clip(mask, 0, 2)`

**Vấn đề:** Nếu mask nào lưu dạng 0/255, `clip` biến 255 → 2 (khói) một cách im lặng.

**Sửa:** trong `fire_core.validate_mask(mask, path)`:
```python
vals = np.unique(mask)
if not set(vals.tolist()).issubset({0, 1, 2}):
    raise ValueError(f"Mask {path} có giá trị ngoài {{0,1,2}}: {vals[:10]}")
return mask.astype(np.uint8)
```
Dùng ở cả 2 dataset. Khi ảnh/mask đọc lỗi (`None`): **raise** thay vì trả ảnh đen (lỗi đọc file phải lộ ra). Có cờ `strict=True` mặc định.

---

## T4. Sửa motion term trong tiền xử lý

**Vấn đề:** `prev_enhanced` lưu ảnh **đã nhân motion boost**, rồi so với ảnh hiện tại **chưa boost**; frame đầu mỗi clip không có boost → cùng một frame vật lý được xử lý khác nhau tùy vị trí trong cửa sổ trượt.

**Sửa:** tách thành 2 hàm, so sánh trên ảnh *base*:
```python
def enhance_base(img_bgr):
    # bước 1–4 hiện tại (CLAHE đa tỉ lệ, neutrality, DCP, smoke_boost) — GIỮ NGUYÊN THAM SỐ
    ...
    return enhanced_bgr

def apply_motion(cur_base, prev_base, strength=0.15):
    if prev_base is None:
        return cur_base
    diff = cv2.absdiff(cur_base, prev_base)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    w = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32) / 255.0 * strength
    return np.clip(cur_base.astype(np.float32) * (1.0 + w[:, :, None]), 0, 255).astype(np.uint8)
```
Trong `VideoClipDataset`: `base = enhance_base(img)`; `img = apply_motion(base, prev_base)`; `prev_base = base`.

Thêm cờ `CONFIG["use_motion"]` (mặc định `True`) để ablation. Stage 1 chỉ dùng `enhance_base` (tương đương hành vi cũ → **không cần train lại Stage 1 vì T4**). Giữ `enhance_smoke_and_fire_v2` làm wrapper tương thích ngược. Đặt `PREPROCESS_VERSION = "v3.1"`.

---

## T5. Thống nhất letterbox giữa 2 stage

**Vấn đề:** Stage 1 letterbox (giữ tỉ lệ, pad 114); Stage 2 `cv2.resize` thẳng về 512×512 → méo ảnh (đặc biệt video dọc 1080×1920). Encoder Stage 2 đóng băng nên không thích nghi được.

**Việc cần làm:**
1. Chạy T11 trước để biết frame trong `/data/dataset_video_prepared` có kích thước gì.
2. Nếu frame **chưa** là 512×512 → trong `VideoClipDataset`, làm đúng thứ tự Stage 1: `enhance_base` trên ảnh gốc → `letterbox_image_and_mask` (pad ảnh 114, pad mask = `IGNORE_INDEX = 255`).
3. Nếu frame **đã** là 512×512 (đã bị resize méo ở bước chuẩn bị dữ liệu) → **dừng**, ghi vào `FIX_NOTES.md`, hỏi người dùng script tạo `dataset_video_prepared` để sửa bước đó sang letterbox và sinh lại dữ liệu. Không tự sinh lại.
4. Áp dụng `IGNORE_INDEX = 255` cho vùng pad ở **cả Stage 1** (thay pad mask = 0) để metric không đếm pad là nền.

---

## T6. Đóng băng running stats BN của decode head (Stage 2)

**Vấn đề:** `decode_head.batch_norm` đang ở chế độ train với batch = 8 frame gần giống nhau của 1 clip → running mean/var trôi khỏi thống kê tốt của Stage 1.

**Sửa:** override `train()` trong `SpatioTemporalSegFormer`:
```python
def train(self, mode: bool = True):
    super().train(mode)
    if self.freeze_backbone:
        self.segformer.eval()
    if self.freeze_decode_bn:          # mặc định True
        self.decode_head.batch_norm.eval()   # vẫn học affine weight/bias
    return self
```
Xóa dòng `model.segformer.eval()` thủ công trong vòng train (đã có trong override).

---

## T7. Temporal module causal + positional embedding

**Vấn đề:** Attention hai chiều + `Conv1d(padding=1)` đối xứng → dự đoán ở frame t dùng frame tương lai; frame cuối (frame cần cảnh báo khi streaming) nằm ở biên padding. Không có positional embedding → attention gần như bất biến thứ tự.

**Sửa:**
```python
class TemporalRefinementModule(nn.Module):
    def __init__(self, in_channels=768, hidden_channels=128, num_heads=4,
                 dropout=0.1, max_T=16, causal=True):
        super().__init__()
        self.causal = causal
        self.in_proj = nn.Conv2d(in_channels, hidden_channels, 1)
        self.pool = nn.AvgPool2d(2, 2)
        self.temp_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_T, hidden_channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # norm1, temp_attn, norm2, mlp, out_proj, gamma: giữ nguyên như cũ

    def forward(self, z, B, T):
        BT, C, H, W = z.shape
        feat = self.pool(self.in_proj(z))
        _, hid, h, w = feat.shape
        x = feat.view(B, T, hid, h, w).permute(0, 3, 4, 2, 1).reshape(B * h * w, hid, T)
        pad = (2, 0) if self.causal else (1, 1)
        x = x + self.temp_conv(F.pad(x, pad))
        seq = x.permute(0, 2, 1) + self.pos_embed[:, :T]
        mask = None
        if self.causal:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=z.device), diagonal=1)
        n = self.norm1(seq)
        a, _ = self.temp_attn(n, n, n, attn_mask=mask, need_weights=False)
        seq = seq + a
        seq = seq + self.mlp(self.norm2(seq))
        out = seq.view(B, h, w, T, hid).permute(0, 3, 4, 1, 2).reshape(BT, hid, h, w)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return z + self.gamma * self.out_proj(out)
```
- Thêm `CONFIG["causal"] = True`. Với γ = 0, output vẫn đúng bằng Stage 1 (test ở T13).
- Ghi rõ trong log: checkpoint Stage 2 cũ **không resume được** sau T7 (T1 đã chặn).
- Sửa docstring sai ("16 frames" → "T frames").

---

## T8. Dice loss theo batch, bỏ lớp nền, ignore pad (cả 2 stage)

**Vấn đề:** Dice tính theo từng ảnh và gồm lớp nền; trên ảnh âm tính, Dice lớp lửa/khói = `1/(Σp+1)` → vài pixel sai đã làm loss ≈ 1, rất nhiễu, góp phần sinh nhiều đốm nhỏ (over-segmentation, FAR cao).

**Sửa:** một hàm dùng chung:
```python
IGNORE_INDEX = 255

def dice_loss_fg(logits, targets, classes=(1, 2), smooth=1.0):
    probs = F.softmax(logits.float(), dim=1)
    valid = (targets != IGNORE_INDEX)
    losses = []
    for c in classes:
        p = probs[:, c] * valid
        t = ((targets == c) & valid).float()
        inter = (p * t).sum()
        card = p.sum() + t.sum()
        losses.append(1.0 - (2 * inter + smooth) / (card + smooth))
    return torch.stack(losses).mean()

def seg_ce_dice_loss(logits, targets, class_weights, dice_weight=1.0):
    ce = F.cross_entropy(logits, targets, weight=class_weights, ignore_index=IGNORE_INDEX)
    dl = dice_loss_fg(logits, targets)
    return ce + dice_weight * dl, ce, dl
```
Thêm `CONFIG["dice_mode"] = "batch_fg"`, giữ được chế độ cũ `"per_image_all"` qua cờ để so sánh. Class weights giữ nguyên.

---

## T9. Temporal smoothness chỉ phạt nơi GT không đổi

**Vấn đề:** Loss hiện phạt mọi thay đổi xác suất giữa frame, kể cả lúc lửa/khói **bắt đầu xuất hiện** — trái với mục tiêu phát hiện sớm.

**Sửa** (trong `SpatioTemporalLoss`):
```python
probs_seq = probs.view(b, t, c, h, w)
diff = (probs_seq[:, 1:] - probs_seq[:, :-1]).abs().sum(dim=2)      # (b, t-1, h, w)
tg = targets.view(b, t, h, w)
stable = ((tg[:, 1:] == tg[:, :-1]) & (tg[:, 1:] != IGNORE_INDEX)).float()
loss_temp = (diff * stable).sum() / stable.sum().clamp_min(1.0)
```
Trọng số `0.05` giữ nguyên.

---

## T10. Các sửa nhỏ

1. **Grad accumulation tail** (Stage 2): `if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):`.
2. **Pin version**: cả 2 file dùng `transformers==4.40.0` (Stage 2 đang 4.38.2).
3. **Entrypoint Stage 1**: thêm `@app.local_entrypoint() def main(): train_stage1.remote(git_commit=...)`.
4. **History log**: mỗi epoch append dict `{epoch, train_loss, ce, dice, temp, val_loss, iou_bg, iou_fire, iou_smoke, fg_miou, gamma, lr_groups}` vào `/data/logs/<stage>_history_<RUN_ID>.json`, `volume.commit()`.
5. **Metric Stage 1**: tính bằng confusion matrix 3×3 tích lũy (bỏ pixel 255) thay vì `torch.cat` toàn bộ pred (đỡ ~800 MB RAM).
6. **Metric Stage 2 val**: thêm IoU nền và All-mIoU vào log.
7. Xóa việc lưu `mit_b2_spatial_backbone.pth` nếu không nơi nào dùng (grep repo trước; nếu có dùng thì giữ).

---

## T11. `scripts/audit_data.py` (chạy được trên Modal và local)

Xuất `docs/reports/data_audit_<date>.json` + in tóm tắt. Kiểm tra:

1. **Rò rỉ video**: tập video_id của train/val/test (manifest frame và clips) phải rời nhau. Liệt kê giao nếu có.
2. **Rò rỉ ảnh tĩnh**: perceptual hash (`imagehash.phash`) cho mọi ảnh static; báo các cặp khác split có Hamming ≤ 4. Đặc biệt so `static_v4i_val_manifest.csv` với `static_v4i_test_manifest.csv` (số ảnh mỗi lớp trùng khít 82/49/27 — cần xác nhận không trùng ảnh).
3. **Giá trị mask**: `np.unique` cho mọi mask static + video; liệt kê file có giá trị ngoài {0,1,2}.
4. **Kích thước frame** trong `dataset_video_prepared` (min/max/phân bố) — phục vụ T5.
5. **Thống kê pixel** tính lại cho video train/val/test; kiểm tra ràng buộc `BG_pixels ≥ số_frame_rỗng × H × W` (báo cáo hiện tại vi phạm ở tập test: 36.64% nền nhưng 105/208 frame rỗng).
6. **Clips**: số clip mỗi video, stride suy ra, số frame/video; giải thích vì sao val 48 clip từ 17 video.
7. **Ảnh minh họa**: với mỗi file trong `docs/inference_comparisons*/`, trích video_id từ tên và in split của nó (nhiều ảnh hiện đang là video tập **train**).

---

## T12. `src/evaluate_test.py` — đánh giá chuẩn + ablation

Mục tiêu: sinh **một** file kết quả duy nhất, nhất quán, dùng để viết lại báo cáo.

**Cấu hình đánh giá (chạy cả 3 trên cùng tập test video):**
- `stage1_framewise`: Stage 1 checkpoint, xử lý từng frame độc lập.
- `stage2_gamma0`: Stage 2 checkpoint nhưng ép `gamma = 0` (đo đóng góp của fine-tune decode head).
- `stage2_full`: Stage 2 checkpoint nguyên vẹn.

**Yêu cầu:**
1. Mỗi frame duy nhất `(video_id, frame_idx)` chỉ được đếm **một lần**. Với Stage 2, lấy dự đoán từ clip mà frame đó ở **vị trí muộn nhất** (giả lập streaming causal).
2. **Pixel-level**: confusion matrix 3×3 toàn cục (bỏ 255) → IoU, Dice từng lớp, Fg-mIoU, All-mIoU.
3. **Frame-level**: frame dương với lớp c nếu `area(prob_c > τ) ≥ A_min`. Grid `τ ∈ {0.3,0.4,0.5,0.6,0.7}`, `A_min ∈ {0,50,200,500,1000,2000}` px. **Chọn trên val** (tiêu chí: recall ≥ 0.95, FAR nhỏ nhất; nếu không đạt thì F1 lớn nhất), rồi **cố định** áp lên test. Báo TP/FP/TN/FN, Recall, FAR, Precision, F1, MCC, và **base rate** dương.
4. **Video-level**: video dương nếu ≥ k frame liên tiếp dương (k ∈ {1,2,3}, chọn trên val).
5. **Object-level**: giữ LTPR/LPPV/S hiện có, sau khi lọc vùng < `A_min`.
6. **Bootstrap 95% CI** (1000 lần, resample theo **video**) cho Fg-mIoU, Smoke IoU, Fire IoU, frame Recall/FAR.
7. **Bảng theo từng video** (IoU từng lớp, FP frames) — giúp nhìn thấy video nào kéo điểm.
8. **Tốc độ**: đo riêng (a) tiền xử lý CPU/frame, (b) GPU/clip, (c) end-to-end streaming stride=1 với **cache đặc trưng từng frame** (encoder + linear_fuse chạy 1 lần/frame, chỉ temporal + classifier chạy lại). Ghi tên GPU, batch, AMP.
9. Ghi vào output: đường dẫn checkpoint, `run_id`, epoch, `gamma`, `config`, git commit, τ/A_min đã chọn.
10. Output: `docs/reports/eval_<run_id>.json` + `docs/reports/eval_<run_id>.md` (bảng gọn).

Hỗ trợ chạy local (CPU, `--limit N` clip) để debug và chạy Modal cho bản đầy đủ.

---

## T13. `tests/test_core.py` (CPU, không cần internet)

Dùng `SegformerConfig(num_labels=3)` khởi tạo ngẫu nhiên (cấu hình nhỏ mặc định), ảnh 128×128, T=4.

1. **Shape**: `SpatioTemporalSegFormer` trả `(B, T, 3, H, W)`.
2. **Tương đương γ=0**: ở `eval()`, logits Stage 2 (γ=0) == logits HF `SegformerForSemanticSegmentation` upsample về H×W, từng frame (`atol=1e-5`).
3. **Causal**: đặt `gamma=1.0`, `eval()`; thay đổi frame cuối → output các frame trước **không đổi**; với `causal=False` thì có đổi.
4. **BN đóng băng**: sau `model.train()`, `decode_head.batch_norm.training is False` và `running_mean` không đổi sau 1 forward.
5. **Dice**: pixel 255 không ảnh hưởng loss; ảnh toàn nền + dự đoán toàn nền cho Dice loss ≈ 0 theo batch.
6. **Smoothness**: khi GT đổi lớp tại một pixel giữa 2 frame, thay đổi dự đoán ở pixel đó không làm tăng `loss_temp`.
7. **Motion**: `apply_motion(x, None) is x`; cùng một frame gốc, ảnh base không phụ thuộc vị trí trong clip.
8. **validate_mask**: mask chứa 255 → `ValueError`.
9. **Resume ordering (T1)**: hàm nhỏ `build_latest_payload(...)` được gọi sau cập nhật best — test giá trị.

---

## 2. Sau khi agent hoàn thành — việc của người dùng

1. Chạy `scripts/audit_data.py` trên Modal → xử lý mọi rò rỉ/mask lỗi trước khi train.
2. Train lại Stage 1 (vì T5 pad=255 và T8 đổi loss): `modal run --detach src/train_stage1_segformer.py`.
3. Train Stage 2 từ đầu: `modal run --detach src/train_stage2_timesformer.py` (không `--resume`).
4. Chạy `src/evaluate_test.py` cho 3 cấu hình.
5. Viết lại báo cáo **chỉ** từ `eval_<run_id>.json`: một run, một bộ checkpoint, có CI, có baseline Stage 1 frame-wise, ảnh minh họa lấy từ video tập **test**.

## 3. Định nghĩa "xong"

- [ ] Tất cả test T13 pass.
- [ ] `FIX_NOTES.md` liệt kê mọi quyết định/giả định và câu hỏi còn mở.
- [ ] Không còn định nghĩa model/loss/tiền xử lý trùng lặp giữa hai file train.
- [ ] Mỗi checkpoint mới có `run_id` + `config`; resume khác kiến trúc bị chặn.
- [ ] `evaluate_test.py --limit 2` chạy được local trên CPU và sinh đủ file JSON + MD.
