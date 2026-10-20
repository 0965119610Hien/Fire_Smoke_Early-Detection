# FIX_NOTES — triển khai `docs/FIX_PLAN_fire_smoke.md`

Ngày: 2026-10-19. Tất cả test `python -m pytest tests/ -q` pass (20 test, CPU).
Không chạy `modal run`, không xóa / sinh lại dữ liệu, checkpoint hay manifest.

## 0. Quyết định chung

- **Không commit** (theo lựa chọn của người dùng): thư mục làm việc đang có phần sắp xếp lại
  chưa commit (`src/`, `utils/`, `docs/` untracked, ~45 file bị xóa). Thay vì 1 commit/task,
  mọi thay đổi nằm trong working tree để người dùng tự xem diff và commit.
- Vì không commit theo từng task, code được viết thẳng ở trạng thái cuối (T0 → T13), nhưng các
  chế độ cũ vẫn bật lại được bằng cờ để làm ablation: `dice_mode="per_image_all"`,
  `use_motion=False`, `letterbox=False`, `causal=False`.
- File mới / sửa:
  - mới: `src/fire_core.py`, `src/evaluate_test.py`, `scripts/audit_data.py`, `tests/test_core.py`, `FIX_NOTES.md`
  - sửa: `src/train_stage1_segformer.py`, `src/train_stage2_timesformer.py`
  - report sinh ra: `docs/reports/data_audit_2026-10-19.json`, `docs/reports/eval_legacy_limit2.{json,md}`
- Môi trường local: Python 3.11, torch 2.12 CPU, transformers 5.8.1; `modal` **chưa cài** ở local
  nên không kiểm tra được `modal.__version__`. Đã cài thêm `pytest` (pip).

## T0 — `src/fire_core.py`

- Toàn bộ model / dataset / loss / tiền xử lý / metric nằm trong `fire_core`; hai file train chỉ còn
  cấu hình + vòng lặp. Không còn định nghĩa trùng lặp giữa hai file train.
- Đưa vào image Modal bằng `.add_local_file(Path(__file__).parent / "fire_core.py", "/root/fire_core.py")`
  (đặt **cuối** chuỗi build image, sau `pip_install`/`env`, theo yêu cầu của Modal ≥ 0.70).
  Chọn `add_local_file` thay vì `add_local_python_source` vì không phụ thuộc `sys.path` lúc deploy.
  `/root` nằm trong `sys.path` của container. `import fire_core` đặt **bên trong** hàm Modal.
  ⚠️ Cần Modal ≥ 0.67 (có `Image.add_local_file`). Nếu bản Modal của bạn cũ hơn: `pip install -U modal`.
- `build_segformer_b2(pretrained=False)` dựng đúng kiến trúc MiT-B2 offline (để nạp checkpoint khi
  đánh giá mà không cần tải `nvidia/mit-b2`). Đã kiểm tra: checkpoint Stage 1 local nạp `strict=True` OK.

## T1 — Thứ tự checkpoint / resume (Stage 2)

- `fc.epoch_checkpoint_step(...)`: cập nhật kỷ lục → lưu best/fire/smoke → lưu LATEST **cuối cùng**.
  LATEST chứa `run_id`, `config`, optimizer/scheduler/scaler, kỷ lục *sau cập nhật*, `rng`.
- Resume: nạp với `map_location="cpu"` (cần để khôi phục RNG), khôi phục RNG, và **từ chối** nếu
  `config` thiếu hoặc khác ở `ARCH_CONFIG_KEYS = (causal, temporal_hidden, num_heads, max_T, letterbox, preprocess_version)`
  (thêm `max_T` vì nó quyết định shape của `pos_embed`).
- `--resume` khi không có LATEST → in cảnh báo và chạy mới (có archive).

## T2 — Archive + run_id

- `RUN_ID = UTC %Y%m%d-%H%M%S`; `CONFIG` chứa toàn bộ siêu tham số, `preprocess_version`, `git_commit`.
- `git_commit` lấy ở local entrypoint: `git rev-parse --short HEAD` + hậu tố `-dirty` nếu working tree
  có thay đổi chưa commit (hiện tại sẽ là `33d58d8-dirty`) — fallback `"unknown"`.
- Chạy mới → `shutil.move` mọi file tồn tại vào `/data/checkpoints/archive/<old_run_id|legacy>_<timestamp>/`.
  Stage 1 archive `segformer_spatial_best.pth`; Stage 2 archive best/fire/smoke/latest.
  Đã bỏ `backup_before_enhanced_preprocess`.
- Stage 2 ghi thêm `stage1_run_id` / `stage1_epoch` vào config để truy vết checkpoint Stage 1 đã dùng.
- Lưu ý: Stage 1 archive checkpoint cũ ngay khi bắt đầu — nếu run mới chết trước epoch 1 thì Stage 2
  sẽ không tìm thấy `segformer_spatial_best.pth` (bản cũ vẫn nằm trong `archive/`).

## T3 — Mask

- `validate_mask` raise `ValueError` nếu giá trị ngoài {0,1,2}; `read_image`/`read_mask` raise khi đọc lỗi
  (`strict=True` mặc định). Đã bỏ `np.clip(mask, 0, 2)`.
- Audit local: **toàn bộ** mask static + video chỉ có {0,1,2}, mask khớp kích thước ảnh.

## T4 — Motion

- `enhance_base` (bước 1–4, giữ nguyên tham số) + `apply_motion(cur_base, prev_base)`; so sánh trên ảnh base.
  `enhance_smoke_and_fire_v2` giữ làm wrapper (tham số `prev_frame_bgr` giờ là *base* frame trước).
  `PREPROCESS_VERSION = "v3.1"`, `CONFIG["use_motion"] = True`. Stage 1 chỉ dùng `enhance_base` (hành vi cũ).
- Motion được tính ở **độ phân giải gốc** (trước letterbox), cùng thứ tự với Stage 1.

## T5 — Letterbox

- Audit: frame trong `dataset_video_prepared` **không** phải 512×512 (240×426 → 3840×2160; phổ biến
  640×640: 487, 720×405: 432, 1920×1080: 261, 1080×1920: 116, 2160×3840: 89) → làm theo nhánh 2:
  `enhance_base` trên ảnh gốc → motion → `letterbox_image_and_mask` (ảnh pad 114, mask pad 255).
- Pad mask = `IGNORE_INDEX = 255` ở **cả hai stage**; CE dùng `ignore_index=255`, Dice và metric bỏ pad.
- ⚠️ Câu hỏi mở: 487 frame 640×640 có thể đã bị Roboflow resize méo (stretch) từ video gốc 16:9 lúc
  export. Nếu đúng, letterbox không sửa được méo của các frame này — cần xem lại bước export Roboflow.
- Tiền xử lý ở độ phân giải gốc chậm hơn với video 4K (CPU ~0.3 s/frame 1080p) — chấp nhận để
  thống nhất với Stage 1; nếu DataLoader thành nút cổ chai, tăng `num_workers`.

## T6 — BN decode head

- `SpatioTemporalSegFormer.train()` giữ `segformer` và `decode_head.batch_norm` ở eval (vẫn học affine).
  Đã xóa dòng `model.segformer.eval()` thủ công trong vòng train.

## T7 — Temporal causal + positional embedding

- Đúng như đặc tả (Conv1d pad trái 2, `attn_mask` tam giác trên, `pos_embed` `max_T=16`).
  Test: γ=0 ⇒ bằng HF Stage 1 (atol 1e-5); causal ⇒ đổi frame cuối không đổi output frame trước.
- **Checkpoint Stage 2 cũ không resume được** (thiếu `config` → T1 chặn).
- Để vẫn đánh giá được checkpoint cũ: `load_stage2_state_dict(..., legacy_ok=True)` nạp checkpoint
  thiếu `pos_embed` vào module `causal=False` với `pos_embed = 0` — tương đương chính xác module cũ
  (pad đối xứng (1,1) + conv padding 0 ≡ conv padding 1). Chỉ dùng trong `evaluate_test.py`.

## T8 — Dice

- `dice_loss_fg` theo batch, chỉ lớp 1/2, bỏ pad; `seg_ce_dice_loss(..., dice_mode)` dùng cho cả hai stage.
  Chế độ cũ `per_image_all` vẫn có (đã thêm bỏ pad). Class weights giữ nguyên `[0.1, 3.5, 2.0]`.
- CE/softmax tính ở float32 (`logits.float()`), an toàn với AMP.

## T9 — Temporal smoothness

- Chỉ phạt nơi GT ổn định và không phải pad, trọng số 0.05 giữ nguyên.
- Lưu ý: giá trị loss_temp mới = tổng |Δp| theo kênh / số pixel ổn định, nên **thang đo khác** loss cũ
  (`diff.mean()` chia cả cho số lớp C=3) — với cùng trọng số 0.05, đóng góp thực tế lớn hơn ~3 lần
  trên vùng ổn định. Giữ đúng đặc tả; cân nhắc khi so sánh log loss với run cũ.

## T10 — Sửa nhỏ

1. Grad accumulation tail: bước optimizer cả ở batch cuối epoch.
2. Stage 2 pin `transformers==4.40.0` (cùng Stage 1). Image evaluation cũng dùng 4.40.0.
3. Stage 1 có `@app.local_entrypoint() main()` truyền `git_commit`.
4. History: `/data/logs/stage{1,2}_history_<RUN_ID>.json` (append mỗi epoch, `volume.commit()` mỗi epoch).
5. Metric Stage 1 = confusion matrix 3×3 tích lũy (bỏ 255), không còn `torch.cat` toàn bộ pred.
6. Val Stage 2 log thêm IoU nền + All-mIoU.
7. `mit_b2_spatial_backbone.pth`: grep cả repo — chỉ Stage 1 ghi file này, không script nào đọc
   (chỉ README nhắc tới) → đã bỏ việc lưu. README chưa sửa (README đang có thay đổi chưa commit của bạn).
- IoU của lớp không xuất hiện (union = 0) giờ là `NaN` (bỏ qua khi lấy trung bình) thay vì 1.0 như cũ.

## T11 — `scripts/audit_data.py` (đã chạy local, report: `docs/reports/data_audit_2026-10-19.json`)

Chạy: `python scripts/audit_data.py` (local) hoặc `modal run scripts/audit_data.py`.

Kết quả và **việc cần người dùng quyết định trước khi train lại**:

1. **Rò rỉ video theo nguồn**: không trùng `video_id`, nhưng 2 chunk của cùng một video gốc nằm ở 2 split:
   `2016 April - Fire Investigation 1B - Milpitas CA_pure_chunk_02` (train) và `..._chunk_03` (val).
   → Đề xuất chuyển cả hai về cùng một split. (Có 100 thư mục video = 68 train / 18 val / 14 test.)
2. **Rò rỉ ảnh tĩnh nghiêm trọng**: 167 cặp khác split có phash Hamming ≤ 4 (32 cặp hash trùng khít):
   train&val 63, train&test 89, val&test 15; thêm 9 ảnh gốc Roboflow (cùng tên nguồn, khác hash `.rf.`)
   nằm ở nhiều split. Val và test có số ảnh mỗi lớp trùng khít (82/49/27) nhưng **test không có ảnh
   `empty`** (val có 30) và có 15 cặp gần trùng val–test. → Cần chia lại static theo nhóm (phash cluster /
   tên nguồn) rồi train lại Stage 1. Danh sách cặp nằm trong report (`static_leakage.pairs`).
3. Mask: OK (xem T3).
4. Kích thước frame: xem T5.
5. Thống kê pixel video (tính lại từ mask, độ phân giải gốc):
   test: 14 video, 208 frame, 105 frame rỗng (50.5%), **BG 90.09%** / Fire 3.24% / Smoke 6.67%.
   Ràng buộc BG ≥ pixel của frame rỗng thỏa ở cả 3 split → con số 36.64% nền trong báo cáo cũ là sai.
   Manifest khớp mask (0 sai lệch số pixel).
6. **Clips lệch manifest**: `train_clips.csv` thiếu 4 video
   (`12150804_1080_1920_30fps_pure`, `12935692_1080_1920_60fps_pure`, `14003137_2160_3840_30fps_pure`,
   `7cc52846_pexels_heavy_smoke_35241029`), `val_clips.csv` thiếu `14477514_1080_1920_30fps_pure`
   → file clips được sinh từ manifest cũ hơn. Ngoài ra stride bỏ mất đuôi video: train 49, val 32,
   test **8** frame không nằm trong clip nào (không được đánh giá). Val 48 clip / 17 video:
   mỗi video n ≥ 8 frame cho ⌊(n−8)/4⌋+1 clip; video n < 8 cho 1 clip pad.
   → Đề xuất chạy lại `src/create_split_manifests_and_clips.py` (sau khi sửa mục 1) và cân nhắc thêm
   1 clip neo ở cuối mỗi video để phủ đủ frame. **Chưa tự sinh lại** (quy tắc 2).
7. Ảnh minh họa: 50 ảnh trong `docs/inference_comparisons*/` thuộc video **test**; 8 ảnh
   (`sample_26..33_hard_negative_neg_hard_negativ_*`) có tên bị cắt ngắn, khớp với video
   `neg_hard_negatives_*` ở cả 3 split → không xác định được. Nên đặt tên đầy đủ video khi xuất ảnh.

## T12 — `src/evaluate_test.py`

- Local: `python src/evaluate_test.py --limit 2` (CPU, ~4 phút) → `docs/reports/eval_<run_id>_limit2.{json,md}`.
  Modal: `modal run src/evaluate_test.py` (GPU A10G) → ghi `docs/reports/eval_<run_id>.{json,md}` ở local.
- Mặc định local dùng `checkpoints/segformer_spatial_best.pth` và
  `checkpoints/checkpoints/timesformer_spatiotemporal_best.pth` (checkpoint cũ → chế độ legacy, có cảnh báo,
  `run_id = legacy`). Kết quả `eval_legacy_limit2.*` **chỉ để kiểm tra script**, không dùng cho báo cáo.
- Giả định:
  - Frame dương khi `area(prob_c > τ) ≥ max(A_min, 1)` — dùng `max(·,1)` vì với A_min = 0 điều kiện
    `area ≥ 0` luôn đúng. Diện tích tính ở độ phân giải mô hình 512×512, không tính vùng pad.
  - Chọn (τ, A_min) và k **riêng cho từng lớp và từng cấu hình** trên val; "any" = báo động nếu lửa hoặc khói.
  - Object-level: vùng dự đoán < max(A_min, 16) px bị loại; vùng GT giữ ngưỡng 16 px như `metrics_calculator`.
  - `stage1_framewise` dùng tiền xử lý Stage 1 (base, không motion, letterbox); Stage 2 dùng config trong checkpoint.
  - Chỉ frame nằm trong ít nhất một clip được đánh giá (8 frame test bị loại, xem T11.6).
- Tốc độ: (a) tiền xử lý CPU/frame ở độ phân giải gốc, (b) model/clip, (c) streaming stride 1 có cache
  đặc trưng (encoder + linear_fuse 1 lần/frame; temporal chạy lại trên cửa sổ, classifier chỉ cho frame mới —
  đã kiểm tra khớp với forward đầy đủ).

## Các file chưa sửa (ngoài phạm vi đặc tả)

`src/evaluate_all_metrics.py`, `src/infer_and_compare_stages.py`, `src/run_inference_40_samples.py`,
`src/evaluate_stage2_local.py`, `src/inference_realtime.py` vẫn tự định nghĩa model/tiền xử lý **cũ**
(non-causal, resize thẳng, motion v2). Chúng **sẽ không nạp được** checkpoint Stage 2 mới (thiếu `pos_embed`)
và cho kết quả không nhất quán. Dùng `src/evaluate_test.py` cho số liệu báo cáo; nếu cần ảnh minh họa,
các script trên nên được chuyển sang import `fire_core` (chưa làm — hỏi trước).

## Checkpoint local (tham khảo)

- `checkpoints/segformer_spatial_best.pth`: Stage 1 epoch 35, Fg-mIoU 67.44% — nạp được.
- `checkpoints/timesformer_spatiotemporal_best.pth`: kiến trúc ViT-TimeSformer cũ hơn (không phải SpatioTemporalSegFormer).
- `checkpoints/checkpoints/timesformer_*`: SpatioTemporalSegFormer cũ; `latest` ghi epoch 49 trong khi
  `NUM_EPOCHS = 35` và best_smoke epoch 17 > best epoch 13 → xác nhận checkpoint đến từ nhiều lần chạy (lý do T2).
- **Hỏng** (đọc lỗi "invalid header or archive is corrupted"): `checkpoints/timesformer_spatiotemporal_latest.pth`,
  `checkpoints/checkpoints/segformer_spatial_best.pth`.

## Sửa dữ liệu (đã thực hiện, `scripts/fix_data.py`, 2026-10-19 10:58)

Người dùng yêu cầu sửa toàn bộ lỗi dữ liệu. Không xóa gì; mọi bản cũ được sao lưu:
- `data/data_splits/archive/20261019-105826/` — toàn bộ 18 file manifest/clips cũ.
- `data/archive/stretched_frames_20261019-105826/` — 28 thư mục video trước khi un-stretch.

1. **Un-stretch 28 video round 2** (`Frames_video.v1i`): frame + mask 640×640 không có viền pad (bị ép tỉ lệ
   khi xuất) được resize về tỉ lệ gốc (cạnh dài 640; ảnh `INTER_AREA`, mask `INTER_NEAREST`). Kích thước gốc
   lấy từ `data/frames_for_annotation/manifest.csv` (fallback: video trong `data/all_videos_raw`).
   Không frame nào đổi nhãn có/không lửa-khói; số pixel trong manifest được tính lại (383 dòng).
2. **Rò rỉ chunk**: `Milpitas CA_pure_chunk_03` val → train (cùng split với `chunk_02`).
   Video: train 69 / val 17 / test 14 (test **không đổi**: 14 video, 208 frame).
3. **Clips** sinh lại bằng `src/create_split_manifests_and_clips.py` (đã thêm clip neo cuối video,
   bỏ đường dẫn tuyệt đối): train 408 / val 52 / test 40 clip; mọi frame nằm trong ≥ 1 clip.
4. **Ảnh tĩnh chia lại theo nhóm**: union-find trên phash Hamming ≤ 4 + cùng ảnh nguồn Roboflow (962 nhóm),
   phân tầng theo category, giữ tỉ lệ cũ, seed 42. Nhóm giống frame của video val/test (Hamming ≤ 6)
   bị ép vào đúng split đó để Stage 1 không bao giờ train trên frame video đánh giá (30 nhóm).
   Kết quả: train 888 / val 192 / test 158 (test nay có 23 ảnh `empty`); 505/1238 ảnh đổi split.
   → **Tập static val/test đã thay đổi**: số liệu Stage 1 static cũ không còn so sánh được.
5. Audit sau khi sửa: không rò rỉ video / chunk / ảnh tĩnh, mask OK, 0 frame ngoài clip, 0 sai lệch manifest.

Còn lại (không sửa được / không thuộc dữ liệu train):
- 196 ảnh tĩnh 640×640 và 82 ảnh 416×416 có thể cũng bị Roboflow ép tỉ lệ, nhưng không có ảnh gốc để khôi phục.
- 8 ảnh minh họa `sample_26..33_hard_negative_*` có tên video bị cắt ngắn nên không xác định split; các ảnh
  minh họa cũ đều sinh từ checkpoint cũ — sinh lại từ checkpoint mới sau khi train.
- Các tài liệu thống kê cũ trong `docs/` (Dataset_Split_Statistics.md, …) và `data/data_splits/split_statistics.json`
  đã lỗi thời; số liệu đúng nằm trong `docs/reports/data_audit_2026-10-19.json`.

## Chạy nền trên Modal (`--detach`)

- Lần chạy Stage 2 đầu tiên (run `20261019-051452`) bị **hủy** khi tắt máy dù có `--detach`: log Modal ghi
  `Received a cancellation signal while processing input`. Nguyên nhân: local entrypoint gọi `.remote()` và
  chờ kết quả; khi client local bị tắt, lời gọi đang chờ bị hủy theo.
- Sửa: cả 3 entrypoint dùng `.spawn()`. Train: gửi job rồi thoát ngay. Evaluate: `spawn()` + `get()`, và
  hàm remote ghi thêm report vào volume `/reports/` để không mất kết quả nếu máy tắt.
  Đã kiểm chứng bằng app thử (modal 1.4.3): client thoát, function vẫn chạy đến hết trên cloud.
- Stage 1 run `20261019-044921` đã hoàn tất 40 epoch (best epoch 33, val Fg-mIoU 70.22%).
- Lệnh `modal` trên máy này thuộc Python 3.12 (`modal` 1.4.3); `python` mặc định là 3.11 (không có modal).

## Thí nghiệm Stage 2 dense 10 FPS (2026-10-19)

- `scripts/build_dense_clips.py`: mỗi frame có nhãn là frame CUỐI của clip T=8; 7 frame ngữ cảnh 10 FPS lấy
  từ video gốc (mask rỗng → 255, trừ khi chính frame đó có nhãn). Vị trí frame trong video gốc: round 1 khớp
  ảnh tuyệt đối (PSNR 99, 1 FPS), round 2 dùng `frame_orig` của `frames_for_annotation/manifest.csv`
  (video gần tĩnh nên khớp ảnh mơ hồ, tỉ lệ cost 1.02–1.11). Ảnh lưu JPEG q95, cạnh dài ≤ 960.
- Phát hiện: 28 video round 2 trong `dataset_video_prepared` đã được **CLAHE sẵn + kéo giãn 640×640** lúc
  trích frame (`extract_1fps_clahe.py`) → bị CLAHE 2 lần khi train sparse. Bộ dense dùng frame gốc nên không bị.
- Train: `modal run --detach src/train_stage2_timesformer.py --dense` (run `20261019-094419`, cùng siêu tham số),
  checkpoint `timesformer_dense_*.pth`, best epoch 18 (val Fg-mIoU 60.94%), vẫn overfit (val loss 1.1 → 1.5).
- Đánh giá test (208 frame): `docs/reports/eval_20261019-094419_dense10fps.md`, so sánh
  `docs/reports/compare_sparse_vs_dense.md`. Tóm tắt Fg-mIoU: Stage 1 60.2 | sparse full 57.2 | dense γ=0 **62.1** |
  dense full 54.8. Nhánh temporal vẫn làm giảm kết quả (đặc biệt khói: dự đoán khói giả trên video chỉ có lửa).
  Mọi chênh lệch nằm trong CI 95% rộng (14 video test).
- Lỗi đã sửa: frame có nhãn xuất hiện làm ngữ cảnh bị đặt tên `ctx_raw_*` → bị đếm 2 lần khi đánh giá (216 frame);
  đã đổi tên trong CSV (train không bị ảnh hưởng) và trong script.

## Việc của người dùng (thứ tự)

1. Đồng bộ dữ liệu đã sửa lên volume (cần `pip install modal`, đã đăng nhập):
   `python scripts/sync_fixed_data_to_modal.py --dry-run` rồi `python scripts/sync_fixed_data_to_modal.py`
2. Kiểm tra trên volume: `modal run scripts/audit_data.py` (phải sạch như local).
3. `modal run --detach src/train_stage1_segformer.py`
4. `modal run --detach src/train_stage2_timesformer.py` (không `--resume`)
5. `modal run src/evaluate_test.py`
6. Viết lại báo cáo chỉ từ `docs/reports/eval_<run_id>.json`.

## Gộp dữ liệu + chia lại (2026-10-20, `scripts/build_unified_dataset.py`)

- Nguồn mới: `data/Frames_video.v2i.png-mask-semantic` (692 ảnh = 388 ảnh của v1i + 304 ảnh mới thuộc 11 video).
  Ảnh mới được ghép với `frames_for_annotation/<video>/frame_XXXXX.jpg` bằng so khớp pixel (sai khác ~0.5 vs ≥ 8 với ứng viên thứ hai).
- Một quy ước lưu trữ cho mọi frame video: giải mã lại từ video gốc theo `frame_orig` / `anchor_raw_idx`, giữ tỉ lệ,
  cạnh dài ≤ 960, JPEG q95 → `data/dataset_video_unified/<video>/raw_<idx>.jpg`; mask resize nearest. Không còn CLAHE / kéo giãn
  sẵn trong dữ liệu (round 2 trước đây bị CLAHE 2 lần). Tăng cường ảnh chỉ chạy lúc load (`fire_core`).
  Sparse và dense dùng chung thư mục này; `dataset_video_prepared` / `dataset_video_dense` không còn được dùng.
- Frame không được export từ Roboflow (`auto_bg` cũ) không còn coi là nền; chỉ làm ngữ cảnh (mask 255).
- 1864 frame có nhãn / 111 video. Chia theo nhóm nguồn (pexels id / YouTube id / chunk), tối ưu 70/15/15 theo frame lửa / khói / rỗng
  và loại video: train 78 video 1302 frame | val 17 video 280 | test 16 video 282. 43/100 video cũ đổi split
  → **số liệu test cũ không còn so sánh được**.
- Ảnh tĩnh chia lại theo nhóm với cùng tỉ lệ: 855 / 192 / 191 (vẫn ép nhóm giống frame video val/test vào đúng split).
- Audit: `docs/reports/data_audit_2026-10-20.json` — sạch. Bản cũ của data_splits: `data/data_splits/archive/20261020-161110/`.
- Cần làm: `py -3.12 scripts/sync_fixed_data_to_modal.py --unified`, `modal run scripts/audit_data.py`, rồi train lại Stage 1 → Stage 2.

### Chia lại bằng MILP + 5-fold (2026-10-20)

- `build_unified_dataset.py --resplit-only`: test cố định 15% + 5 fold (~17%) theo nhóm nguồn, MILP minimax (scipy/HiGHS) cân bằng
  số frame lửa/khói/rỗng, tổng diện tích lửa/khói và số frame lửa/khói nhỏ/vừa/lớn; số video mỗi loại là ràng buộc cứng.
  Lệch tránh được tối đa 2.78 điểm % (video). Không tránh được: `bucket11` chứa 30% frame khói nhỏ.
- Ảnh tĩnh cùng 6 phần; nhóm giống frame video bị ép theo fold của video (0 xung đột). Vì ảnh tĩnh rỗng (180/181) là frame
  của video âm tính, bài toán chia video cân bằng luôn số ảnh tĩnh gắn với mỗi video → ảnh rỗng 16.6–17.7% mỗi fold.
- `data_splits/` = val fold 0 (lệnh cũ chạy như trước); `data_splits/folds/fold<k>/` cho từng fold.
  Train: `modal run --detach src/train_stage1_segformer.py --fold k`, `... train_stage2_timesformer.py [--dense] --fold k`
  (Stage 2 fold k đọc `segformer_spatial_best_fold<k>.pth`); `modal run src/evaluate_test.py --fold k --stage1-ckpt ... --stage2-ckpt ...`.
- Nguồn cố định của pipeline cũ (dense clips cũ, manifest prepared, v1 mapping): `data/data_splits/legacy_sources/`.
