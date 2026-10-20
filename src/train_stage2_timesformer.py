"""
src/train_stage2_timesformer.py
================================
STAGE 2: Huấn luyện Spatio-Temporal SegFormer trên dữ liệu video chuỗi thời gian
Mục tiêu:
- Spatio-Temporal SegFormer = SegFormer-B2 Spatial Core (Stage 1) + Temporal Refinement Module.
- Nạp 100% trọng số không gian từ Stage 1 (segformer_spatial_best.pth); đóng băng encoder.
- Running stats BatchNorm của decode head được đóng băng (vẫn học affine).
- Temporal module CAUSAL (frame t chỉ nhìn frame <= t) + positional embedding thời gian,
  Zero-Initialized Residual Gating (gamma=0).
- Mixed Precision (FP16 AMP) + Gradient Accumulation trên NVIDIA A100.
- Checkpoint đa mục tiêu, có run_id + config, tự động khôi phục:
  + timesformer_spatiotemporal_best.pth: Fg-mIoU đạt đỉnh mới.
  + timesformer_best_fire.pth / timesformer_best_smoke.pth: IoU lửa / khói đạt kỷ lục.
  + timesformer_spatiotemporal_latest.pth: lưu CUỐI mỗi epoch (sau khi cập nhật kỷ lục) để resume.
- Chạy mới (không --resume): toàn bộ checkpoint cũ được chuyển vào /data/checkpoints/archive/.

Lưu ý: checkpoint Stage 2 trước bản sửa (không causal, không pos_embed, không config)
KHÔNG resume được -> chạy lại từ đầu.

Model / dataset / loss / tiền xử lý dùng chung nằm trong `src/fire_core.py`.

Cách chạy trên Modal GPU:
  modal run --detach src/train_stage2_timesformer.py
  modal run --detach src/train_stage2_timesformer.py --resume
"""

import subprocess
from pathlib import Path

import modal

app = modal.App("fire-smoke-stage2-timesformer")
volume = modal.Volume.from_name("fire_smoke_dataset_v2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "transformers==4.40.0",
        "opencv-python-headless",
        "numpy<2.0.0",
        "Pillow",
        "scikit-learn",
        "tqdm",
        "pandas",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(Path(__file__).parent / "fire_core.py", "/root/fire_core.py")
)


def local_git_commit():
    """Short commit hash of the local checkout (+'-dirty' if uncommitted changes)."""
    try:
        root = Path(__file__).resolve().parent.parent
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        return f"{commit}-dirty" if dirty else commit
    except Exception:
        return "unknown"


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A100",
    cpu=8.0,           # per-frame preprocessing (CLAHE/DCP/motion at source resolution) is the bottleneck
    timeout=86400,
    memory=32768,
)
def train_stage2(resume: bool = False, git_commit: str = "unknown", dense: bool = False):
    import os
    import random
    import numpy as np

    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    import fire_core as fc

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Thiết lập seed
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"🚀 [STAGE 2] Khởi chạy Spatio-Temporal SegFormer trên GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM khả dụng: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print("=" * 80)

    # clip_mode:
    #   "sparse"     -> clips of consecutive labeled frames (1-2 FPS), all frames supervised
    #   "dense10fps" -> one clip per labeled frame: 7 context frames at 10 FPS before it,
    #                   only labeled frames supervised (scripts/build_dense_clips.py)
    CLIP_MODE = "dense10fps" if dense else "sparse"
    DATA_ROOT = Path("/data/dataset_video_dense" if dense else "/data/dataset_video_prepared")
    SPLITS_DIR = Path("/data/data_splits")
    CHECKPOINTS_DIR = Path("/data/checkpoints")
    LOGS_DIR = Path("/data/logs")

    PREFIX = "timesformer_dense" if dense else "timesformer"
    STAGE1_CHECKPOINT = CHECKPOINTS_DIR / "segformer_spatial_best.pth"
    BEST_CHECKPOINT   = CHECKPOINTS_DIR / f"{PREFIX}_spatiotemporal_best.pth"
    BEST_FIRE_CKPT    = CHECKPOINTS_DIR / f"{PREFIX}_best_fire.pth"
    BEST_SMOKE_CKPT   = CHECKPOINTS_DIR / f"{PREFIX}_best_smoke.pth"
    LATEST_CHECKPOINT = CHECKPOINTS_DIR / f"{PREFIX}_spatiotemporal_latest.pth"
    BEST_PATHS = {"best": BEST_CHECKPOINT, "fire": BEST_FIRE_CKPT, "smoke": BEST_SMOKE_CKPT}

    TRAIN_CLIPS_CSV = SPLITS_DIR / ("dense_train_clips.csv" if dense else "train_clips.csv")
    VAL_CLIPS_CSV   = SPLITS_DIR / ("dense_val_clips.csv" if dense else "val_clips.csv")

    assert TRAIN_CLIPS_CSV.exists(), f"Không tìm thấy {TRAIN_CLIPS_CSV}"
    assert VAL_CLIPS_CSV.exists(), f"Không tìm thấy {VAL_CLIPS_CSV}"
    assert DATA_ROOT.exists(), f"Không tìm thấy {DATA_ROOT}"
    assert STAGE1_CHECKPOINT.exists(), f"BẮT BUỘC có checkpoint Stage 1 tại {STAGE1_CHECKPOINT} để khởi tạo Spatial Core!"

    # =========================================================================
    # 0. CONFIG + RUN_ID
    # =========================================================================
    print("⏳ Đang nạp checkpoint Stage 1...")
    s1_ckpt = torch.load(STAGE1_CHECKPOINT, map_location="cpu")

    CONFIG = {
        "stage": 2,
        "clip_mode": CLIP_MODE,
        "data_root": str(DATA_ROOT),
        "model": "nvidia/mit-b2",
        "T": 8,
        "target_size": [512, 512],
        "letterbox": True,
        "mask_pad_value": fc.IGNORE_INDEX,
        "preprocess_version": fc.PREPROCESS_VERSION,
        "use_motion": True,
        "causal": True,
        "temporal_hidden": 128,
        "num_heads": 4,
        "max_T": 16,
        "temporal_dropout": 0.1,
        "freeze_backbone": True,
        "freeze_decode_bn": True,
        "batch_size": 1,
        "grad_accum_steps": 4,
        "num_workers": 8,
        "epochs": 35,
        "lr_head": 2e-5,
        "lr_temporal": 3e-4,
        "weight_decay": 1e-2,
        "eta_min": 1e-6,
        "max_grad_norm": 1.0,
        "class_weights": [0.1, 3.5, 2.0],
        "dice_weight": 1.0,
        "dice_mode": "batch_fg",
        "temp_smooth_weight": 0.05,
        "seed": SEED,
        "git_commit": git_commit,
        "stage1_checkpoint": str(STAGE1_CHECKPOINT),
        "stage1_run_id": s1_ckpt.get("run_id", "legacy"),
        "stage1_epoch": s1_ckpt.get("epoch"),
    }

    resume_data = None
    if resume and LATEST_CHECKPOINT.exists():
        print(f"🔄 Tìm thấy checkpoint mới nhất tại: {LATEST_CHECKPOINT.name}")
        resume_data = torch.load(LATEST_CHECKPOINT, map_location="cpu")
        try:
            # also refuse if Stage 1 was retrained since this run started (weights would be mixed)
            saved_cfg = resume_data.get("config")
            if saved_cfg is not None:
                saved_cfg = {"clip_mode": "sparse", **saved_cfg}
            fc.check_resume_config(saved_cfg, CONFIG,
                                   keys=fc.ARCH_CONFIG_KEYS + ("stage1_run_id", "clip_mode"))
        except ValueError as e:
            print(f"❌ KHÔNG THỂ RESUME: {e}")
            raise
        RUN_ID = resume_data["run_id"]
    else:
        if resume:
            print(f"⚠️ --resume nhưng không có {LATEST_CHECKPOINT.name} -> bắt đầu lần chạy mới.")
        RUN_ID = fc.make_run_id()
        archived = fc.archive_existing_checkpoints(
            [BEST_CHECKPOINT, BEST_FIRE_CKPT, BEST_SMOKE_CKPT, LATEST_CHECKPOINT], CHECKPOINTS_DIR / "archive")
        if archived:
            volume.commit()
            print(f"📦 Đã chuyển toàn bộ checkpoint Stage 2 cũ vào: {archived}")

    print(f"🆔 RUN_ID = {RUN_ID} | clip_mode = {CLIP_MODE} | git = {git_commit} | Stage 1 run = {CONFIG['stage1_run_id']} | preprocess = {fc.PREPROCESS_VERSION}")

    # =========================================================================
    # 1. MÔ HÌNH, DỮ LIỆU, LOSS, OPTIMIZER
    # =========================================================================
    print("⏳ Đang tải kiến trúc nvidia/mit-b2 và nạp trọng số Stage 1 SegFormer...")
    base_segformer = fc.build_segformer_b2(num_labels=fc.NUM_CLASSES, pretrained=True)
    base_segformer.load_state_dict(s1_ckpt["model_state_dict"])
    del s1_ckpt
    print("✅ Đã nạp thành công 100% trọng số không gian đã tối ưu từ Stage 1!")

    model = fc.SpatioTemporalSegFormer(
        base_segformer,
        freeze_backbone=CONFIG["freeze_backbone"],
        freeze_decode_bn=CONFIG["freeze_decode_bn"],
        temporal_hidden=CONFIG["temporal_hidden"],
        num_heads=CONFIG["num_heads"],
        dropout=CONFIG["temporal_dropout"],
        max_T=CONFIG["max_T"],
        causal=CONFIG["causal"],
    ).to(device)

    target_size = tuple(CONFIG["target_size"])
    train_dataset = fc.VideoClipDataset(TRAIN_CLIPS_CSV, DATA_ROOT, target_size=target_size, is_training=True,
                                        use_motion=CONFIG["use_motion"], letterbox=CONFIG["letterbox"])
    val_dataset   = fc.VideoClipDataset(VAL_CLIPS_CSV, DATA_ROOT, target_size=target_size, is_training=False,
                                        use_motion=CONFIG["use_motion"], letterbox=CONFIG["letterbox"])

    BATCH_SIZE = CONFIG["batch_size"]
    GRAD_ACCUM_STEPS = CONFIG["grad_accum_steps"]
    T = CONFIG["T"]
    loader_kw = dict(num_workers=CONFIG["num_workers"], pin_memory=True, persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **loader_kw)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kw)

    print(f"📊 Dataset Clips: Train: {len(train_dataset)} clips | Val: {len(val_dataset)} clips (T={T})")
    print(f"⚙️ Cấu hình: Batch Size = {BATCH_SIZE}, Grad Accum = {GRAD_ACCUM_STEPS} (Effective Batch = {BATCH_SIZE * GRAD_ACCUM_STEPS} clips = {BATCH_SIZE * GRAD_ACCUM_STEPS * T} frames)")

    class_weights = torch.tensor(CONFIG["class_weights"], dtype=torch.float32, device=device)
    criterion = fc.SpatioTemporalLoss(class_weights=class_weights, dice_weight=CONFIG["dice_weight"],
                                      temp_smooth_weight=CONFIG["temp_smooth_weight"], dice_mode=CONFIG["dice_mode"])

    head_params     = [p for n, p in model.decode_head.named_parameters() if p.requires_grad]
    temporal_params = [p for n, p in model.temporal_refinement.named_parameters() if p.requires_grad]

    optimizer = optim.AdamW([
        {"params": head_params,     "lr": CONFIG["lr_head"],     "weight_decay": CONFIG["weight_decay"]},
        {"params": temporal_params, "lr": CONFIG["lr_temporal"], "weight_decay": CONFIG["weight_decay"]},
    ])

    NUM_EPOCHS = CONFIG["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=CONFIG["eta_min"])
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 1
    tracker = fc.BestTracker()

    # 🔄 TỰ ĐỘNG KHÔI PHỤC (FAULT-TOLERANT AUTO-RESUME)
    if resume_data is not None:
        model.load_state_dict(resume_data["model_state_dict"])
        optimizer.load_state_dict(resume_data["optimizer_state_dict"])
        scheduler.load_state_dict(resume_data["scheduler_state_dict"])
        if resume_data.get("scaler_state_dict"):
            scaler.load_state_dict(resume_data["scaler_state_dict"])
        fc.restore_rng_state(resume_data.get("rng"))
        start_epoch = resume_data["epoch"] + 1
        tracker = fc.BestTracker(resume_data["best_val_miou"], resume_data["best_fire_iou"],
                                 resume_data["best_smoke_iou"], resume_data["best_epoch"])
        print(f"⏩ KHÔI PHỤC THÀNH CÔNG! Tiếp tục từ Epoch {start_epoch:02d}/{NUM_EPOCHS}")
        print(f"   (Kỷ lục trước đó: Fg-mIoU: {tracker.best_val_miou*100:.2f}%, Fire: {tracker.best_fire_iou*100:.2f}%, Smoke: {tracker.best_smoke_iou*100:.2f}% ở Epoch {tracker.best_epoch})")
        del resume_data
    else:
        print("🌱 Bắt đầu huấn luyện Stage 2 từ đầu (Kỷ lục mới từ 0.0%)...")

    history_path = LOGS_DIR / f"stage2_{'dense_' if dense else ''}history_{RUN_ID}.json"

    print("\n" + "=" * 80)
    print(f"🔥 BẮT ĐẦU HUẤN LUYỆN SPATIO-TEMPORAL SEGFORMER: EPOCHS {start_epoch:02d} -> {NUM_EPOCHS} TRÊN NVIDIA A100")
    print("=" * 80)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()  # encoder & decode-head BN stay in eval mode (see SpatioTemporalSegFormer.train)

        train_loss = 0.0
        train_ce = 0.0
        train_dice = 0.0
        train_temp = 0.0

        pbar = tqdm(train_loader, desc=f"Stage 2 Epoch {epoch:02d}/{NUM_EPOCHS} [Train]")
        optimizer.zero_grad()

        for step, (clips, masks) in enumerate(pbar):
            clips = clips.to(device)  # (1, T, 3, H, W)
            masks = masks.to(device)  # (1, T, H, W)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(clips)     # (1, T, 3, H, W)
                loss, l_ce, l_dice, l_tmp = criterion(logits, masks)
                loss_scaled = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss_scaled).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()
            train_ce += l_ce.item()
            train_dice += l_dice.item()
            train_temp += l_tmp.item()

            cur_gamma = model.temporal_refinement.gamma.item()
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Dice": f"{l_dice.item():.3f}",
                "gamma": f"{cur_gamma:.4f}"
            })

        lr_groups = [g["lr"] for g in optimizer.param_groups]
        scheduler.step()
        train_loss /= len(train_loader)
        train_ce /= len(train_loader)
        train_dice /= len(train_loader)
        train_temp /= len(train_loader)

        # =====================================================================
        # VALIDATION (confusion matrix 3x3, bỏ pixel pad 255)
        # =====================================================================
        model.eval()
        val_loss = 0.0
        cm = np.zeros((fc.NUM_CLASSES, fc.NUM_CLASSES), dtype=np.int64)

        with torch.no_grad():
            for clips, masks in tqdm(val_loader, desc=f"Stage 2 Epoch {epoch:02d}/{NUM_EPOCHS} [Val]"):
                clips = clips.to(device)
                masks = masks.to(device)

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(clips)
                    loss, _, _, _ = criterion(logits, masks)

                val_loss += loss.item()
                cm += fc.confusion_matrix_3c(torch.argmax(logits, dim=2), masks)

        val_loss /= len(val_loader)
        m = fc.iou_dice_from_cm(cm)
        iou_bg, iou_fire, iou_smoke = m["iou_bg"], m["iou_fire"], m["iou_smoke"]
        fg_miou, all_miou = m["fg_miou"], m["all_miou"]
        cur_gamma = model.temporal_refinement.gamma.item()

        print(f"\n[Stage 2 Epoch {epoch:02d}] Train Loss: {train_loss:.4f} (CE: {train_ce:.3f}, Dice: {train_dice:.3f}, Temp: {train_temp:.4f})")
        print(f"                      Val Loss:   {val_loss:.4f} | IoU BG: {iou_bg*100:.2f}% | IoU Fire: {iou_fire*100:.2f}% | IoU Smoke: {iou_smoke*100:.2f}% | Fg-mIoU: {fg_miou*100:.2f}% | All-mIoU: {all_miou*100:.2f}% | Gamma: {cur_gamma:.4f}")

        fc.append_history(history_path, {
            "epoch": epoch, "train_loss": train_loss, "ce": train_ce, "dice": train_dice, "temp": train_temp,
            "val_loss": val_loss, "iou_bg": iou_bg, "iou_fire": iou_fire, "iou_smoke": iou_smoke,
            "fg_miou": fg_miou, "all_miou": all_miou, "gamma": cur_gamma, "lr_groups": lr_groups,
        })

        # =====================================================================
        # CHECKPOINT: cập nhật kỷ lục -> lưu best/fire/smoke -> lưu LATEST CUỐI CÙNG
        # =====================================================================
        metrics = {"fg_miou": fg_miou, "iou_fire": iou_fire, "iou_smoke": iou_smoke}
        labels = {"best": "TỔNG HỢP (Fg-mIoU)", "fire": "CHUYÊN LỬA", "smoke": "CHUYÊN KHÓI"}

        def save_best(kind):
            torch.save({
                "epoch": epoch,
                "run_id": RUN_ID,
                "config": CONFIG,
                "model_state_dict": model.state_dict(),
                "fg_miou": fg_miou, "all_miou": all_miou,
                "iou_bg": iou_bg, "iou_fire": iou_fire, "iou_smoke": iou_smoke,
                "gamma": cur_gamma,
                "architecture": "SpatioTemporalSegFormer",
            }, BEST_PATHS[kind])
            print(f"  ⭐ [KỶ LỤC {labels[kind]}] Epoch {epoch}: Fg-mIoU {fg_miou*100:.2f}% | Fire {iou_fire*100:.2f}% | Smoke {iou_smoke*100:.2f}% -> {BEST_PATHS[kind].name}")

        def save_latest(payload):
            torch.save({
                **payload,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "rng": fc.capture_rng_state(),
                "architecture": "SpatioTemporalSegFormer",
            }, LATEST_CHECKPOINT)

        fc.epoch_checkpoint_step(epoch, metrics, tracker, save_best, save_latest, RUN_ID, CONFIG, cur_gamma)
        volume.commit()

    print("\n" + "=" * 80)
    print(f"🎉 HUẤN LUYỆN STAGE 2 HOÀN TẤT! (RUN_ID = {RUN_ID})")
    print(f"   Best Epoch: {tracker.best_epoch} với Fg-mIoU = {tracker.best_val_miou*100:.2f}%")
    print(f"   File tốt nhất đã lưu tại: {BEST_CHECKPOINT}")
    print(f"   Lịch sử huấn luyện: {history_path}")
    print("=" * 80)
    return float(tracker.best_val_miou)


@app.local_entrypoint()
def main(resume: bool = False, dense: bool = False):
    # spawn (not remote): with `modal run --detach` the job keeps running on Modal even if
    # this machine sleeps / shuts down. `.remote()` would be cancelled when the client exits.
    call = train_stage2.spawn(resume=resume, git_commit=local_git_commit(), dense=dense)
    print(f"🚀 Đã gửi job Stage 2 ({'dense10fps' if dense else 'sparse'}) lên Modal (call id: {call.object_id}). Có thể tắt máy.")
    print("   Theo dõi: modal app list  |  modal app logs <app-id>")
