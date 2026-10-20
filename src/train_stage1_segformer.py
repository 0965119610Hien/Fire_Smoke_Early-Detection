"""
src/train_stage1_segformer.py
==============================
STAGE 1: Huấn luyện Spatial Core SegFormer-B2 trên dữ liệu ảnh tĩnh (Fire_Smoke.v4i)
Mục tiêu:
- Tối ưu hóa đặc trưng không gian (Spatial Representation) nhận diện đốm lửa nhỏ & khói sớm.
- Combined Loss (Weighted Cross-Entropy + Dice foreground theo batch) khắc phục mất cân bằng điểm ảnh.
- Letterbox giữ tỉ lệ; vùng pad của mask = 255 (bị bỏ qua bởi loss và metric).
- Xuất checkpoint: `/data/checkpoints/segformer_spatial_best.pth` (kèm run_id + config).
- Khi bắt đầu lần chạy mới, checkpoint cũ được chuyển vào `/data/checkpoints/archive/`.

Model / dataset / loss / tiền xử lý dùng chung nằm trong `src/fire_core.py`.

Cách chạy trên Modal GPU:
  modal run --detach src/train_stage1_segformer.py
"""

import subprocess
from pathlib import Path

import modal

app = modal.App("fire-smoke-stage1-segformer")
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
        "einops",
        "scikit-learn",
        "tqdm",
        "pandas",
    )
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
    timeout=86400,
    memory=32768,
)
def train_stage1(git_commit: str = "unknown", fold: int = -1):
    import random
    import numpy as np

    import torch
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    import fire_core as fc

    # Thiết lập seed
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [STAGE 1] Khởi chạy huấn luyện SegFormer-B2 trên Modal GPU: {torch.cuda.get_device_name(0)}")

    DATA_ROOT = Path("/data/Fire_Smoke.v4i/train")
    # fold = -1 -> default layout of data_splits/ (val = fold 0); fold = k -> data_splits/folds/fold<k>/
    SPLITS_DIR = Path("/data/data_splits") / (f"folds/fold{fold}" if fold >= 0 else "")
    FOLD_TAG = f"_fold{fold}" if fold >= 0 else ""
    CHECKPOINTS_DIR = Path("/data/checkpoints")
    LOGS_DIR = Path("/data/logs")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    BEST_CHECKPOINT = CHECKPOINTS_DIR / f"segformer_spatial_best{FOLD_TAG}.pth"

    TRAIN_CSV = SPLITS_DIR / "static_v4i_train_manifest.csv"
    VAL_CSV   = SPLITS_DIR / "static_v4i_val_manifest.csv"

    assert TRAIN_CSV.exists(), f"Không tìm thấy {TRAIN_CSV}"
    assert VAL_CSV.exists(), f"Không tìm thấy {VAL_CSV}"
    assert DATA_ROOT.exists(), f"Không tìm thấy {DATA_ROOT}"

    RUN_ID = fc.make_run_id()
    CONFIG = {
        "stage": 1,
        "model": "nvidia/mit-b2",
        "num_labels": fc.NUM_CLASSES,
        "target_size": [512, 512],
        "letterbox": True,
        "mask_pad_value": fc.IGNORE_INDEX,
        "preprocess_version": fc.PREPROCESS_VERSION,
        "use_motion": False,
        "batch_size": 8,
        "num_workers": 4,
        "epochs": 40,
        "lr_backbone": 6e-5,
        "lr_head": 6e-4,
        "weight_decay": 0.01,
        "eta_min": 1e-6,
        "class_weights": [0.1, 3.5, 2.0],
        "dice_weight": 1.0,
        "dice_mode": "batch_fg",
        "seed": SEED,
        "git_commit": git_commit,
        "fold": fold,
    }
    print(f"🆔 RUN_ID = {RUN_ID} | git = {git_commit} | preprocess = {fc.PREPROCESS_VERSION}")

    # Lần chạy mới: chuyển checkpoint cũ vào archive để mọi file best đến từ cùng 1 run
    archived = fc.archive_existing_checkpoints([BEST_CHECKPOINT], CHECKPOINTS_DIR / "archive")
    if archived:
        volume.commit()
        print(f"📦 Đã chuyển checkpoint Stage 1 cũ vào: {archived}")

    # =========================================================================
    # 1. DATALOADERS
    # =========================================================================
    train_dataset = fc.StaticSegDataset(TRAIN_CSV, DATA_ROOT, target_size=(512, 512), is_training=True)
    val_dataset   = fc.StaticSegDataset(VAL_CSV, DATA_ROOT, target_size=(512, 512), is_training=False)

    BATCH_SIZE = CONFIG["batch_size"]
    NUM_WORKERS = CONFIG["num_workers"]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print(f"📊 Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # =========================================================================
    # 2. MÔ HÌNH, LOSS, OPTIMIZER
    # =========================================================================
    print("⏳ Tải SegFormer-B2 pretrained ImageNet...")
    model = fc.build_segformer_b2(num_labels=fc.NUM_CLASSES, pretrained=True)
    model.to(device)

    # Class weights: BG: 0.1, Fire: 3.5, Smoke: 2.0
    class_weights = torch.tensor(CONFIG["class_weights"], dtype=torch.float32, device=device)

    def criterion(logits, targets):
        return fc.seg_ce_dice_loss(logits, targets, class_weights,
                                   dice_weight=CONFIG["dice_weight"], dice_mode=CONFIG["dice_mode"])

    # Differential Learning Rate: Backbone nhỏ, Decode head lớn
    backbone_params = [p for n, p in model.named_parameters() if "segformer" in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters() if "decode_head" in n and p.requires_grad]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": CONFIG["lr_backbone"], "weight_decay": CONFIG["weight_decay"]},
        {"params": head_params,     "lr": CONFIG["lr_head"],     "weight_decay": CONFIG["weight_decay"]},
    ])

    NUM_EPOCHS = CONFIG["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=CONFIG["eta_min"])

    history_path = LOGS_DIR / f"stage1_history{FOLD_TAG}_{RUN_ID}.json"

    # =========================================================================
    # 3. VÒNG LẶP HUẤN LUYỆN
    # =========================================================================
    best_val_miou = 0.0
    best_epoch = 0

    print("\n" + "=" * 70)
    print(f"🔥 BẮT ĐẦU HUẤN LUYỆN STAGE 1: {NUM_EPOCHS} EPOCHS TRÊN MODAL GPU")
    print("=" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_ce = 0.0
        train_dice = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} [Train]")
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(pixel_values=imgs).logits  # (B, 3, H/4, W/4)
            logits_upsampled = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss, l_ce, l_dice = criterion(logits_upsampled, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_ce += l_ce.item()
            train_dice += l_dice.item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "CE": f"{l_ce.item():.3f}", "Dice": f"{l_dice.item():.3f}"})

        lr_groups = [g["lr"] for g in optimizer.param_groups]
        scheduler.step()
        train_loss /= len(train_loader)
        train_ce /= len(train_loader)
        train_dice /= len(train_loader)

        # =====================================================================
        # VALIDATION (confusion matrix 3x3 tích lũy, bỏ pixel pad 255)
        # =====================================================================
        model.eval()
        val_loss = 0.0
        cm = np.zeros((fc.NUM_CLASSES, fc.NUM_CLASSES), dtype=np.int64)

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} [Val]"):
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(pixel_values=imgs).logits
                logits_upsampled = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

                loss, _, _ = criterion(logits_upsampled, masks)
                val_loss += loss.item()
                cm += fc.confusion_matrix_3c(torch.argmax(logits_upsampled, dim=1), masks)

        val_loss /= len(val_loader)
        m = fc.iou_dice_from_cm(cm)
        ious, dices = m["iou"], m["dice"]
        fg_miou, all_miou = m["fg_miou"], m["all_miou"]

        print(f"\n[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"          IoU: Fire={ious[1]*100:.2f}%, Smoke={ious[2]*100:.2f}%, BG={ious[0]*100:.2f}% | Fg-mIoU={fg_miou*100:.2f}% | All-mIoU={all_miou*100:.2f}%")
        print(f"          Dice: Fire={dices[1]*100:.2f}%, Smoke={dices[2]*100:.2f}%")

        fc.append_history(history_path, {
            "epoch": epoch, "train_loss": train_loss, "ce": train_ce, "dice": train_dice,
            "val_loss": val_loss, "iou_bg": ious[0], "iou_fire": ious[1], "iou_smoke": ious[2],
            "fg_miou": fg_miou, "all_miou": all_miou, "lr_groups": lr_groups,
        })

        # Lưu checkpoint khi Foreground mIoU cải thiện
        if fg_miou > best_val_miou:
            best_val_miou = fg_miou
            best_epoch = epoch
            print(f"  ⭐ ĐẠT ĐỈNH MỚI! (Fg-mIoU: {fg_miou*100:.2f}% ở Epoch {epoch}) -> Đang lưu checkpoint...")
            torch.save({
                "epoch": epoch,
                "run_id": RUN_ID,
                "config": CONFIG,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "fg_miou": fg_miou,
                "all_miou": all_miou,
                "ious": ious,
                "dices": dices,
                "confusion_matrix": cm.tolist(),
            }, BEST_CHECKPOINT)
            print(f"  💾 Đã lưu checkpoint: {BEST_CHECKPOINT.name}")

        volume.commit()

    print("\n" + "=" * 70)
    print(f"🎉 HUẤN LUYỆN STAGE 1 HOÀN TẤT! (RUN_ID = {RUN_ID})")
    print(f"   Best Epoch: {best_epoch} với Fg-mIoU = {best_val_miou*100:.2f}%")
    print(f"   File đã lưu tại: {BEST_CHECKPOINT}")
    print(f"   Lịch sử huấn luyện: {history_path}")
    print("=" * 70)
    return float(best_val_miou)


@app.local_entrypoint()
def main(fold: int = -1):
    # spawn (not remote): with `modal run --detach` the job keeps running on Modal even if
    # this machine sleeps / shuts down. `.remote()` would be cancelled when the client exits.
    call = train_stage1.spawn(git_commit=local_git_commit(), fold=fold)
    print(f"🚀 Đã gửi job Stage 1{f' fold {fold}' if fold >= 0 else ''} lên Modal (call id: {call.object_id}). Có thể tắt máy.")
    print("   Theo dõi: modal app list  |  modal app logs <app-id>")
