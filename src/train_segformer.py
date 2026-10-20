"""
=============================================================================
BASELINE C — SegFormer-B2 Semantic Segmentation  (Fire & Smoke Detection)
=============================================================================
Chiến lược:
  - Input:   Ảnh tĩnh từ dataset_seg_prepared/ (đã split & augment offline)
  - Output:  Segmentation mask 3-class:
               0 = Background
               1 = Fire
               2 = Smoke
  - Backbone: MiT-B2 (Mix Transformer) pretrained ImageNet
  - Decoder:  All-MLP Lightweight Decoder (SegFormer — NVIDIA 2021)
  - Post-proc: Mask → BBox extraction (cho downstream tracking & mAP so sánh)

Các điểm kỹ thuật quan trọng:
  1. Dataset: PNG Semantic Mask (grayscale index 0/1/2) — đã augment offline
  2. Input size 512×512
  3. Differential LR: backbone 6e-5, decode head 6e-4
  4. Loss: CrossEntropyLoss (class-weighted) + DiceLoss
  5. SegFormer output = H/4 × W/4 → cần upsample về H×W khi eval
  6. WeightedRandomSampler: oversample ảnh có fire/smoke nhỏ

Chuẩn bị dữ liệu (chạy local trước):
  python prepare_segmentation_dataset.py
  python upload_seg_dataset_to_modal.py

Chạy trên Modal:
  modal run src/train_segformer.py

Evaluation:
  - Segmentation: mIoU, IoU_fire, IoU_smoke, Dice_fire, Dice_smoke
  - Detection:    mAP@0.5 (sau khi chuyển mask → BBox để so sánh với Baseline A/B)
============================================================================="""

import modal
import os

app = modal.App("fire-smoke-segformer-baseline-c")
vol = modal.Volume.from_name("fire_smoke_dataset")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "transformers==4.40.0",    # SegFormer từ HuggingFace
        "opencv-python-headless",
        "numpy<2.0.0",             # Pin numpy <2.0 để tương thích với PyTorch 2.2.2 C-API
        "Pillow",
        "einops",
        "scikit-learn",
        "tqdm",
    )
)


# ===========================================================================
# MODAL FUNCTION — chạy trên A10G (~24 GB VRAM)
# ===========================================================================
@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="A10G",
    timeout=86400,   # 24h tối đa
    memory=32768,    # 32 GB RAM
)
def train_segformer():
    import copy
    import glob
    import random
    import re

    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from PIL import Image, ImageEnhance
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    from transformers import SegformerForSemanticSegmentation

    print("=" * 70)
    print("🚀  BASELINE C — SegFormer-B2  (GPU: A10G)")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device : {device}")
    if torch.cuda.is_available():
        print(f"   GPU    : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # =========================================================================
    # 1. CẤU HÌNH
    # =========================================================================
    CFG = {
        # --- Paths trên Modal volume /data ---
        # Dataset đã split & augment offline bởi prepare_segmentation_dataset.py
        "train_img_dir": "/data/fire_smoke_seg_prepared/train/images",
        "train_msk_dir": "/data/fire_smoke_seg_prepared/train/masks",
        "val_img_dir":   "/data/fire_smoke_seg_prepared/val/images",
        "val_msk_dir":   "/data/fire_smoke_seg_prepared/val/masks",
        "test_img_dir":  "/data/fire_smoke_seg_prepared/test/images",
        "test_msk_dir":  "/data/fire_smoke_seg_prepared/test/masks",
        "save_dir":      "/data",

        # --- Model ---
        "num_classes": 3,         # 0=background, 1=fire, 2=smoke
        "img_size":    512,       # Khớp với prepare_segmentation_dataset.py IMG_SIZE

        # --- Training ---
        "batch_size":    16,      # A10G ~24GB VRAM với AMP FP16
        "epochs":        35,      # SegFormer hội tụ nhanh
        "lr_backbone":   6e-5,    # Differential LR: backbone (đã pretrained)
        "lr_decode":     6e-4,    # Decode head (học mới hoàn toàn)
        "weight_decay":  1e-4,
        "patience":      5,       # Early stopping

        # --- DataLoader ---
        "num_workers": 4,
        "pin_memory":  True,

        # --- Sampling ---
        # Trọng số sampling: ảnh có fire/smoke nhỏ được bốc nhiều hơn
        # (tương ứng với compute_sample_weights trong notebook EDA)
        "small_fg_threshold_pct": 5.0,  # < 5% diện tích → weight_small
        "weight_bg_only":         1.0,
        "weight_fg_large":        2.0,
        "weight_fg_small":        5.0,
    }

    # =========================================================================
    # 2. ĐỌC PNG MASK (Grayscale index — đã chuẩn bị offline)
    # =========================================================================
    def load_mask_png(mask_path: str) -> np.ndarray:
        """
        Đọc PNG mask grayscale được tạo bởi prepare_segmentation_dataset.py.

        Format:
          Giá trị pixel = class index
            0 = background
            1 = fire
            2 = smoke

        Returns:
            np.ndarray [H, W] uint8, values in {0, 1, 2}
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Trả về mask toàn background nếu file không đọc được
            return np.zeros((512, 512), dtype=np.uint8)
        return mask.astype(np.uint8)

    # =========================================================================
    # 3. DATASET
    # =========================================================================
    class FireSmokeSegDataset(Dataset):
        """
        Dataset cho SegFormer Segmentation.

        Đọc dataset đã chuẩn bị offline bởi prepare_segmentation_dataset.py:
          - Ảnh: PNG RGB (đã resize 512×512, letterbox padding)
          - Mask: PNG Grayscale với giá trị pixel = class index {0=bg, 1=fire, 2=smoke}

        Không thực hiện augmentation nội bộ (đã làm offline).
        Chỉ cần: normalize ImageNet + convert to Tensor.
        """

        def __init__(self, img_dir: str, msk_dir: str, img_size: int = 512):
            self.img_dir  = img_dir
            self.msk_dir  = msk_dir
            self.img_size = img_size

            self.img_files = sorted([
                f for f in os.listdir(img_dir)
                if f.lower().endswith(".png")
            ])
            print(f"  Loaded {len(self.img_files):,} samples  ←  {img_dir}")

        def __len__(self):
            return len(self.img_files)

        def __getitem__(self, idx):
            img_name  = self.img_files[idx]
            stem      = os.path.splitext(img_name)[0]

            # ── Load ảnh ──────────────────────────────────────────────────────
            img_path = os.path.join(self.img_dir, img_name)
            img_bgr  = cv2.imread(img_path)
            if img_bgr is None:
                img_bgr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # ── Đảm bảo đúng kích thước (safety check) ────────────────────────
            if img_rgb.shape[:2] != (self.img_size, self.img_size):
                img_rgb = cv2.resize(img_rgb, (self.img_size, self.img_size),
                                     interpolation=cv2.INTER_LINEAR)

            # ── Load PNG mask grayscale ────────────────────────────────────────
            msk_path = os.path.join(self.msk_dir, f"{stem}.png")
            mask     = load_mask_png(msk_path)
            if mask.shape != (self.img_size, self.img_size):
                mask = cv2.resize(mask, (self.img_size, self.img_size),
                                  interpolation=cv2.INTER_NEAREST)

            # ── Normalize ImageNet mean/std ────────────────────────────────────
            mean    = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std     = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_nrm = (img_rgb.astype(np.float32) / 255.0 - mean) / std

            # ── Convert to Tensor ──────────────────────────────────────────────
            img_t  = torch.from_numpy(img_nrm).permute(2, 0, 1).float()  # [3, H, W]
            mask_t = torch.from_numpy(mask).long()                         # [H, W]

            return img_t, mask_t

    # =========================================================================
    # 4. LOSS: CrossEntropy (class-weighted) + Dice
    # =========================================================================
    class DiceLoss(nn.Module):
        """
        Dice Loss giảm thiểu tác động của class imbalance:
        vùng background >>> vùng lửa/khói → nếu chỉ dùng CE mô hình
        sẽ ưu tiên dự đoán background, bỏ qua lửa/khói nhỏ.
        """
        def __init__(self, num_classes: int = 3, eps: float = 1e-6):
            super().__init__()
            self.num_classes = num_classes
            self.eps         = eps

        def forward(self, logits: torch.Tensor, targets: torch.Tensor):
            # logits:  [B, C, H, W]
            # targets: [B, H, W] long
            probs = F.softmax(logits, dim=1)                                  # [B, C, H, W]
            tgt_oh = F.one_hot(targets, self.num_classes).permute(0,3,1,2).float()  # [B, C, H, W]

            dice_scores = []
            for c in range(self.num_classes):
                p = probs[:, c]
                t = tgt_oh[:, c]
                inter = (p * t).sum()
                dice  = (2 * inter + self.eps) / (p.sum() + t.sum() + self.eps)
                dice_scores.append(dice)

            return 1.0 - torch.stack(dice_scores).mean()

    class CombinedLoss(nn.Module):
        """60% CrossEntropy + 40% Dice"""
        def __init__(self, num_classes=3):
            super().__init__()
            # class_weights: background nhẹ (0.4), fire/smoke nặng (2.5/2.0)
            # vì background chiếm đa số pixels → không cần nhấn mạnh
            weights = torch.tensor([0.4, 2.5, 2.0], dtype=torch.float32)
            self.ce   = nn.CrossEntropyLoss(weight=weights)
            self.dice = DiceLoss(num_classes=num_classes)

        def forward(self, logits, targets):
            return 0.6 * self.ce(logits, targets) + 0.4 * self.dice(logits, targets)

    # =========================================================================
    # 5. METRICS: mIoU, Dice
    # =========================================================================
    def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, nc: int = 3):
        """
        preds, targets: [N, H, W] long — batch predictions và ground truth
        Returns dict với mIoU, IoU per class, Dice per class
        """
        eps = 1e-6
        ious, dices = [], []
        for c in range(nc):
            p = (preds   == c)
            t = (targets == c)
            tp = (p & t).sum().item()
            fp = (p & ~t).sum().item()
            fn = (~p & t).sum().item()
            ious.append(tp / (tp + fp + fn + eps))
            dices.append(2 * tp / (2 * tp + fp + fn + eps))

        return {
            "mIoU":       sum(ious) / nc,
            "IoU_bg":     ious[0],
            "IoU_fire":   ious[1],
            "IoU_smoke":  ious[2],
            "Dice_fire":  dices[1],
            "Dice_smoke": dices[2],
        }

    # =========================================================================
    # 6. MODEL — SegFormer-B2 (HuggingFace)
    # =========================================================================
    print("\n📦 Tải SegFormer-B2 từ HuggingFace (pretrained ImageNet)...")
    id2label = {0: "background", 1: "fire", 2: "smoke"}
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels          = CFG["num_classes"],
        id2label            = id2label,
        label2id            = label2id,
        ignore_mismatched_sizes = True,
    ).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params:     {n_total/1e6:.2f}M")
    print(f"   Trainable params: {n_train/1e6:.2f}M")

    # =========================================================================
    # 7. DATA LOADERS + WEIGHTED SAMPLER
    # =========================================================================
    from torch.utils.data import WeightedRandomSampler

    print("\n📂 Chuẩn bị DataLoaders...")
    train_ds = FireSmokeSegDataset(CFG["train_img_dir"], CFG["train_msk_dir"],
                                    CFG["img_size"])
    val_ds   = FireSmokeSegDataset(CFG["val_img_dir"],   CFG["val_msk_dir"],
                                    CFG["img_size"])
    test_ds  = FireSmokeSegDataset(CFG["test_img_dir"],  CFG["test_msk_dir"],
                                    CFG["img_size"])

    # ── WeightedRandomSampler: oversample ảnh có fire/smoke nhỏ ───────────
    # Tính sample weight từ mask: ảnh foreground nhỏ (<5% diện tích) → weight cao
    print("   Tính sample weights cho WeightedRandomSampler...")
    sample_weights = []
    msk_dir = CFG["train_msk_dir"]
    for img_name in train_ds.img_files:
        stem     = os.path.splitext(img_name)[0]
        msk_path = os.path.join(msk_dir, f"{stem}.png")
        mask     = load_mask_png(msk_path)
        total    = mask.size
        pct_fg   = (np.sum(mask > 0) / total) * 100.0
        if pct_fg == 0:
            w = CFG["weight_bg_only"]
        elif pct_fg < CFG["small_fg_threshold_pct"]:
            w = CFG["weight_fg_small"]    # đốm lửa/khói nhỏ → oversample nhiều
        else:
            w = CFG["weight_fg_large"]    # vùng lớn → oversample vừa
        sample_weights.append(w)

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )
    print(f"   Weights phân bố: bg={sample_weights.count(CFG['weight_bg_only'])} | "
          f"fg_large={sample_weights.count(CFG['weight_fg_large'])} | "
          f"fg_small={sample_weights.count(CFG['weight_fg_small'])}")

    train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"],
                           sampler=sampler,            # thay shuffle=True
                           num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"],
                           drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False,
                           num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"])
    test_dl  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False,
                           num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"])

    print(f"   Train : {len(train_ds):,} samples → {len(train_dl)} batches/epoch")
    print(f"   Val   : {len(val_ds):,} samples → {len(val_dl)} batches/epoch")
    print(f"   Test  : {len(test_ds):,} samples → {len(test_dl)} batches/epoch")

    # =========================================================================
    # 8. OPTIMIZER & SCHEDULER — Differential LR
    # =========================================================================
    backbone_params = list(model.segformer.parameters())
    decode_params   = list(model.decode_head.parameters())

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": CFG["lr_backbone"]},
        {"params": decode_params,   "lr": CFG["lr_decode"]},
    ], weight_decay=CFG["weight_decay"])

    # PolynomialLR: chuẩn trong mọi segmentation paper (MMSeg, SegFormer paper)
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=CFG["epochs"], power=0.9
    )

    criterion = CombinedLoss(num_classes=CFG["num_classes"]).to(device)

    # Automatic Mixed Precision (AMP FP16) cho A10G GPU — Tăng tốc 3x
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # =========================================================================
    # 9. TRAINING LOOP
    # =========================================================================
    print(f"\n🏋️  Bắt đầu training {CFG['epochs']} epochs  (AMP FP16 Enabled ⚡ | early stop={CFG['patience']})\n")
    best_miou         = 0.0
    epochs_no_improve = 0
    best_weights      = None
    history           = []

    for epoch in range(1, CFG["epochs"] + 1):

        # ── TRAIN ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for imgs, masks in tqdm(train_dl, desc=f"Ep {epoch:3d}/{CFG['epochs']} TRAIN", leave=False):
            imgs  = imgs.to(device, non_blocking=True)    # [B, 3, 512, 512]
            masks = masks.to(device, non_blocking=True)   # [B, 512, 512]

            optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Forward
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(pixel_values=imgs)
                logits  = outputs.logits  # [B, 3, 128, 128]

                # Downsample mask về kích thước output của decoder (H/4 × W/4)
                lh, lw = logits.shape[2], logits.shape[3]
                masks_dn = F.interpolate(
                    masks.unsqueeze(1).float(), size=(lh, lw), mode="nearest"
                ).squeeze(1).long()

                loss = criterion(logits, masks_dn)

            # Scaled Backward & Step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dl)
        scheduler.step()

        # ── VALIDATION ────────────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        all_preds   = []
        all_targets = []

        with torch.no_grad():
            for imgs, masks in tqdm(val_dl, desc=f"Ep {epoch:3d}/{CFG['epochs']} VAL", leave=False):
                imgs  = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(pixel_values=imgs)
                    logits  = outputs.logits

                    # Loss: downsampled mask
                    lh, lw = logits.shape[2], logits.shape[3]
                    masks_dn = F.interpolate(
                        masks.unsqueeze(1).float(), size=(lh, lw), mode="nearest"
                    ).squeeze(1).long()
                    v_loss = criterion(logits, masks_dn)

                val_loss += v_loss.item()

                # Metrics: upsample logits về 512×512
                logits_up = F.interpolate(
                    logits.float(), size=(CFG["img_size"], CFG["img_size"]),
                    mode="bilinear", align_corners=False
                )
                all_preds.append(logits_up.argmax(dim=1).cpu())
                all_targets.append(masks.cpu())

        avg_val_loss = val_loss / len(val_dl)
        all_preds    = torch.cat(all_preds)
        all_targets  = torch.cat(all_targets)
        m            = compute_metrics(all_preds, all_targets, CFG["num_classes"])

        log = (
            f"Ep {epoch:3d}  |  TrLoss {avg_train_loss:.4f}  |  VaLoss {avg_val_loss:.4f}  "
            f"|  mIoU {m['mIoU']:.4f}  |  IoU_fire {m['IoU_fire']:.4f}  "
            f"|  IoU_smoke {m['IoU_smoke']:.4f}  |  Dice_fire {m['Dice_fire']:.4f}"
        )
        print(log)
        history.append({"epoch": epoch, "train_loss": avg_train_loss,
                         "val_loss": avg_val_loss, **m})

        # ── Early Stopping ────────────────────────────────────────────────────
        if m["mIoU"] > best_miou:
            best_miou    = m["mIoU"]
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"  🌟 New best mIoU: {best_miou:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  ⏳ No improve {epochs_no_improve}/{CFG['patience']}")
            if epochs_no_improve >= CFG["patience"]:
                print(f"\n🛑 EARLY STOPPING at epoch {epoch}")
                break

    # =========================================================================
    # 10. LƯU MODEL TỐT NHẤT
    # =========================================================================
    if best_weights:
        model.load_state_dict(best_weights)

    existing = glob.glob(f"{CFG['save_dir']}/segformer_b2_fire_smoke_v*.pth")
    max_v = max(
        [int(re.search(r"_v(\d+)\.pth$", f).group(1)) for f in existing
         if re.search(r"_v(\d+)\.pth$", f)],
        default=0
    )
    save_path = f"{CFG['save_dir']}/segformer_b2_fire_smoke_v{max_v + 1}.pth"

    torch.save({
        "model_state_dict": model.state_dict(),
        "best_miou":        best_miou,
        "config":           CFG,
        "history":          history,
    }, save_path)

    print(f"\n💾 Saved: {save_path}")
    print(f"   Best mIoU (val): {best_miou:.4f}")

    # =========================================================================
    # 11. POST-PROCESSING — Mask → BBox (cho so sánh mAP với Baseline A/B)
    # =========================================================================
    def mask_to_bboxes(mask_pred: np.ndarray):
        """
        Segmentation Mask → Bounding Boxes (YOLO format, normalized).
        Dùng Connected Component Analysis để tách từng vùng lửa/khói riêng biệt.

        Args:
            mask_pred: np.ndarray [H, W] values {0=bg, 1=fire, 2=smoke}
        Returns:
            bboxes: list of [cls_id, cx_norm, cy_norm, w_norm, h_norm]
        """
        h, w = mask_pred.shape
        bboxes = []
        for mask_val, cls_id in [(1, 0), (2, 1)]:     # fire=1→cls0, smoke=2→cls1
            binary = (mask_pred == mask_val).astype(np.uint8)
            if binary.sum() == 0:
                continue
            n_cc, _, stats, _ = cv2.connectedComponentsWithStats(binary)
            for i in range(1, n_cc):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 64:      # Bỏ qua region < 8×8 pixels (nhiễu)
                    continue
                x1 = stats[i, cv2.CC_STAT_LEFT]
                y1 = stats[i, cv2.CC_STAT_TOP]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]
                bboxes.append([
                    cls_id,
                    (x1 + bw / 2) / w,
                    (y1 + bh / 2) / h,
                    bw / w,
                    bh / h,
                ])
        return bboxes

    # Gắn hàm vào model để dễ dùng ở inference
    model.mask_to_bboxes = mask_to_bboxes

    print("\n✅ mask_to_bboxes() sẵn sàng để dùng trong inference & tracking.")

    # =========================================================================
    # 12. ĐÁNH GIÁ TRÊN TEST SET
    # =========================================================================
    print("\n📊 Đánh giá trên TEST SET...")
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for imgs, masks in tqdm(test_dl, desc="Test eval"):
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(pixel_values=imgs)
                logits  = outputs.logits
            logits_up = F.interpolate(
                logits.float(), size=(CFG["img_size"], CFG["img_size"]),
                mode="bilinear", align_corners=False
            )
            test_preds.append(logits_up.argmax(dim=1).cpu())
            test_targets.append(masks.cpu())

    test_preds   = torch.cat(test_preds)
    test_targets = torch.cat(test_targets)
    test_m       = compute_metrics(test_preds, test_targets, CFG["num_classes"])
    print(f"  TEST mIoU     : {test_m['mIoU']:.4f}")
    print(f"  TEST IoU_fire : {test_m['IoU_fire']:.4f}")
    print(f"  TEST IoU_smoke: {test_m['IoU_smoke']:.4f}")
    print(f"  TEST Dice_fire: {test_m['Dice_fire']:.4f}")

    print("\n" + "=" * 70)
    print("🎉  BASELINE C TRAINING HOÀN TẤT!")
    print(f"    Checkpoint: segformer_b2_fire_smoke_v{max_v + 1}.pth")
    print(f"    Best Val mIoU : {best_miou:.4f}")
    print(f"    Test mIoU     : {test_m['mIoU']:.4f}")
    print("=" * 70)


# ===========================================================================
# LOCAL ENTRYPOINT
# ===========================================================================
@app.local_entrypoint()
def main():
    print("🚀 Đang gửi Baseline C job lên Modal (GPU: A10G)...")
    train_segformer.remote()
