"""
src/static_dataset.py
=====================
PyTorch Dataset & DataLoader cho tập ảnh tĩnh Fire_Smoke.v4i (Stage 1: Spatial Core SegFormer):
1. Đọc cặp ảnh tĩnh và mask từ Fire_Smoke.v4i.png-mask-semantic/train.
2. Tiền xử lý không gian chuẩn hóa:
   - Letterbox Resize giữ nguyên tỷ lệ khung hình gốc (512x512 hoặc 640x640), đệm xám (114, 114, 114).
   - Tăng cường độ tương phản cục bộ CLAHE trên kênh L (không gian màu LAB).
   - Chuẩn hóa ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
3. Tăng cường dữ liệu thích ứng vật lý (Physics-aware Data Augmentation):
   - Random Horizontal Flip (p=0.5).
   - Random Photometric Jitter (Brightness & Contrast +-20%).
   - Random Gaussian Blur nhẹ (3x3, p=0.2).
   - TUYỆT ĐỐI KHÔNG lật dọc (Vertical Flip = False) để bảo toàn chiều bay lên của khói và lửa.
4. Hàm mất mát kết hợp đa nhiệm (Combined Loss: Weighted Cross-Entropy + Dice Loss).
"""

import os
import sys
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r"E:\NCKH\Code\Fire_Smoke_Early-Detection")
DATA_DIR = BASE_DIR / "data"
V4_DIR   = DATA_DIR / "Fire_Smoke.v4i.png-mask-semantic" / "train"
SPLITS_DIR = DATA_DIR / "data_splits"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def letterbox_image_and_mask(img, mask, target_size=(512, 512), pad_color=(114, 114, 114)):
    """
    Letterbox resize cả ảnh và mask đồng thời, giữ nguyên tỷ lệ khung hình.
    """
    th, tw = target_size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    # Resize
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    # Padding
    pad_w = tw - nw
    pad_h = th - nh
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    img_padded = cv2.copyMakeBorder(
        img_resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=pad_color
    )
    mask_padded = cv2.copyMakeBorder(
        mask_resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0
    )

    return img_padded, mask_padded

def apply_clahe_lab(img_bgr, clip_limit=2.0, tile_grid=(8, 8)):
    """Tăng cường tương phản cục bộ trên kênh Lightness (LAB)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    cl = clahe.apply(l)
    enhanced = cv2.merge([cl, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

class FireSmokeStaticDataset(Dataset):
    """
    Dataset cho ảnh tĩnh Stage 1 (SegFormer Spatial Core).
    """
    def __init__(
        self,
        manifest_csv_path,
        data_root=V4_DIR,
        target_size=(512, 512),
        is_training=False,
        apply_clahe=True,
        augment=True
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.is_training = is_training
        self.apply_clahe = apply_clahe
        self.augment = augment and is_training

        if isinstance(manifest_csv_path, (str, Path)):
            self.df = pd.read_csv(manifest_csv_path)
        elif isinstance(manifest_csv_path, pd.DataFrame):
            self.df = manifest_csv_path
        else:
            raise ValueError("manifest_csv_path must be a file path or DataFrame.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_p = self.data_root / row['filename']
        mask_p = self.data_root / row['mask_filename']

        # 1. Đọc ảnh & mask
        img = cv2.imread(str(img_p))
        if img is None:
            img = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

        mask = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)
        if mask is None:
            mask = np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)

        # 2. CLAHE tăng cường tương phản
        if self.apply_clahe:
            img = apply_clahe_lab(img)

        # 3. Letterbox Resize (giữ nguyên aspect ratio)
        img, mask = letterbox_image_and_mask(img, mask, target_size=self.target_size)

        # 4. Data Augmentation thích ứng vật lý (chỉ cho tập Train)
        if self.augment:
            # a. Lật ngang (Horizontal Flip, p=0.5)
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

            # b. Biến thiên độ sáng/tương phản (Photometric jitter +-20%)
            if random.random() < 0.5:
                alpha = random.uniform(0.80, 1.20)
                beta = random.uniform(-20, 20)
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

            # c. Làm mờ nhẹ (Gaussian Blur p=0.2)
            if random.random() < 0.2:
                img = cv2.GaussianBlur(img, (3, 3), 0)

        # 5. Chuyển BGR sang RGB và chuẩn hóa Tensor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_norm = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float() # (C, H, W)

        mask_tensor = torch.from_numpy(mask.astype(np.int64)).long() # (H, W)

        has_fire = float(row['has_fire'])
        has_smoke = float(row['has_smoke'])
        is_empty = 1.0 if (has_fire == 0 and has_smoke == 0) else 0.0
        label_tensor = torch.tensor([has_fire, has_smoke, is_empty], dtype=torch.float32)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": label_tensor,
            "filename": row['filename'],
            "category": row['category']
        }

class DiceLoss(nn.Module):
    """Multi-class Dice Loss chống mất cân bằng diện tích đối tượng nhỏ."""
    def __init__(self, smooth=1e-5, ignore_index=-100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: (B, num_classes, H, W)
        # targets: (B, H, W)
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(targets.clamp(min=0), num_classes=num_classes) # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float() # (B, C, H, W)

        dice_per_class = []
        # Bỏ qua class background khi tính dice nếu muốn, ở đây tính class 1 (fire) và 2 (smoke)
        for c in range(1, num_classes):
            p = probs[:, c, :, :].contiguous().view(-1)
            t = targets_one_hot[:, c, :, :].contiguous().view(-1)
            intersection = (p * t).sum()
            dice = (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_per_class.append(1.0 - dice)

        return torch.mean(torch.stack(dice_per_class)) if dice_per_class else torch.tensor(0.0)

class CombinedLoss(nn.Module):
    """
    Combined Loss: Weighted CrossEntropy + Dice Loss
    """
    def __init__(self, class_weights=None, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice, loss_ce, loss_dice

def get_static_dataloaders(
    batch_size=8,
    target_size=(512, 512),
    num_workers=0,
    pin_memory=False
):
    """Tạo 3 DataLoader chuẩn Train, Val, Test cho tập ảnh tĩnh v4i."""
    train_csv = SPLITS_DIR / "static_v4i_train_manifest.csv"
    val_csv   = SPLITS_DIR / "static_v4i_val_manifest.csv"
    test_csv  = SPLITS_DIR / "static_v4i_test_manifest.csv"

    train_ds = FireSmokeStaticDataset(train_csv, target_size=target_size, is_training=True, augment=True)
    val_ds   = FireSmokeStaticDataset(val_csv,   target_size=target_size, is_training=False, augment=False)
    test_ds  = FireSmokeStaticDataset(test_csv,  target_size=target_size, is_training=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print("[*] Kiểm thử khởi tạo Dataset & DataLoader ảnh tĩnh v4i...")
    tr_l, va_l, te_l = get_static_dataloaders(batch_size=4)
    print(f"  • Train : {len(tr_l.dataset)} ảnh ({len(tr_l)} batches)")
    print(f"  • Val   : {len(va_l.dataset)} ảnh ({len(va_l)} batches)")
    print(f"  • Test  : {len(te_l.dataset)} ảnh ({len(te_l)} batches)")

    sample_b = next(iter(tr_l))
    print(f"\n[*] Kiểm tra kích thước Batch:")
    print(f"  • image.shape: {sample_b['image'].shape} (Batch, Channels, Height, Width)")
    print(f"  • mask.shape : {sample_b['mask'].shape} (Batch, Height, Width)")
    print(f"  • label.shape: {sample_b['label'].shape}")

    # Kiểm tra Loss function
    dummy_logits = torch.randn(4, 3, 512, 512)
    criterion = CombinedLoss()
    total_loss, ce_loss, d_loss = criterion(dummy_logits, sample_b['mask'])
    print(f"\n[*] Kiểm tra hàm mất mát Combined Loss:")
    print(f"  • Total Loss: {total_loss.item():.4f} (CE: {ce_loss.item():.4f}, Dice: {d_loss.item():.4f})")
    print("✅ Kiểm thử static_dataset.py thành công!")
