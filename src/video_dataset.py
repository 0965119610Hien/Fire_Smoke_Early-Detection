"""
src/video_dataset.py
====================
PyTorch Spatio-Temporal Dataset & DataLoader cho bài toán phát hiện sớm lửa và khói trên Video:
1. Nạp chuỗi T frame (mặc định T=8) kèm T mask phân vùng ngữ nghĩa (0: BG, 1: Fire, 2: Smoke).
2. Chuẩn hóa không gian (Spatial):
   - Kích thước chuẩn hóa 640x640 (hoặc tùy biến 512x512).
   - Tăng cường tương phản cục bộ CLAHE trên kênh L (không gian màu LAB).
   - Chuẩn hóa tensor ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
3. Tăng cường dữ liệu thích ứng vật lý (Physics-aware Data Augmentation):
   - Lật ngang đồng bộ chuỗi (Horizontal Flip, p=0.5).
   - TUYỆT ĐỐI KHÔNG lật dọc (Vertical Flip = False) để bảo toàn hướng bốc lên của khói và lửa.
   - Biến thiên ánh sáng (Brightness/Contrast jitter) đồng bộ cho toàn bộ T frame trong clip.
   - Thêm nhiễu làm mờ (Motion/Gaussian Blur, p=0.2).
"""

import os
import sys
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r"E:\NCKH\Code\Fire_Smoke_Early-Detection")
DATA_DIR = BASE_DIR / "data"
PREPARED_DIR = DATA_DIR / "dataset_video_prepared"
SPLITS_DIR = DATA_DIR / "data_splits"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def apply_clahe_lab(img_bgr, clip_limit=2.0, tile_grid=(8, 8)):
    """Tăng cường tương phản trên kênh Lightness của không gian LAB."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    cl = clahe.apply(l)
    enhanced = cv2.merge([cl, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

class FireSmokeVideoDataset(Dataset):
    """
    Dataset nạp chuỗi thời gian video (Spatio-Temporal Video Clips).
    Mỗi mẫu trả về gồm chuỗi T frames và T semantic segmentation masks.
    """
    def __init__(
        self,
        clips_csv_path,
        data_root=PREPARED_DIR,
        target_size=(640, 640),
        clip_len=8,
        is_training=False,
        apply_clahe=True,
        augment=True
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.clip_len = clip_len
        self.is_training = is_training
        self.apply_clahe = apply_clahe
        self.augment = augment and is_training

        if isinstance(clips_csv_path, (str, Path)):
            self.clips_df = pd.read_csv(clips_csv_path)
        elif isinstance(clips_csv_path, pd.DataFrame):
            self.clips_df = clips_csv_path
        else:
            raise ValueError("clips_csv_path must be a file path or DataFrame.")

    def __len__(self):
        return len(self.clips_df)

    def __getitem__(self, idx):
        row = self.clips_df.iloc[idx]
        img_paths_rel = row['image_paths'].split(";")
        mask_paths_rel = row['mask_paths'].split(";")

        frames_list = []
        masks_list = []

        for img_rel, mask_rel in zip(img_paths_rel, mask_paths_rel):
            img_p = self.data_root / img_rel
            mask_p = self.data_root / mask_rel

            # 1. Đọc ảnh
            img = cv2.imread(str(img_p))
            if img is None:
                img = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            else:
                if self.apply_clahe:
                    img = apply_clahe_lab(img)
                if (img.shape[1], img.shape[0]) != self.target_size:
                    img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 2. Đọc mask
            mask = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)
            if mask is None:
                mask = np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)
            else:
                if (mask.shape[1], mask.shape[0]) != self.target_size:
                    mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

            frames_list.append(img)
            masks_list.append(mask)

        # 3. Áp dụng Augmentation thích ứng vật lý đồng bộ cho cả chuỗi (Sequence-level)
        if self.augment:
            # a. Lật ngang (Horizontal Flip) đồng bộ p=0.5
            if random.random() < 0.5:
                frames_list = [cv2.flip(f, 1) for f in frames_list]
                masks_list = [cv2.flip(m, 1) for m in masks_list]

            # b. Biến thiên độ sáng/tương phản đồng bộ (Brightness/Contrast jitter)
            if random.random() < 0.5:
                alpha = random.uniform(0.85, 1.15) # Tương phản
                beta = random.uniform(-15, 15)      # Độ sáng
                frames_list = [np.clip(alpha * f + beta, 0, 255).astype(np.uint8) for f in frames_list]

            # c. Làm mờ nhẹ (Gaussian Blur) đồng bộ p=0.2
            if random.random() < 0.2:
                frames_list = [cv2.GaussianBlur(f, (3, 3), 0) for f in frames_list]

        # 4. Chuẩn hóa Tensor
        # frames: (T, H, W, C) -> (T, C, H, W)
        frames_np = np.stack(frames_list, axis=0).astype(np.float32) / 255.0
        frames_np = (frames_np - IMAGENET_MEAN) / IMAGENET_STD
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()

        # masks: (T, H, W) -> int64 (0: BG, 1: Fire, 2: Smoke)
        masks_np = np.stack(masks_list, axis=0).astype(np.int64)
        masks_tensor = torch.from_numpy(masks_np).long()

        # 5. Nhãn cấp độ Clip (Multi-label classification: has_fire, has_smoke, is_empty)
        has_fire = float(row['has_fire'])
        has_smoke = float(row['has_smoke'])
        is_empty = 1.0 if (has_fire == 0 and has_smoke == 0) else 0.0
        clip_label = torch.tensor([has_fire, has_smoke, is_empty], dtype=torch.float32)

        return {
            "frames": frames_tensor,         # (T, 3, H, W)
            "masks": masks_tensor,           # (T, H, W)
            "clip_label": clip_label,       # (3,)
            "video_name": row['video_name'],
            "category": row['category'],
            "start_idx": row['start_idx'],
            "end_idx": row['end_idx'],
        }

def get_video_dataloaders(
    batch_size=4,
    target_size=(640, 640),
    num_workers=0,
    pin_memory=True
):
    """
    Hàm tiện ích tạo 3 DataLoader chuẩn (Train, Val, Test) cho toàn bộ pipeline.
    """
    train_csv = SPLITS_DIR / "train_clips.csv"
    val_csv   = SPLITS_DIR / "val_clips.csv"
    test_csv  = SPLITS_DIR / "test_clips.csv"

    train_ds = FireSmokeVideoDataset(
        train_csv, target_size=target_size, is_training=True, augment=True
    )
    val_ds = FireSmokeVideoDataset(
        val_csv, target_size=target_size, is_training=False, augment=False
    )
    test_ds = FireSmokeVideoDataset(
        test_csv, target_size=target_size, is_training=False, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print("[*] Kiểm thử khởi tạo Dataset & DataLoader...")
    tr_l, va_l, te_l = get_video_dataloaders(batch_size=2)
    print(f"  • Train Loader: {len(tr_l.dataset)} clips ({len(tr_l)} batches)")
    print(f"  • Val Loader  : {len(va_l.dataset)} clips ({len(va_l)} batches)")
    print(f"  • Test Loader : {len(te_l.dataset)} clips ({len(te_l)} batches)")

    sample_batch = next(iter(tr_l))
    print(f"\n[*] Kích thước Batch mẫu (Train):")
    print(f"  • frames.shape    : {sample_batch['frames'].shape} (Batch, Time, Channels, Height, Width)")
    print(f"  • masks.shape     : {sample_batch['masks'].shape} (Batch, Time, Height, Width)")
    print(f"  • clip_label.shape: {sample_batch['clip_label'].shape}")
    print(f"  • video_names     : {sample_batch['video_name']}")
    print("✅ Kiểm thử video_dataset.py thành công!")
