"""
src/verify_pipeline.py
======================
Kiểm tra toàn diện Pipeline và xuất ảnh trực quan hóa mẫu (8 frames clip progression)
kèm lớp phủ Semantic Mask (Đỏ: Lửa, Xanh/Cyan: Khói).
Lưu tại: docs/sample_pipeline_clip_visualization.png
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from video_dataset import get_video_dataloaders, FireSmokeVideoDataset, PREPARED_DIR, SPLITS_DIR

BASE_DIR = Path(r"E:\NCKH\Code\Fire_Smoke_Early-Detection")
DOCS_DIR = BASE_DIR / "docs"

def main():
    print("=" * 70)
    print("  KIỂM TRA PIPELINE DỮ LIỆU & XUẤT ẢNH TRỰC QUAN HÓA CHUỖI THỜI GIAN")
    print("=" * 70)

    # 1. Khởi tạo DataLoader
    train_loader, val_loader, test_loader = get_video_dataloaders(batch_size=2, pin_memory=False)

    print(f"[*] DataLoader đã sẵn sàng:")
    print(f"  • Train : {len(train_loader.dataset)} clips ({len(train_loader)} batches)")
    print(f"  • Val   : {len(val_loader.dataset)} clips ({len(val_loader)} batches)")
    print(f"  • Test  : {len(test_loader.dataset)} clips ({len(test_loader)} batches)")

    # 2. Tìm một clip mẫu có cả Lửa và Khói để trực quan hóa
    train_ds = train_loader.dataset
    target_idx = None
    for idx in range(len(train_ds)):
        row = train_ds.clips_df.iloc[idx]
        if row['category'] == 'both':
            target_idx = idx
            break

    if target_idx is None:
        target_idx = 0

    sample = train_ds[target_idx]
    frames = sample['frames'] # (8, 3, 640, 640)
    masks = sample['masks']   # (8, 640, 640)
    vname = sample['video_name']
    cat = sample['category']

    print(f"\n[*] Chọn clip mẫu để vẽ: Video '{vname}' (Category: {cat}, Start: {sample['start_idx']}, End: {sample['end_idx']})")

    # ImageNet de-normalization
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for t in range(8):
        # Convert tensor to uint8 image
        f_t = frames[t].permute(1, 2, 0).numpy() # (H, W, 3)
        f_t = (f_t * std + mean) * 255.0
        f_t = np.clip(f_t, 0, 255).astype(np.uint8)

        m_t = masks[t].numpy() # (H, W)

        # Tạo overlay màu cho mask
        # 1: Fire -> Đỏ (255, 30, 30)
        # 2: Smoke -> Xanh cyan (0, 200, 255)
        overlay = f_t.copy()
        fire_mask = (m_t == 1)
        smoke_mask = (m_t == 2)

        overlay[smoke_mask] = (0.5 * overlay[smoke_mask] + 0.5 * np.array([0, 200, 255])).astype(np.uint8)
        overlay[fire_mask]  = (0.4 * overlay[fire_mask]  + 0.6 * np.array([255, 40, 40])).astype(np.uint8)

        fire_px = int(np.sum(fire_mask))
        smoke_px = int(np.sum(smoke_mask))

        ax = axes[t]
        ax.imshow(overlay)
        status_txt = f"Frame {t+1} (t={sample['start_idx'] + t}s)\n"
        if fire_px > 0 and smoke_px > 0:
            status_txt += f"🔥 Fire: {fire_px}px | 💨 Smoke: {smoke_px}px"
        elif fire_px > 0:
            status_txt += f"🔥 Fire: {fire_px}px"
        elif smoke_px > 0:
            status_txt += f"💨 Smoke: {smoke_px}px"
        else:
            status_txt += "⬛ Normal (Background)"

        ax.set_title(status_txt, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle(f"Trực Quan Hóa Chuỗi Clip Thời Gian (T=8s) - Video: {vname}", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    out_fig_p = DOCS_DIR / "sample_pipeline_clip_visualization.png"
    plt.savefig(out_fig_p, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[+] Đã xuất ảnh trực quan hóa chuỗi clip thành công:")
    print(f"    👉 {out_fig_p}")
    print("=" * 70)

if __name__ == "__main__":
    main()
