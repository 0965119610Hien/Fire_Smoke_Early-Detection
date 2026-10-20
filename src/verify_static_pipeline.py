"""
src/verify_static_pipeline.py
=============================
Trực quan hóa mẫu kiểm tra Pipeline ảnh tĩnh Fire_Smoke.v4i:
Hiển thị 4 mẫu tiêu biểu (Both, Fire Only, Smoke Only, Small Fire Spot)
với lớp phủ Semantic Mask màu (Đỏ = Lửa, Cyan = Khói) và thông số diện tích.
Lưu tại: docs/sample_static_v4i_pipeline_visualization.png
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

from static_dataset import get_static_dataloaders, FireSmokeStaticDataset

BASE_DIR = Path(r"E:\NCKH\Code\Fire_Smoke_Early-Detection")
DOCS_DIR = BASE_DIR / "docs"

def main():
    print("=" * 70)
    print("  XUẤT ẢNH TRỰC QUAN HÓA PIPELINE ẢNH TĨNH V4I (SEGFORMER INPUT)")
    print("=" * 70)

    train_loader, val_loader, test_loader = get_static_dataloaders(batch_size=4)
    train_ds = train_loader.dataset

    # Tìm 4 mẫu đại diện: both, fire_only, smoke_only, small_fire
    samples = []
    found_cats = set()

    for idx in range(len(train_ds)):
        row = train_ds.df.iloc[idx]
        cat = row['category']
        if cat in ['both', 'fire_only', 'smoke_only'] and cat not in found_cats:
            samples.append((idx, cat, row))
            found_cats.add(cat)
        if len(samples) == 3:
            break

    # Thêm 1 mẫu fire nhỏ
    small_fire_idx = train_ds.df[train_ds.df['fire_pixels'] > 0].sort_values(by='fire_pixels').index[10]
    samples.append((small_fire_idx, 'small_fire', train_ds.df.iloc[small_fire_idx]))

    # ImageNet de-normalization
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    for i, (idx, label_tag, row) in enumerate(samples):
        data = train_ds[idx]
        img_t = data['image'].permute(1, 2, 0).numpy()
        img_rgb = (img_t * std + mean) * 255.0
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

        mask = data['mask'].numpy()

        overlay = img_rgb.copy()
        fire_mask = (mask == 1)
        smoke_mask = (mask == 2)

        overlay[smoke_mask] = (0.5 * overlay[smoke_mask] + 0.5 * np.array([0, 200, 255])).astype(np.uint8)
        overlay[fire_mask]  = (0.4 * overlay[fire_mask]  + 0.6 * np.array([255, 30, 30])).astype(np.uint8)

        fire_px = int(np.sum(fire_mask))
        smoke_px = int(np.sum(smoke_mask))

        ax = axes[i]
        ax.imshow(overlay)
        title = f"Mẫu {i+1}: {row['category'].upper()}\n"
        if fire_px > 0 and smoke_px > 0:
            title += f"Fire: {fire_px}px | Smoke: {smoke_px}px"
        elif fire_px > 0:
            title += f"Fire: {fire_px}px ({row['fire_ratio']:.2f}%)"
        elif smoke_px > 0:
            title += f"Smoke: {smoke_px}px ({row['smoke_ratio']:.2f}%)"
        else:
            title += "Empty Background"

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle("Trực Quan Hóa Mẫu Pipeline Ảnh Tĩnh (Fire_Smoke.v4i) - Đầu Vào Huấn Luyện SegFormer", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    out_p = DOCS_DIR / "sample_static_v4i_pipeline_visualization.png"
    plt.savefig(out_p, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[+] Đã xuất ảnh trực quan hóa mẫu thành công:")
    print(f"    👉 {out_p}")
    print("=" * 70)

if __name__ == "__main__":
    main()
