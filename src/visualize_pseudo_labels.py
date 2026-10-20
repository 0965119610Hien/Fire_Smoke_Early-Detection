"""
visualize_pseudo_labels.py
==========================
Vẽ Sample Grid trực quan hóa chất lượng Pseudo-label (Ảnh gốc + Mask + Overlay)
từ toàn bộ 20 video đã gán nhãn trong Modal Volume (/data/video_pseudo_labels/).

Tạo 2 biểu đồ trực quan:
  1. pseudo_labels_20_videos_grid.png:
     - Grid hiển thị đại diện cho TOÀN BỘ 20 video (20 hàng × 3 cột: Original | Mask | Overlay)
  2. pseudo_labels_diverse_highlights.png:
     - Grid chọn lọc 16 frames tiêu biểu (Fire, Smoke, Early Smoke, Hard Negative) phóng to rõ nét.

Chạy:
  modal run src/visualize_pseudo_labels.py
"""

import modal
import os
from pathlib import Path

app = modal.App("fire-smoke-visualize-pseudo-labels")
vol = modal.Volume.from_name("fire_smoke_dataset")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "opencv-python-headless",
        "numpy<2.0.0",
        "Pillow",
        "matplotlib",
        "pandas",
        "tqdm",
    )
)


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=1800,
    memory=8192,
)
def generate_pseudo_label_grids():
    import glob
    import os
    from pathlib import Path

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    print("=" * 70)
    print("🎨 TẠO SAMPLE GRID: ẢNH + MASK OVERLAY TỪ 20 VIDEO")
    print("=" * 70)

    ROOT = Path("/data/video_pseudo_labels")
    manifest_path = ROOT / "manifest.csv"

    if not manifest_path.exists():
        print(f"❌ Không tìm thấy manifest tại {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    print(f"📊 Đã đọc {len(df)} frames từ {df['video'].nunique()} videos.")

    # Bảng màu cho mask overlay
    # 0 = Background (trong suốt)
    # 1 = Fire (Cam/Đỏ rực: #FF3D00)
    # 2 = Smoke (Xanh lam/Xám: #00E5FF hoặc #B0BEC5)
    COLOR_FIRE = np.array([255, 61, 0], dtype=np.uint8)     # RGB
    COLOR_SMOKE = np.array([0, 229, 255], dtype=np.uint8)   # RGB

    def create_overlay(img_rgb, mask, alpha=0.5):
        overlay = img_rgb.copy()
        fire_mask = (mask == 1)
        smoke_mask = (mask == 2)

        if fire_mask.any():
            overlay[fire_mask] = (
                (1 - alpha) * overlay[fire_mask] + alpha * COLOR_FIRE
            ).astype(np.uint8)

        if smoke_mask.any():
            overlay[smoke_mask] = (
                (1 - alpha) * overlay[smoke_mask] + alpha * COLOR_SMOKE
            ).astype(np.uint8)

        # Vẽ viền contour nhẹ để làm nổi bật ranh giới
        contours_fire, _ = cv2.findContours(
            fire_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours_fire, -1, (255, 255, 255), 1)

        contours_smoke, _ = cv2.findContours(
            smoke_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours_smoke, -1, (255, 255, 0), 1)

        return overlay

    def colorize_mask(mask):
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        color_mask[mask == 1] = COLOR_FIRE
        color_mask[mask == 2] = COLOR_SMOKE
        return color_mask

    # =========================================================================
    # 1. GRID TOÀN BỘ 20 VIDEO (20 ROWS x 3 COLS)
    # =========================================================================
    videos = sorted(df["video"].unique())
    n_vids = len(videos)
    print(f"\n🖼️ Đang render Grid cho {n_vids} videos...")

    fig, axes = plt.subplots(n_vids, 3, figsize=(15, 4.2 * n_vids))
    fig.patch.set_facecolor("#121212")

    for i, vid in enumerate(videos):
        vid_df = df[df["video"] == vid]
        # Ưu tiên chọn frame có cả fire và smoke, hoặc fire, hoặc smoke
        cand = vid_df[(vid_df["has_fire"] == 1) & (vid_df["has_smoke"] == 1)]
        if len(cand) == 0:
            cand = vid_df[vid_df["has_fire"] == 1]
        if len(cand) == 0:
            cand = vid_df[vid_df["has_smoke"] == 1]
        if len(cand) == 0:
            cand = vid_df

        # Lấy frame ở khoảng giữa để tiêu biểu
        sample_row = cand.iloc[len(cand) // 2]

        frame_p = ROOT / sample_row["frame_file"]
        mask_p = ROOT / sample_row["mask_file"]

        img_bgr = cv2.imread(str(frame_p))
        if img_bgr is None:
            img_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((512, 512), dtype=np.uint8)

        overlay = create_overlay(img_rgb, mask, alpha=0.55)
        mask_rgb = colorize_mask(mask)

        short_title = vid.replace("yt_", "")
        if len(short_title) > 32:
            short_title = short_title[:29] + "..."

        # Col 1: Original Image
        ax1 = axes[i, 0] if n_vids > 1 else axes[0]
        ax1.imshow(img_rgb)
        ax1.set_title(f"[{i+1:02d}] {short_title}\nOriginal Frame", color="white", fontsize=10, fontweight="bold")
        ax1.axis("off")

        # Col 2: Colorized Mask
        ax2 = axes[i, 1] if n_vids > 1 else axes[1]
        ax2.imshow(mask_rgb)
        f_px = sample_row["fire_pixels"]
        s_px = sample_row["smoke_pixels"]
        cov = sample_row["coverage_pct"]
        ax2.set_title(f"Pseudo Mask (Fire: {f_px:,}px | Smoke: {s_px:,}px | {cov:.1f}%)", color="#00E5FF", fontsize=9)
        ax2.axis("off")

        # Col 3: Overlay
        ax3 = axes[i, 2] if n_vids > 1 else axes[2]
        ax3.imshow(overlay)
        ax3.set_title(f"SegFormer-B2 Overlay", color="#FF9100", fontsize=10, fontweight="bold")
        ax3.axis("off")

    plt.tight_layout()
    out_all_grid = "/data/pseudo_labels_20_videos_grid.png"
    plt.savefig(out_all_grid, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"   ✓ Đã lưu: {out_all_grid}")

    # =========================================================================
    # 2. GRID 16 TIÊU BIỂU (4x4 HIGHLIGHTS)
    # =========================================================================
    print("\n🖼️ Đang render Grid 16 Highlights đa dạng...")
    # Chọn 16 samples đa dạng: 6 Fire rõ, 6 Smoke rõ, 4 Cả hai
    both_df = df[(df["has_fire"] == 1) & (df["has_smoke"] == 1)].sample(min(4, len(df)), random_state=42)
    fire_df = df[(df["has_fire"] == 1) & (df["has_smoke"] == 0)].sample(min(6, len(df)), random_state=42)
    smoke_df = df[(df["has_fire"] == 0) & (df["has_smoke"] == 1)].sample(min(6, len(df)), random_state=42)

    highlights = pd.concat([both_df, fire_df, smoke_df]).head(16)

    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    fig.patch.set_facecolor("#181818")
    fig.suptitle(
        "🔥 SegFormer-B2 Pseudo-Labeling Highlights (16 Sample Frames from Videos)\n"
        "[ Cam: Fire | Xanh lam: Smoke ]",
        color="white", fontsize=16, fontweight="bold", y=0.98
    )

    for idx, (_, row) in enumerate(highlights.iterrows()):
        ax = axes[idx // 4, idx % 4]
        frame_p = ROOT / row["frame_file"]
        mask_p = ROOT / row["mask_file"]

        img_bgr = cv2.imread(str(frame_p))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else np.zeros((512, 512, 3), dtype=np.uint8)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((512, 512), dtype=np.uint8)

        overlay = create_overlay(img_rgb, mask, alpha=0.6)

        ax.imshow(overlay)
        label_tag = []
        if row["has_fire"]:
            label_tag.append(f"Fire: {row['fire_pixels']:,}px")
        if row["has_smoke"]:
            label_tag.append(f"Smoke: {row['smoke_pixels']:,}px")
        if not label_tag:
            label_tag.append("Background")

        v_name = row["video"][:20]
        ax.set_title(f"{v_name}\n" + " | ".join(label_tag), color="#EEEEEE", fontsize=9)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_highlights = "/data/pseudo_labels_diverse_highlights.png"
    plt.savefig(out_highlights, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"   ✓ Đã lưu: {out_highlights}")

    vol.commit()
    print("\n✅ Hoàn thành tạo tất cả Sample Grid!")

    # Đọc bytes để gửi về máy local
    with open(out_all_grid, "rb") as f:
        bytes_all_grid = f.read()
    with open(out_highlights, "rb") as f:
        bytes_highlights = f.read()

    return {
        "grid_20_videos": out_all_grid,
        "grid_highlights": out_highlights,
        "bytes_all_grid": bytes_all_grid,
        "bytes_highlights": bytes_highlights,
    }


@app.local_entrypoint()
def main():
    print("🚀 Đang khởi chạy tạo Sample Grid trực quan trên Modal...")
    res = generate_pseudo_label_grids.remote()
    
    local_dir = Path("visualizations")
    local_dir.mkdir(exist_ok=True)
    
    p1 = local_dir / "pseudo_labels_20_videos_grid.png"
    with open(p1, "wb") as f:
        f.write(res["bytes_all_grid"])
    print(f"\n💾 Đã tải về máy local: {p1} ({len(res['bytes_all_grid'])/1e6:.2f} MB)")

    p2 = local_dir / "pseudo_labels_diverse_highlights.png"
    with open(p2, "wb") as f:
        f.write(res["bytes_highlights"])
    print(f"💾 Đã tải về máy local: {p2} ({len(res['bytes_highlights'])/1e6:.2f} MB)")
    print("\n🎉 Bạn có thể mở 2 file ảnh trong thư mục visualizations/ để xem chi tiết!")
