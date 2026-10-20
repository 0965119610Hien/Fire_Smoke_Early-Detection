"""
=============================================================================
SEGFORMER-B2 VISUALIZATION SCRIPT — Sample Grid trên Tập Test Chuẩn
=============================================================================
Mục tiêu:
  1. Load checkpoint SegFormer-B2 tốt nhất (tự động tìm version mới nhất v2/v3...)
  2. Đọc toàn bộ ảnh & mask từ tập Test chuẩn:
       /data/fire_smoke_seg_prepared/test/images/
       /data/fire_smoke_seg_prepared/test/masks/
  3. Dự đoán segmentation mask trên toàn bộ 141 ảnh test.
  4. Tạo 2 Sample Grid siêu nét (Dark Theme chuyên nghiệp):
       - testset_sample_grid_24.png:
           24 mẫu tiêu biểu (Fire, Smoke, Cả hai, Negative)
           Mỗi mẫu gồm 4 cột: [Ảnh gốc] | [Ground Truth] | [Dự đoán SegFormer] | [Overlay + IoU/Dice]
       - testset_montage_all_highlights.png:
           Grid 6×6 (36 mẫu) Overlay kết quả dự đoán trên các trường hợp khó/đa dạng.
  5. Tự động tải 2 file hình ảnh về thư mục `visualizations/` trên máy local.

Màu sắc quy ước:
  - 🔴 Đỏ cam (#FF3D00): Vùng Lửa (Fire)
  - 🔵 Xanh lam (#00E5FF): Vùng Khói (Smoke)

Chạy:
  modal run src/visualize_segformer.py
=============================================================================
"""

import modal
import os
from pathlib import Path

app = modal.App("fire-smoke-segformer-visualization")
vol = modal.Volume.from_name("fire_smoke_dataset")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "transformers==4.40.0",
        "opencv-python-headless",
        "numpy<2.0.0",
        "Pillow",
        "matplotlib",
        "tqdm",
    )
)


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="A10G",
    timeout=3600,
    memory=16384,
)
def visualize_test_samples():
    import glob
    import os
    import re
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformers import SegformerForSemanticSegmentation

    print("=" * 70)
    print("🎨 TRỰC QUAN HÓA SAMPLE GRID — SEGFORMER-B2 TRÊN TẬP TEST")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Tìm checkpoint mới nhất
    ckpt_files = sorted(glob.glob("/data/segformer_b2_fire_smoke_v*.pth"))
    if not ckpt_files:
        print("❌ Không tìm thấy checkpoint SegFormer nào!")
        return None

    def _ver(p):
        m = re.search(r"_v(\d+)\.pth$", p)
        return int(m.group(1)) if m else 0

    checkpoint_path = max(ckpt_files, key=_ver)
    print(f"\n📦 Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    img_size = cfg.get("img_size", 512)

    id2label = {0: "background", 1: "fire", 2: "smoke"}
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("   ✓ Load model thành công!")

    # 2. Đọc Test Set chuẩn
    TEST_IMG_DIR = "/data/fire_smoke_seg_prepared/test/images"
    TEST_MSK_DIR = "/data/fire_smoke_seg_prepared/test/masks"

    if not os.path.exists(TEST_IMG_DIR):
        print(f"❌ Không tìm thấy thư mục: {TEST_IMG_DIR}")
        return None

    img_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".png")])
    total_samples = len(img_files)
    print(f"📂 Tổng số mẫu test: {total_samples}")

    COLOR_FIRE = np.array([255, 61, 0], dtype=np.uint8)     # RGB
    COLOR_SMOKE = np.array([0, 229, 255], dtype=np.uint8)   # RGB

    def colorize_mask(mask):
        h, w = mask.shape
        c_mask = np.zeros((h, w, 3), dtype=np.uint8)
        c_mask[mask == 1] = COLOR_FIRE
        c_mask[mask == 2] = COLOR_SMOKE
        return c_mask

    def create_overlay(img_rgb, mask, alpha=0.55):
        overlay = img_rgb.copy()
        m_fire = (mask == 1)
        m_smoke = (mask == 2)

        if m_fire.any():
            overlay[m_fire] = ((1 - alpha) * overlay[m_fire] + alpha * COLOR_FIRE).astype(np.uint8)
            contours, _ = cv2.findContours(m_fire.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

        if m_smoke.any():
            overlay[m_smoke] = ((1 - alpha) * overlay[m_smoke] + alpha * COLOR_SMOKE).astype(np.uint8)
            contours, _ = cv2.findContours(m_smoke.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 0), 1)

        return overlay

    # 3. Chạy Inference trên toàn bộ Test Set
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    samples_data = []

    print("\n🧪 Đang chạy dự đoán SegFormer trên toàn bộ tập Test...")
    for f_name in tqdm(img_files):
        stem = os.path.splitext(f_name)[0]
        img_p = os.path.join(TEST_IMG_DIR, f_name)
        msk_p = os.path.join(TEST_MSK_DIR, f"{stem}.png")

        img_bgr = cv2.imread(img_p)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[:2] != (img_size, img_size):
            img_rgb = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        gt_mask = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            gt_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        elif gt_mask.shape[:2] != (img_size, img_size):
            gt_mask = cv2.resize(gt_mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        # Chuẩn bị tensor input
        nrm = (img_rgb.astype(np.float32) / 255.0 - mean) / std
        inp = torch.from_numpy(nrm).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = model(pixel_values=inp).logits
            logits_up = F.interpolate(logits.float(), size=(img_size, img_size), mode="bilinear", align_corners=False)
            pred_mask = logits_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Tính IoU riêng cho mẫu này
        def compute_sample_iou(p_m, g_m, c_id):
            p = (p_m == c_id)
            g = (g_m == c_id)
            inter = (p & g).sum()
            union = (p | g).sum()
            if union == 0:
                return 1.0 if inter == 0 else 0.0
            return inter / (union + 1e-7)

        iou_f = compute_sample_iou(pred_mask, gt_mask, 1)
        iou_s = compute_sample_iou(pred_mask, gt_mask, 2)

        has_gt_fire = (gt_mask == 1).sum() > 20
        has_gt_smoke = (gt_mask == 2).sum() > 50
        has_pr_fire = (pred_mask == 1).sum() > 20
        has_pr_smoke = (pred_mask == 2).sum() > 50

        category = "bg"
        if has_gt_fire and has_gt_smoke:
            category = "both"
        elif has_gt_fire:
            category = "fire"
        elif has_gt_smoke:
            category = "smoke"

        samples_data.append({
            "name": f_name,
            "stem": stem,
            "img_rgb": img_rgb,
            "gt_mask": gt_mask,
            "pred_mask": pred_mask,
            "iou_fire": iou_f,
            "iou_smoke": iou_s,
            "category": category,
            "has_gt_fire": has_gt_fire,
            "has_gt_smoke": has_gt_smoke,
        })

    # =========================================================================
    # 4. GRID 24 MẪU TIÊU BIỂU (4 CỘT: ẢNH GỐC | GROUND TRUTH | DỰ ĐOÁN | OVERLAY)
    # =========================================================================
    print("\n🖼️ Đang vẽ Sample Grid 24 mẫu tiêu biểu (4 cột)...")

    # Lấy phân bổ mẫu: 8 Fire, 8 Smoke, 5 Both, 3 Negative
    both_list = [s for s in samples_data if s["category"] == "both"]
    fire_list = [s for s in samples_data if s["category"] == "fire"]
    smoke_list = [s for s in samples_data if s["category"] == "smoke"]
    bg_list = [s for s in samples_data if s["category"] == "bg"]

    selected_24 = []
    selected_24.extend(both_list[:min(5, len(both_list))])
    selected_24.extend(fire_list[:min(8, len(fire_list))])
    selected_24.extend(smoke_list[:min(8, len(smoke_list))])
    selected_24.extend(bg_list[:min(3, len(bg_list))])

    # Nếu chưa đủ 24 thì bổ sung
    if len(selected_24) < 24:
        rem = [s for s in samples_data if s not in selected_24]
        selected_24.extend(rem[: 24 - len(selected_24)])

    N_ROWS = len(selected_24)
    fig, axes = plt.subplots(N_ROWS, 4, figsize=(18, 4.2 * N_ROWS))
    fig.patch.set_facecolor("#101010")

    col_headers = [
        "1. Original Image (Input)",
        "2. Ground Truth Annotation",
        "3. SegFormer-B2 Prediction",
        "4. Prediction Overlay & Score",
    ]

    for col_idx, h_text in enumerate(col_headers):
        axes[0, col_idx].text(
            0.5, 1.15, h_text,
            ha="center", va="bottom",
            transform=axes[0, col_idx].transAxes,
            color="#FFCC00", fontsize=12, fontweight="bold"
        )

    for i, s in enumerate(selected_24):
        img_rgb = s["img_rgb"]
        gt_c = colorize_mask(s["gt_mask"])
        pr_c = colorize_mask(s["pred_mask"])
        ovl = create_overlay(img_rgb, s["pred_mask"], alpha=0.55)

        short_name = s["name"]
        if len(short_name) > 24:
            short_name = short_name[:21] + "..."

        # Col 1: Original
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"[{i+1:02d}] {short_name}", color="white", fontsize=9)
        axes[i, 0].axis("off")

        # Col 2: GT
        axes[i, 1].imshow(gt_c)
        cat_str = s["category"].upper()
        axes[i, 1].set_title(f"GT: {cat_str}", color="#B0BEC5", fontsize=9)
        axes[i, 1].axis("off")

        # Col 3: Pred
        axes[i, 2].imshow(pr_c)
        axes[i, 2].set_title("SegFormer Mask", color="#00E5FF", fontsize=9)
        axes[i, 2].axis("off")

        # Col 4: Overlay + Score
        axes[i, 3].imshow(ovl)
        score_info = []
        if s["has_gt_fire"]:
            score_info.append(f"Fire IoU: {s['iou_fire']*100:.1f}%")
        if s["has_gt_smoke"]:
            score_info.append(f"Smoke IoU: {s['iou_smoke']*100:.1f}%")
        score_text = " | ".join(score_info) if score_info else "Background Match"
        axes[i, 3].set_title(score_text, color="#76FF03", fontsize=9, fontweight="bold")
        axes[i, 3].axis("off")

    plt.tight_layout()
    out_grid_24 = "/data/testset_sample_grid_24.png"
    plt.savefig(out_grid_24, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"   ✓ Đã lưu: {out_grid_24}")

    # =========================================================================
    # 5. MONTAGE 36 MẪU (6x6 OVERLAY GRID)
    # =========================================================================
    print("\n🖼️ Đang vẽ Montage 36 mẫu đa dạng (6×6 Grid)...")
    montage_samples = samples_data[:36]
    fig, axes = plt.subplots(6, 6, figsize=(22, 22))
    fig.patch.set_facecolor("#141414")
    fig.suptitle(
        "🔥 SegFormer-B2 Test Set Evaluation Montage (36 Samples Overlay)\n"
        "[ Cam: Fire | Xanh lam: Smoke | Viền trắng: Biên giới phân đoạn ]",
        color="white", fontsize=16, fontweight="bold", y=0.99
    )

    for idx, s in enumerate(montage_samples):
        ax = axes[idx // 6, idx % 6]
        ovl = create_overlay(s["img_rgb"], s["pred_mask"], alpha=0.6)
        ax.imshow(ovl)

        tag = []
        if (s["pred_mask"] == 1).sum() > 20:
            tag.append("Fire")
        if (s["pred_mask"] == 2).sum() > 50:
            tag.append("Smoke")
        tag_str = " + ".join(tag) if tag else "Clean BG"

        ax.set_title(f"#{idx+1:02d} {tag_str}", color="#EEEEEE", fontsize=9)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_montage = "/data/testset_montage_all_highlights.png"
    plt.savefig(out_montage, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"   ✓ Đã lưu: {out_montage}")

    vol.commit()

    # Đọc bytes để truyền về local
    with open(out_grid_24, "rb") as f:
        bytes_24 = f.read()
    with open(out_montage, "rb") as f:
        bytes_montage = f.read()

    return {
        "grid_24_path": out_grid_24,
        "montage_path": out_montage,
        "bytes_24": bytes_24,
        "bytes_montage": bytes_montage,
        "total_test": total_samples,
    }


@app.local_entrypoint()
def main():
    print("🚀 Đang khởi chạy trực quan hóa SegFormer trên Tập Test Modal...")
    res = visualize_test_samples.remote()

    if not res:
        print("❌ Lỗi trong quá trình chạy.")
        return

    local_dir = Path("visualizations")
    local_dir.mkdir(exist_ok=True)

    p1 = local_dir / "testset_sample_grid_24.png"
    with open(p1, "wb") as f:
        f.write(res["bytes_24"])
    print(f"\n💾 Đã tải về máy local: {p1} ({len(res['bytes_24'])/1e6:.2f} MB)")

    p2 = local_dir / "testset_montage_all_highlights.png"
    with open(p2, "wb") as f:
        f.write(res["bytes_montage"])
    print(f"💾 Đã tải về máy local: {p2} ({len(res['bytes_montage'])/1e6:.2f} MB)")

    print("\n🎉 Hoàn tất! Bạn có thể mở 2 file ảnh trong thư mục visualizations/ để xem chi tiết!")
