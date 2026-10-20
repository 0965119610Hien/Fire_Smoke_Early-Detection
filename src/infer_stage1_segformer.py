"""
src/infer_stage1_segformer.py
==============================
Chạy suy luận (Inference) và trực quan hóa chuyên sâu mô hình Stage 1 (SegFormer-B2)
trên tập TEST ĐỘC LẬP (158 ảnh tĩnh chưa từng thấy - Unseen Test Set):
- Đọc test manifest: /data/data_splits/static_v4i_test_manifest.csv
- Nạp checkpoint tốt nhất: /data/checkpoints/segformer_spatial_best.pth
- Tạo ảnh so sánh chi tiết 3-Panel: [Ảnh Gốc] | [Ground Truth] | [Dự Đoán SegFormer + IoU]
- Tạo bảng Montage Grid tổng hợp 16 mẫu tiêu biểu (Fire Only, Smoke Only, Cả hai)
- Tự động tải toàn bộ ảnh kết quả về máy: docs/segformer_test_predictions/

Cách chạy trên Modal GPU:
  modal run src/infer_stage1_segformer.py
"""

import modal
import os
import sys

app = modal.App("fire-smoke-infer-segformer")
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
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "pandas",
    )
)

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def run_segformer_test_inference(num_samples: int = 16):
    import cv2
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import torch
    import torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"🚀 [STAGE 1 INFERENCE] Suy luận SegFormer-B2 trên tập Test tĩnh (GPU: {torch.cuda.get_device_name(0)})")
    print("=" * 80)

    DATA_DIR = Path("/data")
    SPLITS_DIR = DATA_DIR / "data_splits"
    CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
    V4I_DIR = DATA_DIR / "Fire_Smoke.v4i" / "train"
    OUT_DIR = DATA_DIR / "segformer_test_predictions"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    S1_CKPT = CHECKPOINTS_DIR / "segformer_spatial_best.pth"
    TEST_CSV = SPLITS_DIR / "static_v4i_test_manifest.csv"

    assert S1_CKPT.exists(), f"Không tìm thấy checkpoint Stage 1 tại {S1_CKPT}"
    assert TEST_CSV.exists(), f"Không tìm thấy {TEST_CSV}"
    assert V4I_DIR.exists(), f"Không tìm thấy {V4I_DIR}"

    print(f"📦 Đang nạp mô hình SegFormer-B2 từ {S1_CKPT}...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", num_labels=3, ignore_mismatched_sizes=True
    ).to(device)
    ckpt = torch.load(S1_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    df_test = pd.read_csv(TEST_CSV)
    print(f"📄 Tổng số ảnh trong tập Test: {len(df_test)}")

    # Chọn mẫu đại diện đồng đều các nhóm: fire_only, smoke_only, both
    df_fire = df_test[df_test['category'] == 'fire_only']
    df_smoke = df_test[df_test['category'] == 'smoke_only']
    df_both = df_test[df_test['category'] == 'both']

    samples = []
    samples.extend(df_both.head(4).to_dict('records'))
    samples.extend(df_fire.head(6).to_dict('records'))
    samples.extend(df_smoke.head(6).to_dict('records'))

    if len(samples) < num_samples:
        remaining = df_test[~df_test['filename'].isin([s['filename'] for s in samples])].head(num_samples - len(samples))
        samples.extend(remaining.to_dict('records'))

    samples = samples[:num_samples]
    print(f"🎯 Đang xử lý và tạo ảnh phân tích cho {len(samples)} mẫu tiêu biểu...")

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    COLOR_FIRE = (0, 0, 255)       # Đỏ BGR
    COLOR_SMOKE = (0, 165, 255)    # Vàng cam BGR

    def overlay_prediction(img_bgr, mask, alpha=0.5):
        overlay = img_bgr.copy()
        overlay[mask == 1] = ((1 - alpha) * overlay[mask == 1] + alpha * np.array(COLOR_FIRE)).astype(np.uint8)
        overlay[mask == 2] = ((1 - alpha) * overlay[mask == 2] + alpha * np.array(COLOR_SMOKE)).astype(np.uint8)

        # Viền sắc nét
        for cls_id, col in [(1, COLOR_FIRE), (2, COLOR_SMOKE)]:
            bin_m = (mask == cls_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, col, 2)
        return overlay

    def add_label(img, text, color=(255, 255, 255), bg_color=(25, 25, 25)):
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 36), bg_color, -1)
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        return img

    results_files = {}
    montage_tiles = []

    with torch.no_grad():
        for idx, row in enumerate(tqdm(samples, desc="Inference SegFormer")):
            img_p = V4I_DIR / row["filename"]
            mask_p = V4I_DIR / row["mask_filename"]

            img = cv2.imread(str(img_p))
            mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue

            h, w = img.shape[:2]
            scale = min(512 / w, 512 / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            mask_r = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

            pad_w = 512 - nw
            pad_h = 512 - nh
            img_pad = cv2.copyMakeBorder(img_r, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            mask_pad = cv2.copyMakeBorder(mask_r, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2, cv2.BORDER_CONSTANT, value=0)
            mask_pad = np.clip(mask_pad, 0, 2).astype(np.uint8)

            img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_norm = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
            tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

            outputs = model(pixel_values=tensor)
            logits = outputs.logits
            logits_up = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
            pred = torch.argmax(logits_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Tính IoU
            def compute_iou(p, g, c):
                inter = np.sum((p == c) & (g == c))
                union = np.sum((p == c) | (g == c))
                return (inter / union * 100) if union > 0 else (100.0 if np.sum(g == c) == 0 and np.sum(p == c) == 0 else 0.0)

            iou_f = compute_iou(pred, mask_pad, 1)
            iou_s = compute_iou(pred, mask_pad, 2)

            # 3 Panels: Raw | Ground Truth | Prediction
            p1 = add_label(img_pad.copy(), f"Raw Image [{row['category']}]", (255, 255, 255))
            p2 = overlay_prediction(img_pad.copy(), mask_pad)
            p2 = add_label(p2, "Ground Truth Mask (Red:Fire, Orange:Smoke)", (0, 255, 0))
            p3 = overlay_prediction(img_pad.copy(), pred)
            p3 = add_label(p3, f"SegFormer-B2 (Fire IoU:{iou_f:.1f}%, Smoke IoU:{iou_s:.1f}%)", (0, 255, 255))

            row_panel = np.hstack([p1, p2, p3])

            header = np.zeros((45, 512 * 3, 3), dtype=np.uint8)
            cv2.putText(
                header,
                f"STAGE 1 SEGFORMER TEST PREDICTION | Sample #{idx+1:02d}: {row['filename'][:40]}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            final_sample = np.vstack([header, row_panel])

            out_name = f"segformer_test_sample_{idx+1:02d}_{row['category']}.jpg"
            out_path = OUT_DIR / out_name
            cv2.imwrite(str(out_path), final_sample, [cv2.IMWRITE_JPEG_QUALITY, 92])

            with open(out_path, "rb") as f:
                results_files[out_name] = f.read()

            # Chuẩn bị tile cho Montage (kích thước nhỏ gọn 384x128)
            small_tile = cv2.resize(row_panel, (768, 256))
            montage_tiles.append(small_tile)

    # Ghép bảng Montage Grid 4x4 hoặc 4x2
    if len(montage_tiles) >= 8:
        grid_rows = []
        for r in range(0, min(8, len(montage_tiles)), 2):
            row_img = np.hstack([montage_tiles[r], montage_tiles[r+1]])
            grid_rows.append(row_img)
        montage_full = np.vstack(grid_rows)

        montage_name = "segformer_testset_montage_highlights.jpg"
        montage_path = OUT_DIR / montage_name
        cv2.imwrite(str(montage_path), montage_full, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(montage_path, "rb") as f:
            results_files[montage_name] = f.read()
        print(f"  🎨 Đã tạo ảnh tổng hợp Montage: {montage_name}")

    volume.commit()
    print("=" * 80)
    print(f"🎉 SUY LUẬN TEST SEGFORMER HOÀN TẤT! Đã tạo {len(results_files)} ảnh.")
    print("=" * 80)
    return results_files

@app.local_entrypoint()
def main(num_samples: int = 12):
    from pathlib import Path
    print("🚀 Bắt đầu gọi hàm suy luận SegFormer trên Modal GPU...")
    image_dict = run_segformer_test_inference.remote(num_samples=num_samples)

    local_dir = Path("docs/segformer_test_predictions")
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Đang lưu {len(image_dict)} ảnh suy luận SegFormer về: {local_dir}")
    for fname, img_bytes in image_dict.items():
        local_p = local_dir / fname
        with open(local_p, "wb") as f:
            f.write(img_bytes)
        print(f"  💾 Đã lưu: {local_p}")

    print("\n" + "=" * 80)
    print(f"✨ TẤT CẢ ẢNH SEGFORMER TEST ĐÃ CÓ TẠI: docs/segformer_test_predictions/")
    print("=" * 80)
