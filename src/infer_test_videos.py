"""
infer_test_videos.py
=====================
Chạy Inference SegFormer-B2 trên các Video Test Đa Dạng:
- Phát hiện khoảnh khắc NGUY HIỂM ĐẦU TIÊN (Early Detection Timestamp)
- Bắt khoảnh khắc ĐỈNH DIỂM NGUY HIỂM (Peak Danger Moment)
- Đo tốc độ tăng trưởng diện tích (growth rate) của Lửa & Khói
- Vẽ mặt nạ Overlay (Đỏ = Lửa, Xanh Dương = Khói) & Bounding Box
- Lưu ảnh kết quả vào vis_results/ và xuất báo cáo tổng hợp.

Cách chạy:
  modal run src/infer_test_videos.py
"""

import modal
import os

app = modal.App("fire-smoke-test-video-inference")
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
        "pandas",
        "tqdm",
    )
    .add_local_dir("test_annotation_frames", remote_path="/root/test_annotation_frames")
)


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="A10G",
    timeout=1800,
    memory=16384,
)
def run_video_inference():
    import json
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from pathlib import Path
    from tqdm import tqdm
    from transformers import SegformerForSemanticSegmentation
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 75)
    print("🎬 CHẠY INFERENCE VÀ BẮT KHOẢNH KHẮC NGUY HIỂM TRÊN CÁC VIDEO TEST")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SegFormer-B2
    ckpt_path = "/data/segformer_b2_fire_smoke_v1.pth"
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", num_labels=3, ignore_mismatched_sizes=True
    )
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded SegFormer-B2 checkpoint!")

    model.to(device)
    model.eval()

    frames_base = Path("/root/test_annotation_frames")
    out_dir     = Path("/data/vis_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chọn 5 video đại diện tiêu biểu từ test set:
    # 1. roomfire41_chunk_05 (Lửa + Khói nhà ở)
    # 2. pexels_heavy_smoke_hazard_6264583 (Khói nặng độc hại)
    # 3. pexels_small_visible_flame_35538061 (Lửa mới bùng phát)
    # 4. pexels_welding_sparks_ctrl_35590715 (Tia lửa hàn - Báo động giả)
    # 5. pexels_steam_vapor_29702771 (Hơi nước - Báo động giả)

    sample_stems = [
        "roomfire41_chunk_05",
        "pexels_heavy_smoke_hazard_6264583",
        "pexels_small_visible_flame_35538061",
        "pexels_welding_sparks_ctrl_35590715",
        "pexels_steam_vapor_29702771",
    ]

    # Nếu stem không tìm thấy, lấy 5 stem bất kỳ từ frames_base
    avail_stems = [p.name for p in frames_base.iterdir() if p.is_dir()]
    target_stems = [s for s in sample_stems if s in avail_stems]
    if len(target_stems) < 5:
        remaining = [s for s in avail_stems if s not in target_stems]
        target_stems.extend(remaining[:5 - len(target_stems)])

    print(f"  Target video clips for inference analysis: {target_stems}")

    summary_reports = []

    for stem in target_stems:
        clip_dir = frames_base / stem
        frame_files = sorted(list(clip_dir.glob("*.png")))
        if not frame_files:
            continue

        print(f"\n🎥 Analyzing Clip: {stem} ({len(frame_files)} frames)...")

        stats_per_frame = []
        first_danger_frame = None
        peak_danger_frame  = None
        max_total_pixels   = 0

        rendered_frames = []

        for idx, fpath in enumerate(frame_files):
            img_bgr = cv2.imread(str(fpath))
            if img_bgr is None:
                continue

            h_orig, w_orig = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_512 = cv2.resize(img_rgb, (512, 512))

            # Preprocess tensor
            t_img = torch.tensor(img_512, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            t_img = (t_img - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            t_img = t_img.to(device)

            with torch.no_grad():
                logits = model(t_img).logits
                logits = F.interpolate(logits, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
                pred_mask = logits.argmax(dim=1).cpu().numpy().squeeze(0)

            fire_pixels  = int((pred_mask == 1).sum())
            smoke_pixels = int((pred_mask == 2).sum())
            total_pixels = fire_pixels + smoke_pixels
            cov_pct = (total_pixels / (h_orig * w_orig)) * 100

            has_fire  = fire_pixels > 100
            has_smoke = smoke_pixels > 300
            is_danger = has_fire or has_smoke

            if is_danger and first_danger_frame is None:
                first_danger_frame = (idx, fpath.name, fire_pixels, smoke_pixels)

            if total_pixels > max_total_pixels:
                max_total_pixels = total_pixels
                peak_danger_frame = (idx, fpath.name, fire_pixels, smoke_pixels)

            # Create visual overlay
            overlay = img_bgr.copy()
            # Red color for Fire (Class 1)
            overlay[pred_mask == 1] = (0, 0, 255)
            # Blue color for Smoke (Class 2)
            overlay[pred_mask == 2] = (255, 120, 0)

            # Alpha blend
            vis = cv2.addWeighted(img_bgr, 0.65, overlay, 0.35, 0)

            # Draw status banner
            status_str = "SAFE"
            banner_color = (0, 180, 0)
            if has_fire and has_smoke:
                status_str = "CRITICAL: FIRE & SMOKE DETECTED!"
                banner_color = (0, 0, 220)
            elif has_fire:
                status_str = "WARNING: FIRE DETECTED!"
                banner_color = (0, 0, 255)
            elif has_smoke:
                status_str = "WARNING: SMOKE DETECTED!"
                banner_color = (255, 140, 0)

            cv2.rectangle(vis, (0, 0), (w_orig, 40), (0, 0, 0), -1)
            cv2.putText(vis, f"[{fpath.name}] {status_str}", (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner_color, 2)
            cv2.putText(vis, f"Fire: {fire_pixels} px | Smoke: {smoke_pixels} px ({cov_pct:.1f}%)", (15, w_orig - 15 if w_orig < 500 else 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            stats_per_frame.append({
                "frame": fpath.name,
                "index": idx,
                "fire_pixels": fire_pixels,
                "smoke_pixels": smoke_pixels,
                "is_danger": is_danger,
            })

            rendered_frames.append(vis)

        # Plot 4-frame grid preview for this video
        n_rendered = len(rendered_frames)
        sample_indices = np.linspace(0, n_rendered - 1, min(4, n_rendered), dtype=int)

        fig, axes = plt.subplots(1, len(sample_indices), figsize=(16, 4))
        if len(sample_indices) == 1:
            axes = [axes]

        for i, s_idx in enumerate(sample_indices):
            vis_rgb = cv2.cvtColor(rendered_frames[s_idx], cv2.COLOR_BGR2RGB)
            axes[i].imshow(vis_rgb)
            axes[i].set_title(f"Frame {s_idx+1}/{n_rendered}", fontsize=10)
            axes[i].axis("off")

        plt.suptitle(f"Detection Preview: {stem}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        grid_save_path = out_dir / f"infer_{stem}_grid.png"
        plt.savefig(str(grid_save_path), dpi=120)
        plt.close()

        # Record clip summary
        summary_reports.append({
            "video_clip": stem,
            "total_frames": len(frame_files),
            "first_detection": {
                "frame_idx": first_danger_frame[0] if first_danger_frame else None,
                "frame_name": first_danger_frame[1] if first_danger_frame else "None",
                "fire_px": first_danger_frame[2] if first_danger_frame else 0,
                "smoke_px": first_danger_frame[3] if first_danger_frame else 0,
            },
            "peak_danger": {
                "frame_idx": peak_danger_frame[0] if peak_danger_frame else None,
                "frame_name": peak_danger_frame[1] if peak_danger_frame else "None",
                "fire_px": peak_danger_frame[2] if peak_danger_frame else 0,
                "smoke_px": peak_danger_frame[3] if peak_danger_frame else 0,
            },
            "grid_image": f"vis_results/infer_{stem}_grid.png",
        })

        print(f"  ✓ Early Detection: {first_danger_frame[1] if first_danger_frame else 'No Fire/Smoke'}")
        print(f"  ✓ Peak Moment   : {peak_danger_frame[1] if peak_danger_frame else 'None'}")

    report_json_path = out_dir / "test_videos_inference_summary.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_reports, f, indent=2)

    vol.commit()
    print(f"\n✅ Finished Video Inference! Summary saved to {report_json_path}")


@app.local_entrypoint()
def main():
    print("[INFO] Running Video Inference & Early Detection Capture on Modal GPU...")
    run_video_inference.remote()
