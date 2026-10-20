"""
video_pseudo_label.py
=====================
Pipeline gán nhãn pseudo-label cho 20 video đầu tiên từ video_train/.

Quy trình trên Modal GPU (A10G):
  1. Upload 20 video lên container qua add_local_dir
  2. Extract frames từ mỗi video (1 frame/giây, resize 512×512)
  3. Chạy SegFormer-B2 inference (batch) → per-frame mask PNG
  4. Lưu kết quả lên Modal Volume:
       /data/video_pseudo_labels/
         <video_stem>/
           frames/frame_000001.png   ← ảnh frame (512×512)
           masks/frame_000001_mask.png  ← mask grayscale 0/1/2
     + manifest.csv tổng hợp toàn bộ
     + stats.json thống kê

Output mask format: PNG Grayscale (0=bg, 1=fire, 2=smoke)
— khớp với format train_temporal.py

Chạy:
  modal run src/video_pseudo_label.py

Sau khi xong có thể train TimeSformer:
  modal run src/train_temporal.py
"""

import modal
import os
import glob
import shutil
import tempfile

# ─── Chọn 20 video đầu tiên (trimmed) ────────────────────────────────────────
VIDEO_DIR   = "video_train"
N_VIDEOS    = 20
FPS_EXTRACT = 1          # 1 frame/giây — đủ để học temporal, không quá nhiều data
RESIZE      = 512        # khớp với train_segformer.py img_size

# Lấy danh sách 20 video _trimmed.mp4 đầu tiên theo tên
all_trimmed = sorted(glob.glob(os.path.join(VIDEO_DIR, "*_trimmed.mp4")))
if len(all_trimmed) < N_VIDEOS:
    all_fire = sorted(glob.glob(os.path.join(VIDEO_DIR, "fire*.mp4")))
    all_trimmed = (all_trimmed + all_fire)[:N_VIDEOS]
else:
    all_trimmed = all_trimmed[:N_VIDEOS]

VIDEO_FILES_LOCAL = all_trimmed
VIDEO_BASENAMES   = {os.path.basename(v) for v in VIDEO_FILES_LOCAL}

print(f"[LOCAL] Sẽ xử lý {len(VIDEO_FILES_LOCAL)} video:")
for v in VIDEO_FILES_LOCAL:
    size_mb = os.path.getsize(v) / 1e6
    print(f"  • {os.path.basename(v)}  ({size_mb:.1f} MB)")
total_mb = sum(os.path.getsize(v) for v in VIDEO_FILES_LOCAL) / 1e6
print(f"[LOCAL] Tổng upload: {total_mb:.0f} MB")

# ─── Tạo thư mục tạm chứa đúng 20 video để upload ───────────────────────────
# Modal add_local_dir không hỗ trợ filter lambda, nên copy sang temp dir
_TEMP_VIDEO_DIR = os.path.join("video_train", "_upload_20")
os.makedirs(_TEMP_VIDEO_DIR, exist_ok=True)
for v in VIDEO_FILES_LOCAL:
    dst = os.path.join(_TEMP_VIDEO_DIR, os.path.basename(v))
    if not os.path.exists(dst):
        shutil.copy2(v, dst)

# ─── Modal setup ──────────────────────────────────────────────────────────────
app = modal.App("fire-smoke-video-pseudo-label")
vol = modal.Volume.from_name("fire_smoke_dataset")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")                          # extract frames bằng ffmpeg
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "transformers==4.40.0",
        "opencv-python-headless",
        "numpy<2.0.0",
        "Pillow",
        "tqdm",
    )
    # Chỉ upload 20 video đã copy vào _upload_20/
    .add_local_dir(_TEMP_VIDEO_DIR, remote_path="/root/videos")
)


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="A10G",
    timeout=7200,     # 2h — 20 videos × ~60s mỗi video
    memory=16384,
    cpu=4.0,
)
def run_video_pseudo_label(n_videos: int = 20, fps: int = 1, resize: int = 512):
    """
    Trên Modal GPU:
      1. Extract frames từ video bằng ffmpeg
      2. Inference SegFormer-B2 → pseudo-label mask
      3. Lưu vào /data/video_pseudo_labels/
    """
    import csv
    import glob
    import json
    import re
    import subprocess
    import time
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformers import SegformerForSemanticSegmentation

    print("=" * 70)
    print("🎬 VIDEO PSEUDO-LABEL PIPELINE — SegFormer-B2")
    print(f"   GPU   : {torch.cuda.get_device_name(0)}")
    print(f"   FPS   : {fps} frame/s extract")
    print(f"   Resize: {resize}×{resize}")
    print("=" * 70)

    device = torch.device("cuda")
    OUT_ROOT = Path("/data/video_pseudo_labels")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── 1. Load SegFormer-B2 checkpoint ──────────────────────────────────────
    # Tự động tìm checkpoint mới nhất
    ckpt_files = sorted(glob.glob("/data/segformer_b2_fire_smoke_v*.pth"))
    if not ckpt_files:
        print("❌ Không tìm thấy SegFormer checkpoint trong Volume!")
        print("   Hãy chạy train_segformer.py trước.")
        return

    def _ver(p):
        m = re.search(r"_v(\d+)\.pth$", p)
        return int(m.group(1)) if m else 0
    ckpt_path = max(ckpt_files, key=_ver)
    print(f"\n📦 Checkpoint: {ckpt_path}")

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=3,
        id2label={0: "background", 1: "fire", 2: "smoke"},
        label2id={"background": 0, "fire": 1, "smoke": 2},
        ignore_mismatched_sizes=True,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print("   ✓ Model loaded.")

    # ── 2. Lấy danh sách video ───────────────────────────────────────────────
    video_dir = Path("/root/videos")
    video_files = sorted(video_dir.glob("*.mp4"))[:n_videos]
    print(f"\n📂 Tìm thấy {len(video_files)} video để xử lý:")
    for v in video_files:
        size_mb = v.stat().st_size / 1e6
        print(f"   • {v.name}  ({size_mb:.1f} MB)")

    # ── 3. Xử lý từng video ──────────────────────────────────────────────────
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    global_manifest = []
    global_stats = {
        "total_frames": 0,
        "n_fire": 0, "n_smoke": 0, "n_both": 0, "n_empty": 0,
        "videos_processed": [],
    }

    for vid_path in tqdm(video_files, desc="Videos"):
        stem = vid_path.stem.replace("_trimmed", "")  # bỏ hậu tố _trimmed
        out_dir = OUT_ROOT / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"🎬 {vid_path.name}")

        # ── Extract frames bằng ffmpeg ────────────────────────────────────
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(vid_path),
            "-vf", f"fps={fps},scale={resize}:{resize}:flags=lanczos",
            "-q:v", "2",
            str(frames_dir / "frame_%06d.png"),
            "-loglevel", "error"
        ]
        t0 = time.time()
        result = subprocess.run(ffmpeg_cmd, capture_output=True)
        if result.returncode != 0:
            print(f"   ❌ ffmpeg error: {result.stderr.decode()[:200]}")
            continue

        frame_paths = sorted(frames_dir.glob("frame_*.png"))
        n_frames = len(frame_paths)
        elapsed_extract = time.time() - t0
        print(f"   ✓ Extracted {n_frames} frames in {elapsed_extract:.1f}s")

        if n_frames == 0:
            print("   ⚠️  No frames extracted, skipping.")
            continue

        # ── Inference SegFormer (batch=16) ───────────────────────────────
        masks_dir = out_dir / "masks"
        masks_dir.mkdir(exist_ok=True)

        vid_stats = {"fire": 0, "smoke": 0, "both": 0, "empty": 0}
        batch_size = 16

        t1 = time.time()
        for i in range(0, n_frames, batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            batch_tensors = []
            valid_meta    = []

            for fp in batch_paths:
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Đảm bảo đúng kích thước
                if img_rgb.shape[:2] != (resize, resize):
                    img_rgb = cv2.resize(img_rgb, (resize, resize),
                                         interpolation=cv2.INTER_LINEAR)
                nrm = (img_rgb.astype(np.float32) / 255.0 - mean) / std
                batch_tensors.append(torch.from_numpy(nrm).permute(2, 0, 1).float())
                valid_meta.append(fp)

            if not batch_tensors:
                continue

            inp = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logits = model(pixel_values=inp).logits   # [B, 3, H/4, W/4]
                logits_up = F.interpolate(
                    logits.float(), size=(resize, resize),
                    mode="bilinear", align_corners=False
                )
                preds = logits_up.argmax(dim=1).cpu().numpy().astype(np.uint8)  # [B, H, W]

            for fp, pred_mask in zip(valid_meta, preds):
                # Lưu mask PNG grayscale
                mask_name = fp.stem + "_mask.png"
                mask_path = masks_dir / mask_name
                cv2.imwrite(str(mask_path), pred_mask)

                fire_px  = int((pred_mask == 1).sum())
                smoke_px = int((pred_mask == 2).sum())
                has_fire  = fire_px  > 50
                has_smoke = smoke_px > 200

                if has_fire and has_smoke:
                    vid_stats["both"]  += 1
                elif has_fire:
                    vid_stats["fire"]  += 1
                elif has_smoke:
                    vid_stats["smoke"] += 1
                else:
                    vid_stats["empty"] += 1

                total_px = pred_mask.size
                global_manifest.append({
                    "video":         stem,
                    "frame_file":    str(fp.relative_to(OUT_ROOT)).replace("\\", "/"),
                    "mask_file":     str(mask_path.relative_to(OUT_ROOT)).replace("\\", "/"),
                    "fire_pixels":   fire_px,
                    "smoke_pixels":  smoke_px,
                    "coverage_pct":  round((fire_px + smoke_px) / total_px * 100, 2),
                    "has_fire":      int(has_fire),
                    "has_smoke":     int(has_smoke),
                })

        elapsed_infer = time.time() - t1
        fps_infer = n_frames / elapsed_infer if elapsed_infer > 0 else 0

        print(f"   ✓ Inference: {n_frames} masks in {elapsed_infer:.1f}s ({fps_infer:.1f} fps)")
        print(f"   Distribution: fire={vid_stats['fire']} smoke={vid_stats['smoke']} "
              f"both={vid_stats['both']} empty={vid_stats['empty']}")

        # Cập nhật global stats
        global_stats["total_frames"] += n_frames
        global_stats["n_fire"]       += vid_stats["fire"]
        global_stats["n_smoke"]      += vid_stats["smoke"]
        global_stats["n_both"]       += vid_stats["both"]
        global_stats["n_empty"]      += vid_stats["empty"]
        global_stats["videos_processed"].append({
            "video": stem,
            "frames": n_frames,
            **vid_stats
        })

    # ── 4. Lưu manifest CSV và stats JSON ────────────────────────────────────
    manifest_path = OUT_ROOT / "manifest.csv"
    if global_manifest:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(global_manifest[0].keys()))
            writer.writeheader()
            writer.writerows(global_manifest)
        print(f"\n✓ Manifest CSV: {manifest_path} ({len(global_manifest)} rows)")

    stats_path = OUT_ROOT / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, indent=2, ensure_ascii=False)

    vol.commit()

    # ── 5. Tổng kết ──────────────────────────────────────────────────────────
    n_total = global_stats["total_frames"]
    print("\n" + "=" * 70)
    print("✅ PSEUDO-LABELING HOÀN THÀNH!")
    print("=" * 70)
    print(f"  Videos processed : {len(global_stats['videos_processed'])}")
    print(f"  Total frames     : {n_total:,}")
    print(f"  Fire only        : {global_stats['n_fire']:,}  ({100*global_stats['n_fire']/max(n_total,1):.1f}%)")
    print(f"  Smoke only       : {global_stats['n_smoke']:,}  ({100*global_stats['n_smoke']/max(n_total,1):.1f}%)")
    print(f"  Fire + Smoke     : {global_stats['n_both']:,}  ({100*global_stats['n_both']/max(n_total,1):.1f}%)")
    print(f"  Background only  : {global_stats['n_empty']:,}  ({100*global_stats['n_empty']/max(n_total,1):.1f}%)")
    print(f"  Output path      : /data/video_pseudo_labels/")
    print("=" * 70)
    print(f"\n📋 Bước tiếp theo:")
    print(f"   modal run src/train_temporal.py")
    print("=" * 70)

    return global_stats


@app.local_entrypoint()
def main():
    print(f"🚀 Uploading {len(VIDEO_FILES_LOCAL)} videos lên Modal và chạy pseudo-labeling...")
    for v in VIDEO_FILES_LOCAL:
        print(f"   • {os.path.basename(v)}")
    print()
    result = run_video_pseudo_label.remote(
        n_videos=N_VIDEOS,
        fps=FPS_EXTRACT,
        resize=RESIZE,
    )
    print(f"\n✅ Done! Total frames: {result['total_frames']:,}")
