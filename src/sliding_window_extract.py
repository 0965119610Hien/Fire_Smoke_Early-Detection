"""
Sliding Window Frame Extraction + CLAHE Enhancement
====================================================
Pipeline:
  1. Lay tat ca video tu 2 nguon:
       - dataset_annotation_round2/videos/     (62 video moi)
       - video_train/ (yt_*, fire*, Weihnacht*) (102 video cu Round 2)
  2. Voi moi video:
       - Chia sliding window: clip_len=20s, stride=10s (overlap 10s)
       - Voi moi clip: extract 2 FPS -> frames
       - Apply CLAHE LAB enhancement (nhat quan voi enhanced_proxy_frames)
       - Luu vao roboflow_upload/{video_stem}/clip_{i:03d}/frame_{j:04d}.jpg
  3. Xuat manifest CSV tong hop

Cau truc output:
  roboflow_upload/
  +-- battery_thermal_runaway_VWODdROCajU/
  |   +-- clip_000/
  |   |   +-- frame_0000.jpg
  |   |   +-- ...
  |   +-- clip_001/
  +-- manifest.csv
"""

import cv2
import os
import csv
import glob
import numpy as np

# === CONFIG ===
BASE_DIR   = r"e:\NCKH\Code\Fire_Smoke_Early-Detection"
OUT_DIR    = os.path.join(BASE_DIR, "roboflow_upload")
CLIP_LEN   = 20      # giay - moi cua so
STRIDE     = 10      # giay - buoc truot (overlap = CLIP_LEN - STRIDE = 10s)
FPS_SAMPLE = 2       # frame/giay trich xuat trong moi clip
IMG_SIZE   = 640     # resize ve 640x640 de upload Roboflow
JPEG_Q     = 92      # chat luong JPEG

# === NGUON VIDEO ===
SOURCE_NEW = os.path.join(BASE_DIR, "dataset_annotation_round2", "videos")
SOURCE_OLD = os.path.join(BASE_DIR, "video_train")

# Chi lay early-detection incidents tu video_train
OLD_PREFIXES = (
    "yt_indoor_building",
    "yt_outdoor_industrial",
    "yt_vehicle_transport",
    "yt_warehouse_factory",
    "fire",
    "Weihnacht",
)


def clahe_lab(frame_bgr):
    """Apply CLAHE tren kenh L cua khong gian mau LAB."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def collect_videos():
    videos = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4"):
        for p in glob.glob(os.path.join(SOURCE_NEW, ext)):
            videos.append({"path": p, "source": "round2_new"})
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4"):
        for p in glob.glob(os.path.join(SOURCE_OLD, ext)):
            name = os.path.basename(p)
            if any(name.startswith(pfx) for pfx in OLD_PREFIXES):
                videos.append({"path": p, "source": "round2_old"})
    return videos


def process_video(video_info, writer, stats):
    path   = video_info["path"]
    source = video_info["source"]
    stem   = os.path.splitext(os.path.basename(path))[0]
    stem_safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in stem)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  WARNING: Cannot open: {os.path.basename(path)}")
        stats["failed"] += 1
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps     = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0 or total_frames <= 0:
        cap.release()
        stats["failed"] += 1
        return

    duration   = total_frames / orig_fps
    frame_step = max(1, round(orig_fps / FPS_SAMPLE))

    clip_idx = 0
    start_sec = 0.0
    clips_for_video = 0
    last_frame_count = 0

    while start_sec < duration:
        end_sec  = min(start_sec + CLIP_LEN, duration)
        clip_dur = end_sec - start_sec

        if clip_dur < 5.0:
            break

        clip_dir = os.path.join(OUT_DIR, stem_safe, f"clip_{clip_idx:03d}")
        os.makedirs(clip_dir, exist_ok=True)

        start_frame = int(start_sec * orig_fps)
        end_frame   = int(end_sec   * orig_fps)
        frame_count = 0

        for fno in range(start_frame, end_frame, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame     = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            frame_enh = clahe_lab(frame)
            out_name  = f"frame_{frame_count:04d}.jpg"
            out_path  = os.path.join(clip_dir, out_name)
            cv2.imwrite(out_path, frame_enh, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            ts = start_sec + (fno - start_frame) / orig_fps
            writer.writerow({
                "video_stem":     stem,
                "source":         source,
                "clip_idx":       clip_idx,
                "frame_idx":      frame_count,
                "frame_orig":     fno,
                "timestamp_sec":  round(ts, 3),
                "clip_start_sec": round(start_sec, 3),
                "clip_end_sec":   round(end_sec, 3),
                "rel_path":       os.path.relpath(out_path, BASE_DIR),
            })
            frame_count += 1
            stats["total_frames"] += 1

        last_frame_count = frame_count
        clips_for_video += 1
        clip_idx += 1
        start_sec += STRIDE

        if duration <= CLIP_LEN:
            break

    cap.release()
    stats["total_clips"]  += clips_for_video
    stats["total_videos"] += 1
    print(f"  OK  {stem_safe[:55]:<55} | {duration:6.1f}s -> {clips_for_video} clips, ~{last_frame_count} fr/clip")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")

    videos = collect_videos()
    print(f"\n{'='*72}")
    print(f"  Sliding Window Frame Extraction --- {len(videos)} videos")
    print(f"  Clip: {CLIP_LEN}s | Stride: {STRIDE}s | Overlap: {CLIP_LEN - STRIDE}s")
    print(f"  Sample: {FPS_SAMPLE} FPS | Output: {IMG_SIZE}x{IMG_SIZE} JPEG q={JPEG_Q}")
    print(f"  Out dir: {OUT_DIR}")
    print(f"{'='*72}\n")

    stats = {"total_videos": 0, "total_clips": 0, "total_frames": 0, "failed": 0}

    fieldnames = [
        "video_stem", "source", "clip_idx", "frame_idx",
        "frame_orig", "timestamp_sec", "clip_start_sec", "clip_end_sec", "rel_path",
    ]

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, vinfo in enumerate(sorted(videos, key=lambda x: x["path"]), 1):
            print(f"[{i:3d}/{len(videos)}] ", end="")
            process_video(vinfo, writer, stats)

    print(f"\n{'='*72}")
    print(f"  HOAN THANH")
    print(f"  Videos xu ly : {stats['total_videos']}")
    print(f"  Clips tao ra : {stats['total_clips']}")
    print(f"  Frames tong  : {stats['total_frames']:,}")
    print(f"  Loi          : {stats['failed']}")
    print(f"  Manifest     : {manifest_path}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
