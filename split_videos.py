"""
Script cat video thanh cac clip nho ~15-20 giay voi overlap 3 giay.
- Input:  d:\zip_dataset\dataset_merged\dataset_merged\dataset_merged\
- Output: d:\zip_dataset\dataset_split\  (giu nguyen cau truc class_0/class_1/...)
"""

import cv2
import os
import sys
import time
from pathlib import Path

INPUT_ROOT  = Path(r"D:\zip_dataset\dataset_merged\dataset_merged\dataset_merged")
OUTPUT_ROOT = Path(r"D:\zip_dataset\dataset_split")

CLIP_DURATION  = 15
MAX_DURATION   = 20
OVERLAP        = 3
MIN_CLIP_SECS  = 5

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
OUTPUT_EXT = ".mp4"
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


def get_video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        return None
    duration = frames / fps
    return fps, int(frames), duration


def compute_segments(duration_sec, fps,
                     clip_sec=CLIP_DURATION, overlap_sec=OVERLAP,
                     min_clip_sec=MIN_CLIP_SECS, max_clip_sec=MAX_DURATION):
    stride = clip_sec - overlap_sec
    clip_frames   = int(clip_sec   * fps)
    stride_frames = int(stride     * fps)
    total_frames  = int(duration_sec * fps)
    min_frames    = int(min_clip_sec * fps)
    max_frames    = int(max_clip_sec * fps)

    segments = []
    start = 0
    while start < total_frames:
        end = min(start + clip_frames, total_frames)
        segments.append((start, end))
        if end >= total_frames:
            break
        start += stride_frames

    if len(segments) > 1:
        last_start, last_end = segments[-1]
        if (last_end - last_start) < min_frames:
            prev_start, prev_end = segments[-2]
            new_end = min(prev_end + (last_end - last_start), total_frames)
            new_end = min(new_end, prev_start + max_frames)
            segments[-2] = (prev_start, new_end)
            segments.pop()

    return segments


def write_clip(cap, out_path, start_frame, end_frame, fps, width, height):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), FOURCC, fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    writer.release()


def process_video(video_path, output_dir, stats):
    info = get_video_info(video_path)
    if info is None:
        print(f"  [SKIP] Cannot read: {video_path.name}")
        stats["skipped"] += 1
        return

    fps, total_frames, duration = info

    cap = cv2.VideoCapture(str(video_path))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem = video_path.stem

    if duration <= CLIP_DURATION:
        out_path = output_dir / f"{stem}_clip001{OUTPUT_EXT}"
        write_clip(cap, out_path, 0, total_frames, fps, width, height)
        stats["clips_written"] += 1
        cap.release()
        return

    segments = compute_segments(duration, fps)
    for idx, (sf, ef) in enumerate(segments, start=1):
        out_name = f"{stem}_clip{idx:03d}{OUTPUT_EXT}"
        out_path = output_dir / out_name
        clip_dur = (ef - sf) / fps
        write_clip(cap, out_path, sf, ef, fps, width, height)
        print(f"    -> clip {idx:03d}: {sf/fps:.1f}s - {ef/fps:.1f}s  ({clip_dur:.1f}s)")
        stats["clips_written"] += 1

    cap.release()
    stats["videos_processed"] += 1


def main():
    if not INPUT_ROOT.exists():
        print(f"[ERROR] INPUT_ROOT not found: {INPUT_ROOT}")
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    video_files = sorted([
        p for p in INPUT_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ])

    total_videos = len(video_files)
    print(f"Found {total_videos} videos in {INPUT_ROOT}")
    print(f"Output -> {OUTPUT_ROOT}")
    print(f"Config: clip={CLIP_DURATION}s, overlap={OVERLAP}s, stride={CLIP_DURATION-OVERLAP}s")
    print("=" * 70)

    stats = {"videos_processed": 0, "clips_written": 0, "skipped": 0}
    t0 = time.time()

    for i, vpath in enumerate(video_files, start=1):
        rel = vpath.relative_to(INPUT_ROOT)
        out_dir = OUTPUT_ROOT / rel.parent

        elapsed = time.time() - t0
        eta_str = ""
        if i > 1:
            avg = elapsed / (i - 1)
            eta = avg * (total_videos - i + 1)
            eta_str = f"  ETA ~{eta/60:.1f} min"

        print(f"[{i:4d}/{total_videos}] {rel}{eta_str}")
        process_video(vpath, out_dir, stats)

    elapsed_total = time.time() - t0
    print("=" * 70)
    print(f"Done!")
    print(f"  Videos processed : {stats['videos_processed']}")
    print(f"  Clips written    : {stats['clips_written']}")
    print(f"  Skipped          : {stats['skipped']}")
    print(f"  Time             : {elapsed_total/60:.1f} minutes")
    print(f"  Output dir       : {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
