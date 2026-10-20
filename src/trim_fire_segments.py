"""
=============================================================================
STRICT FIRE & SMOKE INCIDENT VIDEO TRIMMER — Color & Luminance Chromatic Trimmer
=============================================================================
PURPOSE:
  STRICTLY trim and extract ONLY the exact video sub-segment where fire or smoke
  is actively present (t_start to t_end), discarding all earlier/later irrelevant minutes.

STRICT DETECTION RULES:
  1. Fire Chromatic Detector (HSV + YCrCb Color Spaces):
     Detects red/orange/yellow high-luminance flame pixels (Fire Area >= 80 px).
  2. Smoke Expansion Detector (Saturation + Low-Contrast Motion):
     Detects gray/cloudy expanding smoke plumes (Smoke Area >= 250 px).
  3. Precise Boundary Trimming:
     t_start = First second where fire or smoke is detected (with a tight 2.0s buffer).
     t_end   = Last second where fire or smoke is active.

USAGE:
  python src/trim_fire_segments.py --input filtered_tracking_dataset/high_confidence --output trimmed_fire_incidents --auto
=============================================================================
"""

import os
import sys
import json
import csv
import subprocess
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

# ── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trim_fire_segments.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# ── Check FFmpeg Availability ──────────────────────────────────────────────
def check_ffmpeg() -> bool:
    try:
        res = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except Exception:
        return False

HAS_FFMPEG = check_ffmpeg()

# ── Strict Fire Color Mask (HSV + YCrCb) ───────────────────────────────────
def detect_fire_pixels(bgr_frame: np.ndarray) -> int:
    """Đếm số pixel ngọn lửa màu đỏ/cam/vàng có độ sáng cao."""
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)

    # Khung màu HSV cho ngọn lửa
    lower_fire1 = np.array([0, 100, 180], dtype=np.uint8)
    upper_fire1 = np.array([30, 255, 255], dtype=np.uint8)
    lower_fire2 = np.array([165, 100, 180], dtype=np.uint8)
    upper_fire2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
    mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
    hsv_fire_mask = cv2.bitwise_or(mask1, mask2)

    # Khung YCrCb bổ trợ cho lửa
    Y = ycrcb[:, :, 0]
    Cr = ycrcb[:, :, 1]
    Cb = ycrcb[:, :, 2]
    ycrcb_mask = (Y > 160) & (Cr > 140) & (Cb < 140) & (Cr >= Cb)

    final_mask = cv2.bitwise_and(hsv_fire_mask, hsv_fire_mask, mask=ycrcb_mask.astype(np.uint8) * 255)
    return int(cv2.countNonZero(final_mask))

# ── Strict Smoke Mask (Low Saturation + Gray Cloud Expansion) ──────────────
def detect_smoke_pixels(bgr_frame: np.ndarray, prev_gray: Optional[np.ndarray]) -> int:
    """Đếm số pixel đám khói mờ đục màu xám có chuyển động khuếch tán."""
    if prev_gray is None:
        return 0

    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Khói có độ bão hòa màu thấp (S < 60) và độ sáng vừa-cao (100 < V < 235)
    smoke_color_mask = (S < 60) & (V > 100) & (V < 235)

    # Chuyển động khuếch tán
    delta = cv2.absdiff(prev_gray, gray)
    motion_mask = delta > 15

    final_smoke = smoke_color_mask & motion_mask
    return int(np.sum(final_smoke))

# ── Strict Auto Incident Segment Timestamps Detector ──────────────────────
def detect_strict_incident_timestamps(video_path: Path, buffer_sec: float = 2.0) -> Tuple[float, float, float]:
    """
    Quét video 1 fps, phát hiện chính xác giây t_start đầu tiên có khói/lửa xuất hiện
    và t_end cuối cùng kết thúc đám cháy.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0, 0.0, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_dur = total_frames / fps if fps > 0 else 0.0

    if total_dur <= 0:
        cap.release()
        return 0.0, 0.0, 0.0

    frame_step = max(1, int(fps))
    first_incident_sec = None
    last_incident_sec = None

    prev_gray = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        current_sec = frame_count / fps

        if frame_count % frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            fire_px = detect_fire_pixels(frame)
            smoke_px = detect_smoke_pixels(frame, prev_gray)

            # Ngưỡng nghiêm ngặt: Lửa >= 80px HOẶC Khói >= 250px
            if fire_px >= 80 or smoke_px >= 250:
                if first_incident_sec is None:
                    first_incident_sec = current_sec
                last_incident_sec = current_sec

            prev_gray = gray

        frame_count += 1

    cap.release()

    if first_incident_sec is None:
        # Nếu hoàn toàn không phát hiện thấy khói/lửa -> Giữ nguyên video
        return 0.0, total_dur, total_dur

    # Cắt gọn với buffer 2.0s trước t_start để chỉ giữ mốc bắt đầu
    start_sec = max(0.0, first_incident_sec - buffer_sec)
    end_sec = min(total_dur, (last_incident_sec or total_dur) + 3.0)

    return start_sec, end_sec, total_dur

# ── Video Trimming Engine ─────────────────────────────────────────────────
def trim_video(src_path: Path, dst_path: Path, start_sec: float, end_sec: float) -> bool:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dur = max(1.0, end_sec - start_sec)

    if HAS_FFMPEG:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.2f}",
            "-i", str(src_path),
            "-t", f"{dur:.2f}",
            "-c", "copy",
            str(dst_path)
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and dst_path.exists() and dst_path.stat().st_size > 100000:
                return True
        except Exception as ex:
            log.debug(f"FFmpeg stream copy error: {ex}")

    return _trim_opencv(src_path, dst_path, start_sec, end_sec)

def _trim_opencv(src_path: Path, dst_path: Path, start_sec: float, end_sec: float) -> bool:
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (w, h))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        writer.write(frame)
        current_frame += 1

    cap.release()
    writer.release()
    return dst_path.exists() and dst_path.stat().st_size > 100000

# ── Main Processor ─────────────────────────────────────────────────────────
def process_trimming(input_dir: Path, output_dir: Path, auto_mode: bool, csv_file: Optional[Path] = None, buffer_sec: float = 2.0):
    output_dir.mkdir(parents=True, exist_ok=True)
    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    video_files = [p for p in input_dir.rglob("*") if p.suffix.lower() in video_extensions]

    log.info(f"[+] Found {len(video_files)} videos in '{input_dir}' to strictly trim...")
    log.info("=" * 70)

    manual_timestamps = {}
    if csv_file and csv_file.exists():
        log.info(f"[+] Reading manual timestamps from CSV: {csv_file}")
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename") or row.get("video_name")
                if fname:
                    st = float(row.get("start_sec", 0.0))
                    et = float(row.get("end_sec", 0.0))
                    manual_timestamps[Path(fname).name] = (st, et)

    manifest_records = []

    for idx, vid_path in enumerate(video_files, 1):
        rel_path = vid_path.relative_to(input_dir)
        dst_name = f"{vid_path.stem}_trimmed.mp4"
        dst_path = output_dir / rel_path.parent / dst_name

        if vid_path.name in manual_timestamps:
            st, et = manual_timestamps[vid_path.name]
            mode_desc = "Manual CSV"
        elif auto_mode:
            st, et, tot_dur = detect_strict_incident_timestamps(vid_path, buffer_sec=buffer_sec)
            mode_desc = "Strict Fire/Smoke Detection"
        else:
            st = 0.0
            cap = cv2.VideoCapture(str(vid_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            tot = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            et = tot / fps if fps > 0 else 60.0
            cap.release()
            mode_desc = "Full Length"

        log.info(f"[{idx}/{len(video_files)}] Strictly Trimming: {rel_path.name}")
        log.info(f"  -> Incident Range: {st:.1f}s to {et:.1f}s (Clip Length: {(et-st):.1f}s) [{mode_desc}]")

        ok = trim_video(vid_path, dst_path, st, et)

        if ok:
            log.info(f"  [OK] Saved strict clip: {dst_path.name}")
            manifest_records.append({
                "original_video": str(rel_path),
                "trimmed_video": str(dst_path.relative_to(output_dir)),
                "start_sec": round(st, 1),
                "end_sec": round(et, 1),
                "duration_sec": round(et - st, 1),
                "mode": mode_desc
            })
        else:
            log.warning(f"  [ERROR] Could not trim video: {rel_path.name}")

    manifest_csv = output_dir / "trimmed_fire_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "original_video", "trimmed_video", "start_sec", "end_sec", "duration_sec", "mode"
        ])
        writer.writeheader()
        writer.writerows(manifest_records)

    print("\n" + "=" * 70)
    print(">>> STRICT VIDEO TRIMMING COMPLETE")
    print("=" * 70)
    print(f"  • Strictly Processed Clips: {len(manifest_records)} / {len(video_files)}")
    print(f"  - Output Folder          : {output_dir.resolve()}")
    print(f"  - Manifest CSV           : {manifest_csv.resolve()}")
    print("=" * 70)

# ── CLI Interface ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Strictly trim relevant fire and smoke incident video segments"
    )
    ap.add_argument("--input", default="filtered_tracking_dataset/high_confidence", help="Input folder containing high_confidence videos")
    ap.add_argument("--output", default="trimmed_fire_incidents", help="Output directory for trimmed video clips")
    ap.add_argument("--auto", action="store_true", help="Automatically detect smoke/fire start and end timestamps")
    ap.add_argument("--buffer", type=float, default=2.0, help="Buffer padding in seconds before smoke onset")
    ap.add_argument("--csv", default=None, help="Manual CSV file containing start_sec and end_sec timestamps")

    args = ap.parse_args()

    process_trimming(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        auto_mode=args.auto,
        buffer_sec=args.buffer,
        csv_file=Path(args.csv) if args.csv else None
    )
