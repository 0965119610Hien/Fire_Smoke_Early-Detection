"""
filter_pure_fire_smoke.py
=========================
Quét từng video trong --input, phát hiện các đoạn có khói/lửa,
CẮT BỎ phần không liên quan rồi GỘP lại thành 1 file *_pure.mp4.

Cách hoạt động:
  1. Đọc tuần tự frame (không seek) → tránh lỗi prev_gray.
  2. Phát hiện lửa qua HSV+YCrCb, khói qua low-saturation mask.
  3. Gộp các khoảng active cách nhau ≤ gap_tolerance giây.
  4. Dùng ffmpeg (nếu có) để extract+concat; fallback sang OpenCV.

Chạy:
  python src/filter_pure_fire_smoke.py --input trimmed_fire_incidents --output pure_fire_smoke_clips
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Logging ────────────────────────────────────────────────────────────────
# Force UTF-8 on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("filter_pure_fire_smoke.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


# ── FFmpeg check ───────────────────────────────────────────────────────────
def _has_ffmpeg() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return r.returncode == 0
    except FileNotFoundError:
        return False

HAS_FFMPEG = _has_ffmpeg()


# ── Per-frame detectors ────────────────────────────────────────────────────
def _fire_pixels(bgr: np.ndarray) -> int:
    """Đếm pixel lửa qua HSV (hue cam/đỏ) AND YCrCb (Cr cao, Cb thấp)."""
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    m1 = cv2.inRange(hsv, np.array([0,   90, 150], np.uint8),
                           np.array([35, 255, 255], np.uint8))
    m2 = cv2.inRange(hsv, np.array([160,  90, 150], np.uint8),
                           np.array([180, 255, 255], np.uint8))
    hue_mask = cv2.bitwise_or(m1, m2)

    Y, Cr, Cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    color_ok = ((Y > 140) & (Cr > 130) & (Cb < 145) & (Cr.astype(int) >= Cb.astype(int)))
    combined = cv2.bitwise_and(hue_mask, hue_mask,
                               mask=color_ok.astype(np.uint8) * 255)
    return int(cv2.countNonZero(combined))


def _smoke_pixels(bgr: np.ndarray) -> int:
    """Đếm pixel khói: low-saturation, mid-brightness (xám), không cần motion."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    mask = (S < 70) & (V > 80) & (V < 230)
    return int(np.sum(mask))


# ── Active interval scanner ────────────────────────────────────────────────
def scan_intervals(
    video_path: Path,
    scan_fps: int = 5,
    fire_thresh: int = 30,
    smoke_thresh: int = 5_000,
    min_dur: float = 2.0,
    gap_tol: float = 2.0,
    buffer: float = 0.5,
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Trả về (list[( start_sec, end_sec )], total_duration_sec).
    Đọc tuần tự, bước nhảy frame_step để đạt scan_fps.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_dur = n_frames / orig_fps if orig_fps > 0 else 0.0

    frame_step = max(1, int(round(orig_fps / scan_fps)))
    active_secs: List[float] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            t = idx / orig_fps
            fp = _fire_pixels(frame)
            sp = _smoke_pixels(frame)
            if fp >= fire_thresh or sp >= smoke_thresh:
                active_secs.append(t)
        idx += 1

    cap.release()

    if not active_secs:
        return [], total_dur

    # Gộp khoảng liên tiếp
    intervals: List[Tuple[float, float]] = []
    seg_start = active_secs[0]
    seg_end   = active_secs[0]

    for t in active_secs[1:]:
        if t - seg_end <= gap_tol:
            seg_end = t
        else:
            if seg_end - seg_start >= min_dur:
                intervals.append((
                    max(0.0, seg_start - buffer),
                    min(total_dur, seg_end + buffer),
                ))
            seg_start = t
            seg_end   = t

    if seg_end - seg_start >= min_dur:
        intervals.append((
            max(0.0, seg_start - buffer),
            min(total_dur, seg_end + buffer),
        ))

    return intervals, total_dur


# ── FFmpeg helpers ─────────────────────────────────────────────────────────
def _ffmpeg_extract(src: Path, dst: Path, start: float, end: float) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t",  f"{end - start:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(dst),
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 500


def _ffmpeg_concat(clips: List[Path], dst: Path) -> bool:
    if not clips:
        return False
    if len(clips) == 1:
        import shutil
        shutil.copy2(str(clips[0]), str(dst))
        return dst.exists() and dst.stat().st_size > 500

    list_file = dst.with_suffix(".txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for c in clips:
            f.write(f"file '{c.as_posix()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(dst),
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    list_file.unlink(missing_ok=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 500


# ── OpenCV fallback ────────────────────────────────────────────────────────
def _opencv_extract_concat(
    src: Path, dst: Path, intervals: List[Tuple[float, float]]
) -> bool:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))

    for (st, et) in intervals:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(st * fps))
        end_frame = int(et * fps)
        while True:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if pos > end_frame:
                break
            ok, frm = cap.read()
            if not ok:
                break
            writer.write(frm)

    cap.release()
    writer.release()
    return dst.exists() and dst.stat().st_size > 500


# ── Save one video (ffmpeg preferred, opencv fallback) ─────────────────────
def save_pure_video(
    src: Path,
    dst: Path,
    intervals: List[Tuple[float, float]],
    tmp_dir: Path,
) -> bool:
    if HAS_FFMPEG:
        tmp_clips: List[Path] = []
        for i, (st, et) in enumerate(intervals):
            tmp = tmp_dir / f"__seg_{src.stem}_{i}.mp4"
            if _ffmpeg_extract(src, tmp, st, et):
                tmp_clips.append(tmp)

        if tmp_clips:
            ok = _ffmpeg_concat(tmp_clips, dst)
            for c in tmp_clips:
                c.unlink(missing_ok=True)
            if ok:
                return True
            log.warning(f"    ffmpeg concat failed → opencv fallback")

    return _opencv_extract_concat(src, dst, intervals)


# ── Main pipeline ──────────────────────────────────────────────────────────
def run(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    video_files = sorted(
        p for p in input_dir.iterdir()   # flat scan (single level)
        if p.is_file() and p.suffix.lower() in VIDEO_EXT
    )
    total = len(video_files)
    log.info(f"Found {total} videos in '{input_dir}' (flat scan)")
    log.info(f"FFmpeg available: {HAS_FFMPEG}")
    log.info("=" * 70)

    records = []
    n_saved = n_skip = n_err = 0

    for idx, src in enumerate(video_files, 1):
        log.info(f"[{idx:3d}/{total}] {src.name}")

        intervals, total_dur = scan_intervals(src)

        if not intervals:
            log.warning(f"       -> SKIP (no fire/smoke detected)")
            n_skip += 1
            continue

        active_dur = sum(et - st for st, et in intervals)
        purity = active_dur / max(total_dur, 0.001) * 100
        log.info(f"       -> {len(intervals)} segment(s) | "
                 f"{active_dur:.1f}s / {total_dur:.1f}s ({purity:.0f}% active)")
        for i, (st, et) in enumerate(intervals, 1):
            log.info(f"          seg{i}: {st:.1f}s - {et:.1f}s ({et-st:.1f}s)")

        dst = output_dir / f"{src.stem}_pure.mp4"

        # Resume: skip se file da ton tai
        if dst.exists() and dst.stat().st_size > 500:
            active_dur = sum(et - st for st, et in intervals)
            purity = active_dur / max(total_dur, 0.001) * 100
            size_mb = dst.stat().st_size / 1_048_576
            log.info(f"       [SKIP-EXISTS] {dst.name} ({size_mb:.1f} MB)")
            n_saved += 1
            records.append({
                "original_video":      src.name,
                "pure_video":          dst.name,
                "num_segments":        len(intervals),
                "active_duration_sec": round(active_dur, 1),
                "total_duration_sec":  round(total_dur, 1),
                "purity_pct":          round(purity, 1),
                "segments":            " | ".join(f"{st:.1f}-{et:.1f}s" for st, et in intervals),
            })
            continue

        ok = save_pure_video(src, dst, intervals, tmp_dir)

        if ok:
            size_mb = dst.stat().st_size / 1_048_576
            log.info(f"       -> OK  {dst.name}  ({size_mb:.1f} MB)")
            n_saved += 1
            records.append({
                "original_video":      src.name,
                "pure_video":          dst.name,
                "num_segments":        len(intervals),
                "active_duration_sec": round(active_dur, 1),
                "total_duration_sec":  round(total_dur, 1),
                "purity_pct":          round(purity, 1),
                "segments":            " | ".join(f"{st:.1f}-{et:.1f}s"
                                                   for st, et in intervals),
            })
        else:
            log.error(f"       → ERROR saving {dst.name}")
            n_err += 1

    # Cleanup tmp
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    # Manifest CSV
    csv_path = output_dir / "pure_fire_smoke_manifest.csv"
    fieldnames = ["original_video", "pure_video", "num_segments",
                  "active_duration_sec", "total_duration_sec",
                  "purity_pct", "segments"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)

    total_active_min = sum(float(r["active_duration_sec"]) for r in records) / 60

    print("\n" + "=" * 70)
    print("  PURE FIRE & SMOKE EXTRACTION — DONE")
    print("=" * 70)
    print(f"  Scanned  : {total}")
    print(f"  Saved    : {n_saved}")
    print(f"  Skipped  : {n_skip}  (no fire/smoke)")
    print(f"  Errors   : {n_err}")
    print(f"  Total active footage : {total_active_min:.1f} min")
    print(f"  Output   : {output_dir.resolve()}")
    print(f"  Manifest : {csv_path.resolve()}")
    print("=" * 70)


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="trimmed_fire_incidents",
                    help="Flat folder of input videos")
    ap.add_argument("--output", default="pure_fire_smoke_clips",
                    help="Output folder for pure clips")
    args = ap.parse_args()
    run(Path(args.input), Path(args.output))
