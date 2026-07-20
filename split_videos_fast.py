"""
Script cat video thanh clip ~15-20 giay, overlap 3 giay.
Dung ffmpeg voi -c copy (khong re-encode) -> rat nhanh.
- Input:  D:\zip_dataset\dataset_merged\dataset_merged\dataset_merged
- Output: D:\zip_dataset\dataset_split  (giu cau truc class_0/class_1/...)
"""

import subprocess
import sys
import time
from pathlib import Path

FFMPEG = r"D:\zip_dataset\venv38\lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win64-v4.2.2.exe"

INPUT_ROOT  = Path(r"D:\zip_dataset\dataset_merged\dataset_merged\dataset_merged")
OUTPUT_ROOT = Path(r"D:\zip_dataset\dataset_split")

CLIP_DURATION = 15      # giay
OVERLAP       = 3       # giay
MIN_CLIP_SECS = 5       # clip cuoi ngan hon nay thi gop vao clip truoc
MAX_CLIP_SECS = 20      # cap tren clip cuoi

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
OUTPUT_EXT = ".mp4"


def get_duration(path):
    """Lay thoi luong video (giay) bang ffprobe."""
    result = subprocess.run(
        [FFMPEG, "-v", "quiet", "-print_format", "json",
         "-show_entries", "format=duration", "-i", str(path)],
        capture_output=True, text=True
    )
    # ffmpeg -i in stderr, ffprobe in stdout
    # dung ffmpeg -i de lay thong tin
    result2 = subprocess.run(
        [FFMPEG, "-i", str(path)],
        capture_output=True, text=True
    )
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", result2.stderr)
    if not m:
        return None
    h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mi * 60 + s


def compute_segments(duration: float):
    """
    Tinh danh sach (start_sec, end_sec) cho tung clip.
    stride = CLIP_DURATION - OVERLAP
    """
    stride = CLIP_DURATION - OVERLAP
    segments = []
    start = 0.0
    while start < duration:
        end = min(start + CLIP_DURATION, duration)
        segments.append((start, end))
        if end >= duration:
            break
        start += stride

    # Neu clip cuoi qua ngan, gop vao clip truoc
    if len(segments) > 1:
        ls, le = segments[-1]
        if (le - ls) < MIN_CLIP_SECS:
            ps, pe = segments[-2]
            new_end = min(pe + (le - ls), duration)
            new_end = min(new_end, ps + MAX_CLIP_SECS)
            segments[-2] = (ps, new_end)
            segments.pop()

    return segments


def cut_clip(input_path: Path, output_path: Path, start: float, end: float) -> bool:
    """Dung ffmpeg -ss -to -c copy de cat clip (sieu nhanh, khong re-encode)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(input_path),
        "-c", "copy",
        "-avoid_negative_ts", "1",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def process_video(vpath: Path, out_dir: Path, stats: dict):
    duration = get_duration(vpath)
    if duration is None or duration <= 0:
        print(f"  [SKIP] Khong lay duoc duration: {vpath.name}")
        stats["skipped"] += 1
        return

    stem = vpath.stem

    if duration <= CLIP_DURATION:
        out_path = out_dir / f"{stem}_clip001{OUTPUT_EXT}"
        ok = cut_clip(vpath, out_path, 0, duration)
        if ok:
            stats["clips_written"] += 1
        stats["videos_processed"] += 1
        return

    segments = compute_segments(duration)
    for idx, (s, e) in enumerate(segments, 1):
        out_name = f"{stem}_clip{idx:03d}{OUTPUT_EXT}"
        out_path = out_dir / out_name
        ok = cut_clip(vpath, out_path, s, e)
        clip_dur = e - s
        status = "OK" if ok else "FAIL"
        print(f"    -> [{status}] clip {idx:03d}: {s:.1f}s - {e:.1f}s  ({clip_dur:.1f}s)")
        if ok:
            stats["clips_written"] += 1

    stats["videos_processed"] += 1


def main():
    if not Path(FFMPEG).exists():
        print(f"[ERROR] FFmpeg khong tim thay tai: {FFMPEG}")
        sys.exit(1)

    if not INPUT_ROOT.exists():
        print(f"[ERROR] INPUT_ROOT khong ton tai: {INPUT_ROOT}")
        sys.exit(1)

    # Lay danh sach video chua co trong OUTPUT_ROOT
    video_files = sorted([
        p for p in INPUT_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    total = len(video_files)
    print(f"Tim thay {total} video trong {INPUT_ROOT}")
    print(f"Output -> {OUTPUT_ROOT}")
    print(f"Config : clip={CLIP_DURATION}s | overlap={OVERLAP}s | stride={CLIP_DURATION - OVERLAP}s")
    print("=" * 70)

    stats = {"videos_processed": 0, "clips_written": 0, "skipped": 0}
    t0 = time.time()

    for i, vpath in enumerate(video_files, 1):
        rel     = vpath.relative_to(INPUT_ROOT)
        out_dir = OUTPUT_ROOT / rel.parent

        # Skip neu tat ca clip tuong ung da ton tai (resume support)
        stem = vpath.stem
        existing = list(out_dir.glob(f"{stem}_clip*.mp4")) if out_dir.exists() else []
        if existing:
            print(f"[{i:4d}/{total}] SKIP (da co {len(existing)} clip): {rel}")
            stats["clips_written"] += len(existing)
            stats["videos_processed"] += 1
            continue

        elapsed = time.time() - t0
        eta_str = ""
        if i > 1:
            avg = elapsed / (i - 1)
            eta = avg * (total - i + 1)
            eta_str = f"  ETA ~{eta/60:.1f}min"

        print(f"[{i:4d}/{total}] {rel}{eta_str}")
        process_video(vpath, out_dir, stats)

    total_time = time.time() - t0
    print("=" * 70)
    print(f"HOAN THANH!")
    print(f"  Videos xu ly : {stats['videos_processed']}")
    print(f"  Clips tao ra : {stats['clips_written']}")
    print(f"  Bo qua       : {stats['skipped']}")
    print(f"  Thoi gian    : {total_time/60:.1f} phut")
    print(f"  Output dir   : {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
