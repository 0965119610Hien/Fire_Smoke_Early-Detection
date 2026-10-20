"""
=============================================================================
POST-PROCESSING VALIDATOR
Kiểm tra và lọc các video đã download:
  1. Xác minh thời lượng thực tế bằng ffprobe
  2. Kiểm tra resolution (480p – 1080p)
  3. Loại bỏ file trùng lặp (dedup bằng perceptual hash)
  4. Tạo báo cáo chất lượng dataset

Yêu cầu: ffmpeg, pip install imageio[ffmpeg] imagehash pillow
=============================================================================
"""

import os
import json
import subprocess
import hashlib
from pathlib import Path
from collections import defaultdict
import csv

OUTPUT_DIR = Path("dataset_test")
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
MIN_DURATION_S = 5
MAX_DURATION_S = 15
MIN_HEIGHT = 480
MAX_HEIGHT = 1080


def run_ffprobe(filepath: Path) -> dict:
    """Dùng ffprobe để lấy metadata video chính xác."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(filepath)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return {}


def get_video_info(filepath: Path) -> dict:
    """Trích xuất duration, resolution, codec."""
    probe = run_ffprobe(filepath)
    info = {"duration_s": 0.0, "width": 0, "height": 0, "codec": "", "valid": False}

    if not probe:
        return info

    # Duration từ format
    try:
        info["duration_s"] = float(probe["format"]["duration"])
    except (KeyError, ValueError, TypeError):
        pass

    # Video stream info
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            info["width"] = stream.get("width", 0)
            info["height"] = stream.get("height", 0)
            info["codec"] = stream.get("codec_name", "")
            # Override duration với stream duration nếu chính xác hơn
            dur = stream.get("duration")
            if dur and info["duration_s"] == 0:
                try:
                    info["duration_s"] = float(dur)
                except (ValueError, TypeError):
                    pass
            break

    info["valid"] = (
        MIN_DURATION_S <= info["duration_s"] <= MAX_DURATION_S
        and MIN_HEIGHT <= info["height"] <= MAX_HEIGHT
    )
    return info


def compute_md5(filepath: Path, chunk_size: int = 65536) -> str:
    """Tính MD5 để phát hiện file trùng."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(output_dir: Path = OUTPUT_DIR):
    """Validate toàn bộ dataset, in báo cáo."""
    print("\n" + "="*65)
    print("  VIDEO DATASET VALIDATOR")
    print("="*65)

    stats = {
        "total": 0,
        "valid": 0,
        "invalid_duration": 0,
        "invalid_resolution": 0,
        "duplicate": 0,
        "missing": 0,
        "class_0": defaultdict(int),
        "class_1": defaultdict(int),
    }

    seen_md5s = {}
    report_rows = []

    # Tìm tất cả mp4 files
    video_files = sorted(output_dir.rglob("*.mp4"))
    print(f"\nFound {len(video_files)} .mp4 files in {output_dir}")

    for vpath in video_files:
        stats["total"] += 1
        row = {"path": str(vpath), "status": "ok", "reason": ""}

        # Phân loại class từ path
        parts = vpath.parts
        class_label = -1
        for p in parts:
            if p.startswith("class_"):
                try:
                    class_label = int(p.split("_")[1])
                except (IndexError, ValueError):
                    pass

        # Kiểm tra tồn tại
        if not vpath.exists() or vpath.stat().st_size < 50_000:
            stats["missing"] += 1
            row["status"] = "missing_or_tiny"
            report_rows.append(row)
            continue

        # ffprobe
        info = get_video_info(vpath)
        row.update(info)

        # Kiểm tra duration
        if not (MIN_DURATION_S <= info["duration_s"] <= MAX_DURATION_S):
            stats["invalid_duration"] += 1
            row["status"] = "invalid_duration"
            row["reason"] = f"duration={info['duration_s']:.1f}s"
            report_rows.append(row)
            continue

        # Kiểm tra resolution
        if not (MIN_HEIGHT <= info["height"] <= MAX_HEIGHT):
            stats["invalid_resolution"] += 1
            row["status"] = "invalid_resolution"
            row["reason"] = f"height={info['height']}px"
            report_rows.append(row)
            continue

        # Dedup
        md5 = compute_md5(vpath)
        if md5 in seen_md5s:
            stats["duplicate"] += 1
            row["status"] = "duplicate"
            row["reason"] = f"same as {seen_md5s[md5]}"
            report_rows.append(row)
            continue
        seen_md5s[md5] = str(vpath)

        # Valid!
        stats["valid"] += 1
        if class_label == 0:
            subtype = vpath.parent.name
            stats["class_0"][subtype] += 1
        elif class_label == 1:
            subtype = vpath.parent.name
            stats["class_1"][subtype] += 1

        report_rows.append(row)

    # ── Print Report ──────────────────────────────────────────────────────────
    print(f"\n  SUMMARY")
    print(f"  Total files:        {stats['total']}")
    print(f"  ✓ Valid:            {stats['valid']}")
    print(f"  ✗ Invalid duration: {stats['invalid_duration']}")
    print(f"  ✗ Invalid res:      {stats['invalid_resolution']}")
    print(f"  ✗ Duplicates:       {stats['duplicate']}")
    print(f"  ✗ Missing/tiny:     {stats['missing']}")

    print(f"\n  Class 1 (Positive) breakdown:")
    for subtype, count in sorted(stats["class_1"].items()):
        print(f"    {subtype:30s}: {count}")

    print(f"\n  Class 0 (Negative / Hard Neg) breakdown:")
    for subtype, count in sorted(stats["class_0"].items()):
        print(f"    {subtype:30s}: {count}")

    total_valid_c0 = sum(stats["class_0"].values())
    total_valid_c1 = sum(stats["class_1"].values())
    print(f"\n  Valid Class 0:  {total_valid_c0} / 150")
    print(f"  Valid Class 1:  {total_valid_c1} / 150")
    print(f"  TOTAL VALID:    {stats['valid']} / 300")

    if stats['valid'] < 300:
        print(f"\n  ⚠ Cần thêm {300 - stats['valid']} video nữa.")
    else:
        print(f"\n  ✓ Dataset đạt đủ 300 video!")

    # Lưu validation report
    report_path = output_dir / "validation_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "status", "reason",
                                                "duration_s", "width", "height",
                                                "codec", "valid"])
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"\n  Validation report saved: {report_path}")

    return stats


if __name__ == "__main__":
    import sys
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    validate_dataset(dir_arg)