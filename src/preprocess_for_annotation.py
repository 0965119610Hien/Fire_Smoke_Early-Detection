"""
preprocess_for_annotation.py
============================
Tiền xử lý toàn bộ video từ pure_fire_smoke_clips/ thành các frame ảnh chuẩn
cho việc gán nhãn (Annotation) bằng SegFormer.

5 BƯỚC TIỀN XỬ LÝ:
  1. Video Chunking     -- Cắt video > 120s thành chunk 30s
  2. Smart Frame Sample -- Lấy mẫu 1-2 FPS hoặc theo chỉ số SSIM
  3. Quality Filter     -- Loại frame tối (brightness), mờ (blur), lỗi nén
  4. Letterbox Resize   -- Chuẩn hóa về 512x512 giữ nguyên tỷ lệ khung hình
  5. CLAHE Enhancement  -- Tăng tương phản khói mỏng trên CCTV

OUTPUT STRUCTURE:
  annotation_frames/
  ├── <video_stem>/
  │   ├── frame_000001.png
  │   ├── frame_000002.png
  │   └── ...
  └── preprocessing_manifest.csv

CHẠY:
  python src/preprocess_for_annotation.py --input pure_fire_smoke_clips --output annotation_frames
  python src/preprocess_for_annotation.py --input pure_fire_smoke_clips --output annotation_frames --fps 1 --target_size 512 --no_clahe
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("preprocess_for_annotation.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


# ── BƯỚC 4: Letterbox Resize ─────────────────────────────────────────────────
def letterbox_resize(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    """
    Resize ảnh về target_size x target_size giữ nguyên tỷ lệ khung hình (Aspect Ratio).
    Phần còn lại được pad bằng viền đen.
    """
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    top  = (target_size - nh) // 2
    left = (target_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


# ── BƯỚC 5: CLAHE Enhancement ────────────────────────────────────────────────
def apply_clahe(bgr: np.ndarray, clip_limit: float = 2.0,
                grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Tăng cường tương phản khói mỏng bằng CLAHE trên kênh L (color space LAB).
    Giúp làm nổi rõ làn khói mờ trên nền xám của camera CCTV.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ── BƯỚC 3: Quality Filter ───────────────────────────────────────────────────
def is_frame_valid(frame: np.ndarray,
                   min_brightness: float = 15.0,
                   min_blur_var: float = 40.0) -> Tuple[bool, str]:
    """
    Kiểm tra chất lượng frame:
      - Loại frame quá tối (mean brightness < min_brightness)
      - Loại frame bị mờ nhòe do camera rung (Laplacian variance < min_blur_var)
    Trả về (True, "ok") nếu hợp lệ, (False, reason) nếu loại bỏ.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < min_brightness:
        return False, f"too_dark (brightness={brightness:.1f})"
    if blur_var < min_blur_var:
        return False, f"too_blurry (lap_var={blur_var:.1f})"
    return True, "ok"


# ── BƯỚC 2: SSIM-based Frame Sampler ─────────────────────────────────────────
def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Tính Structural Similarity Index (SSIM) giữa 2 frame grayscale."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    # Resize nhỏ xuống để tính nhanh
    ga = cv2.resize(ga, (128, 128))
    gb = cv2.resize(gb, (128, 128))
    score, _ = ssim(ga, gb, full=True)
    return float(score)


# ── BƯỚC 1: Video Chunker ────────────────────────────────────────────────────
def get_video_info(video_path: Path) -> Tuple[float, float, int]:
    """Lấy thông tin FPS, Duration, Total Frames của video."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n / fps if fps > 0 else 0.0, n


def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    target_fps: float = 1.0,
    use_ssim: bool = True,
    ssim_threshold: float = 0.92,
    target_size: int = 512,
    apply_clahe_flag: bool = True,
    min_brightness: float = 15.0,
    min_blur_var: float = 40.0,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    frame_offset: int = 0,
) -> int:
    """
    Trích xuất frame từ 1 video (hoặc 1 chunk của video):
      - start_sec, end_sec: Khoảng thời gian cần trích xuất
      - frame_offset: Số thứ tự bắt đầu của frame trong tên file

    Trả về số lượng frame đã lưu thành công.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"Cannot open video: {video_path}")
        return 0

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_dur = total_frames / orig_fps

    if end_sec is None:
        end_sec = total_dur

    start_frame = int(start_sec * orig_fps)
    end_frame   = int(min(end_sec * orig_fps, total_frames - 1))

    # Bước nhảy: đọc frame theo target_fps
    frame_step = max(1, int(round(orig_fps / target_fps)))

    out_dir.mkdir(parents=True, exist_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_frame: Optional[np.ndarray] = None
    saved  = 0
    idx    = start_frame

    while idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (idx - start_frame) % frame_step == 0:
            # Bước 3: Quality Filter
            valid, reason = is_frame_valid(frame, min_brightness, min_blur_var)
            if not valid:
                idx += 1
                continue

            # Bước 2: SSIM sampling (bỏ qua frame quá giống frame trước)
            if use_ssim and prev_frame is not None:
                sim = compute_ssim(prev_frame, frame)
                if sim > ssim_threshold:
                    prev_frame = frame.copy()  # Update prev_frame to move sliding window forward!
                    idx += 1
                    continue

            prev_frame = frame.copy()

            # Bước 4: Letterbox Resize
            processed = letterbox_resize(frame, target_size)

            # Bước 5: CLAHE Enhancement
            if apply_clahe_flag:
                processed = apply_clahe(processed)

            # Lưu frame
            frame_name = f"frame_{frame_offset + saved:06d}.png"
            out_path = out_dir / frame_name
            cv2.imwrite(str(out_path), processed, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            saved += 1

        idx += 1

    cap.release()
    return saved


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def run_preprocessing(
    input_dir: Path,
    output_dir: Path,
    target_fps: float = 1.0,
    use_ssim: bool = True,
    ssim_threshold: float = 0.92,
    target_size: int = 512,
    apply_clahe_flag: bool = True,
    chunk_max_sec: float = 120.0,   # Video dài hơn giá trị này sẽ bị chia chunk
    chunk_size_sec: float = 30.0,   # Độ dài mỗi chunk (giây)
    min_brightness: float = 15.0,
    min_blur_var: float = 40.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_files = sorted(p for p in input_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    total = len(video_files)

    log.info(f"Found {total} videos in '{input_dir}'")
    log.info(f"Settings: target_fps={target_fps}, ssim_thresh={ssim_threshold}, "
             f"target_size={target_size}, clahe={apply_clahe_flag}, "
             f"chunk_max={chunk_max_sec}s, chunk_size={chunk_size_sec}s")
    log.info("=" * 70)

    manifest_rows = []
    total_frames_saved = 0

    for vid_idx, vid_path in enumerate(video_files, 1):
        fps, duration, n_frames = get_video_info(vid_path)
        log.info(f"[{vid_idx:3d}/{total}] {vid_path.name} ({duration:.1f}s, {fps:.0f}fps)")

        # BƯỚC 1: Video Chunking
        if duration > chunk_max_sec:
            chunks = [(s, min(s + chunk_size_sec, duration))
                      for s in np.arange(0, duration, chunk_size_sec)]
            log.info(f"         -> Long video ({duration:.0f}s): splitting into {len(chunks)} chunks of {chunk_size_sec}s")
        else:
            chunks = [(0.0, duration)]

        vid_frames_saved = 0
        for c_idx, (start, end) in enumerate(chunks):
            # Tên thư mục: <video_stem> hoặc <video_stem>_chunk_XX
            if len(chunks) == 1:
                chunk_dir = output_dir / vid_path.stem
            else:
                chunk_dir = output_dir / f"{vid_path.stem}_chunk_{c_idx + 1:02d}"

            saved = extract_frames_from_video(
                video_path=vid_path,
                out_dir=chunk_dir,
                target_fps=target_fps,
                use_ssim=use_ssim,
                ssim_threshold=ssim_threshold,
                target_size=target_size,
                apply_clahe_flag=apply_clahe_flag,
                min_brightness=min_brightness,
                min_blur_var=min_blur_var,
                start_sec=start,
                end_sec=end,
                frame_offset=vid_frames_saved,
            )

            log.info(f"         -> Chunk [{c_idx + 1}/{len(chunks)}] "
                     f"({start:.0f}s-{end:.0f}s): {saved} frames saved -> {chunk_dir.name}/")
            vid_frames_saved += saved

            manifest_rows.append({
                "video":          vid_path.name,
                "chunk_dir":      chunk_dir.name,
                "start_sec":      round(start, 1),
                "end_sec":        round(end, 1),
                "frames_saved":   saved,
            })

        total_frames_saved += vid_frames_saved

    # Ghi Manifest CSV
    csv_path = output_dir / "preprocessing_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video", "chunk_dir", "start_sec", "end_sec", "frames_saved"])
        w.writeheader()
        w.writerows(manifest_rows)

    print("\n" + "=" * 70)
    print("  PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Videos processed    : {total}")
    print(f"  Total frames saved  : {total_frames_saved:,}")
    print(f"  Frame size          : {target_size} x {target_size} px (PNG)")
    print(f"  Output directory    : {output_dir.resolve()}")
    print(f"  Manifest CSV        : {csv_path.resolve()}")
    print("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Preprocess fire/smoke video clips into annotation-ready frames."
    )
    ap.add_argument("--input",       default="pure_fire_smoke_clips",
                    help="Input folder of pure fire/smoke video clips")
    ap.add_argument("--output",      default="annotation_frames",
                    help="Output folder for extracted frames (PNG)")
    ap.add_argument("--fps",         type=float, default=1.0,
                    help="Target sampling rate in frames-per-second (default: 1.0)")
    ap.add_argument("--ssim",        type=float, default=0.92,
                    help="SSIM threshold to skip near-duplicate frames (default: 0.92)")
    ap.add_argument("--no_ssim",     action="store_true",
                    help="Disable SSIM-based adaptive sampling (use fixed FPS only)")
    ap.add_argument("--target_size", type=int, default=512,
                    help="Output frame resolution (default: 512 -> 512x512)")
    ap.add_argument("--no_clahe",    action="store_true",
                    help="Disable CLAHE smoke contrast enhancement")
    ap.add_argument("--chunk_max",   type=float, default=120.0,
                    help="Max video duration before chunking (seconds, default: 120)")
    ap.add_argument("--chunk_size",  type=float, default=30.0,
                    help="Chunk length in seconds for long videos (default: 30)")
    ap.add_argument("--min_brightness", type=float, default=15.0,
                    help="Minimum frame mean brightness (default: 15)")
    ap.add_argument("--min_blur",    type=float, default=15.0,
                    help="Minimum Laplacian variance for blur detection (default: 15)")

    args = ap.parse_args()

    run_preprocessing(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        target_fps=args.fps,
        use_ssim=not args.no_ssim,
        ssim_threshold=args.ssim,
        target_size=args.target_size,
        apply_clahe_flag=not args.no_clahe,
        chunk_max_sec=args.chunk_max,
        chunk_size_sec=args.chunk_size,
        min_brightness=args.min_brightness,
        min_blur_var=args.min_blur,
    )
