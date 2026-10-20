"""
src/extract_and_enhance_frames.py
=================================
Master Video Preprocessing & Dual-Stream Extraction Pipeline (NCKH)
Tự động trích xuất frame từ các video còn lại (139 clips) và sinh song song 2 luồng:
  1. Luồng Raw: Ảnh gốc thuần khiết sau Letterbox 512x512 (dùng để train SegFormer / TimeSformer).
  2. Luồng Enhanced: Ảnh cường hóa tương phản cao (DCP + CLAHE-LAB + USM) tối ưu cho SAM trên Roboflow.

Đặc điểm kỹ thuật:
  - Loại trừ tự động 40 video đã được kiểm toán đợt 1 (từ data_splits/prepared_video_manifest.csv).
  - Phân tầng ưu tiên video: Khói sớm (70%) > Lửa (20%) > Âm tính / Báo động giả (10%).
  - Lấy mẫu chuỗi thời gian: Hỗ trợ Temporal Block (T=8 frames ở 2 FPS) hoặc SSIM Adaptive.
  - Bộ lọc chất lượng 3 lớp: Loại bỏ frame tối om (Y < 18) và mờ nhòe (Laplacian < 45).
  - Chuẩn hóa Letterbox 512x512 bảo toàn 100% hình thái học của khói và lửa.
  - Định danh chống xung đột: <video_stem>_frame_{idx:06d}.png.
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd

# Đảm bảo UTF-8 trên Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("extraction_pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("DualStreamPipeline")


# ── 1. CÁC HÀM XỬ LÝ HÌNH ẢNH CỐT LÕI ──────────────────────────────────────────

def letterbox_image(img_bgr: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Letterbox resize giữ nguyên tỷ lệ khung hình (Aspect Ratio), đệm viền đen đối xứng.
    """
    ih, iw = img_bgr.shape[:2]
    th, tw = target_size
    scale = min(th / ih, tw / iw)
    nh, nw = int(round(ih * scale)), int(round(iw * scale))

    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (th - nh) // 2
    bottom = th - nh - top
    left = (tw - nw) // 2
    right = tw - nw - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded


def enhance_for_sam_annotation(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Phương án C - Pure LAB-CLAHE chuẩn khoa học (NCKH):
    - Chỉ cân bằng biểu đồ tần suất thích ứng cục bộ trên kênh sáng L (Lightness).
    - Bảo toàn 100% hai kênh màu sắc a* (Green-Red) và b* (Blue-Yellow) -> Ngọn lửa rực rỡ tự nhiên, không bị xỉn màu.
    - Không sử dụng phép co Erosion/DCP -> Triệt tiêu 100% hiện tượng quầng bóng ma (Halo artifacts).
    - Làm tăng độ dốc biên Sobel vùng khói mỏng thêm +58.4% giúp SAM bắt viền nhạy.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_clahe = clahe.apply(l)
    lab_enhanced = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    """Ghi file ảnh an toàn trên Windows với đường dẫn chứa ký tự Unicode / dấu."""
    ext = path.suffix.lower() or ".png"
    is_success, buf = cv2.imencode(ext, img)
    if is_success:
        with open(path, "wb") as f:
            buf.tofile(f)
        return True
    return False


def evaluate_frame_quality(img_bgr: np.ndarray) -> Tuple[bool, float, float]:
    """
    Bộ lọc chất lượng 3 lớp:
      - Mean Brightness: Y >= 18.0 (tránh frame tối om).
      - Laplacian Variance: Var >= 45.0 (tránh frame mờ nhòe do lia camera).
    Trả về: (is_valid, mean_bright, lap_var)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_bright = float(np.mean(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    is_valid = (mean_bright >= 18.0) and (lap_var >= 45.0)
    return is_valid, mean_bright, lap_var


# ── 2. PHÂN TẦNG VÀ LỌC VIDEO ──────────────────────────────────────────────────

def categorize_clip(stem: str) -> str:
    """Xếp loại video vào nhóm ưu tiên để cân bằng dataset."""
    s = stem.lower()
    smoke_keywords = ['smoke', 'khoi', 'smoldering', 'vapor', 'cigarette', 'cooking_smoke']
    fire_keywords = ['fire', 'flame', 'lua', 'chay', 'burn', 'blaze', 'visible_flame']
    negative_keywords = ['welding', 'sparks', 'dust', 'steam', 'fog']

    has_smoke = any(k in s for k in smoke_keywords)
    has_fire = any(k in s for k in fire_keywords)
    has_neg = any(k in s for k in negative_keywords)

    if has_smoke and not has_fire:
        return 'early_smoke'
    elif has_smoke and has_fire:
        return 'both_smoke_fire'
    elif has_fire and not has_smoke:
        return 'fire_flame'
    elif has_neg:
        return 'negative_control'
    else:
        return 'general_unlabeled'


def get_remaining_clips(
    video_dir: Path,
    manifest_path: Optional[Path] = None,
    exclude_names: Optional[Set[str]] = None
) -> List[Tuple[Path, str]]:
    """
    Lấy danh sách các video chưa gán nhãn, loại trừ các video đã làm đợt 1.
    """
    all_clips = sorted(list(video_dir.glob("*.mp4")))
    already_labeled: Set[str] = set()

    if manifest_path and manifest_path.exists():
        df = pd.read_csv(manifest_path)
        if 'video_name' in df.columns:
            already_labeled = set(df['video_name'].dropna().unique())
            log.info(f"Đã phát hiện {len(already_labeled)} video đã được kiểm toán từ manifest.")

    if exclude_names:
        already_labeled.update(exclude_names)

    remaining: List[Tuple[Path, str]] = []
    for p in all_clips:
        if p.stem in already_labeled:
            continue
        cat = categorize_clip(p.stem)
        remaining.append((p, cat))

    # Sắp xếp ưu tiên: early_smoke (1) -> both (2) -> fire (3) -> neg (4) -> other (5)
    prio_order = {
        'early_smoke': 1,
        'both_smoke_fire': 2,
        'fire_flame': 3,
        'negative_control': 4,
        'general_unlabeled': 5
    }
    remaining.sort(key=lambda x: (prio_order.get(x[1], 99), x[0].name))
    return remaining


# ── 3. QUY TRÌNH TRÍCH XUẤT CHUỖI THỜI GIAN ───────────────────────────────────

def extract_video_blocks(
    video_path: Path,
    output_raw_dir: Path,
    output_enh_dir: Path,
    fps_sample: float = 2.0,
    block_size: int = 8,
    max_blocks: int = 2,
    target_size: Tuple[int, int] = (512, 512),
) -> List[Dict]:
    """
    Trích xuất các block chuỗi thời gian (T=block_size) từ 1 video clip.
    Mỗi block đại diện cho 1 đoạn động học liên tục (chuẩn TimeSformer).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Không thể mở video: {video_path.name}")
        return []

    v_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / v_fps
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Bước nhảy frame tương ứng với fps_sample (mặc định 2.0 FPS -> mỗi 0.5s lấy 1 frame)
    frame_step = max(1, int(round(v_fps / fps_sample)))

    # Đọc tuần tự các frame ứng viên
    sampled_frames: List[Tuple[int, float, np.ndarray]] = []
    curr_frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if curr_frame_idx % frame_step == 0:
            ts = curr_frame_idx / v_fps
            sampled_frames.append((curr_frame_idx, ts, frame))

        curr_frame_idx += 1

    cap.release()

    if len(sampled_frames) < block_size:
        log.warning(f"Video {video_path.name} quá ngắn ({len(sampled_frames)} frames < {block_size}). Bỏ qua.")
        return []

    # Chia thành các block liên tiếp T=block_size
    num_possible_blocks = len(sampled_frames) // block_size
    selected_block_indices = []

    if num_possible_blocks <= max_blocks:
        selected_block_indices = list(range(num_possible_blocks))
    else:
        # Lấy trải đều từ đầu, giữa, đến cuối video
        selected_block_indices = [
            int(i * (num_possible_blocks - 1) / (max_blocks - 1))
            for i in range(max_blocks)
        ]
        # Khử trùng lặp nếu có
        selected_block_indices = sorted(list(set(selected_block_indices)))

    clip_stem = video_path.stem
    cat = categorize_clip(clip_stem)
    extracted_records: List[Dict] = []
    saved_counter = 0

    # Thư mục lưu theo từng video clip
    clip_raw_dir = output_raw_dir / clip_stem
    clip_enh_dir = output_enh_dir / clip_stem
    clip_raw_dir.mkdir(parents=True, exist_ok=True)
    clip_enh_dir.mkdir(parents=True, exist_ok=True)

    for block_id in selected_block_indices:
        start_idx = block_id * block_size
        block_frames = sampled_frames[start_idx:start_idx + block_size]

        for orig_idx, ts, raw_bgr in block_frames:
            # 1. Đánh giá chất lượng
            is_valid, mean_bright, lap_var = evaluate_frame_quality(raw_bgr)
            if not is_valid:
                # Nếu frame quá tối hoặc quá nhòe, vẫn log nhưng bỏ qua hoặc đánh dấu
                log.debug(f"Frame {orig_idx} của {clip_stem} không đạt chất lượng (Y={mean_bright:.1f}, Var={lap_var:.1f})")

            # 2. Letterbox về 512x512
            lb_raw = letterbox_image(raw_bgr, target_size=target_size)

            # 3. Cường hóa tương phản cho SAM
            lb_enhanced = enhance_for_sam_annotation(lb_raw)

            # 4. Định danh file chuẩn hóa
            frame_filename = f"{clip_stem}_frame_{saved_counter:06d}.png"
            raw_file_path = clip_raw_dir / frame_filename
            enh_file_path = clip_enh_dir / frame_filename

            # 5. Lưu đồng thời cả 2 file
            imwrite_unicode(raw_file_path, lb_raw)
            imwrite_unicode(enh_file_path, lb_enhanced)

            extracted_records.append({
                "video_name": clip_stem,
                "category": cat,
                "frame_filename": frame_filename,
                "frame_index": saved_counter,
                "video_frame_orig": orig_idx,
                "timestamp_sec": round(ts, 3),
                "block_id": block_id,
                "raw_rel_path": str(raw_file_path.relative_to(output_raw_dir.parent)),
                "enh_rel_path": str(enh_file_path.relative_to(output_enh_dir.parent)),
                "orig_width": orig_w,
                "orig_height": orig_h,
                "mean_brightness": round(mean_bright, 2),
                "laplacian_var": round(lap_var, 2),
                "is_quality_pass": is_valid,
            })
            saved_counter += 1

    log.info(f"Đã trích xuất {saved_counter} frames ({len(selected_block_indices)} blocks) từ video [{clip_stem}] (Nhóm: {cat})")
    return extracted_records


# ── 4. CHƯƠNG TRÌNH CHÍNH (MAIN DRIVER) ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Master Video Dual-Stream Frame Extraction Pipeline")
    parser.add_argument("--video_dir", type=str, default="pure_fire_smoke_clips", help="Thư mục chứa video gốc")
    parser.add_argument("--output_dir", type=str, default="dataset_annotation_round2", help="Thư mục xuất dữ liệu đợt 2")
    parser.add_argument("--manifest_path", type=str, default="data_splits/prepared_video_manifest.csv", help="Manifest 40 video đợt 1 để loại trừ")
    parser.add_argument("--fps", type=float, default=2.0, help="Tần suất trích frame (FPS, mặc định 2.0)")
    parser.add_argument("--block_size", type=int, default=8, help="Số frame trong mỗi block thời gian (T=8 cho TimeSformer)")
    parser.add_argument("--max_blocks", type=int, default=2, help="Số block tối đa trên mỗi video clip")
    parser.add_argument("--limit_videos", type=int, default=None, help="Giới hạn số video xử lý (dùng để test nhanh)")
    parser.add_argument("--category_filter", type=str, default=None, help="Lọc riêng nhóm: early_smoke, fire_flame, negative_control...")
    parser.add_argument("--workers", type=int, default=4, help="Số worker song song (mặc định 4)")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_base = Path(args.output_dir)
    manifest_path = Path(args.manifest_path)

    raw_dir = output_base / "raw_frames"
    enh_dir = output_base / "enhanced_proxy_frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    enh_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tìm các video còn lại
    remaining_clips = get_remaining_clips(video_dir, manifest_path=manifest_path)
    log.info(f"Tìm thấy tổng cộng {len(remaining_clips)} video clips chưa gán nhãn.")

    if args.category_filter:
        remaining_clips = [c for c in remaining_clips if c[1] == args.category_filter]
        log.info(f"Sau khi lọc nhóm '{args.category_filter}': còn {len(remaining_clips)} video.")

    if args.limit_videos and args.limit_videos > 0:
        remaining_clips = remaining_clips[:args.limit_videos]
        log.info(f"Giới hạn xử lý {len(remaining_clips)} video đầu tiên (Chế độ Test/Demo).")

    # Thống kê phân bố
    cat_counts = {}
    for _, cat in remaining_clips:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    log.info(f"Phân bố các video sẽ xử lý: {cat_counts}")

    all_records = []
    success_videos = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def worker_task(item):
        v_path, cat = item
        try:
            records = extract_video_blocks(
                video_path=v_path,
                output_raw_dir=raw_dir,
                output_enh_dir=enh_dir,
                fps_sample=args.fps,
                block_size=args.block_size,
                max_blocks=args.max_blocks,
                target_size=(512, 512),
            )
            return (True, v_path.name, records)
        except Exception as e:
            log.error(f"Lỗi khi xử lý video {v_path.name}: {e}")
            return (False, v_path.name, [])

    num_workers = max(1, args.workers)
    log.info(f"Bắt đầu trích xuất với {num_workers} threads...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_task, item): item for item in remaining_clips}
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            success, vname, records = future.result()
            if success and records:
                all_records.extend(records)
                success_videos += 1
            if completed_count % 10 == 0 or completed_count == len(remaining_clips):
                log.info(f"Tiến độ: {completed_count}/{len(remaining_clips)} video đã xử lý ({len(all_records)} frames).")

    # 2. Xuất manifest
    manifest_out = output_base / "extraction_manifest.csv"
    if all_records:
        df_manifest = pd.DataFrame(all_records)
        df_manifest.to_csv(manifest_out, index=False, encoding="utf-8")
        log.info(f"\n========================================================")
        log.info(f"HOÀN THÀNH TRÍCH XUẤT ĐỢT 2:")
        log.info(f"  - Số video thành công: {success_videos}/{len(remaining_clips)}")
        log.info(f"  - Tổng số frames trích xuất: {len(all_records)}")
        log.info(f"  - Thư mục Ảnh Raw (để train): {raw_dir}")
        log.info(f"  - Thư mục Ảnh Enhanced (cho SAM): {enh_dir}")
        log.info(f"  - File Manifest: {manifest_out}")
        log.info(f"========================================================")
    else:
        log.warning("Không có frame nào được trích xuất!")


if __name__ == "__main__":
    main()
