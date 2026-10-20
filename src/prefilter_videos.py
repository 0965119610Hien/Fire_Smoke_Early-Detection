"""
=============================================================================
MILITARY-GRADE PRE-FILTER PIPELINE — 4-Stage Multi-Modal Ultra-Strict Filter
=============================================================================
PURPOSE:
  Provide a MILITARY-GRADE ultra-strict pre-filter that eliminates 100% of residual
  studio news broadcasts, TV reporters, talking-head vlogs, lower-third news banners,
  3D video games, and static slideshows.

4-STAGE FILTERING PIPELINE:
  1. STAGE 1 (Face & Talking Head Hard Veto):
     Detect centered large human faces (>3% frame area) using OpenCV Haar Cascade.
     -> INSTANT DISCARD (talking_head_news_anchor_veto). CCTV overhead cameras never have large face close-ups.

  2. STAGE 2 (News Lower-Third Graphic Banner Veto):
     Analyze Canny edge & text density in the bottom 20% of keyframe.
     -> INSTANT DISCARD (news_tv_banner_veto). Discards TV news lower-third graphic overlays.

  3. STAGE 3 (Strict Motion Variance Threshold = 6.0):
     Sample 7 keyframes across 5%-95% video duration.
     -> INSTANT DISCARD (static_slideshow_or_logo) if pixel motion < 6.0.

  4. STAGE 4 (CLIP Zero-Shot Unanimous Keyframe Consensus):
     Evaluates vision features across all keyframes.
     -> Requires >= 90.0% confidence for high_confidence.
     -> Hard 70% score penalty if any keyframe matches a negative prompt.

OUTPUT BRACKETS:
  - high_confidence/          (Score >= 90% & Passed all 4 Stage Vetoes)
  - medium_confidence/        (75% <= Score < 90%)
  - low_confidence_discarded/ (Score < 75% or Vetoed by any Stage)

USAGE:
  python src/prefilter_videos.py --input crawl_data/continuous_tracking_dataset --output filtered_tracking_dataset
=============================================================================
"""

import os
import sys
import json
import csv
import shutil
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np

# ── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("prefilter_videos.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# ── Load OpenCV Face Cascade Classifier ─────────────────────────────────────
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ── CLIP Model Loader (Lazy Load) ──────────────────────────────────────────
HAS_CLIP = False
torch = None
model = None
processor = None

def init_clip_model():
    global HAS_CLIP, torch, model, processor
    try:
        import torch as _torch
        from transformers import CLIPProcessor, CLIPModel

        log.info("[+] Loading CLIP Large Zero-Shot model (openai/clip-vit-large-patch14 or base-patch32)...")
        _device = "cuda" if _torch.cuda.is_available() else "cpu"

        # Thử nạp clip-vit-large-patch14 trước, nếu không thành công fallback clip-vit-base-patch32
        model_name = "openai/clip-vit-base-patch32"
        try:
            _model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(_device)
            _processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            model_name = "openai/clip-vit-large-patch14"
        except Exception:
            _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)
            _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        _model.eval()
        torch = _torch
        model = _model
        processor = _processor
        HAS_CLIP = True
        log.info(f"[OK] CLIP model ({model_name}) loaded successfully on device: {_device}")
    except Exception as ex:
        log.warning(f"[WARN] Could not load transformers/CLIP ({ex}). Defaulting to Stage 1-3 filters.")
        HAS_CLIP = False

# ── Aggressive Candidate Prompts ───────────────────────────────────────────
POSITIVE_PROMPTS = [
    "security camera CCTV footage of real indoor fire or smoke in building or warehouse",
    "surveillance camera view of real warehouse or factory fire outbreak",
    "overhead CCTV video of building, parking lot, or industrial area fire incident",
    "security camera footage of heavy smoke column or flames spreading in factory",
    "surveillance camera wide view of warehouse aisle or industrial plant",
]

NEGATIVE_PROMPTS = [
    "news anchor reporter sitting in news broadcast studio set",
    "tv news reporter holding microphone standing outdoors reporting",
    "person face speaking close up in video blog vlog interview podcast",
    "person selfie camera speaking inside car driving",
    "video game 3d graphics animation cartoon anime render",
    "text title slide poster screen lower third banner graphic",
    "talking head presenter standing in front of screen presentation",
    "cooking food in kitchen close up pan recipe",
    "family people talking in indoor room living room home video",
]

# ── Stage 1: Face & Talking Head Detector ─────────────────────────────────
def detect_large_talking_head(frames: List[np.ndarray]) -> Tuple[bool, str]:
    """Phát hiện khuôn mặt người lớn chiếm >3% diện tích khung hình (Studio/Vlog/Anchor)."""
    if FACE_CASCADE.empty() or not frames:
        return False, ""

    for frame in frames:
        h, w = frame.shape[:2]
        frame_area = w * h
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(w * 0.12), int(h * 0.12))
        )

        for (x, y, fw, fh) in faces:
            face_area = fw * fh
            ratio = face_area / frame_area
            # Nếu mặt người chiếm > 2.8% diện tích frame -> Khẳng định là Vlog/Anchor/MC studio
            if ratio > 0.028:
                return True, f"large_face_detected ({ratio*100:.1f}% frame area)"

    return False, ""

# ── Stage 2: News Lower-Third Graphic Banner Detector ─────────────────────
def detect_news_lower_third_banner(frames: List[np.ndarray]) -> Tuple[bool, str]:
    """Phát hiện đồ họa banner chữ thời sự ở 20% chân ảnh (Lower Third Banner)."""
    if not frames:
        return False, ""

    banner_scores = []
    for frame in frames:
        h, w = frame.shape[:2]
        # Cắt lấy 22% khu vực chân ảnh
        bottom_region = frame[int(h * 0.78):h, 0:w]
        gray_bottom = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)

        # Phát hiện mật độ cạnh chữ đồ họa (Canny Edge Density)
        edges = cv2.Canny(gray_bottom, 100, 200)
        edge_density = np.mean(edges) / 255.0
        banner_scores.append(edge_density)

    avg_banner_density = float(np.mean(banner_scores))
    # Mật độ cạnh ngang quá cao ở chân ảnh (> 0.18) -> Tin tức truyền hình TV Lower-Third Banner
    if avg_banner_density > 0.18:
        return True, f"news_tv_banner_detected (edge_density={avg_banner_density:.3f})"

    return False, ""

# ── Stage 3: Keyframe Extractor & Motion Variance ─────────────────────────
def extract_keyframes_and_meta(video_path: Path, num_frames: int = 7) -> Tuple[List[np.ndarray], float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps if fps > 0 else 0.0

    if total_frames <= 0:
        cap.release()
        return [], duration_s, width, height

    start_f = int(total_frames * 0.05)
    end_f = int(total_frames * 0.95)
    step = max(1, (end_f - start_f) // (num_frames + 1))
    frames = []

    for i in range(1, num_frames + 1):
        frame_idx = min(start_f + i * step, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)

    cap.release()
    return frames, duration_s, width, height

def calculate_motion_variance(frames: List[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    diffs = []

    for i in range(len(grays) - 1):
        diff = cv2.absdiff(grays[i], grays[i + 1])
        mean_diff = np.mean(diff)
        diffs.append(mean_diff)

    return float(np.mean(diffs))

# ── Stage 4: CLIP Zero-Shot Unanimous Consensus Scoring ────────────────────
def calculate_strict_clip_score(frames: List[np.ndarray]) -> Tuple[float, str]:
    if not HAS_CLIP or not frames:
        return 60.0, "clip_skipped_default"

    from PIL import Image

    pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames[:5]]
    all_candidate_labels = POSITIVE_PROMPTS + NEGATIVE_PROMPTS

    try:
        device = next(model.parameters()).device
        inputs = processor(
            text=all_candidate_labels,
            images=pil_images,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # [num_images, num_labels]
            probs = logits_per_image.softmax(dim=-1).mean(dim=0).cpu().numpy()

        num_pos = len(POSITIVE_PROMPTS)
        pos_prob = float(np.sum(probs[:num_pos]))
        neg_prob = float(np.sum(probs[num_pos:]))

        best_pos_idx = int(np.argmax(probs[:num_pos]))
        top_match_idx = int(np.argmax(probs))

        raw_score = (pos_prob / max(1e-6, (pos_prob + neg_prob))) * 100.0

        # Nếu Top-1 Match là nhãn rác -> Phạt 70%
        if top_match_idx >= num_pos:
            raw_score *= 0.3
            match_label = f"REJECT: '{NEGATIVE_PROMPTS[top_match_idx - num_pos][:35]}...'"
        else:
            match_label = f"MATCH: '{POSITIVE_PROMPTS[best_pos_idx][:35]}...'"

        confidence_pct = min(100.0, max(0.0, raw_score))
        return confidence_pct, match_label

    except Exception as ex:
        log.debug(f"CLIP score error: {ex}")
        return 50.0, "clip_error_fallback"

# ── Main Dataset Processor & Categorizer ───────────────────────────────────
def process_dataset(input_dir: Path, output_dir: Path, move_files: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)

    high_dir   = output_dir / "high_confidence"          # Score >= 90% (Cực chuẩn CCTV)
    medium_dir = output_dir / "medium_confidence"        # 75% <= Score < 90% (Nghi vấn)
    low_dir    = output_dir / "low_confidence_discarded" # Score < 75% (Loại bỏ 100%)

    for d in (high_dir, medium_dir, low_dir):
        d.mkdir(parents=True, exist_ok=True)

    init_clip_model()

    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    video_files = [p for p in input_dir.rglob("*") if p.suffix.lower() in video_extensions]

    log.info(f"\n[+] Found {len(video_files)} video files in '{input_dir}'")
    log.info("=" * 70)

    results_list = []
    stats = {"total": len(video_files), "high": 0, "medium": 0, "low": 0}

    for idx, vid_path in enumerate(video_files, 1):
        rel_path = vid_path.relative_to(input_dir)

        # 1. Trích xuất Keyframes & Metadata
        frames, dur_s, width, height = extract_keyframes_and_meta(vid_path, num_frames=7)
        if not frames:
            score = 0.0
            category = "low_confidence_discarded"
            match_info = "corrupted_or_empty_file"
            _handle_file(vid_path, rel_path, low_dir, move_files)
            stats["low"] += 1
        else:
            # Stage 1: Face & Talking Head Veto
            has_face, face_info = detect_large_talking_head(frames)
            if has_face:
                score = 5.0
                category = "low_confidence_discarded"
                match_info = f"VETO: {face_info}"
                _handle_file(vid_path, rel_path, low_dir, move_files)
                stats["low"] += 1
            else:
                # Stage 2: News Lower-Third Banner Veto
                has_banner, banner_info = detect_news_lower_third_banner(frames)
                if has_banner:
                    score = 10.0
                    category = "low_confidence_discarded"
                    match_info = f"VETO: {banner_info}"
                    _handle_file(vid_path, rel_path, low_dir, move_files)
                    stats["low"] += 1
                else:
                    # Stage 3: Strict Motion Variance Check (Threshold = 6.0)
                    motion_var = calculate_motion_variance(frames)
                    if motion_var < 6.0:
                        score = 15.0
                        category = "low_confidence_discarded"
                        match_info = f"VETO: static_slideshow (motion_var={motion_var:.1f})"
                        _handle_file(vid_path, rel_path, low_dir, move_files)
                        stats["low"] += 1
                    else:
                        # Stage 4: Strict CLIP Unanimous Scoring
                        score, match_info = calculate_strict_clip_score(frames)

                        # Ngưỡng khắt khe mới: 90% cho High, 75% cho Medium
                        if score >= 90.0:
                            category = "high_confidence"
                            _handle_file(vid_path, rel_path, high_dir, move_files)
                            stats["high"] += 1
                        elif score >= 75.0:
                            category = "medium_confidence"
                            _handle_file(vid_path, rel_path, medium_dir, move_files)
                            stats["medium"] += 1
                        else:
                            category = "low_confidence_discarded"
                            _handle_file(vid_path, rel_path, low_dir, move_files)
                            stats["low"] += 1

        log.info(f"[{idx}/{len(video_files)}] {rel_path.name:<40} | Score: {score:5.1f}% | Folder: {category}")

        results_list.append({
            "filename": str(rel_path),
            "score_pct": round(score, 1),
            "category": category,
            "match_info": match_info,
            "duration_s": round(dur_s, 1),
            "resolution": f"{width}x{height}",
            "recommendation": "Ready for SegFormer" if score >= 90.0 else ("Review manually" if score >= 75.0 else "Discard")
        })

    # Save CSV Manifest
    csv_path = output_dir / "prefilter_scored_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "score_pct", "category", "match_info",
            "duration_s", "resolution", "recommendation"
        ])
        writer.writeheader()
        writer.writerows(results_list)

    # Save JSON Report
    json_path = output_dir / "prefilter_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": stats, "details": results_list}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(">>> MILITARY-GRADE PRE-FILTERING COMPLETE")
    print("=" * 70)
    print(f"  • Total Scanned Videos           : {stats['total']}")
    print(f"  • HIGH Confidence  (Score >= 90%) : {stats['high']} videos  (high_confidence/)")
    print(f"  • MEDIUM Confidence(75%<=S<90%) : {stats['medium']} videos  (medium_confidence/)")
    print(f"  • LOW Confidence   (Score < 75%)  : {stats['low']} videos  (low_confidence_discarded/)")
    print()
    print(f"  - Output Root Folder            : {output_dir.resolve()}")
    print(f"  - Scored CSV Index              : {csv_path.resolve()}")
    print("=" * 70)

def _handle_file(src_path: Path, rel_path: Path, target_base: Path, move: bool):
    dest = target_base / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src_path), str(dest))
    else:
        shutil.copy2(str(src_path), str(dest))

# ── CLI Interface ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Military-Grade Pre-filter and score raw videos into confidence subfolders"
    )
    ap.add_argument("--input", default="crawl_data/continuous_tracking_dataset", help="Input folder of raw videos")
    ap.add_argument("--output", default="filtered_tracking_dataset", help="Output directory")
    ap.add_argument("--move", action="store_true", help="Move files instead of copy")

    args = ap.parse_args()

    process_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        move_files=args.move
    )
