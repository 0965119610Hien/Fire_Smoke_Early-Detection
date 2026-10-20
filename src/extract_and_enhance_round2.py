"""
Script to extract and preprocess frames for curated videos into dataset_annotation_round2.
- Sampling rate: 1 FPS (every 1.0s) matching TimeSformer 8-frame sequence window.
- Letterbox resize: 512x512 preserving aspect ratio with black padding.
- Enhanced Proxy Frames: CLAHE in LAB color space (clipLimit=2.0, tileGridSize=(8,8)).
- Quality Assessment: Laplacian variance & mean brightness.
- Manifest update: appends to dataset_annotation_round2/extraction_manifest.csv.
"""
import os
import sys
import json
import time
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROUND2_DIR = Path(r"e:\NCKH\Code\Fire_Smoke_Early-Detection\dataset_annotation_round2")
VIDEOS_DIR = ROUND2_DIR / "videos"
RAW_DIR = ROUND2_DIR / "raw_frames"
ENH_DIR = ROUND2_DIR / "enhanced_proxy_frames"
MANIFEST_CSV = ROUND2_DIR / "extraction_manifest.csv"
STATS_JSON = Path(r"e:\NCKH\Code\Fire_Smoke_Early-Detection\crawled_early_fire_transition\curated_continuous_incidents\numbered_videos_stats.json")

def letterbox_resize(img, target_size=(512, 512)):
    h, w = img.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    
    top = (th - nh) // 2
    bottom = th - nh - top
    left = (tw - nw) // 2
    right = tw - nw - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded

def apply_clahe_lab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def load_category_map():
    cat_map = {}
    if STATS_JSON.exists():
        with open(STATS_JSON, 'r', encoding='utf-8') as f:
            stats = json.load(f)
            for s in stats:
                fn = s['filename']
                cat = s['category']
                cat_map[fn] = cat
                base_stem = Path(fn).stem
                cat_map[base_stem] = cat
    return cat_map

def run_extraction_and_enhancement():
    print("=" * 70)
    print(">> BAT DAU TRICH XUAT FRAME & TIEN XU LY ENHANCED PROXY CHO ROUND 2")
    print(f"   - Thư mục video nguồn: {VIDEOS_DIR}")
    print(f"   - Tần số trích xuất: 1 FPS (1 giây / 1 frame)")
    print(f"   - Kích thước chuẩn hóa: 512 x 512 (Letterbox)")
    print(f"   - Tăng cường tương phản: CLAHE LAB (clipLimit=2.0, tileGridSize=(8,8))")
    print("=" * 70)

    # 1. Backup manifest
    if MANIFEST_CSV.exists():
        backup_manifest = ROUND2_DIR / "extraction_manifest_backup.csv"
        shutil.copy2(str(MANIFEST_CSV), str(backup_manifest))
        print(f"[INFO] Đã sao lưu manifest hiện tại vào: {backup_manifest.name}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ENH_DIR.mkdir(parents=True, exist_ok=True)

    cat_map = load_category_map()
    video_files = sorted(list(VIDEOS_DIR.glob("*.mp4")))
    total_videos = len(video_files)
    print(f"[INFO] Tìm thấy {total_videos} video trong {VIDEOS_DIR.name}/")

    manifest_records = []
    total_frames_extracted = 0
    total_pass_frames = 0
    t0_all = time.time()

    for v_idx, v_path in enumerate(video_files, 1):
        v_stem = v_path.stem
        # Determine category
        cat = cat_map.get(v_path.name, cat_map.get(v_stem, "continuous_fire"))
        if "battery" in v_stem: cat = "battery_thermal_runaway"
        elif "electrical" in v_stem: cat = "electrical_ignition"
        elif "kitchen" in v_stem: cat = "kitchen_stove"
        elif "lab" in v_stem: cat = "lab_experiments"
        elif "smoldering" in v_stem: cat = "smoldering_transition"
        elif "vehicle" in v_stem: cat = "vehicle_engine_fire"
        elif "wildfire" in v_stem: cat = "wildfire_forest_ignition"

        v_raw_dir = RAW_DIR / v_stem
        v_enh_dir = ENH_DIR / v_stem
        v_raw_dir.mkdir(parents=True, exist_ok=True)
        v_enh_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            print(f"[{v_idx:2d}/{total_videos}] LOI: Khong mo duoc video {v_path.name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps if fps > 0 else 0.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_step = max(1, int(round(fps))) # 1 FPS
        frame_idx = 0
        extracted_for_vid = 0
        pass_for_vid = 0

        # Sample at 1.0s interval
        sec_idx = 0
        while True:
            target_frame_num = int(round(sec_idx * fps))
            if target_frame_num >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
            ret, frame = cap.read()
            if not ret or frame is None:
                sec_idx += 1
                continue

            timestamp_sec = round(sec_idx * 1.0, 1)
            block_id = int(timestamp_sec // 8) # 8-second TimeSformer block
            frame_filename = f"{v_stem}_frame_{frame_idx:06d}.png"

            # 1. Letterbox 512x512 Raw
            raw_512 = letterbox_resize(frame, (512, 512))

            # 2. CLAHE 512x512 Enhanced Proxy
            enh_512 = apply_clahe_lab(raw_512)

            # 3. Quality Metrics
            gray = cv2.cvtColor(raw_512, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(round(np.mean(gray), 2))
            laplacian_var = float(round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2))
            is_quality_pass = bool(laplacian_var >= 45.0 and 20.0 <= mean_brightness <= 245.0)

            # Save frames
            raw_out_path = v_raw_dir / frame_filename
            enh_out_path = v_enh_dir / frame_filename
            cv2.imwrite(str(raw_out_path), raw_512, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(str(enh_out_path), enh_512, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            # Record
            raw_rel = f"raw_frames\\{v_stem}\\{frame_filename}"
            enh_rel = f"enhanced_proxy_frames\\{v_stem}\\{frame_filename}"

            manifest_records.append({
                "video_name": v_stem,
                "category": cat,
                "frame_filename": frame_filename,
                "frame_index": frame_idx,
                "video_frame_orig": target_frame_num,
                "timestamp_sec": timestamp_sec,
                "block_id": block_id,
                "raw_rel_path": raw_rel,
                "enh_rel_path": enh_rel,
                "orig_width": orig_w,
                "orig_height": orig_h,
                "mean_brightness": mean_brightness,
                "laplacian_var": laplacian_var,
                "is_quality_pass": is_quality_pass
            })

            frame_idx += 1
            extracted_for_vid += 1
            if is_quality_pass:
                pass_for_vid += 1
            sec_idx += 1

        cap.release()
        total_frames_extracted += extracted_for_vid
        total_pass_frames += pass_for_vid
        print(f"[{v_idx:2d}/{total_videos}] {v_stem[:45]:45s} | {duration_s:5.1f}s -> {extracted_for_vid:3d} frames ({pass_for_vid:3d} pass)")

    # 4. Append to manifest
    new_df = pd.DataFrame(manifest_records)
    if MANIFEST_CSV.exists():
        old_df = pd.read_csv(MANIFEST_CSV)
        # Avoid duplicate video_name
        existing_vids = set(old_df['video_name'].unique())
        new_records = new_df[~new_df['video_name'].isin(existing_vids)]
        combined_df = pd.concat([old_df, new_records], ignore_index=True)
        combined_df.to_csv(MANIFEST_CSV, index=False, encoding='utf-8')
        print(f"\n[INFO] Đã gộp vào {MANIFEST_CSV.name}: {len(old_df)} frames cũ + {len(new_records)} frames mới = {len(combined_df)} tổng cộng.")
    else:
        new_df.to_csv(MANIFEST_CSV, index=False, encoding='utf-8')
        print(f"\n[INFO] Đã tạo mới {MANIFEST_CSV.name} với {len(new_df)} frames.")

    elapsed = time.time() - t0_all
    print("=" * 70)
    print("HOAN THANH TRICH XUAT & TIEN XU LY ENHANCED PROXY CHO 62 VIDEO:")
    print(f"   - Tổng số video đã xử lý: {total_videos}")
    print(f"   - Tổng số frame trích xuất: {total_frames_extracted} frames")
    print(f"   - Số frame đạt chuẩn chất lượng: {total_pass_frames} frames ({total_pass_frames/max(1,total_frames_extracted)*100:.1f}%)")
    print(f"   - Thời gian xử lý: {elapsed:.1f}s ({elapsed/60:.2f} phút)")
    print(f"   - Thư mục Raw: {RAW_DIR}")
    print(f"   - Thư mục Enhanced: {ENH_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    run_extraction_and_enhancement()
