"""
prepare_final_video_dataset.py
==============================
Chuẩn hóa và đóng gói dataset video hoàn chỉnh theo yêu cầu:
1. Loại bỏ 2 video: 
   - 3dd42c9c_pexels_heavy_smoke_35069292_pure
   - 14196327_1080_1920_24fps_pure
2. Video '2016 April - Fire Investigation 1B - Milpitas CA_pure_chunk_03': 
   - Bỏ frame_000000.png, giữ lại 30 frame (frame_000001 -> frame_000030)
3. Các video còn lại thiếu frame không có lửa/khói:
   - Tự động sinh mask all-zeros (Background = 0) cho các frame âm tính này
4. Lưu toàn bộ dataset chuẩn vào: 'dataset_video_prepared/<video_name>/'
5. Xuất manifest và thống kê phân bổ dữ liệu.
"""

import os
import sys
import shutil
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

EXCLUDE_VIDEOS = {
    "3dd42c9c_pexels_heavy_smoke_35069292_pure",
    "14196327_1080_1920_24fps_pure",
}

SPECIAL_DROP_FRAMES = {
    "2016 April - Fire Investigation 1B - Milpitas CA_pure_chunk_03": {"frame_000000.png"}
}

def get_thumb_and_info(img_path):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        thumb = cv2.resize(img, (16, 16)).flatten()
        return (h, w, thumb, img_path)
    except Exception:
        return None

def main():
    base_dir = Path(r"E:\NCKH\Code\Fire_Smoke_Early-Detection")
    v3i_dir = base_dir / "fire_smoke_video.v3i.png-mask-semantic" / "train"
    out_dir = base_dir / "dataset_video_prepared"
    
    print("=" * 70, flush=True)
    print("  XÂY DỰNG BỘ DỮ LIỆU VIDEO HOÀN CHỈNH (DATASET_VIDEO_PREPARED)", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Quét nhãn từ v3i
    v3i_imgs = sorted([f for f in v3i_dir.glob("*.jpg")] + 
                      [f for f in v3i_dir.glob("*.png") if "_mask" not in f.name])
    print(f"[*] Tổng số frame nhãn trong v3i: {len(v3i_imgs)}", flush=True)
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        v3i_features = [res for res in executor.map(get_thumb_and_info, v3i_imgs) if res is not None]
        
    # 2. Quét video gốc
    candidate_roots = [
        base_dir / "test_annotation_frames",
        base_dir / "annotation_frames",
        base_dir / "final_video_frames",
    ]
    
    all_cand_paths = []
    video_to_source_frames = defaultdict(dict)
    
    for root in candidate_roots:
        if not root.exists():
            continue
        for vid_folder in root.iterdir():
            if not vid_folder.is_dir() or vid_folder.name in EXCLUDE_VIDEOS:
                continue
            vid_name = vid_folder.name
            for f in vid_folder.glob("*.png"):
                if "_mask" in f.name or "_aug_" in f.name:
                    continue
                # Kiểm tra drop frame đặc biệt
                if vid_name in SPECIAL_DROP_FRAMES and f.name in SPECIAL_DROP_FRAMES[vid_name]:
                    continue
                video_to_source_frames[vid_name][f.name] = f
                all_cand_paths.append(f)
                
    indexed_candidates = defaultdict(list)
    with ThreadPoolExecutor(max_workers=16) as executor:
        for res in executor.map(get_thumb_and_info, all_cand_paths):
            if res is not None:
                h, w, thumb, p = res
                indexed_candidates[(h, w)].append((thumb, p.parent.name, p.name, p))
                
    # 3. Match v3i frames -> (video_name, frame_name, mask_path)
    rf_to_video_map = {} # (vid_name, frame_name) -> (v3i_img_path, v3i_mask_path)
    
    for h, w, thumb, rf_p in v3i_features:
        candidates = indexed_candidates.get((h, w), [])
        best_diff = 1e9
        best_match = None
        for c_thumb, vid_name, frame_name, p in candidates:
            diff = np.mean(np.abs(thumb.astype(float) - c_thumb.astype(float)))
            if diff < best_diff:
                best_diff = diff
                best_match = (vid_name, frame_name)
                if diff < 1.0:
                    break
        if best_match and best_diff < 12.0:
            vid_name, frame_name = best_match
            if vid_name not in EXCLUDE_VIDEOS:
                mask_p = rf_p.parent / f"{rf_p.stem}_mask.png"
                if mask_p.exists():
                    rf_to_video_map[(vid_name, frame_name)] = (rf_p, mask_p)
                    
    print(f"[*] Đã ánh xạ thành công: {len(rf_to_video_map)} nhãn hợp lệ vào danh sách video.", flush=True)
    
    # 4. Tạo thư mục output và copy/tạo mask
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_records = []
    total_copied_labeled = 0
    total_generated_bg = 0
    
    # Lấy danh sách video có mặt trong tập nhãn
    active_videos = sorted(set(vid for vid, _ in rf_to_video_map.keys()))
    print(f"[*] Số lượng video hợp lệ chuẩn bị đóng gói: {len(active_videos)} video", flush=True)
    
    for vid_name in active_videos:
        vid_out = out_dir / vid_name
        vid_out.mkdir(parents=True, exist_ok=True)
        
        all_frames_in_vid = sorted(video_to_source_frames[vid_name].keys())
        
        for frame_idx, frame_name in enumerate(all_frames_in_vid):
            src_img_p = video_to_source_frames[vid_name][frame_name]
            dst_img_p = vid_out / frame_name
            dst_mask_p = vid_out / f"{Path(frame_name).stem}_mask.png"
            
            # Copy ảnh gốc
            shutil.copy2(src_img_p, dst_img_p)
            
            # Kiểm tra xem frame có nhãn trong v3i không
            if (vid_name, frame_name) in rf_to_video_map:
                _, src_mask_p = rf_to_video_map[(vid_name, frame_name)]
                shutil.copy2(src_mask_p, dst_mask_p)
                mask_type = "labeled"
                total_copied_labeled += 1
            else:
                # Frame âm tính (Background) -> Tạo mask 0
                img = cv2.imread(str(src_img_p))
                h, w = img.shape[:2]
                bg_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.imwrite(str(dst_mask_p), bg_mask)
                mask_type = "auto_bg"
                total_generated_bg += 1
                
            # Đọc mask để tính stats
            mask = cv2.imread(str(dst_mask_p), cv2.IMREAD_UNCHANGED)
            fire_px = int(np.sum(mask == 1))
            smoke_px = int(np.sum(mask == 2))
            has_fire = 1 if fire_px > 0 else 0
            has_smoke = 1 if smoke_px > 0 else 0
            
            category = "both" if (has_fire and has_smoke) else ("fire_only" if has_fire else ("smoke_only" if has_smoke else "empty"))
            
            manifest_records.append({
                "video_name": vid_name,
                "frame_index": frame_idx,
                "frame_name": frame_name,
                "image_path": f"{vid_name}/{frame_name}",
                "mask_path": f"{vid_name}/{dst_mask_p.name}",
                "mask_type": mask_type,
                "has_fire": has_fire,
                "has_smoke": has_smoke,
                "fire_pixels": fire_px,
                "smoke_pixels": smoke_px,
                "category": category
            })
            
    df_manifest = pd.DataFrame(manifest_records)
    manifest_csv = base_dir / "data_splits" / "prepared_video_manifest.csv"
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    df_manifest.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 70, flush=True)
    print("  TỔNG KẾT ĐÓNG GÓI BỘ DỮ LIỆU VIDEO HOÀN TẤT", flush=True)
    print("=" * 70, flush=True)
    print(f"  • Tổng số video hợp lệ           : {len(active_videos)} video", flush=True)
    print(f"  • Tổng số frame hoàn chỉnh       : {len(df_manifest)} frames (100% có mask)", flush=True)
    print(f"  • Frame có nhãn gốc (Roboflow)   : {total_copied_labeled} frames", flush=True)
    print(f"  • Frame âm tính tự sinh mask 0   : {total_generated_bg} frames", flush=True)
    print(f"  • Thư mục dữ liệu chuẩn          : {out_dir}", flush=True)
    print(f"  • File Manifest chi tiết         : {manifest_csv}", flush=True)
    
    print("\nPhân bổ category các frame:", flush=True)
    for cat, count in df_manifest["category"].value_counts().items():
        print(f"  - {cat:12s}: {count} frames ({count/len(df_manifest)*100:.1f}%)", flush=True)
        
    return df_manifest

if __name__ == "__main__":
    main()
