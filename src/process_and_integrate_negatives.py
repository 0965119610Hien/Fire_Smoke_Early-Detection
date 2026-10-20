"""
src/process_and_integrate_negatives.py
======================================
Tự động hóa hoàn toàn quy trình xử lý dữ liệu âm tính (Normal & Hard Negatives):
1. Quét toàn bộ video âm tính (24 hard negatives + 8 normal ambient CCTV).
2. Trích xuất 16 frames liên tục/chuẩn hóa cho mỗi video vào dataset_video_prepared/<video_name>/
3. Sinh mask toàn số 0 (frame_%06d_mask.png) với giá trị 0 = background.
4. Cập nhật prepared_video_manifest_with_splits.csv (chia Train 70%, Val 15%, Test 15%).
5. Sinh lại toàn bộ các bộ sliding window clips: train_clips.csv, val_clips.csv, test_clips.csv.
6. Đồng bộ mẫu âm tính vào static_v4i_train_manifest.csv và static_v4i_val_manifest.csv cho Stage 1 SegFormer.
"""

import os
import sys
import glob
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE = Path(r"E:\NCKH\Code\Fire_Smoke_Early-Detection")
DATA_DIR = BASE / "data"
PREPARED_DIR = DATA_DIR / "dataset_video_prepared"
SPLITS_DIR = DATA_DIR / "data_splits"

HN_DIR = DATA_DIR / "crawled_early_fire_transition" / "by_category" / "09_hard_negatives" / "hard_negatives"
NORM_DIR = DATA_DIR / "crawled_normal_ambient"

V4I_TRAIN_DIR = DATA_DIR / "Fire_Smoke.v4i.png-mask-semantic" / "train"

MANIFEST_WITH_SPLITS = SPLITS_DIR / "prepared_video_manifest_with_splits.csv"
STATIC_TRAIN_CSV = SPLITS_DIR / "static_v4i_train_manifest.csv"
STATIC_VAL_CSV = SPLITS_DIR / "static_v4i_val_manifest.csv"

FRAMES_PER_VIDEO = 16  # Chuẩn 16 frames để tạo đúng 2 clips hoàn chỉnh (T=8) không padding

def sanitize_name(name):
    clean = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    return clean[:45]

def extract_frames_and_masks(video_path, out_folder, num_frames=16):
    out_folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Không mở được video: {video_path.name}")
        return []
    
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_f <= 0 or fps <= 0:
        cap.release()
        return []
    
    # Chọn khoảng thời gian ổn định (bỏ 5% đầu và 5% cuối)
    start_f = int(total_f * 0.05)
    end_f = int(total_f * 0.95)
    if end_f - start_f < num_frames:
        start_f = 0
        end_f = total_f
    
    # Tính bước nhảy khung hình
    step = max(1, (end_f - start_f) // num_frames)
    indices = [min(total_f - 1, start_f + i * step) for i in range(num_frames)]
    
    records = []
    for frame_idx, f_no in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_no)
        ret, frame = cap.read()
        if not ret or frame is None:
            # Fallback nếu đọc lỗi: frame màu đen
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
        h, w = frame.shape[:2]
        # Giới hạn kích thước tối đa 720p để tiết kiệm bộ nhớ
        if max(h, w) > 720:
            scale = 720.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
            
        f_name = f"frame_{frame_idx:06d}.png"
        m_name = f"frame_{frame_idx:06d}_mask.png"
        
        img_out = out_folder / f_name
        mask_out = out_folder / m_name
        
        cv2.imwrite(str(img_out), frame)
        # Tạo mask rỗng toàn bộ pixel = 0 (Background)
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(str(mask_out), empty_mask)
        
        records.append({
            'frame_index': frame_idx,
            'frame_name': f_name,
            'image_path': f"{out_folder.name}/{f_name}",
            'mask_path': f"{out_folder.name}/{m_name}",
            'width': w,
            'height': h,
            'mask_type': 'empty',
            'has_fire': 0,
            'has_smoke': 0,
            'fire_pixels': 0,
            'smoke_pixels': 0,
            'category': 'empty'
        })
        
    cap.release()
    return records

def generate_clips(frames_df, clip_len=8, stride=2):
    frames = frames_df.sort_values(by='frame_index').to_dict('records')
    n = len(frames)
    clips = []
    if n == 0:
        return clips
    for s_idx in range(0, n - clip_len + 1, stride):
        sub = frames[s_idx : s_idx + clip_len]
        has_f = int(any(f['has_fire'] == 1 for f in sub))
        has_s = int(any(f['has_smoke'] == 1 for f in sub))
        cat = 'both' if (has_f and has_s) else ('fire_only' if has_f else ('smoke_only' if has_s else 'empty'))
        clips.append({
            'video_name': frames[0]['video_name'],
            'start_idx': s_idx,
            'end_idx': s_idx + clip_len - 1,
            'clip_length': clip_len,
            'frame_names': ";".join(f['frame_name'] for f in sub),
            'image_paths': ";".join(f['image_path'] for f in sub),
            'mask_paths': ";".join(f['mask_path'] for f in sub),
            'has_fire': has_f,
            'has_smoke': has_s,
            'category': cat,
            'is_padded': False
        })
    return clips

def main():
    print("=" * 70)
    print("🚀 BẮT ĐẦU TỔNG HỢP, TRÍCH XUẤT VÀ TÍCH HỢP DỮ LIỆU ÂM TÍNH (NEGATIVE)")
    print("=" * 70)
    
    # 1. Thu thập danh sách video âm tính
    hn_videos = sorted(list(HN_DIR.glob("*.mp4")))
    norm_videos = sorted([p for p in NORM_DIR.glob("*.mp4") if not p.name.endswith(('.part', '.ytdl'))])
    
    all_neg_videos = hn_videos + norm_videos
    print(f"[*] Tìm thấy tổng cộng {len(all_neg_videos)} video âm tính:")
    print(f"    - Hard Negatives (hơi nước, sương mù, xe cộ, hàn xì): {len(hn_videos)} videos")
    print(f"    - Normal Ambient (văn phòng, ngã tư, kho, đường phố, cây): {len(norm_videos)} videos")
    
    # 2. Phân chia Train (70%), Val (15%), Test (15%)
    random.seed(42)
    shuffled_vids = list(all_neg_videos)
    random.shuffle(shuffled_vids)
    
    n_total = len(shuffled_vids)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_vids = set(shuffled_vids[:n_train])
    val_vids = set(shuffled_vids[n_train:n_train + n_val])
    test_vids = set(shuffled_vids[n_train + n_val:])
    
    print(f"[*] Phân bổ Video âm tính: Train={len(train_vids)}, Val={len(val_vids)}, Test={len(test_vids)}")
    
    # 3. Trích xuất frames và tạo mask trống
    new_frame_records = []
    for idx, v_path in enumerate(shuffled_vids, 1):
        stem = sanitize_name(v_path.stem)
        v_folder_name = f"neg_{stem}"
        out_folder = PREPARED_DIR / v_folder_name
        
        split = "train" if v_path in train_vids else ("val" if v_path in val_vids else "test")
        
        recs = extract_frames_and_masks(v_path, out_folder, num_frames=FRAMES_PER_VIDEO)
        if len(recs) == 0:
            continue
            
        for r in recs:
            r['video_name'] = v_folder_name
            r['split'] = split
            new_frame_records.append(r)
            
        print(f"  [{idx:02d}/{n_total}] ✅ {v_folder_name:<40} -> {len(recs)} frames ({split})")
        
    print(f"\n[+] Tổng số frames âm tính trích xuất thành công: {len(new_frame_records)} frames.")
    
    # 4. Cập nhật prepared_video_manifest_with_splits.csv
    df_existing = pd.read_csv(MANIFEST_WITH_SPLITS)
    # Loại bỏ các records neg cũ nếu đã có
    df_existing = df_existing[~df_existing['video_name'].str.startswith('neg_')]
    
    df_new = pd.DataFrame(new_frame_records)
    # Khớp chính xác các cột của manifest
    cols = ['video_name','frame_index','frame_name','image_path','mask_path','mask_type',
            'has_fire','has_smoke','fire_pixels','smoke_pixels','category','split']
    df_new_matched = df_new[cols]
    
    df_updated = pd.concat([df_existing, df_new_matched], ignore_index=True)
    df_updated.to_csv(MANIFEST_WITH_SPLITS, index=False, encoding="utf-8-sig")
    print(f"✅ Đã cập nhật {MANIFEST_WITH_SPLITS.name}: {len(df_updated)} frames ({df_updated['video_name'].nunique()} videos).")
    print(f"    - Phân bố categories:\n{df_updated['category'].value_counts().to_string()}")
    
    # 5. Xuất train_manifest.csv, val_manifest.csv, test_manifest.csv
    for sp in ['train', 'val', 'test']:
        sp_df = df_updated[df_updated['split'] == sp]
        sp_df.to_csv(SPLITS_DIR / f"{sp}_manifest.csv", index=False, encoding="utf-8-sig")
        
    # 6. Tạo lại Spatio-Temporal Sliding Window Clips (T=8)
    print("\n--- TẠO LẠI SLIDING WINDOW CLIPS CHO STAGE 2 ---")
    strides = {'train': 2, 'val': 4, 'test': 4}
    for sp in ['train', 'val', 'test']:
        sp_df = df_updated[df_updated['split'] == sp]
        sp_clips = []
        for v_name, v_group in sp_df.groupby('video_name'):
            # Với negative clips tập train, dùng stride=4 để tránh áp đảo positive
            s = 4 if (sp == 'train' and v_name.startswith('neg_')) else strides[sp]
            v_clips = generate_clips(v_group, clip_len=8, stride=s)
            sp_clips.extend(v_clips)
            
        clips_df = pd.DataFrame(sp_clips)
        clips_p = SPLITS_DIR / f"{sp}_clips.csv"
        clips_df.to_csv(clips_p, index=False, encoding="utf-8-sig")
        print(f"  [XUẤT CLIPS] {clips_p.name:<18} -> {len(clips_df):>4} clips (Categories: {dict(clips_df['category'].value_counts())})")
        
    # 7. Đồng bộ mẫu âm tính vào Static Manifest cho Stage 1 SegFormer
    print("\n--- ĐỒNG BỘ MẪU ÂM TÍNH VÀO STAGE 1 (SEGFORMER) ---")
    if STATIC_TRAIN_CSV.exists() and STATIC_VAL_CSV.exists() and V4I_TRAIN_DIR.exists():
        df_static_train = pd.read_csv(STATIC_TRAIN_CSV)
        df_static_val = pd.read_csv(STATIC_VAL_CSV)
        
        # Loại bỏ các record neg cũ
        df_static_train = df_static_train[~df_static_train['filename'].str.startswith('neg_')]
        df_static_val = df_static_val[~df_static_val['filename'].str.startswith('neg_')]
        
        train_neg_frames = [r for r in new_frame_records if r['split'] == 'train']
        val_neg_frames = [r for r in new_frame_records if r['split'] == 'val']
        
        # Chọn 150 frames train và 30 frames val
        sampled_train_neg = train_neg_frames[:150]
        sampled_val_neg = val_neg_frames[:30]
        
        new_static_train = []
        for r in sampled_train_neg:
            src_img = PREPARED_DIR / r['image_path']
            src_mask = PREPARED_DIR / r['mask_path']
            
            dst_img_name = f"neg_{r['video_name']}_{r['frame_name']}"
            dst_mask_name = f"neg_{r['video_name']}_{r['frame_name'].replace('.png', '_mask.png')}"
            
            shutil.copy2(src_img, V4I_TRAIN_DIR / dst_img_name)
            shutil.copy2(src_mask, V4I_TRAIN_DIR / dst_mask_name)
            
            new_static_train.append({
                'filename': dst_img_name,
                'mask_filename': dst_mask_name,
                'width': r['width'],
                'height': r['height'],
                'total_pixels': r['width'] * r['height'],
                'fire_pixels': 0,
                'smoke_pixels': 0,
                'fire_ratio': 0.0,
                'smoke_ratio': 0.0,
                'has_fire': 0,
                'has_smoke': 0,
                'category': 'empty',
                'split': 'train'
            })
            
        new_static_val = []
        for r in sampled_val_neg:
            src_img = PREPARED_DIR / r['image_path']
            src_mask = PREPARED_DIR / r['mask_path']
            
            dst_img_name = f"neg_{r['video_name']}_{r['frame_name']}"
            dst_mask_name = f"neg_{r['video_name']}_{r['frame_name'].replace('.png', '_mask.png')}"
            
            shutil.copy2(src_img, V4I_TRAIN_DIR / dst_img_name)
            shutil.copy2(src_mask, V4I_TRAIN_DIR / dst_mask_name)
            
            new_static_val.append({
                'filename': dst_img_name,
                'mask_filename': dst_mask_name,
                'width': r['width'],
                'height': r['height'],
                'total_pixels': r['width'] * r['height'],
                'fire_pixels': 0,
                'smoke_pixels': 0,
                'fire_ratio': 0.0,
                'smoke_ratio': 0.0,
                'has_fire': 0,
                'has_smoke': 0,
                'category': 'empty',
                'split': 'val'
            })
            
        df_static_train_updated = pd.concat([df_static_train, pd.DataFrame(new_static_train)], ignore_index=True)
        df_static_val_updated = pd.concat([df_static_val, pd.DataFrame(new_static_val)], ignore_index=True)
        
        df_static_train_updated.to_csv(STATIC_TRAIN_CSV, index=False, encoding="utf-8-sig")
        df_static_val_updated.to_csv(STATIC_VAL_CSV, index=False, encoding="utf-8-sig")
        print(f"✅ Đã thêm {len(new_static_train)} ảnh âm tính vào Stage 1 Train Manifest (Tổng: {len(df_static_train_updated)})")
        print(f"✅ Đã thêm {len(new_static_val)} ảnh âm tính vào Stage 1 Val Manifest (Tổng: {len(df_static_val_updated)})")
    else:
        print("⚠️ Không tìm thấy thư mục tĩnh V4i để copy ảnh âm tính Stage 1.")
        
    print("\n🎉 HOÀN THÀNH TOÀN BỘ QUY TRÌNH TÍCH HỢP DỮ LIỆU ÂM TÍNH!")

if __name__ == "__main__":
    main()
