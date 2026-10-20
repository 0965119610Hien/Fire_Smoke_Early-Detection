"""
src/create_split_manifests_and_clips.py
=======================================
1. Xuất 3 file manifest riêng biệt cho Train, Val, Test từ prepared_video_manifest_with_splits.csv.
2. Trích xuất chỉ mục Spatio-Temporal Sliding Window Clips (T=8 frames, stride S=2 cho Train, S=4 cho Val/Test).
   Thêm 1 clip neo ở cuối video khi stride không chia hết, để MỌI frame đều nằm trong ít nhất 1 clip.
3. Xuất train_clips.csv, val_clips.csv, test_clips.csv vào data/data_splits/.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
SPLITS_DIR = DATA_DIR / "data_splits"

MANIFEST_WITH_SPLITS = SPLITS_DIR / "prepared_video_manifest_with_splits.csv"

def generate_clips_for_video(vid_df, clip_len=8, stride=2, cover_tail=True):
    """
    Sinh các sliding window clip từ danh sách frame của 1 video.
    Nếu video ngắn hơn clip_len, lặp lại các frame cuối để đủ clip_len.
    cover_tail: thêm clip [n-clip_len, n-1] nếu các cửa sổ trượt bỏ sót đuôi video.
    """
    frames = vid_df.sort_values(by='frame_index').to_dict('records')
    n = len(frames)
    clips = []
    
    if n == 0:
        return clips
        
    if n < clip_len:
        # Replicate padding for short videos
        padded_frames = frames + [frames[-1]] * (clip_len - n)
        has_f = int(any(f['has_fire'] == 1 for f in padded_frames))
        has_s = int(any(f['has_smoke'] == 1 for f in padded_frames))
        cat = 'both' if (has_f and has_s) else ('fire_only' if has_f else ('smoke_only' if has_s else 'empty'))
        clips.append({
            'video_name': frames[0]['video_name'],
            'start_idx': 0,
            'end_idx': n - 1,
            'clip_length': clip_len,
            'frame_names': ";".join(f['frame_name'] for f in padded_frames),
            'image_paths': ";".join(f['image_path'] for f in padded_frames),
            'mask_paths': ";".join(f['mask_path'] for f in padded_frames),
            'has_fire': has_f,
            'has_smoke': has_s,
            'category': cat,
            'is_padded': True
        })
        return clips
        
    # Sliding window
    starts = list(range(0, n - clip_len + 1, stride))
    if cover_tail and starts[-1] != n - clip_len:
        starts.append(n - clip_len)
    for s_idx in starts:
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
    print("  KHỞI TẠO BỘ CHỈ MỤC TRAIN / VAL / TEST VÀ SLIDING WINDOW CLIPS")
    print("=" * 70)

    df = pd.read_csv(MANIFEST_WITH_SPLITS)
    print(f"[*] Đã tải manifest: {len(df)} frames từ {df['video_name'].nunique()} videos.")

    # 1. Xuất file manifest theo từng split
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split].copy()
        out_p = SPLITS_DIR / f"{split}_manifest.csv"
        split_df.to_csv(out_p, index=False, encoding="utf-8-sig")
        print(f"  [XUẤT MANIFEST] {out_p.name:<20} | {split_df['video_name'].nunique():>2} videos | {len(split_df):>4} frames")

    # 2. Sinh các Sliding Window Clips (T=8)
    # Stride: Train=2, Val=4, Test=4
    strides = {'train': 2, 'val': 4, 'test': 4}
    
    print("\n--- 2. TẠO SLIDING WINDOW CLIPS (T=8 frames) ---")
    clip_stats = {}
    
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        stride = strides[split]
        all_clips = []
        
        for vid, vid_group in split_df.groupby('video_name'):
            v_clips = generate_clips_for_video(vid_group, clip_len=8, stride=stride)
            all_clips.extend(v_clips)
            
        df_clips = pd.DataFrame(all_clips)
        out_clip_p = SPLITS_DIR / f"{split}_clips.csv"
        df_clips.to_csv(out_clip_p, index=False, encoding="utf-8-sig")
        
        clip_stats[split] = {
            'total_clips': len(df_clips),
            'fire_only': int((df_clips['category'] == 'fire_only').sum()),
            'smoke_only': int((df_clips['category'] == 'smoke_only').sum()),
            'both': int((df_clips['category'] == 'both').sum()),
            'empty': int((df_clips['category'] == 'empty').sum()),
        }
        
        print(f"  [XUẤT CLIPS] {out_clip_p.name:<18} (Stride={stride}) | {len(df_clips):>4} clips "
              f"(fire={clip_stats[split]['fire_only']}, smoke={clip_stats[split]['smoke_only']}, both={clip_stats[split]['both']}, empty={clip_stats[split]['empty']})")

    # Lưu json summary
    summary_p = SPLITS_DIR / "dataset_summary.json"
    with open(summary_p, "w", encoding="utf-8") as f:
        json.dump(clip_stats, f, indent=2)
    print(f"\n[*] Đã lưu tóm tắt thống kê: {summary_p.name}")
    print("=" * 70)

if __name__ == "__main__":
    main()
