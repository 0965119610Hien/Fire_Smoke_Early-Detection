"""
utils/merge_round2_video_dataset.py
===================================
Tự động gom nhóm các frame mới được gán nhãn từ Roboflow (Frames_video.v1i):
1. Đọc ánh xạ v1_matched_mapping.csv để đưa từng frame + mask về đúng folder video của nó.
2. Chuẩn hóa tên frame theo quy chuẩn: frame_%06d.png và frame_%06d_mask.png.
3. Sinh mask all-zero (background = 0) cho các frame âm tính (chưa có lửa/khói) để giữ chuỗi video liên tục.
4. Gộp toàn bộ vào dataset_video_prepared (40 video cũ + 28 video mới = 68 video).
5. Cập nhật prepared_video_manifest.csv và xuất báo cáo thống kê chi tiết toàn bộ bộ dữ liệu.
"""

import os
import sys
import re
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    base_dir = Path(r"E:\NCKH\Code\Fire_Smoke_Early-Detection")
    data_dir = base_dir / "data" if (base_dir / "data").exists() else base_dir
    v1_train = data_dir / "Frames_video.v1i.png-mask-semantic" / "train"
    annot_dir = data_dir / "frames_for_annotation"
    out_dir = data_dir / "dataset_video_prepared"
    mapping_csv = data_dir / "data_splits" / "v1_matched_mapping.csv"
    manifest_csv = data_dir / "data_splits" / "prepared_video_manifest.csv"

    print("=" * 70)
    print("  GỘP DỮ LIỆU GÁN NHÃN MỚI VÀO DATASET_VIDEO_PREPARED (40 + 28 VIDEOS)")
    print("=" * 70)

    df_map = pd.read_csv(mapping_csv)
    n_vids = df_map['video_name'].nunique()
    print(f"[*] Đã tải ánh xạ: {len(df_map)} frame đã gán nhãn thuộc {n_vids} video mới.")

    labeled_lookup = {}
    for _, row in df_map.iterrows():
        v_name = row['video_name']
        o_name = row['orig_name']
        rf_img_p = v1_train / row['rf_img']
        rf_mask_p = v1_train / row['rf_mask']
        labeled_lookup[(v_name, o_name)] = (rf_img_p, rf_mask_p)

    new_manifest_records = []
    videos_processed = sorted(df_map['video_name'].unique())

    total_new_labeled = 0
    total_new_auto_bg = 0

    for idx, vid_name in enumerate(videos_processed, 1):
        vid_src_dir = annot_dir / vid_name
        vid_dst_dir = out_dir / vid_name
        vid_dst_dir.mkdir(parents=True, exist_ok=True)

        src_frames = sorted(os.listdir(vid_src_dir))
        vid_labeled_count = 0
        vid_bg_count = 0

        for f_name in src_frames:
            if not f_name.endswith(('.jpg', '.png')):
                continue

            idx_match = re.search(r'frame_(\d+)', f_name)
            frame_idx = int(idx_match.group(1)) if idx_match else 0
            std_frame_name = f"frame_{frame_idx:06d}.png"
            std_mask_name = f"frame_{frame_idx:06d}_mask.png"

            dst_img_p = vid_dst_dir / std_frame_name
            dst_mask_p = vid_dst_dir / std_mask_name

            src_img_p = vid_src_dir / f_name
            img = cv2.imread(str(src_img_p))
            if img is None:
                continue
            h, w = img.shape[:2]
            cv2.imwrite(str(dst_img_p), img)

            if (vid_name, f_name) in labeled_lookup:
                _, src_mask_p = labeled_lookup[(vid_name, f_name)]
                mask = cv2.imread(str(src_mask_p), cv2.IMREAD_UNCHANGED)
                if mask is None or mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) if mask is not None else np.zeros((h, w), dtype=np.uint8)
                cv2.imwrite(str(dst_mask_p), mask)
                mask_type = "labeled"
                total_new_labeled += 1
                vid_labeled_count += 1
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.imwrite(str(dst_mask_p), mask)
                mask_type = "auto_bg"
                total_new_auto_bg += 1
                vid_bg_count += 1

            fire_px = int(np.sum(mask == 1))
            smoke_px = int(np.sum(mask == 2))
            has_fire = 1 if fire_px > 0 else 0
            has_smoke = 1 if smoke_px > 0 else 0

            category = "both" if (has_fire and has_smoke) else ("fire_only" if has_fire else ("smoke_only" if has_smoke else "empty"))

            new_manifest_records.append({
                "video_name": vid_name,
                "frame_index": frame_idx,
                "frame_name": std_frame_name,
                "image_path": f"{vid_name}/{std_frame_name}",
                "mask_path": f"{vid_name}/{std_mask_name}",
                "mask_type": mask_type,
                "has_fire": has_fire,
                "has_smoke": has_smoke,
                "fire_pixels": fire_px,
                "smoke_pixels": smoke_px,
                "category": category
            })

        print(f"[{idx:02d}/{len(videos_processed)}] Video: {vid_name[:40]} -> {vid_labeled_count} labeled, {vid_bg_count} auto_bg")

    print("\n" + "=" * 70)
    print("  KẾT NỐI VÀ CẬP NHẬT BÁO CÁO TOÀN BỘ DATASET")
    print("=" * 70)

    # Đọc manifest cũ
    df_old = pd.read_csv(manifest_csv)
    df_new = pd.DataFrame(new_manifest_records)
    df_combined = pd.concat([df_old, df_new], ignore_index=True)

    # Lưu backup và ghi đè manifest
    shutil.copy2(manifest_csv, base_dir / "data_splits" / "prepared_video_manifest_backup_40vids.csv")
    df_combined.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    print(f"[*] Đã lưu manifest hợp nhất: {manifest_csv}")

    # Xuất thống kê chi tiết theo từng video
    video_summary = []
    for vid, group in df_combined.groupby('video_name'):
        video_summary.append({
            "video_name": vid,
            "total_frames": len(group),
            "labeled_frames": int((group['mask_type'] == 'labeled').sum()),
            "auto_bg_frames": int((group['mask_type'] == 'auto_bg').sum()),
            "fire_frames": int(group['has_fire'].sum()),
            "smoke_frames": int(group['has_smoke'].sum()),
            "both_frames": int(((group['has_fire'] == 1) & (group['has_smoke'] == 1)).sum()),
            "empty_frames": int(((group['has_fire'] == 0) & (group['has_smoke'] == 0)).sum()),
            "total_fire_pixels": int(group['fire_pixels'].sum()),
            "total_smoke_pixels": int(group['smoke_pixels'].sum()),
        })

    df_vid_summary = pd.DataFrame(video_summary)
    summary_csv = base_dir / "data_splits" / "video_dataset_summary_68vids.csv"
    df_vid_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("  THỐNG KÊ TỔNG THỂ BỘ DỮ LIỆU VIDEO HOÀN CHỈNH (68 VIDEOS)")
    print("=" * 70)
    print(f"  • Tổng số video              : {df_combined['video_name'].nunique()} video (40 cũ + 28 mới)")
    print(f"  • Tổng số frame có mask      : {len(df_combined)} frames")
    print(f"    - Frame có nhãn Roboflow   : {(df_combined['mask_type'] == 'labeled').sum()} frames")
    print(f"    - Frame âm tính (auto 0)   : {(df_combined['mask_type'] == 'auto_bg').sum()} frames")
    print("\nPhân bổ category toàn bộ dataset:")
    for cat, cnt in df_combined['category'].value_counts().items():
        pct = cnt / len(df_combined) * 100
        print(f"  - {cat:12s}: {cnt:5d} frames ({pct:5.1f}%)")

    print(f"\n[*] Đã lưu bảng tổng kết chi tiết từng video: {summary_csv}")

if __name__ == "__main__":
    main()
