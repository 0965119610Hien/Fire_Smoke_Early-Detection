"""
check_and_export_missing_frames.py
==================================
Thống kê đối chiếu nhãn từ fire_smoke_video.v2i.png-mask-semantic với video gốc:
1. Ánh xạ 668 frame đã gán nhãn về đúng Video ID gốc.
2. Kiểm tra xem mỗi video đã có đủ nhãn cho tất cả frame chưa.
3. Xuất các frame bị thiếu (chưa có nhãn) ra thư mục riêng: 'unlabeled_missed_frames/'
4. Xuất báo cáo CSV & JSON chi tiết.
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def get_thumbnail_and_size(img_path):
    """Đọc ảnh nhanh và tạo thumbnail 16x16 để đối chiếu nhanh chóng."""
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
    
    # Cho phép truyền đường dẫn dataset qua CLI argument hoặc dùng v3i mặc định
    if len(sys.argv) > 1:
        v_name = sys.argv[1]
        v_dir = Path(v_name) if Path(v_name).is_absolute() else base_dir / v_name
    else:
        v_dir = base_dir / "fire_smoke_video.v3i.png-mask-semantic"
        
    v_train_dir = v_dir / "train" if (v_dir / "train").exists() else v_dir
    out_dir = base_dir / "unlabeled_missed_frames"
    
    # Xóa sạch thư mục xuất cũ để cập nhật chính xác nhất
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70, flush=True)
    print(f"  ĐỐI CHIẾU DỮ LIỆU & XUẤT FRAME CHƯA GÁN NHÃN: {v_dir.name}", flush=True)
    print("=" * 70, flush=True)
    
    rf_imgs = sorted([f for f in v_train_dir.glob("*.jpg")] + 
                     [f for f in v_train_dir.glob("*.png") if "_mask" not in f.name])
    print(f"[*] Tổng số ảnh đã gán nhãn trong {v_dir.name}: {len(rf_imgs)}", flush=True)
    
    # 1. Đọc thumbnail cho 668 ảnh Roboflow
    print("[*] Đang nạp và trích xuất đặc trưng của 668 ảnh nhãn...")
    t0 = time.time()
    rf_features = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(get_thumbnail_and_size, rf_imgs)
        for res in results:
            if res is not None:
                rf_features.append(res)
    print(f"    -> Xong trong {time.time() - t0:.2f}s (Nạp thành công {len(rf_features)} ảnh)")
    
    # 2. Quét và lập chỉ mục các thư mục video gốc thực tế
    candidate_roots = [
        base_dir / "test_annotation_frames",
        base_dir / "annotation_frames",
        base_dir / "final_video_frames",
    ]
    
    all_cand_paths = []
    video_to_all_frames = defaultdict(dict) # video_name -> {frame_name: full_path}
    
    print("[*] Đang quét danh sách frame gốc từ các thư mục video...")
    for root in candidate_roots:
        if not root.exists():
            continue
        for vid_folder in root.iterdir():
            if not vid_folder.is_dir():
                continue
            vid_name = vid_folder.name
            for f in vid_folder.glob("*.png"):
                if "_mask" in f.name or "_aug_" in f.name:
                    continue
                video_to_all_frames[vid_name][f.name] = f
                all_cand_paths.append(f)
    
    print(f"[*] Tổng số video gốc tìm thấy: {len(video_to_all_frames)} video")
    print(f"[*] Tổng số frame gốc cần lập chỉ mục: {len(all_cand_paths)} frames")
    
    # 3. Lập chỉ mục song song
    print("[*] Đang lập chỉ mục đặc trưng frame gốc (Multi-threading)...")
    t0 = time.time()
    indexed_candidates = defaultdict(list) # (h, w) -> list of (thumb, vid_name, frame_name, path)
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(get_thumbnail_and_size, all_cand_paths)
        for res in results:
            if res is not None:
                h, w, thumb, p = res
                vid_name = p.parent.name
                frame_name = p.name
                indexed_candidates[(h, w)].append((thumb, vid_name, frame_name, p))
                
    print(f"    -> Lập chỉ mục hoàn tất trong {time.time() - t0:.2f}s")
    
    # 4. Đối chiếu từng ảnh nhãn với frame gốc
    print("[*] Đang đối chiếu 668 ảnh nhãn với video gốc...")
    matched_results = []
    unmatched_rf = []
    
    for h, w, thumb, rf_p in rf_features:
        candidates = indexed_candidates.get((h, w), [])
        if not candidates:
            # Thử tìm các kích thước xấp xỉ nếu có letterbox
            unmatched_rf.append(rf_p.name)
            continue
            
        best_diff = 1e9
        best_match = None
        
        for c_thumb, vid_name, frame_name, p in candidates:
            diff = np.mean(np.abs(thumb.astype(float) - c_thumb.astype(float)))
            if diff < best_diff:
                best_diff = diff
                best_match = (vid_name, frame_name, p)
                if diff < 1.0: # Khớp chính xác
                    break
                    
        if best_match and best_diff < 12.0:
            vid_name, frame_name, p = best_match
            matched_results.append({
                "rf_filename": rf_p.name,
                "video_name": vid_name,
                "frame_name": frame_name,
                "source_path": str(p),
                "match_diff": round(best_diff, 2)
            })
        else:
            unmatched_rf.append(rf_p.name)
            
    print(f"  ✅ Đã khớp thành công: {len(matched_results)} / {len(rf_features)} frames")
    if unmatched_rf:
        print(f"  ⚠️ Số frame chưa khớp được: {len(unmatched_rf)}")
        
    # 5. Phân nhóm theo Video ID và kiểm tra độ đầy đủ
    # Gom các frame đã gán nhãn theo video
    video_labeled_frames = defaultdict(set)
    for m in matched_results:
        video_labeled_frames[m["video_name"]].add(m["frame_name"])
        
    print("\n" + "=" * 70)
    print("  KẾT QUẢ THỐNG KÊ CHI TIẾT THEO TỪNG VIDEO")
    print("=" * 70)
    
    summary_rows = []
    total_missing_frames = 0
    total_exported = 0
    
    # Duyệt qua các video xuất hiện trong tập gán nhãn
    for vid_name in sorted(video_labeled_frames.keys()):
        all_frames_in_vid = sorted(video_to_all_frames[vid_name].keys())
        labeled_set = video_labeled_frames[vid_name]
        
        missing_frames = [f for f in all_frames_in_vid if f not in labeled_set]
        total_vid_frames = len(all_frames_in_vid)
        num_labeled = len(labeled_set)
        num_missing = len(missing_frames)
        total_missing_frames += num_missing
        
        status = "FULL (Đầy đủ 100%)" if num_missing == 0 else f"MISSING ({num_missing} frames)"
        
        # Nếu có frame bị thiếu nhãn, xuất ra thư mục riêng
        if num_missing > 0:
            vid_out_dir = out_dir / vid_name
            vid_out_dir.mkdir(parents=True, exist_ok=True)
            for m_frame in missing_frames:
                src_f = video_to_all_frames[vid_name][m_frame]
                dst_f = vid_out_dir / m_frame
                shutil.copy2(src_f, dst_f)
                total_exported += 1
                
        summary_rows.append({
            "video_name": vid_name,
            "total_frames": total_vid_frames,
            "labeled_frames": num_labeled,
            "missing_frames": num_missing,
            "status": "FULL" if num_missing == 0 else "INCOMPLETE",
            "completion_rate": f"{(num_labeled / total_vid_frames * 100):.1f}%" if total_vid_frames > 0 else "0%",
            "missing_frame_list": "; ".join(missing_frames[:10]) + ("..." if len(missing_frames) > 10 else "")
        })
        
    df_summary = pd.DataFrame(summary_rows)
    csv_out_path = base_dir / "video_labeling_verification_report.csv"
    df_summary.to_csv(csv_out_path, index=False, encoding="utf-8-sig")
    
    # In bảng tổng kết ra màn hình
    full_count = len(df_summary[df_summary["status"] == "FULL"])
    incomplete_count = len(df_summary[df_summary["status"] == "INCOMPLETE"])
    
    print(f"\n[*] Tổng số video có frame trong tập nhãn: {len(df_summary)}")
    print(f"  - Số video ĐÃ GÁN NHÃN ĐỦ 100%       : {full_count} video")
    print(f"  - Số video BỊ MISS (thiếu frame nhãn) : {incomplete_count} video")
    print(f"  - Tổng số frame bị miss đã xuất ra   : {total_exported} frames")
    print(f"  - Thư mục lưu frame bị thiếu          : {out_dir}")
    print(f"  - Báo cáo CSV chi tiết                : {csv_out_path}")
    
    print("\n[Chi tiết Top 15 Video bị thiếu frame]:")
    incomplete_df = df_summary[df_summary["status"] == "INCOMPLETE"]
    for idx, row in incomplete_df.head(15).iterrows():
        print(f"  • {row['video_name']}: Gán {row['labeled_frames']}/{row['total_frames']} ({row['completion_rate']}) -> Thiếu: {row['missing_frames']} frames")
        
    return df_summary

if __name__ == "__main__":
    main()
