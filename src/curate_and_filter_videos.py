"""
src/curate_and_filter_videos.py
===============================
Bộ lọc dữ liệu thô toàn diện và Phân loại Curation cho 350 Video:
1. Lọc trùng video (Visual Fingerprint Deduplication):
   - Trích xuất vân tay thị giác (Temporal Fingerprint) qua 5 mốc thời gian (15%, 35%, 50%, 65%, 85%).
   - Đo khoảng cách sai phân MSE & Cosine similarity giữa các cặp video.
   - Giữ lại video có độ phân giải/chất lượng cao nhất trong nhóm trùng, đưa bản sao vào quarantine.
2. Lọc video đứng hình / slide ảnh (Motion / Static Slideshow Filter):
   - Đo lượng dịch chuyển điểm ảnh trung bình qua thời gian. Loại các video chỉ có ảnh tĩnh + nhạc.
3. Lọc video cắt cảnh quá nhiều (Excessive Scene Cuts / Compilation Montage):
   - Đo tương quan histogram màu giữa các khung hình liên tiếp. Loại video montage biên tập quá nhiều cảnh.
4. Lọc độ phân giải thấp / video hỏng (Corrupted / Low-Resolution):
   - Loại video không mở được, W < 480 hoặc H < 360, frame < 30.
5. Lọc video đã cháy to từ đầu (Already Engulfed Inferno):
   - Loại video đã bốc cháy dữ dội ngay từ giây đầu (> 5% diện tích).
6. Phân bổ vào thư mục curated_for_review/:
   - 01_GOLDEN_TRANSITIONS/       : Video chuyển tiếp chuẩn mẫu mực nhất (chia theo 8 loại)
   - 02_HARD_NEGATIVES_CLEAN/     : Video âm tính sạch không khói/lửa (chống báo động giả)
   - 03_AMBIENT_FIRE_GENERAL/     : Video cháy thực tế nhưng diễn biến tĩnh/chậm
   - REJECTED_QUARANTINE/         : Các video bị loại (chia theo thư mục lý do: duplicates, static, cuts, engulfed)
7. Xuất báo cáo curation_report.csv và lưới ảnh 3 mốc (Early -> Mid -> Late) cho golden transitions.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

base_dir = Path("crawled_early_fire_transition")
videos_dir = base_dir / "videos"
manifest_path = base_dir / "crawled_manifest.csv"
review_dir = base_dir / "curated_for_review"

# Thư mục đích
golden_dir = review_dir / "01_GOLDEN_TRANSITIONS"
neg_dir = review_dir / "02_HARD_NEGATIVES_CLEAN"
ambient_dir = review_dir / "03_AMBIENT_FIRE_GENERAL"
quarantine_dir = review_dir / "REJECTED_QUARANTINE"

quarantine_dup_dir = quarantine_dir / "duplicates"
quarantine_static_dir = quarantine_dir / "static_slideshows"
quarantine_cuts_dir = quarantine_dir / "excessive_cuts"
quarantine_engulfed_dir = quarantine_dir / "already_engulfed"
quarantine_corrupt_dir = quarantine_dir / "corrupted_or_low_res"

for d in [golden_dir, neg_dir, ambient_dir, quarantine_dup_dir, quarantine_static_dir, 
          quarantine_cuts_dir, quarantine_engulfed_dir, quarantine_corrupt_dir]:
    d.mkdir(parents=True, exist_ok=True)

# ── 1. ĐỌC MANIFEST & CHUẨN BỊ ──────────────────────────────────────────────
df = pd.read_csv(manifest_path)
print(f"Tổng số video trong manifest: {len(df)}")

video_records = []
fingerprints: Dict[str, np.ndarray] = {}
video_metadata: Dict[str, Dict] = {}

print("\n[BƯỚC 1/4] Đang phân tích kỹ thuật từng video (Resolution, Motion, Scene Cuts, Fingerprints)...")

for idx, row in df.iterrows():
    v_id = row["video_id"]
    cat = row["category"]
    filename = row["filename"]
    
    # Tìm file thực tế
    v_path = videos_dir / filename
    if not v_path.exists():
        # Tìm trong by_category nếu đã gom
        found = list(base_dir.glob(f"**/{filename}"))
        if found:
            v_path = found[0]
        else:
            video_records.append({
                "video_id": v_id,
                "filename": filename,
                "category": cat,
                "title": row.get("title", ""),
                "status": "REJECTED",
                "rejection_reason": "file_not_found",
                "width": 0, "height": 0, "fps": 0, "total_frames": 0,
                "motion_score": 0.0, "scene_cuts": 0,
                "duplicate_of": "",
                "is_transition": False
            })
            continue

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        video_records.append({
            "video_id": v_id,
            "filename": filename,
            "category": cat,
            "title": row.get("title", ""),
            "status": "REJECTED",
            "rejection_reason": "cannot_open_file",
            "width": 0, "height": 0, "fps": 0, "total_frames": 0,
            "motion_score": 0.0, "scene_cuts": 0,
            "duplicate_of": "",
            "is_transition": False
        })
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps

    # Kiểm tra độ phân giải quá thấp hoặc hỏng
    if total_frames < 30 or width < 400 or height < 280:
        cap.release()
        video_records.append({
            "video_id": v_id,
            "filename": filename,
            "category": cat,
            "title": row.get("title", ""),
            "status": "REJECTED",
            "rejection_reason": "corrupted_or_low_res",
            "width": width, "height": height, "fps": round(fps, 1), "total_frames": total_frames,
            "motion_score": 0.0, "scene_cuts": 0,
            "duplicate_of": "",
            "is_transition": False,
            "source_path": str(v_path)
        })
        continue

    # Lấy 5 khung hình để tạo Fingerprint (15%, 35%, 50%, 65%, 85%)
    fp_pcts = [0.15, 0.35, 0.50, 0.65, 0.85]
    fp_frames = []
    prev_gray = None
    motion_diffs = []
    
    # Đồng thời lấy 8 khung hình để kiểm tra Scene Cuts qua Histogram Correlation
    hist_corr_list = []
    prev_hist = None

    for p in np.linspace(0.10, 0.90, 9):
        f_idx = int(total_frames * p)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Đổi kích thước 64x64 grayscale cho motion
        gray_small = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = np.mean(np.abs(gray_small.astype(np.float32) - prev_gray.astype(np.float32)))
            motion_diffs.append(diff)
        prev_gray = gray_small

        # Histogram cho scene cut
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            hist_corr_list.append(corr)
        prev_hist = hist

    # Lấy 5 frame chính xác cho Fingerprint
    for p in fp_pcts:
        f_idx = int(total_frames * p)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            small_fp = cv2.cvtColor(cv2.resize(frame, (32, 32)), cv2.COLOR_BGR2GRAY)
            fp_frames.append(small_fp.flatten() / 255.0)
        else:
            fp_frames.append(np.zeros(32 * 32, dtype=np.float32))

    cap.release()

    # Tính điểm chuyển động (Motion score)
    motion_score = float(np.mean(motion_diffs)) if motion_diffs else 0.0

    # Đếm số lần cắt cảnh đột ngột (Scene cut nếu correlation < 0.35)
    scene_cuts = sum(1 for c in hist_corr_list if c < 0.35)

    # Ghép temporal fingerprint
    temporal_fp = np.concatenate(fp_frames)
    fingerprints[v_id] = temporal_fp

    early_fire = float(row.get("early_fire_pct", 0.0))
    late_fire = float(row.get("late_fire_pct", 0.0))
    is_trans = bool(row.get("is_transition_verified", False))

    video_metadata[v_id] = {
        "video_id": v_id,
        "filename": filename,
        "category": cat,
        "title": row.get("title", ""),
        "duration_s": round(duration_s, 1),
        "width": width,
        "height": height,
        "fps": round(fps, 1),
        "total_frames": total_frames,
        "early_fire_pct": early_fire,
        "late_fire_pct": late_fire,
        "fire_growth_ratio": float(row.get("fire_growth_ratio", 1.0)),
        "is_transition_raw": is_trans,
        "transition_reason": str(row.get("transition_reason", "")),
        "motion_score": round(motion_score, 2),
        "scene_cuts": scene_cuts,
        "source_path": str(v_path),
        "filesize_bytes": v_path.stat().st_size if v_path.exists() else 0
    }

print(f"-> Đã trích xuất đặc trưng cho {len(video_metadata)} video.")

# ── 2. PHÁT HIỆN TRÙNG LẶP (DEDUPLICATION) ──────────────────────────────────
print("\n[BƯỚC 2/4] Đang quét trùng lặp quang học (Temporal Fingerprint Matching)...")

duplicate_map: Dict[str, str] = {}  # duplicate_id -> primary_id
processed_ids = list(fingerprints.keys())
n_vids = len(processed_ids)

for i in range(n_vids):
    id_a = processed_ids[i]
    if id_a in duplicate_map:
        continue
    fp_a = fingerprints[id_a]

    for j in range(i + 1, n_vids):
        id_b = processed_ids[j]
        if id_b in duplicate_map:
            continue
        fp_b = fingerprints[id_b]

        # Tính Mean Absolute Error giữa 2 temporal fingerprints
        mae = float(np.mean(np.abs(fp_a - fp_b)))

        # Tính cosine similarity
        norm_a = np.linalg.norm(fp_a)
        norm_b = np.linalg.norm(fp_b)
        cosine_sim = float(np.dot(fp_a, fp_b) / (norm_a * norm_b + 1e-7))

        # Tiêu chí trùng lặp: MAE cực nhỏ (< 0.045) hoặc Cosine cực cao (> 0.94)
        if mae < 0.042 or cosine_sim > 0.945:
            meta_a = video_metadata[id_a]
            meta_b = video_metadata[id_b]

            # Giữ video có độ phân giải x số frame cao hơn (chất lượng tốt hơn)
            qual_a = meta_a["width"] * meta_a["height"] * meta_a["filesize_bytes"]
            qual_b = meta_b["width"] * meta_b["height"] * meta_b["filesize_bytes"]

            if qual_a >= qual_b:
                primary_id = id_a
                dup_id = id_b
            else:
                primary_id = id_b
                dup_id = id_a

            duplicate_map[dup_id] = primary_id
            print(f"  ⚡ Phát hiện TRÙNG LẶP (Sim: {cosine_sim:.3f}, MAE: {mae:.4f}):")
            print(f"     [Chính] {primary_id} ({video_metadata[primary_id]['title'][:40]})")
            print(f"     [Trùng] {dup_id} ({video_metadata[dup_id]['title'][:40]})")

print(f"-> Tổng số video bị trùng lặp phát hiện: {len(duplicate_map)}")

# ── 3. ÁP DỤNG CÁC BỘ LỌC CHẤT LƯỢNG & PHÂN LOẠI ────────────────────────────
print("\n[BƯỚC 3/4] Đang áp dụng các bộ lọc chất lượng thô (Motion, Cuts, Early Flame)...")

curated_results = []

for v_id, meta in video_metadata.items():
    cat = meta["category"]
    filename = meta["filename"]
    src_file = Path(meta["source_path"])
    motion = meta["motion_score"]
    cuts = meta["scene_cuts"]
    early_fire = meta["early_fire_pct"]
    is_trans_raw = meta["is_transition_raw"]

    status = "ACCEPTED"
    reject_reason = "none"
    target_dest: Optional[Path] = None

    # 1. Kiểm tra trùng lặp
    if v_id in duplicate_map:
        status = "REJECTED"
        reject_reason = f"duplicate_of_{duplicate_map[v_id]}"
        target_dest = quarantine_dup_dir / filename

    # 2. Kiểm tra video đứng hình / slideshow (motion < 1.5)
    elif motion < 1.2:
        status = "REJECTED"
        reject_reason = f"static_slideshow (motion={motion})"
        target_dest = quarantine_static_dir / filename

    # 3. Kiểm tra cắt cảnh quá nhiều (scene cuts >= 4 trong 8 điểm lấy mẫu)
    elif cuts >= 4:
        status = "REJECTED"
        reject_reason = f"excessive_scene_cuts ({cuts}_cuts)"
        target_dest = quarantine_cuts_dir / filename

    # 4. Kiểm tra cháy to từ đầu (> 4.5% diện tích ngay ở 8% thời lượng)
    elif early_fire > 4.5 and cat != "hard_negatives":
        status = "REJECTED"
        reject_reason = f"already_engulfed_inferno ({early_fire:.1f}%)"
        target_dest = quarantine_engulfed_dir / filename

    # NẾU VƯỢT QUA TẤT CẢ CÁC BỘ LỌC TRÊN:
    else:
        status = "ACCEPTED"
        if cat == "hard_negatives":
            target_dest = neg_dir / filename
            group_name = "02_HARD_NEGATIVES_CLEAN"
        elif is_trans_raw:
            cat_clean = cat.replace("01_", "").replace("02_", "").replace("03_", "").replace("04_", "").replace("05_", "").replace("06_", "").replace("07_", "").replace("08_", "").replace("09_", "")
            target_dest = golden_dir / cat_clean / filename
            target_dest.parent.mkdir(parents=True, exist_ok=True)
            group_name = "01_GOLDEN_TRANSITIONS"
        else:
            target_dest = ambient_dir / filename
            group_name = "03_AMBIENT_FIRE_GENERAL"

    # Sao chép file vào thư mục đích tương ứng
    if target_dest and src_file.exists():
        target_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, target_dest)

    meta_copy = meta.copy()
    meta_copy["curation_status"] = status
    meta_copy["rejection_reason"] = reject_reason
    meta_copy["duplicate_of"] = duplicate_map.get(v_id, "")
    meta_copy["curated_path"] = str(target_dest.relative_to(base_dir)).replace("\\", "/") if target_dest else ""
    meta_copy["curated_group"] = group_name if status == "ACCEPTED" else f"REJECTED_{reject_reason.split()[0]}"
    curated_results.append(meta_copy)

# ── 4. XUẤT BÁO CÁO VÀ LƯỚI ẢNH REVIEW ─────────────────────────────────────
print("\n[BƯỚC 4/4] Đang tổng hợp báo cáo và tạo lưới ảnh preview cho Golden Transitions...")

report_df = pd.DataFrame(curated_results)
report_csv_path = review_dir / "curation_report.csv"
report_df.to_csv(report_csv_path, index=False, encoding="utf-8")

# Tạo ảnh composite 3 mốc (Early -> Mid -> Late) cho toàn bộ Golden Transitions
golden_previews_dir = review_dir / "golden_previews"
golden_previews_dir.mkdir(parents=True, exist_ok=True)

golden_records = [r for r in curated_results if r["curation_status"] == "ACCEPTED" and r.get("curated_group") == "01_GOLDEN_TRANSITIONS"]
print(f"Đang tạo ảnh preview cho {len(golden_records)} video GOLDEN TRANSITIONS...")

for idx, r in enumerate(golden_records):
    v_path = Path(r["source_path"])
    if not v_path.exists():
        continue
    cap = cv2.VideoCapture(str(v_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 30:
        cap.release()
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.08))
    _, f_early = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.50))
    _, f_mid = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.85))
    _, f_late = cap.read()
    cap.release()

    if f_early is None or f_mid is None or f_late is None:
        continue

    target_h, target_w = 270, 480
    f_early_r = cv2.resize(f_early, (target_w, target_h))
    f_mid_r = cv2.resize(f_mid, (target_w, target_h))
    f_late_r = cv2.resize(f_late, (target_w, target_h))

    # Chú thích
    def draw_box(img, title, subtitle, col):
        im = img.copy()
        cv2.rectangle(im, (0, 0), (target_w, 42), (20, 20, 20), -1)
        cv2.putText(im, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)
        cv2.putText(im, subtitle, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
        return im

    p_early = draw_box(f_early_r, "1. Khoi Dau (8%)", f"Lua: {r['early_fire_pct']:.2f}% | Khoi/Tia", (0, 255, 255))
    p_mid = draw_box(f_mid_r, "2. Phat Trien (50%)", "Tich nhiet / Lan toa", (255, 200, 0))
    p_late = draw_box(f_late_r, "3. Bung Lua (85%)", f"Lua: {r['late_fire_pct']:.2f}% (x{r['fire_growth_ratio']}x)", (0, 0, 255))

    comp = np.hstack([p_early, p_mid, p_late])
    hdr = np.full((48, comp.shape[1], 3), 28, dtype=np.uint8)
    title_str = f"[{r['category'].upper()}] {r['title'][:65]} ({int(r['duration_s'])}s)"
    cv2.putText(hdr, title_str, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    final_img = np.vstack([hdr, comp])

    out_preview = golden_previews_dir / f"golden_{r['category']}_{r['video_id']}.jpg"
    cv2.imencode(".jpg", final_img)[1].tofile(str(out_preview))

print("\n========================================================")
print("BÁO CÁO TỔNG KẾT BỘ LỌC DỮ LIỆU & CURATION (350 VIDEO):")
print(f"  - Tổng số video đã kiểm tra: {len(curated_results)}")
print(f"  - ⭐ GOLDEN TRANSITIONS (Chuyển tiếp sạch mẫu mực) : {len(golden_records)} video")
print(f"  - 🛡️ HARD NEGATIVES (Âm tính sạch, không cháy)      : {len([r for r in curated_results if r['curated_group'] == '02_HARD_NEGATIVES_CLEAN'])} video")
print(f"  - ℹ️ AMBIENT FIRE (Cháy thực tế/chậm)               : {len([r for r in curated_results if r['curated_group'] == '03_AMBIENT_FIRE_GENERAL'])} video")
print(f"\n  ❌ CÁC VIDEO BỊ LOẠI BỎ (REJECTED QUARANTINE):")
reasons = report_df[report_df["curation_status"] == "REJECTED"]["rejection_reason"].value_counts()
for r, c in reasons.items():
    print(f"     + {r:35s}: {c} video")

print(f"\n  📁 Thư mục duyệt dữ liệu cho bạn: {review_dir}")
print(f"  📊 Tệp báo cáo chi tiết: {report_csv_path}")
print(f"  🖼️ Thư mục ảnh preview 3 mốc: {golden_previews_dir}")
print("========================================================")
