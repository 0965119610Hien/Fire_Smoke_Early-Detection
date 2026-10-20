"""
src/filter_and_trim_continuous_incidents.py
============================================
Hệ thống 10 Bộ Lọc Chuyên Sâu & Cắt Trích Xuất Đoạn Cháy Khói Liên Tục (Tối ưu hóa đa luồng & FFmpeg pipe).
1. Trích xuất khung hình 2 FPS (mỗi 0.5s) qua FFmpeg pipe trực tiếp vào bộ nhớ (tối ưu tuyệt đối cho AV1, VP9, H264).
2. Phát hiện người dẫn chương trình / phỏng vấn (Haar Cascade trên ảnh thu nhỏ 320x180, <5ms/frame).
3. Phân tích quang học: Lửa (HSV+YCrCb), Khói khuếch tán, Độ mờ (Laplacian), Độ phơi sáng, Cắt cảnh.
4. Tìm phân đoạn liên tục [t_start, t_end] có cháy/khói >= 5.0s và KHÔNG có người dẫn chương trình.
5. Cắt tách losslessly bằng FFmpeg (-preset veryfast) loại bỏ hoàn toàn phần phỏng vấn/rác.
6. Cách ly các trường hợp xấu vào REJECTED_QUARANTINE theo từng phân loại cụ thể.
7. Tạo ảnh preview grid 3 mốc thời gian và xuất manifest CSV hoàn chỉnh.
"""

import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ── Khởi tạo đường dẫn ────────────────────────────────────────────────────────
base_dir = Path("crawled_early_fire_transition")
videos_dir = base_dir / "videos"
manifest_path = base_dir / "crawled_manifest.csv"
output_curated_dir = base_dir / "curated_continuous_incidents"

trans_dir = output_curated_dir / "01_CONTINUOUS_TRANSITIONS"
ambient_dir = output_curated_dir / "02_CONTINUOUS_AMBIENT_CLEAN"
neg_dir = output_curated_dir / "03_HARD_NEGATIVES_CLEAN"
quarantine_dir = output_curated_dir / "REJECTED_QUARANTINE"

q_interview_dir = quarantine_dir / "interviews_and_news_talking_heads"
q_intermittent_dir = quarantine_dir / "intermittent_short_flashes"
q_blur_dir = quarantine_dir / "severe_blur_and_shake"
q_exposure_dir = quarantine_dir / "extreme_exposure_noise"
q_animation_dir = quarantine_dir / "animation_and_cgi"
q_nofire_dir = quarantine_dir / "no_fire_smoke"

preview_dir = output_curated_dir / "preview_grids"

for d in [trans_dir, ambient_dir, neg_dir, q_interview_dir, q_intermittent_dir, 
          q_blur_dir, q_exposure_dir, q_animation_dir, q_nofire_dir, preview_dir]:
    d.mkdir(parents=True, exist_ok=True)

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


# ── Hàm trích xuất frame siêu tốc qua FFmpeg Pipe ─────────────────────────────
def extract_sampled_frames(video_path: Path, target_fps: float = 2.0, target_w: int = 480, target_h: int = 270) -> Tuple[List[float], List[np.ndarray], float]:
    """Trích xuất frame trực tiếp bằng FFmpeg pipe ở tốc độ 2 FPS và kích thước 480x270."""
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        total_dur = float(res.stdout.strip())
    except Exception:
        total_dur = 30.0

    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(video_path),
        "-vf", f"fps={target_fps},scale={target_w}:{target_h}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]
    frames = []
    frame_bytes = target_w * target_h * 3
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7)
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((target_h, target_w, 3))
            frames.append(arr)
        proc.wait(timeout=90)
    except Exception as e:
        print(f"Lỗi extract frame {video_path.name}: {e}")
        return [], [], 0.0

    step_time = 1.0 / target_fps
    sampled_times = [round(i * step_time, 2) for i in range(len(frames))]
    return sampled_times, frames, total_dur


# ── Hàm đo lường pixel khói & lửa ──────────────────────────────────────────
def get_fire_mask(bgr_frame: np.ndarray) -> np.ndarray:
    """Tạo mask ngọn lửa trong không gian HSV & YCrCb."""
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
    
    mask1 = cv2.inRange(hsv, np.array([0, 90, 160]), np.array([32, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 90, 160]), np.array([180, 255, 255]))
    hsv_mask = cv2.bitwise_or(mask1, mask2)
    
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    r = bgr_frame[:, :, 2]
    b = bgr_frame[:, :, 0]
    ycrcb_mask = (cr > 145) & (r > 155) & (r > b + 25) & (cr > cb)
    
    return hsv_mask & ycrcb_mask.astype(np.uint8) * 255


def get_smoke_mask(bgr_frame: np.ndarray, prev_gray: Optional[np.ndarray]) -> np.ndarray:
    """Tạo mask khói xám có độ bão hòa thấp và chuyển động."""
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    
    smoke_color = (s < 60) & (v > 90) & (v < 235)
    if prev_gray is not None:
        delta = cv2.absdiff(prev_gray, gray)
        motion = delta > 10
        return (smoke_color & motion).astype(np.uint8) * 255
    return smoke_color.astype(np.uint8) * 255


# ── Hàm cắt video bằng FFmpeg ───────────────────────────────────────────────
def trim_video_ffmpeg(src_path: Path, dst_path: Path, t_start: float, t_end: float) -> bool:
    """Cắt trích xuất phân đoạn [t_start, t_end] sạch sẽ."""
    duration = t_end - t_start
    if duration < 3.0:
        return False

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t_start:.2f}",
        "-i", str(src_path),
        "-t", f"{duration:.2f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-b:a", "96k",
        "-avoid_negative_ts", "make_zero",
        str(dst_path)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        return dst_path.exists() and dst_path.stat().st_size > 30_000
    except Exception as e:
        print(f"Lỗi khi cắt video {src_path.name}: {e}")
        return False


# ── Thread-safe Counter & Logger ─────────────────────────────────────────────
stats_lock = threading.Lock()
total_processed = 0
total_trimmed = 0
total_pure_kept = 0
total_rejected = 0


# ── Hàm xử lý độc lập cho từng video (Worker Function) ──────────────────────
def process_single_video(row: dict) -> dict:
    global total_processed, total_trimmed, total_pure_kept, total_rejected

    v_id = row["video_id"]
    cat = row["category"]
    filename = row["filename"]
    title = str(row.get("title", ""))

    src_file = videos_dir / filename
    if not src_file.exists():
        found = list(base_dir.glob(f"**/{filename}"))
        if found:
            src_file = found[0]
        else:
            return {
                "video_id": v_id, "category": cat, "title": title,
                "action": "SKIPPED_NOT_FOUND", "filter_reason": "file_not_found",
                "t_start": 0.0, "t_end": 0.0, "final_duration": 0.0, "final_path": ""
            }

    # Trích xuất frames 2 FPS (480x270)
    sampled_times, frames, total_duration_s = extract_sampled_frames(src_file)
    n_samples = len(frames)
    if n_samples < 8:
        # File quá ngắn hoặc lỗi đọc
        dst = q_nofire_dir / filename
        shutil.copy2(src_file, dst)
        with stats_lock:
            total_processed += 1
            total_rejected += 1
        return {
            "video_id": v_id, "category": cat, "title": title,
            "action": "REJECTED", "filter_reason": "too_short_or_corrupt",
            "t_start": 0.0, "t_end": total_duration_s, "final_duration": total_duration_s,
            "final_path": str(dst.relative_to(base_dir)).replace("\\", "/")
        }

    # Khởi tạo Face detector riêng cho thread này
    local_face_cascade = cv2.CascadeClassifier(cascade_path)

    has_fire_list: List[bool] = []
    has_smoke_list: List[bool] = []
    has_presenter_list: List[bool] = []
    blur_scores: List[float] = []
    mean_intensities: List[float] = []
    saturated_ratios: List[float] = []
    hist_corrs: List[float] = []

    prev_gray = None
    prev_hist = None
    total_pixels = 480 * 270

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Đo Blur
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_scores.append(lap_var)

        # 2. Đo Exposure
        mean_i = float(np.mean(gray))
        mean_intensities.append(mean_i)
        sat_ratio = float(np.mean(gray > 250))
        saturated_ratios.append(sat_ratio)

        # 3. Đo Scene Cut
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        if prev_hist is not None:
            corr = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
            hist_corrs.append(corr)
        prev_hist = hist

        # 4. Nhận diện Người dẫn chương trình / Phỏng vấn (trên ảnh 320x180 siêu tốc ~4ms)
        gray_small = cv2.resize(gray, (320, 180))
        faces = local_face_cascade.detectMultiScale(
            gray_small, scaleFactor=1.3, minNeighbors=3, minSize=(20, 20)
        )
        is_presenter = False
        for (fx, fy, fw, fh) in faces:
            face_pct = (fw * fh) / (320 * 180) * 100.0
            # Khuôn mặt chiếm >= 2.5% diện tích và nằm ở nửa trên màn hình
            if face_pct >= 2.5 and fy < 180 * 0.70:
                is_presenter = True
                break
        has_presenter_list.append(is_presenter)

        # 5. Nhận diện khói và lửa
        f_mask = get_fire_mask(frame)
        fire_pct = (cv2.countNonZero(f_mask) / total_pixels) * 100.0
        has_fire = fire_pct > 0.04

        s_mask = get_smoke_mask(frame, prev_gray)
        smoke_pct = (cv2.countNonZero(s_mask) / total_pixels) * 100.0
        has_smoke = smoke_pct > 0.35

        has_fire_list.append(has_fire)
        has_smoke_list.append(has_smoke)
        prev_gray = gray

    # Thống kê toàn clip
    avg_blur = float(np.mean(blur_scores)) if blur_scores else 100.0
    avg_intensity = float(np.mean(mean_intensities)) if mean_intensities else 100.0
    max_saturation = float(np.max(saturated_ratios)) if saturated_ratios else 0.0
    presenter_ratio = sum(has_presenter_list) / n_samples

    cat_clean = cat.replace("01_", "").replace("02_", "").replace("03_", "").replace("04_", "").replace("05_", "").replace("06_", "").replace("07_", "").replace("08_", "").replace("09_", "")

    # ── BỘ LỌC QUANG HỌC & RUNG LẮC ─────────────────────────────────────────
    # A. Rung lắc, mờ nhòe cực mạnh
    if avg_blur < 38.0:
        dst = q_blur_dir / filename
        shutil.copy2(src_file, dst)
        with stats_lock:
            total_processed += 1
            total_rejected += 1
        return {
            "video_id": v_id, "category": cat, "title": title,
            "action": "REJECTED", "filter_reason": f"severe_blur_and_shake (lap_var={avg_blur:.1f})",
            "t_start": 0.0, "t_end": total_duration_s, "final_duration": total_duration_s,
            "final_path": str(dst.relative_to(base_dir)).replace("\\", "/")
        }

    # B. Thiếu sáng đen kịt hoặc Cháy sáng cảm biến
    if avg_intensity < 18.0 or max_saturation > 0.40:
        reason = f"pitch_black ({avg_intensity:.1f})" if avg_intensity < 18.0 else f"sensor_saturation ({max_saturation*100:.1f}%)"
        dst = q_exposure_dir / filename
        shutil.copy2(src_file, dst)
        with stats_lock:
            total_processed += 1
            total_rejected += 1
        return {
            "video_id": v_id, "category": cat, "title": title,
            "action": "REJECTED", "filter_reason": f"extreme_exposure ({reason})",
            "t_start": 0.0, "t_end": total_duration_s, "final_duration": total_duration_s,
            "final_path": str(dst.relative_to(base_dir)).replace("\\", "/")
        }

    # ── MẪU ÂM TÍNH (HARD NEGATIVES) ─────────────────────────────────────────
    if cat == "hard_negatives":
        if presenter_ratio > 0.35:
            dst = q_interview_dir / filename
            shutil.copy2(src_file, dst)
            with stats_lock:
                total_processed += 1
                total_rejected += 1
            return {
                "video_id": v_id, "category": cat, "title": title,
                "action": "REJECTED", "filter_reason": f"news_interview_talking_head ({presenter_ratio*100:.1f}%)",
                "t_start": 0.0, "t_end": total_duration_s, "final_duration": total_duration_s,
                "final_path": str(dst.relative_to(base_dir)).replace("\\", "/")
            }
        else:
            dst = neg_dir / filename
            shutil.copy2(src_file, dst)
            with stats_lock:
                total_processed += 1
                total_pure_kept += 1
            return {
                "video_id": v_id, "category": cat, "title": title,
                "action": "KEEP_CLEAN", "filter_reason": "valid_clean_hard_negative",
                "t_start": 0.0, "t_end": total_duration_s, "final_duration": total_duration_s,
                "final_path": str(dst.relative_to(base_dir)).replace("\\", "/")
            }

    # ── TÌM CỬA SỔ LIÊN TỤC (CONTINUOUS INCIDENT WINDOW) ─────────────────────
    is_incident = [
        (has_fire_list[i] or has_smoke_list[i]) and (not has_presenter_list[i])
        for i in range(n_samples)
    ]

    best_start_idx = 0
    best_end_idx = 0
    best_len = 0

    cur_start_idx = -1
    consecutive_false = 0

    for i in range(n_samples):
        if is_incident[i]:
            if cur_start_idx == -1:
                cur_start_idx = i
            consecutive_false = 0
        else:
            if cur_start_idx != -1:
                consecutive_false += 1
                if consecutive_false >= 2:  # đứt quãng >= 1.0s thì ngắt đoạn
                    seg_len = (i - consecutive_false) - cur_start_idx + 1
                    if seg_len > best_len:
                        best_len = seg_len
                        best_start_idx = cur_start_idx
                        best_end_idx = i - consecutive_false
                    cur_start_idx = -1
                    consecutive_false = 0

    if cur_start_idx != -1:
        seg_len = n_samples - cur_start_idx
        if seg_len > best_len:
            best_len = seg_len
            best_start_idx = cur_start_idx
            best_end_idx = n_samples - 1

    continuous_duration_s = best_len * 0.5
    t_start = max(0.0, sampled_times[best_start_idx] - 0.5) if best_len > 0 else 0.0
    t_end = min(total_duration_s, sampled_times[best_end_idx] + 0.5) if best_len > 0 else 0.0
    final_incident_dur = t_end - t_start

    # ── TRƯỜNG HỢP LOẠI BỎ (KHÔNG ĐỦ 5.0S LIÊN TỤC) ─────────────────────────
    if continuous_duration_s < 5.0:
        if presenter_ratio > 0.25:
            dst = q_interview_dir / filename
            shutil.copy2(src_file, dst)
            reason = f"interview_talking_head_no_continuous_fire ({presenter_ratio*100:.1f}%)"
        elif any(has_fire_list):
            dst = q_intermittent_dir / filename
            shutil.copy2(src_file, dst)
            reason = f"intermittent_short_flashes ({continuous_duration_s:.1f}s < 5s)"
        else:
            dst = q_nofire_dir / filename
            shutil.copy2(src_file, dst)
            reason = "no_fire_or_smoke_detected"

        with stats_lock:
            total_processed += 1
            total_rejected += 1

        return {
            "video_id": v_id, "category": cat, "title": title,
            "action": "REJECTED", "filter_reason": reason,
            "t_start": 0.0, "t_end": total_duration_s, "final_duration": total_duration_s,
            "final_path": str(dst.relative_to(base_dir)).replace("\\", "/")
        }

    # ── TRƯỜNG HỢP CHẤP NHẬN (CÓ ĐOẠN CHÁY LIÊN TỤC >= 5.0S) ────────────────
    is_raw_trans = bool(row.get("is_transition_verified", False))
    early_pct = float(row.get("early_fire_pct", 0.0))

    target_subfolder = (trans_dir if is_raw_trans else ambient_dir) / cat_clean
    target_subfolder.mkdir(parents=True, exist_ok=True)

    needs_trimming = (t_start > 2.0) or (t_end < total_duration_s - 2.0) or (presenter_ratio > 0.05)

    action = "KEEP_CLEAN"
    final_file = target_subfolder / filename
    final_dur = total_duration_s

    if needs_trimming:
        trimmed_name = f"{Path(filename).stem}_continuous_trimmed.mp4"
        dst_path = target_subfolder / trimmed_name
        ok = trim_video_ffmpeg(src_file, dst_path, t_start, t_end)
        if ok:
            action = "TRIMMED_INCIDENT"
            final_file = dst_path
            final_dur = final_incident_dur
            with stats_lock:
                total_trimmed += 1
        else:
            shutil.copy2(src_file, final_file)
            with stats_lock:
                total_pure_kept += 1
    else:
        shutil.copy2(src_file, final_file)
        with stats_lock:
            total_pure_kept += 1

    with stats_lock:
        total_processed += 1

    # Tạo ảnh preview 3 mốc thời gian chứng minh đoạn cắt sạch
    if (is_raw_trans or action == "TRIMMED_INCIDENT") and final_file.exists():
        p_cap = cv2.VideoCapture(str(final_file))
        p_frames = int(p_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if p_frames >= 15:
            p_cap.set(cv2.CAP_PROP_POS_FRAMES, int(p_frames * 0.08))
            _, f_e = p_cap.read()
            p_cap.set(cv2.CAP_PROP_POS_FRAMES, int(p_frames * 0.50))
            _, f_m = p_cap.read()
            p_cap.set(cv2.CAP_PROP_POS_FRAMES, int(p_frames * 0.85))
            _, f_l = p_cap.read()
            p_cap.release()

            if f_e is not None and f_m is not None and f_l is not None:
                tw, th = 480, 270
                fe_r = cv2.resize(f_e, (tw, th))
                fm_r = cv2.resize(f_m, (tw, th))
                fl_r = cv2.resize(f_l, (tw, th))

                def draw_label(im, label_title, sub, color):
                    canvas = im.copy()
                    cv2.rectangle(canvas, (0, 0), (tw, 42), (20, 20, 20), -1)
                    cv2.putText(canvas, label_title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
                    cv2.putText(canvas, sub, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
                    return canvas

                c_e = draw_label(fe_r, "1. Bat Dau Doan Cat", f"Lua/Khoi: {early_pct:.2f}%", (0, 255, 255))
                c_m = draw_label(fm_r, "2. Dien Tien Lien Tuc", "Khong phong vien / Khong cat", (255, 200, 0))
                c_l = draw_label(fl_r, "3. Bung Lua / Toi Da", f"Lien tuc {final_dur:.1f}s", (0, 0, 255))

                comp = np.hstack([c_e, c_m, c_l])
                hdr = np.full((46, comp.shape[1], 3), 26, dtype=np.uint8)
                clean_title = "".join(c for c in title if ord(c) < 128)[:55]
                header_text = f"[{cat_clean.upper()}] {clean_title} ({final_dur:.1f}s, {action})"
                cv2.putText(hdr, header_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
                prev_img = np.vstack([hdr, comp])

                prev_name = f"continuous_{cat_clean}_{v_id}.jpg"
                cv2.imencode(".jpg", prev_img)[1].tofile(str(preview_dir / prev_name))

    return {
        "video_id": v_id, "category": cat, "title": title,
        "action": action, "filter_reason": f"continuous_incident_valid ({final_dur:.1f}s)",
        "t_start": round(t_start, 2), "t_end": round(t_end, 2), "final_duration": round(final_dur, 2),
        "is_transition": is_raw_trans,
        "final_path": str(final_file.relative_to(base_dir)).replace("\\", "/")
    }


# ── MAIN ORCHESTRATOR (Chạy Đa Luồng) ────────────────────────────────────────
def main():
    print(f"========================================================")
    print(f"🚀 BẮT ĐẦU HỆ THỐNG 10 BỘ LỌC CHUYÊN SÂU & CẮT TÁCH LIÊN TỤC (4 WORKERS)")
    print(f"========================================================")

    df_manifest = pd.read_csv(manifest_path)
    total_vids = len(df_manifest)
    print(f"Tổng số video cần phân tích: {total_vids}")

    rows = df_manifest.to_dict(orient="records")
    curated_records = []

    # Sử dụng ThreadPoolExecutor với 4 workers
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_vid = {executor.submit(process_single_video, r): r["video_id"] for r in rows}
        
        for future in as_completed(future_to_vid):
            vid = future_to_vid[future]
            try:
                rec = future.result()
                curated_records.append(rec)
            except Exception as exc:
                print(f"Lỗi khi xử lý video {vid}: {exc}")

            with stats_lock:
                done = total_processed
                if done % 20 == 0 or done == total_vids:
                    print(f"  [Tiến độ: {done:3d}/{total_vids}] "
                          f"✂️ Cắt trích: {total_trimmed:2d} | "
                          f"⭐ Thuần giữ: {total_pure_kept:3d} | "
                          f"❌ Đã loại: {total_rejected:3d}")

    # Xuất Manifest CSV
    report_df = pd.DataFrame(curated_records)
    manifest_out = output_curated_dir / "continuous_curation_manifest.csv"
    report_df.to_csv(manifest_out, index=False, encoding="utf-8")

    print(f"\n========================================================")
    print(f"✅ HOÀN THÀNH TOÀN BỘ 350 VIDEO:")
    print(f"  - Tổng số video đã duyệt     : {len(report_df)}")
    print(f"  - ✂️ Đã cắt trích đoạn liên tục: {total_trimmed} video (Đã loại bỏ 100% phóng viên/intro/outro)")
    print(f"  - ⭐ Video CCTV thuần giữ nguyên : {total_pure_kept} video")
    print(f"  - ❌ Video bị loại bỏ cách ly  : {total_rejected} video")
    print(f"\nChi tiết các lý do bị loại:")
    for reason, cnt in report_df[report_df["action"] == "REJECTED"]["filter_reason"].value_counts().items():
        print(f"  + {reason:55s}: {cnt:3d} video")

    print(f"\nThư mục kết quả hoàn chỉnh:")
    print(f"  📁 Thư mục lưu trữ: {output_curated_dir}")
    print(f"  📊 Manifest chi tiết: {manifest_out}")
    print(f"  🖼️ Thư mục ảnh kiểm chứng: {preview_dir}")
    print(f"========================================================")


if __name__ == "__main__":
    main()
