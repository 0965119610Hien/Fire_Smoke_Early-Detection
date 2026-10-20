"""
Batch Trimming Pipeline using Method 2 (Optical Active Window + Fire/Smoke Dynamics + Face Suppression).
Applied across all videos in 01_CONTINUOUS_TRANSITIONS and 02_CONTINUOUS_AMBIENT_CLEAN.
"""
import os
import sys
import csv
import time
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np

# Reconfigure stdout for Windows console UTF-8
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(r"e:\NCKH\Code\Fire_Smoke_Early-Detection\crawled_early_fire_transition\curated_continuous_incidents")
BACKUP_DIR = Path(r"e:\NCKH\Code\Fire_Smoke_Early-Detection\crawled_early_fire_transition\backup_before_m2_trim")
REPORT_CSV = Path(r"e:\NCKH\Code\Fire_Smoke_Early-Detection\crawled_early_fire_transition\batch_m2_trimming_report.csv")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def probe_duration(video_path: Path) -> float:
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(res.stdout.strip())
    except:
        return 0.0

def analyze_video_m2(video_path: Path):
    orig_dur = probe_duration(video_path)
    if orig_dur <= 0:
        return 0.0, 0.0, 0.0, False, 0.0
        
    t0 = time.time()
    cmd = [
        'ffmpeg', '-v', 'error',
        '-i', str(video_path),
        '-vf', 'fps=2,scale=320:180',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-'
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7)
        w, h = 320, 180
        fb = w * h * 3
        frames = []
        while True:
            raw = proc.stdout.read(fb)
            if len(raw) < fb:
                break
            frames.append(np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)))
        proc.wait()
    except Exception as e:
        return orig_dur, 0.0, orig_dur, False, 0.0

    n = len(frames)
    if n == 0:
        return orig_dur, 0.0, orig_dur, False, time.time() - t0

    incident_flags = []
    prev_gray = None
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        
        # 1. Face detection (suppress talking heads/interviewers)
        is_face = False
        faces = face_cascade.detectMultiScale(gray, 1.3, 3, minSize=(25, 25))
        for (fx, fy, fw, fh) in faces:
            if (fw * fh) / (w * h) > 0.02 and fy < h * 0.70:
                is_face = True
                break
                
        # 2. Fire HSV mask
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        fire_m1 = cv2.inRange(hsv, (0, 75, 140), (35, 255, 255))
        fire_m2 = cv2.inRange(hsv, (170, 75, 140), (180, 255, 255))
        fire_mask = cv2.bitwise_or(fire_m1, fire_m2)
        fire_ratio = cv2.countNonZero(fire_mask) / (w * h)
        
        # 3. Smoke HSV mask
        smoke_mask = cv2.inRange(hsv, (0, 0, 90), (180, 60, 235))
        smoke_ratio = cv2.countNonZero(smoke_mask) / (w * h)
        
        # 4. Motion score
        motion = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion = np.mean(diff > 12)
        prev_gray = gray
        
        # Activity decision
        if is_face:
            is_active = False
        else:
            is_active = (fire_ratio > 0.0003) or (smoke_ratio > 0.08 and motion > 0.015) or (fire_ratio > 0.0001 and motion > 0.01)
        incident_flags.append(is_active)
        
    # Longest continuous segment with gap tolerance of 3 frames (1.5s)
    best_s, best_e, best_len = 0, 0, 0
    cur_s = -1
    cur_false = 0
    for i in range(n):
        if incident_flags[i]:
            if cur_s == -1:
                cur_s = i
            cur_false = 0
        else:
            if cur_s != -1:
                cur_false += 1
                if cur_false > 3:
                    seg_len = (i - cur_false) - cur_s + 1
                    if seg_len > best_len:
                        best_len = seg_len
                        best_s = cur_s
                        best_e = i - cur_false
                    cur_s = -1
                    cur_false = 0
                    
    if cur_s != -1:
        seg_len = (n - cur_false) - cur_s
        if seg_len > best_len:
            best_len = seg_len
            best_s = cur_s
            best_e = n - 1 - cur_false
            
    proc_time = time.time() - t0
    
    if best_len >= 4: # at least 2 seconds of sustained activity
        t_start = max(0.0, best_s * 0.5 - 2.0)
        t_end = min(orig_dur, (best_e + 1) * 0.5 + 3.0)
        # Needs trimming if cutting >= 3.0s from start or >= 4.0s from end, and duration >= 5.0s
        needs_trim = ((t_start >= 3.0) or ((orig_dur - t_end) >= 4.0)) and ((t_end - t_start) >= 5.0)
    else:
        t_start = 0.0
        t_end = orig_dur
        needs_trim = False
        
    return orig_dur, t_start, t_end, needs_trim, proc_time

def run_pipeline():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    groups = ['01_CONTINUOUS_TRANSITIONS', '02_CONTINUOUS_AMBIENT_CLEAN']
    video_list = []
    for grp in groups:
        grp_dir = BASE_DIR / grp
        for f in grp_dir.rglob('*.mp4'):
            # Ignore temporary files
            if f.name.startswith('temp_'):
                continue
            video_list.append(f)
            
    total_vids = len(video_list)
    print("=" * 70)
    print(f">> BAT DAU CHAY PIPELINE PHUONG PHAP 2 TREN TOAN BO {total_vids} VIDEO")
    print(f"   - Nguon: 01_CONTINUOUS_TRANSITIONS & 02_CONTINUOUS_AMBIENT_CLEAN")
    print(f"   - Thu muc backup video goc: {BACKUP_DIR.name}/")
    print(f"   - File bao cao: {REPORT_CSV.name}")
    print("=" * 70)
    
    results = []
    trimmed_count = 0
    clean_count = 0
    error_count = 0
    total_sec_cut = 0.0
    
    for idx, v_path in enumerate(video_list, 1):
        rel_path = v_path.relative_to(BASE_DIR)
        print(f"[{idx:3d}/{total_vids}] {rel_path} ... ", end="", flush=True)
        
        orig_dur, t_start, t_end, needs_trim, proc_time = analyze_video_m2(v_path)
        
        if orig_dur <= 0:
            print("LOI (khong doc duoc file)")
            error_count += 1
            results.append({
                "file": str(rel_path),
                "orig_duration": 0,
                "trimmed_duration": 0,
                "t_start": 0,
                "t_end": 0,
                "action": "ERROR",
                "seconds_saved": 0,
                "scan_time": proc_time
            })
            continue
            
        if needs_trim:
            # Prepare backup path
            bak_file = BACKUP_DIR / rel_path
            bak_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Temporary trimmed file in the same directory
            temp_trimmed = v_path.parent / f"temp_trimmed_{v_path.name}"
            
            # Run FFmpeg trim
            cut_dur = t_end - t_start
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-ss', f"{t_start:.2f}",
                '-to', f"{t_end:.2f}",
                '-i', str(v_path),
                '-c:v', 'libx264',
                '-crf', '20',
                '-preset', 'fast',
                '-c:a', 'aac',
                '-avoid_negative_ts', '1',
                str(temp_trimmed)
            ]
            res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            # Verify the cut file
            new_dur = probe_duration(temp_trimmed)
            if res.returncode == 0 and temp_trimmed.exists() and new_dur > 2.0:
                # Backup original file
                if not bak_file.exists():
                    shutil.copy2(str(v_path), str(bak_file))
                # Replace with trimmed file
                temp_trimmed.replace(v_path)
                
                sec_saved = orig_dur - new_dur
                total_sec_cut += sec_saved
                trimmed_count += 1
                print(f"DA CAT! ({orig_dur:.1f}s -> {new_dur:.1f}s, giam {sec_saved:.1f}s) [{proc_time:.1f}s]")
                action_str = "TRIMMED"
                final_dur = new_dur
            else:
                if temp_trimmed.exists():
                    temp_trimmed.unlink()
                print(f"LOI FFMPEG (giu nguyen file goc)")
                action_str = "TRIM_FAILED_KEPT_ORIG"
                final_dur = orig_dur
                sec_saved = 0.0
        else:
            clean_count += 1
            final_dur = orig_dur
            sec_saved = 0.0
            action_str = "CLEAN_UNTOUCHED"
            print(f"CHUNG MINH SACH (giu nguyen {orig_dur:.1f}s) [{proc_time:.1f}s]")
            
        results.append({
            "file": str(rel_path),
            "orig_duration": f"{orig_dur:.2f}",
            "trimmed_duration": f"{final_dur:.2f}",
            "t_start": f"{t_start:.2f}",
            "t_end": f"{t_end:.2f}",
            "action": action_str,
            "seconds_saved": f"{sec_saved:.2f}",
            "scan_time": f"{proc_time:.2f}"
        })

    # Write report CSV
    with open(REPORT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "orig_duration", "trimmed_duration", "t_start", "t_end",
            "action", "seconds_saved", "scan_time"
        ])
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 70)
    print("HOAN THANH PIPELINE PHUONG PHAP 2 CHO TOAN BO DATASET:")
    print(f"   - Tong so video kiem tra: {total_vids}")
    print(f"   - So video duoc cat tia (loai bo intro/outro/thua): {trimmed_count}")
    print(f"   - So video da sach san (giu nguyen khong can cat): {clean_count}")
    print(f"   - So video gap loi: {error_count}")
    print(f"   - Tong thoi gian doan thua da duoc loai bo: {total_sec_cut / 60:.2f} phut ({total_sec_cut:.1f} giay)")
    print(f"   - File bao cao chi tiet: {REPORT_CSV}")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
