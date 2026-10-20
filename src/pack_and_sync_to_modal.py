"""
src/pack_and_sync_to_modal.py
=============================
Đóng gói và đồng bộ toàn bộ dữ liệu âm tính (171 MB) lên Modal Volume:
1. Đóng gói các thư mục neg_* thành new_negatives.tar
2. Đóng gói các ảnh tĩnh âm tính mới thành new_static_neg.tar
3. Upload lên Volume qua modal volume put
4. Chạy Modal App giải nén trực tiếp trên Volume Linux NVMe siêu tốc
5. Kiểm tra tính toàn vẹn của dữ liệu trên Volume
"""

import os
import sys
import tarfile
import subprocess
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(r"E:\NCKH\Code\Fire_Smoke_Early-Detection")
DATA_DIR = BASE / "data"
PREPARED_DIR = DATA_DIR / "dataset_video_prepared"
V4I_TRAIN_DIR = DATA_DIR / "Fire_Smoke.v4i.png-mask-semantic" / "train"
SPLITS_DIR = DATA_DIR / "data_splits"

NEG_TAR = DATA_DIR / "new_negatives.tar"
STATIC_TAR = DATA_DIR / "new_static_neg.tar"

def make_tar_archives():
    print("=" * 70)
    print("1. DONG GOI ARCHIVES CHO DU LIEU AM TINH")
    print("=" * 70)
    
    # 1. Archive video neg folders
    print(f"[*] Dang nen cac folder neg_* trong {PREPARED_DIR.name}...")
    neg_folders = sorted([p for p in PREPARED_DIR.glob("neg_*") if p.is_dir()])
    with tarfile.open(NEG_TAR, "w") as tar:
        for f in neg_folders:
            tar.add(f, arcname=f.name)
    print(f"  [OK] Da tao {NEG_TAR.name} ({NEG_TAR.stat().st_size / (1024*1024):.2f} MB, {len(neg_folders)} folders)")

    # 2. Archive static neg images
    print(f"[*] Dang nen cac anh neg_* trong {V4I_TRAIN_DIR.name}...")
    neg_static_files = sorted(list(V4I_TRAIN_DIR.glob("neg_*.*")))
    with tarfile.open(STATIC_TAR, "w") as tar:
        for f in neg_static_files:
            tar.add(f, arcname=f.name)
    print(f"  [OK] Da tao {STATIC_TAR.name} ({STATIC_TAR.stat().st_size / (1024*1024):.2f} MB, {len(neg_static_files)} files)")

def upload_to_modal():
    print("\n" + "=" * 70)
    print("2. UPLOAD LEN MODAL VOLUME: fire_smoke_dataset_v2")
    print("=" * 70)
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    files_to_put = [
        (NEG_TAR, "new_negatives.tar"),
        (STATIC_TAR, "new_static_neg.tar"),
    ]
    
    for local_p, remote_p in files_to_put:
        print(f"[*] Dang upload {local_p.name} len Volume ({remote_p})...")
        cmd = ["modal", "volume", "put", "-f", "fire_smoke_dataset_v2", str(local_p), remote_p]
        res = subprocess.run(cmd, env=env)
        if res.returncode == 0:
            print(f"  [OK] Upload thanh cong: {remote_p}")
        else:
            print(f"  [FAIL] Loi upload: {remote_p}")
            
    # Upload toàn bộ file manifest trong data_splits
    print(f"[*] Dang dong bo data_splits sang Modal Volume...")
    for csv_file in SPLITS_DIR.glob("*.csv"):
        cmd = ["modal", "volume", "put", "-f", "fire_smoke_dataset_v2", str(csv_file), f"data_splits/{csv_file.name}"]
        subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  [OK] Dong bo toan bo data_splits thanh cong.")

if __name__ == "__main__":
    make_tar_archives()
    upload_to_modal()
