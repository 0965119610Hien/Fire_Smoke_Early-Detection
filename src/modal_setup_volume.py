"""
src/modal_setup_volume.py
==========================
Tiện ích quản lý và giải nén dữ liệu trực tiếp trên Modal Volume `fire_smoke_dataset_v2`.
1. Giải nén /data/Fire_Smoke.v4i.zip -> /data/Fire_Smoke.v4i/
2. Giải nén /data/dataset_video_prepared.tar -> /data/dataset_video_prepared/
3. Khởi tạo thư mục /data/checkpoints/ để lưu trọng số Stage 1 và Stage 2.
4. Kiểm tra đối soát tính toàn vẹn (integrity verification) của tất cả splits và files.
"""

import modal
import os
import sys

app = modal.App("fire-smoke-volume-setup")
volume = modal.Volume.from_name("fire_smoke_dataset_v2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas", "tqdm")
)

@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def unpack_and_verify():
    import tarfile
    import zipfile
    import pandas as pd
    from pathlib import Path

    print("=" * 70)
    print("🚀 BẮT ĐẦU CẤU HÌNH VÀ GIẢI NÉN DỮ LIỆU TRÊN MODAL VOLUME: fire_smoke_dataset_v2")
    print("=" * 70)

    data_dir = Path("/data")
    checkpoints_dir = data_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Đã tạo thư mục lưu checkpoints: {checkpoints_dir}")

    # 1. Giải nén Static Dataset (Fire_Smoke.v4i.zip)
    v4i_zip = data_dir / "Fire_Smoke.v4i.zip"
    v4i_target = data_dir / "Fire_Smoke.v4i"
    if v4i_zip.exists():
        print(f"\n📦 Tìm thấy {v4i_zip} ({v4i_zip.stat().st_size / 1024 / 1024:.2f} MB). Bắt đầu giải nén...")
        v4i_target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(v4i_zip, 'r') as zip_ref:
            zip_ref.extractall(v4i_target)
        print(f"✅ Giải nén Static Dataset thành công vào: {v4i_target}")
    else:
        print(f"ℹ️ Không có file nén {v4i_zip} (hoặc đã được giải nén trước đó).")

    # 2. Giải nén Video Dataset (dataset_video_prepared.tar)
    video_tar = data_dir / "dataset_video_prepared.tar"
    video_target = data_dir
    if video_tar.exists():
        print(f"\n📦 Tìm thấy {video_tar} ({video_tar.stat().st_size / 1024 / 1024:.2f} MB). Bắt đầu giải nén...")
        with tarfile.open(video_tar, "r:") as tar_ref:
            tar_ref.extractall(video_target)
        print(f"✅ Giải nén Video Dataset thành công vào: {video_target / 'dataset_video_prepared'}")
    else:
        print(f"ℹ️ Không có file nén {video_tar} (hoặc đã được giải nén trước đó).")

    # Commit thay đổi vào volume
    volume.commit()
    print("💾 Đã commit thay đổi lên Modal Volume!")

    # 3. Đối soát tính toàn vẹn (Verification)
    print("\n🔍 ĐỐI SOÁT TÍNH TOÀN VẸN CỦA DỮ LIỆU:")
    splits_dir = data_dir / "data_splits"

    # Kiểm tra static dataset
    static_train_csv = splits_dir / "static_v4i_train_manifest.csv"
    static_val_csv = splits_dir / "static_v4i_val_manifest.csv"
    if static_train_csv.exists() and (v4i_target / "train").exists():
        df_train = pd.read_csv(static_train_csv)
        df_val = pd.read_csv(static_val_csv)
        print(f"  - Static Train CSV: {len(df_train)} dòng")
        print(f"  - Static Val CSV:   {len(df_val)} dòng")

        # Kiểm tra mẫu 10 ảnh đầu
        train_folder = v4i_target / "train"
        missing_train = sum(1 for f in df_train['filename'] if not (train_folder / f).exists())
        missing_val = sum(1 for f in df_val['filename'] if not (train_folder / f).exists())
        print(f"  - Static Train missing files: {missing_train}")
        print(f"  - Static Val missing files:   {missing_val}")
    else:
        print(f"  ⚠️ Cảnh báo: Chưa tìm thấy thư mục {v4i_target / 'train'} hoặc {static_train_csv}")

    # Kiểm tra video dataset
    video_train_csv = splits_dir / "train_clips.csv"
    video_val_csv = splits_dir / "val_clips.csv"
    video_dir = data_dir / "dataset_video_prepared"
    if video_train_csv.exists() and video_dir.exists():
        df_vid_train = pd.read_csv(video_train_csv)
        df_vid_val = pd.read_csv(video_val_csv)
        print(f"  - Video Train Clips: {len(df_vid_train)} clips (T=8)")
        print(f"  - Video Val Clips:   {len(df_vid_val)} clips (T=8)")

        # Kiểm tra tồn tại thư mục video
        video_folders = [f.name for f in video_dir.iterdir() if f.is_dir()]
        print(f"  - Tổng số video folders thực tế: {len(video_folders)}")
    else:
        print(f"  ⚠️ Cảnh báo: Chưa tìm thấy thư mục {video_dir} hoặc {video_train_csv}")

    print("\n" + "=" * 70)
    print("🎉 HOÀN TẤT THIẾT LẬP VÀ XÁC THỰC MODAL VOLUME!")
    print("=" * 70)
    return True
