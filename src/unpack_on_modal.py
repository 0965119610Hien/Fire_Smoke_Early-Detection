"""
src/unpack_on_modal.py
======================
Chạy trực tiếp trên Modal Container để giải nén dữ liệu mới vào đúng đường dẫn trên Volume.
"""

import modal
import tarfile
from pathlib import Path

app = modal.App("fire-smoke-unpack-negatives")
volume = modal.Volume.from_name("fire_smoke_dataset_v2")

@app.function(
    volumes={"/data": volume},
    timeout=600
)
def unpack_data():
    data_dir = Path("/data")
    print("🚀 Bắt đầu giải nén dữ liệu âm tính trên Modal Volume...")
    
    # 1. Giải nén video negatives
    neg_tar = data_dir / "new_negatives.tar"
    video_target = data_dir / "dataset_video_prepared"
    video_target.mkdir(parents=True, exist_ok=True)
    
    if neg_tar.exists():
        print(f"📦 Đang giải nén {neg_tar} vào {video_target}...")
        with tarfile.open(neg_tar, "r:") as tar:
            tar.extractall(video_target)
        print("  ✅ Giải nén video negatives thành công!")
        neg_tar.unlink()  # Xóa file tar tạm
    else:
        print(f"⚠️ Không tìm thấy {neg_tar}")
        
    # 2. Giải nén static negatives
    static_tar = data_dir / "new_static_neg.tar"
    static_target = data_dir / "Fire_Smoke.v4i" / "train"
    static_target.mkdir(parents=True, exist_ok=True)
    
    if static_tar.exists():
        print(f"📦 Đang giải nén {static_tar} vào {static_target}...")
        with tarfile.open(static_tar, "r:") as tar:
            tar.extractall(static_target)
        print("  ✅ Giải nén static negatives thành công!")
        static_tar.unlink()  # Xóa file tar tạm
    else:
        print(f"⚠️ Không tìm thấy {static_tar}")
        
    # Commit volume
    volume.commit()
    print("🎉 Volume committed! Kiểm tra số lượng:")
    num_videos = len(list(video_target.glob("*")))
    print(f"  - Tổng số video folders trong /data/dataset_video_prepared: {num_videos}")

@app.local_entrypoint()
def main():
    unpack_data.remote()
