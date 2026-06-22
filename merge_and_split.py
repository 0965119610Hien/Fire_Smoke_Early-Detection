import modal
import os
import shutil
import random
import concurrent.futures
from pathlib import Path

app = modal.App("merge-and-split-dataset")
volume = modal.Volume.from_name("fire_smoke_dataset")

@app.function(volumes={"/data": volume}, timeout=7200)
def process_dataset(source_dir="/data/unzipped_data", dest_dir="/data/merged_dataset", split_ratio=(0.7, 0.2, 0.1)):
    print("🔍 Bắt đầu quét tìm tất cả ảnh và label...")
    all_data_pairs = []
    valid_extensions = ('.jpg', '.jpeg', '.png')

    for root, _, files in os.walk(source_dir):
        if 'images' in root:
            for file in files:
                if file.lower().endswith(valid_extensions):
                    img_path = os.path.join(root, file)
                    label_dir = root.replace('/images', '/labels')
                    label_name = os.path.splitext(file)[0] + '.txt'
                    label_path = os.path.join(label_dir, label_name)

                    if os.path.exists(label_path):
                        all_data_pairs.append((img_path, label_path))

    total_files = len(all_data_pairs)
    print(f"✅ Đã tìm thấy {total_files} cặp (Ảnh + Label).")

    if total_files == 0:
        return

    print("🔀 Đang xáo trộn dữ liệu...")
    random.shuffle(all_data_pairs)

    train_count = int(total_files * split_ratio[0])
    val_count = int(total_files * split_ratio[1])
    
    train_data = all_data_pairs[:train_count]
    val_data = all_data_pairs[train_count:train_count + val_count]
    test_data = all_data_pairs[train_count + val_count:]

    print(f"📊 Phân bổ: Train ({len(train_data)}), Val ({len(val_data)}), Test ({len(test_data)})")

    # ==========================================
    # HÀM MỚI: CHÉP FILE ĐA LUỒNG TỐC ĐỘ CAO
    # ==========================================
    def copy_single_pair(idx, img_src, lbl_src, img_dest_dir, lbl_dest_dir):
        orig_img_name = os.path.basename(img_src)
        orig_lbl_name = os.path.basename(lbl_src)
        prefix = f"{idx:06d}_"
        shutil.copy2(img_src, os.path.join(img_dest_dir, prefix + orig_img_name))
        shutil.copy2(lbl_src, os.path.join(lbl_dest_dir, prefix + orig_lbl_name))

    def copy_files_fast(data_list, split_name):
        print(f"⏳ Đang xử lý tập {split_name.upper()} (chế độ đa luồng)...")
        img_dest_dir = os.path.join(dest_dir, split_name, 'images')
        lbl_dest_dir = os.path.join(dest_dir, split_name, 'labels')
        os.makedirs(img_dest_dir, exist_ok=True)
        os.makedirs(lbl_dest_dir, exist_ok=True)

        # Mở 50 luồng để copy cùng lúc
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            for idx, (img_src, lbl_src) in enumerate(data_list):
                futures.append(executor.submit(copy_single_pair, idx, img_src, lbl_src, img_dest_dir, lbl_dest_dir))
            
            # Chờ đến khi tất cả các luồng chạy xong
            concurrent.futures.wait(futures)
        print(f"✅ Đã copy xong tập {split_name.upper()}!")

    # Xóa dữ liệu cũ nếu có
    if os.path.exists(dest_dir):
        print("🧹 Đang dọn dẹp thư mục đích cũ (có thể mất vài phút)...")
        shutil.rmtree(dest_dir)

    # Chạy copy siêu tốc
    copy_files_fast(train_data, 'train')
    copy_files_fast(val_data, 'val')
    copy_files_fast(test_data, 'test')

    yaml_content = f"""
        train: {dest_dir}/train/images
        val: {dest_dir}/val/images
        test: {dest_dir}/test/images

        nc: 2
        names: ['fire', 'smoke']
        """
    with open(os.path.join(dest_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content.strip())

    print(f"🎉 Hoàn tất! Dataset đã sẵn sàng tại: {dest_dir}")

@app.local_entrypoint()
def main():
    process_dataset.remote(split_ratio=(0.7, 0.15, 0.15))