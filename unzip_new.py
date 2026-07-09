import modal
import zipfile
import os

app = modal.App("unzip-new-data")
volume = modal.Volume.from_name("fire_smoke_dataset")

# 👇 THÊM timeout=3600 VÀO DÒNG NÀY (Cho phép chạy tối đa 1 tiếng)
@app.function(volumes={"/data": volume}, timeout=3600)
def unzip_data():
    zip_path = "/data/zips/dataset_merged.zip"
    extract_to = "/data/unzipped_data/dataset_merged" # Tạo thư mục riêng cho nó
    
    if not os.path.exists(zip_path):
        print(f"❌ Không tìm thấy file {zip_path}")
        return

    os.makedirs(extract_to, exist_ok=True)
    print("⏳ Đang giải nén dữ liệu mới (video dung lượng lớn nên có thể mất vài phút)...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
    print(f"✅ Đã giải nén xong vào {extract_to}")

@app.local_entrypoint()
def main():
    unzip_data.remote()