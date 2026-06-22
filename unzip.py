import modal
import zipfile
import os

# Khởi tạo Modal App
app = modal.App("unzip")

# Kết nối với Volume bạn đã tạo
volume = modal.Volume.from_name("fire_smoke_dataset")

@app.function(volumes={"/data": volume}, timeout=3600) # Đặt timeout 1 tiếng để tránh ngắt chừng
def extract_specific_zips(zip_filenames, extract_to_folder="/data/unzipped_data"):
    """
    Hàm này chạy trên server Modal.
    - zip_filenames: Danh sách tên các file zip cần giải nén.
    - extract_to_folder: Thư mục đích để chứa dữ liệu sau khi giải nén.
    """
    # Tạo thư mục đích nếu chưa có
    os.makedirs(extract_to_folder, exist_ok=True)

    for filename in zip_filenames:
        # Đường dẫn tới file zip trên Modal Volume (dựa vào bước 1)
        zip_path = f"/data/zips/{filename}"
        
        if not os.path.exists(zip_path):
            print(f"❌ Lỗi: Không tìm thấy file {zip_path}")
            continue

        print(f"⏳ Đang giải nén {filename}...")
        try:
            # Tạo một thư mục con mang tên file zip để dữ liệu không bị trộn lẫn vào nhau
            folder_name = filename.replace('.zip', '')
            target_dir = os.path.join(extract_to_folder, folder_name)
            os.makedirs(target_dir, exist_ok=True)

            # Thực hiện giải nén
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
                
            print(f"✅ Đã giải nén xong {filename} vào thư mục: {target_dir}")
        except Exception as e:
            print(f"❌ Lỗi khi giải nén {filename}: {e}")

@app.local_entrypoint()
def main():
    # =====================================================================
    # 👇 CHỈNH SỬA TẠI ĐÂY: NHẬP TÊN CÁC FILE ZIP BẠN MUỐN GIẢI NÉN
    # =====================================================================
    files_to_extract = [
        "2.zip", 
        "Video.zip",
        "Indoor Fire Smoke.zip"
    ] 
    
    print(f"Bắt đầu gửi lệnh giải nén {len(files_to_extract)} file lên Modal...")
    
    # Gọi hàm chạy trên Modal
    extract_specific_zips.remote(files_to_extract)
    
    print("Hoàn tất quá trình!")