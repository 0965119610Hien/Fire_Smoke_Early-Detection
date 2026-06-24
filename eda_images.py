import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

def analyze_and_visualize_local_dataset(source_dir, output_dir="eda_results"):
    """
    Script Thống kê và Xuất kết quả báo cáo EDA (Tối ưu cho Local & Quét đa tầng)
    """
    os.makedirs(output_dir, exist_ok=True)
    valid_data_pairs = []
    valid_extensions = ('.jpg', '.jpeg', '.png')

    stats = {
        'total_images': 0, 'corrupt_images': 0, 'missing_labels': 0,
        'fire_only': 0, 'smoke_only': 0, 'both_fire_smoke': 0,
        'background_normal': 0, 'garbage_labels': 0
    }

    print(f"🔍 Đang quét đệ quy toàn bộ dữ liệu trong: {source_dir} ...\n")

    # --- THUẬT TOÁN QUÉT BỌC THÉP TÌM ẢNH ---
    for root, dirs, files in os.walk(source_dir):
        if '__MACOSX' in root:
            continue

        if os.path.basename(root) == 'images':
            parent_dir = os.path.dirname(root)
            label_dir = os.path.join(parent_dir, 'labels')

            for file in files:
                if file.startswith('.'):
                    continue

                if file.lower().endswith(valid_extensions):
                    img_path = os.path.join(root, file)
                    stats['total_images'] += 1

                    # Kiểm tra file hỏng
                    try:
                        with open(img_path, 'rb') as f:
                            f.read(10)
                    except:
                        stats['corrupt_images'] += 1
                        continue

                    label_name = os.path.splitext(file)[0] + '.txt'
                    txt_path = os.path.join(label_dir, label_name)

                    if not os.path.exists(txt_path):
                        stats['missing_labels'] += 1
                        continue

                    # Phân tích file txt
                    has_fire = False
                    has_smoke = False
                    has_garbage = False
                    boxes = []
                    
                    with open(txt_path, 'r') as f:
                        for line in f.readlines():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                boxes.append([float(x) for x in parts])
                                
                                if class_id == 0: has_fire = True
                                elif class_id == 1: has_smoke = True
                                else: has_garbage += True
                                    
                    valid_data_pairs.append((img_path, boxes))
                                    
                    if has_fire and has_smoke: stats['both_fire_smoke'] += 1
                    elif has_fire: stats['fire_only'] += 1
                    elif has_smoke: stats['smoke_only'] += 1
                    else: stats['background_normal'] += 1
                        
                    if has_garbage: stats['garbage_labels'] += 1

    if stats['total_images'] == 0:
        print("❌ Lỗi: Không tìm thấy ảnh nào. Hãy kiểm tra lại đường dẫn RAW_DATASET_DIR.")
        return

    # ==========================================
    # 1. TẠO VÀ LƯU BÁO CÁO TEXT
    # ==========================================
    report_lines = [
        "="*50,
        "📊 BÁO CÁO THỐNG KÊ DỮ LIỆU EDA (LOCAL)",
        "="*50,
        f"Tổng số ảnh quét được: {stats['total_images']}",
        f" ⚠️ Ảnh bị hỏng (Corrupt): {stats['corrupt_images']}",
        f" ⚠️ Ảnh mất file Label: {stats['missing_labels']}",
        f" ⚠️ Ảnh có chứa nhãn rác (ID khác 0, 1): {stats['garbage_labels']}",
        "-" * 50,
        "📌 Phân bổ Nhãn (Dùng cho Giai đoạn 1):",
        f" 🔥 Chỉ có Lửa (Fire Only) : {stats['fire_only']}",
        f" 💨 Chỉ có Khói (Smoke Only): {stats['smoke_only']}",
        f" 🌪️ Có cả Lửa & Khói      : {stats['both_fire_smoke']}",
        f" 🟢 Bình thường (Normal)   : {stats['background_normal']}",
        "="*50
    ]
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    report_path = os.path.join(output_dir, "eda_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"✅ Đã lưu báo cáo văn bản tại: {report_path}")

    # ==========================================
    # 2. VẼ VÀ LƯU BIỂU ĐỒ CỘT
    # ==========================================
    categories = ['Fire Only', 'Smoke Only', 'Both', 'Normal']
    counts = [stats['fire_only'], stats['smoke_only'], stats['both_fire_smoke'], stats['background_normal']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, counts, color=['#e74c3c', '#95a5a6', '#e67e22', '#2ecc71'])
    plt.title('Phân bổ dữ liệu các Lớp (Class Imbalance Check)')
    plt.ylabel('Số lượng ảnh')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(counts)*0.01), int(yval), ha='center', va='bottom', fontweight='bold')
        
    plot1_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu biểu đồ phân bổ tại: {plot1_path}")

    # ==========================================
    # 3. VẼ VÀ HIỂN THỊ ẢNH TRỰC QUAN (LOCAL MODE)
    # ==========================================
    if len(valid_data_pairs) > 0:
        samples = random.sample(valid_data_pairs, min(4, len(valid_data_pairs)))
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle("Kiểm tra Bounding Box (Đỏ=Lửa, Xanh=Khói)", fontsize=16)
        
        for ax, (img_path, boxes) in zip(axes.flatten(), samples):
            # Thêm IMREAD_UNCHANGED và xử lý kênh màu chuẩn cho Local
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            
            for box in boxes:
                class_id, x_center, y_center, width, height = box
                x_min = int((x_center - width/2) * w)
                y_min = int((y_center - height/2) * h)
                box_w = int(width * w)
                box_h = int(height * h)
                
                color = (255, 0, 0) if class_id == 0 else (0, 255, 255)
                label = "Fire" if class_id == 0 else "Smoke"
                
                cv2.rectangle(img, (x_min, y_min), (x_min + box_w, y_min + box_h), color, 3)
                cv2.putText(img, label, (x_min, max(25, y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
            ax.imshow(img)
            ax.axis('off')
            # Cắt ngắn tên file nếu quá dài để hiển thị cho đẹp
            ax.set_title(os.path.basename(img_path)[:30], fontsize=10)
            
        plt.tight_layout()
        
        # Lưu ra file và ĐỒNG THỜI bật cửa sổ lên cho bạn xem ngay lập tức
        plot2_path = os.path.join(output_dir, "bbox_samples.png")
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        print(f"✅ Đã lưu ảnh kiểm tra Bounding Box tại: {plot2_path}")
        print("👀 Đang bật cửa sổ trực quan hình ảnh...")
        plt.show()

if __name__ == "__main__":
    # Thay đường dẫn này bằng thư mục tổng chứa 3 thư mục bạn vừa giải nén
    # Ví dụ: "E:/NCKH/Code/Fire_Smoke_Early-Detection/Data"
    RAW_DATASET_DIR = "Data" 
    
    analyze_and_visualize_local_dataset(RAW_DATASET_DIR)