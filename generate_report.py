import os
import cv2
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import time

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Cấu trúc hiện tại: e:/NCKH/files/Data/Video/ gồm dataset_train và dataset_test
ROOT_DIR = Path('e:/NCKH/files/Data/Video')
TRAIN_ROOT = ROOT_DIR / 'dataset_train'
TEST_ROOT  = ROOT_DIR / 'dataset_test'

REPORT_PATH = Path('e:/NCKH/Code/Fire_Smoke_Early-Detection/dataset_stats_report.md')
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}

def get_video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    size = os.path.getsize(path)
    return {'fps': fps, 'duration': duration, 'width': width, 'height': height, 'size': size}

def scan_dir(root):
    stats = {}
    for label in ['0_normal_noise', '1_controlled_fire', '2_uncontrolled_hazard']:
        ldir = root / label
        if not ldir.exists(): continue
        stats[label] = {}
        for sd in sorted(ldir.iterdir()):
            # Bỏ qua các thư mục không phải subtype chuẩn
            if not sd.is_dir() or sd.name.startswith('_') or sd.name == 'und': continue
            sub_stats = []
            for f in sd.iterdir():
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                    info = get_video_info(f)
                    if info: sub_stats.append(info)
            if sub_stats:
                stats[label][sd.name] = sub_stats
    return stats

def aggregate_stats(dataset):
    total_videos = 0
    total_duration = 0
    total_size = 0
    durations = []
    fps_list = []
    
    label_stats = {}
    subtype_stats = {}
    resolutions = defaultdict(lambda: defaultdict(int))
    
    for label, subtypes in dataset.items():
        label_videos = 0
        label_duration = 0
        label_durations = []
        label_fps = []
        label_sizes = []
        
        subtype_stats[label] = {}
        for subtype, videos in subtypes.items():
            subtype_videos = len(videos)
            subtype_durations = [v['duration'] for v in videos if v['duration'] > 0]
            if not subtype_durations: continue
            
            total_videos += subtype_videos
            label_videos += subtype_videos
            total_duration += sum(subtype_durations)
            label_duration += sum(subtype_durations)
            
            durations.extend(subtype_durations)
            label_durations.extend(subtype_durations)
            
            for v in videos:
                total_size += v['size']
                fps_list.append(v['fps'])
                label_fps.append(v['fps'])
                label_sizes.append(v['size'])
                res = f"{v['width']}x{v['height']}"
                resolutions[label][res] += 1
            
            subtype_stats[label][subtype] = {
                'count': subtype_videos,
                'dur_mean': np.mean(subtype_durations),
                'dur_min': np.min(subtype_durations),
                'dur_max': np.max(subtype_durations),
            }
            
        if label_videos > 0:
            label_stats[label] = {
                'count': label_videos,
                'dur_min': np.min(label_durations),
                'dur_max': np.max(label_durations),
                'dur_mean': np.mean(label_durations),
                'dur_std': np.std(label_durations),
                'fps_mean': np.mean(label_fps),
                'fps_std': np.std(label_fps),
                'size_max': np.max(label_sizes)
            }
            
    return {
        'total_videos': total_videos,
        'total_duration_min': total_duration / 60,
        'total_size_gb': total_size / (1024**3),
        'dur_mean': np.mean(durations) if durations else 0,
        'dur_min': np.min(durations) if durations else 0,
        'dur_max': np.max(durations) if durations else 0,
        'fps_mean': np.mean(fps_list) if fps_list else 0,
        'label_stats': label_stats,
        'subtype_stats': subtype_stats,
        'resolutions': {k: dict(v) for k, v in resolutions.items()}
    }

def main():
    start_time = time.time()
    print(">>> Bắt đầu quét tập TRAIN...")
    train_data = scan_dir(TRAIN_ROOT)
    print(">>> Bắt đầu quét tập TEST...")
    test_data = scan_dir(TEST_ROOT)
    
    print(">>> Đang tính toán số liệu thống kê...")
    train_agg = aggregate_stats(train_data)
    test_agg = aggregate_stats(test_data)

    print(">>> Đang sinh báo cáo Markdown...")
    report = f"""# Báo Cáo Phân Tích Dữ Liệu Chi Tiết & Hướng Tiền Xử Lý Mô Hình
*(Đánh giá chuyên sâu cho mô hình phát hiện cháy sớm V-JEPA / Spatiotemporal)*

> **Cập nhật:** Cấu trúc dữ liệu mới nhất
> **Train path:** `{TRAIN_ROOT}`
> **Test path:** `{TEST_ROOT}`
> Ngày tạo báo cáo: {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Thống Kê Số Liệu Toàn Diện

Dữ liệu đã được phân loại thành 3 nhãn chính (3-class classification). Dưới đây là bức tranh tổng thể về số lượng và tính chất video thực tế:

### Tổng quan
| Metric | Train (Học) | Test (Kiểm thử) |
|---|---|---|
| **Tổng số Video** | **{train_agg['total_videos']} video** | **{test_agg['total_videos']} video** |
| **Tổng thời lượng** | {train_agg['total_duration_min']:.1f} phút | {test_agg['total_duration_min']:.1f} phút |
| **Tổng dung lượng** | {train_agg['total_size_gb']:.1f} GB | {test_agg['total_size_gb']:.1f} GB |
| **Thời lượng trung bình** | **{train_agg['dur_mean']:.2f} giây** | **{test_agg['dur_mean']:.2f} giây** |

### Phân bố theo nhãn (Class Distribution)
| Nhãn | Train | Tỉ lệ Train | Test | Tỉ lệ Test |
|---|---|---|---|---|
| **0_normal_noise** | {train_agg['label_stats'].get('0_normal_noise', {}).get('count', 0)} | {train_agg['label_stats'].get('0_normal_noise', {}).get('count', 0) / max(1, train_agg['total_videos']) * 100:.1f}% | {test_agg['label_stats'].get('0_normal_noise', {}).get('count', 0)} | {test_agg['label_stats'].get('0_normal_noise', {}).get('count', 0) / max(1, test_agg['total_videos']) * 100:.1f}% |
| **1_controlled_fire** | {train_agg['label_stats'].get('1_controlled_fire', {}).get('count', 0)} | {train_agg['label_stats'].get('1_controlled_fire', {}).get('count', 0) / max(1, train_agg['total_videos']) * 100:.1f}% | {test_agg['label_stats'].get('1_controlled_fire', {}).get('count', 0)} | {test_agg['label_stats'].get('1_controlled_fire', {}).get('count', 0) / max(1, test_agg['total_videos']) * 100:.1f}% |
| **2_uncontrolled_hazard** | {train_agg['label_stats'].get('2_uncontrolled_hazard', {}).get('count', 0)} | {train_agg['label_stats'].get('2_uncontrolled_hazard', {}).get('count', 0) / max(1, train_agg['total_videos']) * 100:.1f}% | {test_agg['label_stats'].get('2_uncontrolled_hazard', {}).get('count', 0)} | {test_agg['label_stats'].get('2_uncontrolled_hazard', {}).get('count', 0) / max(1, test_agg['total_videos']) * 100:.1f}% |

---

## 2. Thống kê chi tiết & Phân bổ Subtype

### A. Chi tiết tập TRAIN
| Label | Videos | Dur min | Dur max | Dur mean | Dur stdev | FPS mean |
|---|---|---|---|---|---|---|
"""
    for lbl in ['0_normal_noise', '1_controlled_fire', '2_uncontrolled_hazard']:
        s = train_agg['label_stats'].get(lbl)
        if s:
            report += f"| `{lbl}` | {s['count']} | {s['dur_min']:.2f}s | {s['dur_max']:.2f}s | {s['dur_mean']:.2f}s | {s['dur_std']:.2f}s | {s['fps_mean']:.2f} |\n"

    report += f"| **TOTAL** | **{train_agg['total_videos']}** | **{train_agg['dur_min']:.2f}s** | **{train_agg['dur_max']:.2f}s** | **{train_agg['dur_mean']:.2f}s** | — | {train_agg['fps_mean']:.2f} |\n\n"
    
    report += "### B. Chi tiết tập TEST\n| Label | Videos | Dur min | Dur max | Dur mean | Dur stdev | FPS mean |\n|---|---|---|---|---|---|---|\n"
    for lbl in ['0_normal_noise', '1_controlled_fire', '2_uncontrolled_hazard']:
        s = test_agg['label_stats'].get(lbl)
        if s:
            report += f"| `{lbl}` | {s['count']} | {s['dur_min']:.2f}s | {s['dur_max']:.2f}s | {s['dur_mean']:.2f}s | {s['dur_std']:.2f}s | {s['fps_mean']:.2f} |\n"
    report += f"| **TOTAL** | **{test_agg['total_videos']}** | **{test_agg['dur_min']:.2f}s** | **{test_agg['dur_max']:.2f}s** | **{test_agg['dur_mean']:.2f}s** | — | {test_agg['fps_mean']:.2f} |\n\n"

    report += """---

## 3. Mô Tả Chi Tiết Ngữ Nghĩa Của Từng Nhãn

Phần này phân tích ngữ nghĩa vật lý của các video trong từng nhãn dựa trên số liệu thực tế được quét:

### 🟢 Nhãn 0: Normal Noise (Không có cháy rủi ro)
Nhãn quan trọng nhất để hệ thống ép tỷ lệ báo động giả (False Positive) xuống mức thấp.
- **`cigarette_vape_smoke`:** Khói thuốc lá, vape bốc lên và tan nhanh vào không khí. 
- **`dust_particles` / `steam_vapor`:** Hạt bụi bay lơ lửng gần ống kính (phản xạ đèn hồng ngoại) và hơi nước (như từ ấm đun nước). Rất dễ nhầm với khói sương.
- **`cooking_smoke_normal`:** Khói sinh hoạt nhà bếp thông thường, mật độ mỏng.
- **`optical_noise` / `daily_activity_fp`:** Các hoạt động nhiễu quang học (đèn xe lướt qua), mờ ống kính, nhện giăng tơ...
- **`welding_sparks_noise`:** Tia lửa hàn cường độ thấp.


### 🟡 Nhãn 1: Controlled Fire (Cháy/Khói Có Kiểm Soát)
Phân biệt sự cháy có chủ đích, có sự giám sát của con người.
- **`welding_sparks_ctrl`:** Máy cắt/hàn kim loại công nghiệp phát ra vệt lửa dài liên tục.
- **`heavy_smoke_controlled`:** Khói đốt rác, cỏ ngoài đồng, khói nhà máy. Mật độ dày nhưng an toàn.
- **`small_visible_flame`:** Lửa bếp gas, nến, ngọn lửa củi trại nhỏ nhắn.
- **`cooking_smoke_threshold`:** Khói bếp đặc (đồ ăn khét), mấp mé ngưỡng báo động.

### 🔴 Nhãn 2: Uncontrolled Hazard (Nguy Hiểm/Ngoài Tầm Kiểm Soát)
Yêu cầu còi báo động kích hoạt tức thì.
- **`residential_indoor_fire` / `large_visible_flame`:** Ngọn lửa lớn bùng phát trong phòng khách, kho.
- **`heavy_smoke_hazard` / `toxic_material_smoke`:** Khói đặc cuồn cuộn che khuất tầm nhìn, khói từ nhựa/hóa chất (màu đen/đục).
- **`diffused_smoke_spreading`:** Khói tràn lan khắp các hành lang, trần nhà (thời điểm hệ thống PCCC truyền thống mới bắt đầu nhận diện).
- **`early_smoke_real_fire` / `daily_ctx_real_fire`:** Giai đoạn bắt đầu chập cháy của các vật liệu thực tế.

---

## 4. Phân Tích Kỹ Thuật (Lỗ Hổng Dữ Liệu) & Hướng Tiền Xử Lý

Dựa trên việc đọc metadata (FPS, Dimensions, Duration) của toàn bộ video, hệ thống vạch ra 3 vấn đề lớn và các bước tiền xử lý bắt buộc cho mô hình V-JEPA / TimeSformer:

### A. Độ Lệch Thời Lượng (Duration Shift)
- **Vấn đề:** 
  - Train có clip ngắn nhất là **{train_agg['dur_min']:.1f} giây**, trung bình **{train_agg['dur_mean']:.1f} giây**.
  - Test có clip ngắn nhất là **{test_agg['dur_min']:.1f} giây**, trung bình **{test_agg['dur_mean']:.1f} giây**.
- **Hệ quả:** Mô hình Spatiotemporal học theo chuỗi thời gian. Dạy trên clip ngắn rồi test trên clip dài 40s sẽ làm hỏng Attention Map.
- **Hướng giải quyết:**
  - **Train:** Bỏ qua (Filter) toàn bộ clip < 5 giây.
  - **Test (Inference):** Dùng Sliding Window. Cắt video test dài thành các đoạn 10s (stride = 5s). Đưa từng đoạn vào mô hình, lấy điểm Max hoặc Majority Vote.

### B. Xung Đột Tốc Độ Khung Hình (FPS Inconsistency)
- **Vấn đề:** FPS của dữ liệu không đồng nhất (biến động từ 24fps đến 60fps). Độ lệch chuẩn FPS (STDEV) ở Train có lúc lên tới **{train_agg['label_stats'].get('1_controlled_fire', {}).get('fps_std', 0):.2f}**.
- **Hệ quả:** Nếu lấy frame theo index (ví dụ bốc frame số 10, 20), thời gian thực tế thu được sẽ sai lệch hoàn toàn giữa video 24fps và 60fps.
- **Hướng giải quyết:** Bắt buộc áp dụng **Uniform Time Sampling (Lấy mẫu theo giây tuyệt đối)**. Nếu lấy 16 frames, phải cắt chính xác ở các mốc thời gian $t = 0, \frac{T}{16}, \frac{2T}{16}...$ thay vì đếm frame.

### C. Độ Lệch Khung Hình (Spatial Resolution Chaos)
- **Vấn đề:** 
"""
    for lbl in ['0_normal_noise']:
        res = train_agg['resolutions'].get(lbl, {})
        top_res = sorted(res.items(), key=lambda x: x[1], reverse=True)[:4]
        res_str = ", ".join([f"{k}" for k,v in top_res])
        report += f"  - Dữ liệu chứa rất nhiều độ phân giải khác nhau, bao gồm cả quay ngang (Landscape) và quay dọc (Portrait) như: {res_str}...\n"
        
    report += """- **Hệ quả:** Nếu dùng lệnh `cv2.resize(224, 224)` thông thường, các video quay dọc 9:16 sẽ bị bóp méo thành hình vuông 1:1, phá vỡ hình dáng ngọn lửa và luồng khói.
- **Hướng giải quyết:** 
  - **Short-side Resize + CenterCrop:** Thu nhỏ cạnh ngắn nhất về 256px, rồi cắt chính giữa khung hình 224x224px.
  - **Letterbox (Zero Padding):** Thêm viền đen 2 bên để giữ nguyên tỉ lệ video ban đầu, không làm mất bối cảnh.
"""

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f">>> XONG! Báo cáo đã được lưu tại: {REPORT_PATH}")
    print(f">>> Tổng thời gian chạy: {time.time() - start_time:.1f} giây")

if __name__ == '__main__':
    main()
