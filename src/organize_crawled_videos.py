"""
src/organize_crawled_videos.py
==============================
Gom và tổ chức 350 video vừa crawl vào các thư mục phân loại chuyên nghiệp:
- Thư mục gốc: crawled_early_fire_transition/by_category/
- 9 Kịch bản (Categories):
    01_electrical_ignition
    02_battery_thermal_runaway
    03_smoldering_transition
    04_kitchen_stove
    05_industrial_workplace
    06_vehicle_engine_fire
    07_wildfire_forest_ignition
    08_lab_experiments
    09_hard_negatives
- Trong mỗi kịch bản chia 2 nhóm:
    + transition_verified/      : Video chuẩn chuyển tiếp (khói mỏng/đốm nhỏ -> bùng lửa)
    + standard_or_developed/   : Video cháy tĩnh, khói chậm, hoặc đã cháy to từ trước
- Cập nhật crawled_manifest.csv với đường dẫn mới (organized_filepath).
"""

import shutil
import sys
from pathlib import Path
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

base_dir = Path("crawled_early_fire_transition")
videos_dir = base_dir / "videos"
manifest_path = base_dir / "crawled_manifest.csv"
target_base_dir = base_dir / "by_category"
target_base_dir.mkdir(parents=True, exist_ok=True)

CATEGORY_FOLDERS = {
    "electrical_ignition": "01_electrical_ignition",
    "battery_thermal_runaway": "02_battery_thermal_runaway",
    "smoldering_transition": "03_smoldering_transition",
    "kitchen_stove": "04_kitchen_stove",
    "industrial_workplace": "05_industrial_workplace",
    "vehicle_engine_fire": "06_vehicle_engine_fire",
    "wildfire_forest_ignition": "07_wildfire_forest_ignition",
    "wildfire_ignition": "07_wildfire_forest_ignition",  # Gộp wildfire cũ
    "lab_experiments": "08_lab_experiments",
    "hard_negatives": "09_hard_negatives",
}

if not manifest_path.exists():
    print("Không tìm thấy crawled_manifest.csv!")
    sys.exit(1)

df = pd.read_csv(manifest_path)
print(f"Bắt đầu gom và phân loại {len(df)} video vào thư mục: {target_base_dir}...")

organized_paths = []
copied_count = 0
missing_count = 0

for idx, row in df.iterrows():
    cat = row["category"]
    folder_name = CATEGORY_FOLDERS.get(cat, f"other_{cat}")
    
    is_verified = bool(row.get("is_transition_verified", False))
    reason = str(row.get("transition_reason", ""))
    
    if cat == "hard_negatives":
        subfolder = "hard_negatives"
    elif is_verified:
        subfolder = "transition_verified"
    else:
        subfolder = "standard_or_developed"
        
    cat_dir = target_base_dir / folder_name / subfolder
    cat_dir.mkdir(parents=True, exist_ok=True)
    
    filename = row["filename"]
    src_file = videos_dir / filename
    
    # Nếu file đã di chuyển hoặc nằm ở src_file
    dest_file = cat_dir / filename
    
    if src_file.exists():
        # Copy sang thư mục phân loại
        shutil.copy2(src_file, dest_file)
        copied_count += 1
        organized_paths.append(str(dest_file.relative_to(base_dir)).replace("\\", "/"))
    elif dest_file.exists():
        copied_count += 1
        organized_paths.append(str(dest_file.relative_to(base_dir)).replace("\\", "/"))
    else:
        missing_count += 1
        organized_paths.append("")

df["category_folder"] = df["category"].map(lambda c: CATEGORY_FOLDERS.get(c, c))
df["organized_path"] = organized_paths

# Lưu lại manifest đã cập nhật
df.to_csv(manifest_path, index=False, encoding="utf-8")

print(f"\n========================================================")
print(f"HOÀN TẤT PHÂN LOẠI 350 VIDEO THEO FOLDER TỪNG LOẠI:")
print(f"  - Tổng số video đã gom: {copied_count}")
print(f"  - Số video thiếu (nếu có): {missing_count}")
print(f"  - Thư mục đích: {target_base_dir}")
print(f"\nThống kê chi tiết từng folder kịch bản:")

summary_list = []
for folder in sorted(set(CATEGORY_FOLDERS.values())):
    f_path = target_base_dir / folder
    if not f_path.exists():
        continue
    all_vids = list(f_path.glob("**/*.mp4"))
    trans_vids = list((f_path / "transition_verified").glob("*.mp4")) if (f_path / "transition_verified").exists() else []
    neg_vids = list((f_path / "hard_negatives").glob("*.mp4")) if (f_path / "hard_negatives").exists() else []
    std_vids = list((f_path / "standard_or_developed").glob("*.mp4")) if (f_path / "standard_or_developed").exists() else []
    
    print(f"  📁 [{folder}]: {len(all_vids)} video (⭐ Chuyển tiếp: {len(trans_vids)}, Âm tính: {len(neg_vids)}, Cháy thường/tĩnh: {len(std_vids)})")

print(f"========================================================")
