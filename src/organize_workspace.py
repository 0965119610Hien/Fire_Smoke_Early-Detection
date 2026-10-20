"""
organize_workspace.py
=====================
Script tự động sắp xếp lại cấu trúc dự án Fire_Smoke_Early-Detection cho ngăn nắp.

CẤU TRÚC SAU KHI SẮP XẾP:
  Fire_Smoke_Early-Detection/
  ├── data/
  │   ├── pure_fire_smoke_clips/     (177 video clip thuần khói/lửa)
  │   ├── annotation_frames/         (3,339 PNG frames 512x512)
  │   ├── annotation_labels/         (SegFormer-B2 Masks, Vis, COCO JSON)
  │   └── sample_video_previews/     (20 MP4 video mẫu)
  ├── src/                           (Mã nguồn Python)
  ├── weights/                       (Trọng số mô hình: SAM, YOLOv8-World, SegFormer)
  ├── logs/                          (Các file log hệ thống)
  ├── docs/                          (Tài liệu kiến trúc & báo cáo NCKH)
  └── README.md                      (File tổng quan dự án)
"""

import shutil
from pathlib import Path

root = Path(__file__).parent.parent if Path(__file__).parent.name == "src" else Path(".")

# 1. Create target directories
dirs_to_create = ["data", "weights", "logs", "docs", "docs/reports"]
for d in dirs_to_create:
    (root / d).mkdir(parents=True, exist_ok=True)

# 2. Move log files to logs/
log_files = list(root.glob("*.log"))
for lf in log_files:
    dst = root / "logs" / lf.name
    print(f"Moving log: {lf.name} -> logs/")
    shutil.move(str(lf), str(dst))

# 3. Move weights to weights/
weight_files = ["yolov8s-world.pt", "segformer_b2_fire_smoke_v1.pth"]
for wf in weight_files:
    src_p = root / wf
    if src_p.exists():
        dst_p = root / "weights" / wf
        print(f"Moving weight: {wf} -> weights/")
        shutil.move(str(src_p), str(dst_p))

# 4. Move reports & docs to docs/
docs_files = [
    "Preprocessed_Dataset_Evaluation_Report.md",
    "Segmentation_Improvement_Strategies.md",
    "Video_Preprocessing_Proposals_SegFormer.md",
    "Detection_Tracking_StateEngine_Architecture.md",
    "dataset_statistics.md",
]
for df in docs_files:
    src_p = root / df
    if src_p.exists():
        dst_p = root / "docs" / df
        print(f"Moving doc: {df} -> docs/")
        shutil.move(str(src_p), str(dst_p))

# Move evaluation reports to docs/reports/
report_files = [
    "segformer_test_report.json",
    "segformer_training_history.csv",
    "segformer_learning_curves.png",
]
for rf in report_files:
    src_p = root / rf
    if src_p.exists():
        dst_p = root / "docs" / "reports" / rf
        print(f"Moving report: {rf} -> docs/reports/")
        shutil.move(str(src_p), str(dst_p))

# Remove temp scratch files if any
scratch_files = ["get_grid.py"]
for sf in scratch_files:
    src_p = root / sf
    if src_p.exists():
        src_p.unlink()
        print(f"Cleaned up temporary script: {sf}")

print("\n[OK] WORKSPACE ORGANIZED NEATLY!")
