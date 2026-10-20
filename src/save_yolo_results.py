"""
src/save_yolo_results.py
========================
Script local gọi Modal App để lấy weights đã huấn luyện và ghi báo cáo đánh giá ra file local.
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import modal
from modal_train_yolo import app, train_and_eval_yolo, ROOT, WEIGHTS_DIR, REPORTS_DIR

sys.stdout.reconfigure(encoding='utf-8')

def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = ROOT / "dataset_balanced.zip"
    if not zip_path.exists():
        print(f"[ERROR] Không tìm thấy file zip: {zip_path}")
        return

    zip_bytes = zip_path.read_bytes()
    print("[LOCAL] Đang kết nối Modal App để nhận kết quả huấn luyện và đánh giá...")

    with app.run():
        res = train_and_eval_yolo.remote(zip_bytes=zip_bytes, epochs=30)
        
        # 1. Lưu weights
        best_pt_local = WEIGHTS_DIR / "yolov8_fire_smoke_best.pt"
        best_pt_local.write_bytes(res["best_weights"])
        print(f"[LOCAL] Đã lưu file weights tốt nhất: {best_pt_local} ({best_pt_local.stat().st_size / (1024*1024):.2f} MB)")
        
        # 2. Lưu biểu đồ
        metrics = res["metrics"]
        plots = res["plots"]
        for p_name, p_bytes in plots.items():
            p_local = REPORTS_DIR / f"yolo_test_{p_name}"
            p_local.write_bytes(p_bytes)
            print(f"[LOCAL] Đã lưu biểu đồ: {p_local}")

        # 3. Xuất báo cáo Markdown
        mp_val = metrics['mp'] * 100
        mr_val = metrics['mr'] * 100
        map50_val = metrics['map50'] * 100
        map50_95_val = metrics['map50_95'] * 100

        fire_p = metrics['class_metrics']['fire']['p'] * 100
        fire_r = metrics['class_metrics']['fire']['r'] * 100
        fire_map50 = metrics['class_metrics']['fire']['map50'] * 100
        fire_map50_95 = metrics['class_metrics']['fire']['map50_95'] * 100

        smoke_p = metrics['class_metrics']['smoke']['p'] * 100
        smoke_r = metrics['class_metrics']['smoke']['r'] * 100
        smoke_map50 = metrics['class_metrics']['smoke']['map50'] * 100
        smoke_map50_95 = metrics['class_metrics']['smoke']['map50_95'] * 100

        report_content = f"""# Báo Cáo Đánh Giá Mô Hình YOLOv8s Trên Tập Test

- **Dataset**: `Dataset_Balanced_NoEmpty` (13,044 ảnh sạch)
- **Tập Test**: 1,144 ảnh
- **Mô hình**: YOLOv8s (30 Epochs trained on Modal GPU A10G)

---

## 📊 Kết Quả Đánh Giá Tổng Quan (Overall Metrics)

| Chỉ số (Metric) | Giá trị (Value) |
| :--- | :--- |
| **Precision (mp)** | **{mp_val:.2f}%** |
| **Recall (mr)** | **{mr_val:.2f}%** |
| **mAP@50** | **{map50_val:.2f}%** |
| **mAP@50-95** | **{map50_95_val:.2f}%** |

---

## 🏷️ Kết Quả Chi Tiết Theo Lớp (Per-Class Metrics)

| Lớp (Class) | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :--- | :--- | :--- | :--- |
| 🔥 **Fire (0)** | {fire_p:.2f}% | {fire_r:.2f}% | {fire_map50:.2f}% | {fire_map50_95:.2f}% |
| 💨 **Smoke (1)** | {smoke_p:.2f}% | {smoke_r:.2f}% | {smoke_map50:.2f}% | {smoke_map50_95:.2f}% |

---

## 📁 File Lưu Trữ
- **Model Weights**: [`weights/yolov8_fire_smoke_best.pt`](file:///{best_pt_local.as_posix()})
- **Báo cáo**: [`reports/yolo_test_evaluation_report.md`](file:///{(REPORTS_DIR / 'yolo_test_evaluation_report.md').as_posix()})
"""

        report_path = REPORTS_DIR / "yolo_test_evaluation_report.md"
        report_path.write_text(report_content, encoding="utf-8")
        print(f"[LOCAL] Đã tạo file báo cáo: {report_path}")

    print("\n==========================================================")
    print(" HOÀN THÀNH TẢI KẾT QUẢ VỀ MÁY CỤC BỘ!")
    print(f" -> mAP50: {map50_val:.2f}% | Precision: {mp_val:.2f}% | Recall: {mr_val:.2f}%")
    print(f" -> Weights lưu tại: {best_pt_local}")
    print("==========================================================")

if __name__ == "__main__":
    main()
