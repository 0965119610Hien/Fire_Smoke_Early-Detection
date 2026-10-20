# 🔥 Fire & Smoke Early Detection & Tracking Project (NCKH)

Hệ thống Nghiên cứu Khoa học Phát hiện và Phân vùng Lửa & Khói Sớm dựa trên kiến trúc **2 Giai đoạn (Two-Stage Pipeline)**:
1. **Stage 1 (Spatial Core)**: Mô hình **SegFormer-B2** phân đoạn ngữ nghĩa ảnh tĩnh, tối ưu hóa biểu diễn không gian cho các đốm lửa nhỏ và khói loãng đa môi trường (Đạt **74.92% Fire IoU**, **50.92% Smoke IoU**, **98.17% Recall**, **100% Precision**).
2. **Stage 2 (Spatio-Temporal SegFormer)**: Kế thừa 100% trọng số không gian đã tối ưu từ Stage 1, tích hợp khối **Temporal Refinement Module (1D Temporal Conv + Temporal Multi-Head Attention)** với cơ chế **Zero-Initialized Residual Gating ($\gamma = 0$)** trên chuỗi video ($T=8$ frames) để nhận biết động học thời gian (flame flickering và smoke diffusion) mà không làm suy giảm độ nét của biên vùng.

---

## 📁 Cấu Trúc Thư Mục Chuẩn Hóa

```text
Fire_Smoke_Early-Detection/
├── data/                                         # Dữ liệu sạch đã chuẩn hóa & manifests
│   ├── data_splits/                              # 18 manifests phân chia train/val/test & sequence clips
│   ├── Fire_Smoke.v4i.png-mask-semantic/         # Tập dữ liệu ảnh tĩnh (1,058 ảnh + mask)
│   └── dataset_video_prepared/                   # 68 video thực tế chuẩn hóa (1,153 frames + mask)
├── docs/                                         # Tài liệu NCKH & báo cáo kiến trúc
│   ├── Modal_Two_Stage_Training_Architecture_Guide.md  # Hướng dẫn chi tiết & bảng tra cứu lệnh Modal
│   ├── Dataset_68_Videos_Statistics_and_Preprocessing_Plan.md
│   ├── Static_Dataset_v4i_Statistics_and_Preprocessing_Plan.md
│   ├── segformer_test_predictions/               # Kết quả suy luận Stage 1 trên ảnh tĩnh test
│   └── inference_comparisons/                    # Các ảnh kết xuất trực quan 4-Panel đối sánh Stage 1 & Stage 2
├── checkpoints/                                  # Trọng số cục bộ đã tải xuống từ Cloud
│   ├── segformer_spatial_best.pth                # Trọng số Stage 1 (328.7 MB)
│   ├── mit_b2_spatial_backbone.pth               # Backbone không gian (96.9 MB)
│   └── timesformer_spatiotemporal_best.pth       # Trọng số Stage 2 (246.5 MB)
├── src/                                          # Mã nguồn lõi huấn luyện, đánh giá & suy luận
│   ├── modal_setup_volume.py                     # Tiện ích giải nén & đối soát dữ liệu trên Modal Volume
│   ├── train_stage1_segformer.py                 # Huấn luyện Stage 1 (SegFormer-B2 trên GPU A10G)
│   ├── train_stage2_timesformer.py               # Huấn luyện Stage 2 (Spatio-Temporal SegFormer trên GPU A10G)
│   ├── evaluate_all_metrics.py                   # Đánh giá toàn bộ 13 metrics khoa học
│   ├── metrics_calculator.py                     # Module tính toán các độ đo toán học
│   ├── infer_stage1_segformer.py                 # Suy luận Stage 1 trên tập static test
│   └── infer_and_compare_stages.py               # Suy luận đối sánh & xuất ảnh 4-Panel Grid
├── utils/                                        # Tiện ích tiền xử lý, crawl & trích xuất frame
├── notebooks/                                    # Jupyter Notebooks phân tích dữ liệu & EDA
├── logs/                                         # Báo cáo JSON & thống kê mô hình
└── README.md
```

---

## ⚡ Bảng Tra Cứu Toàn Bộ Lệnh Thực Thi Trên Modal GPU

Tất cả các tác vụ đều được thiết kế chạy trực tiếp trên **Modal Cloud GPU (A10G)** gắn Volume độc lập `fire_smoke_dataset_v2`:

| Bước | Tác vụ | Lệnh thực thi Terminal | GPU | Kết quả đầu ra |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Setup & Đối soát Volume** | `modal run src/modal_setup_volume.py` | CPU | Giải nén dữ liệu & kiểm tra 0 missing files |
| **2** | **Huấn luyện Stage 1** | `modal run src/train_stage1_segformer.py` | A10G | `/data/checkpoints/segformer_spatial_best.pth` |
| **3** | **Huấn luyện Stage 2** | `modal run --detach src/train_stage2_timesformer.py` | A100 | `/data/checkpoints/timesformer_spatiotemporal_best.pth` |
| **4** | **Đánh giá Metrics Stage 1** | `modal run src/evaluate_all_metrics.py --stage 1` | A10G | 13 chỉ số (ACC, TPR, FAR, F1, IoU, FPS...) |
| | **Đánh giá Metrics Stage 2** | `modal run src/evaluate_all_metrics.py --stage 2` | A10G | 13 chỉ số (kèm Detection Latency) |
| | **Đánh giá đối sánh cả 2** | `modal run src/evaluate_all_metrics.py --stage both` | A10G | Bảng đối sánh trực tiếp Stage 1 vs Stage 2 |
| **5** | **Suy luận & Trực quan hóa** | `modal run src/infer_and_compare_stages.py` | A10G | Tải ảnh 4-Panel về `docs/inference_comparisons/` |
| | **Tùy chọn số clip suy luận** | `modal run src/infer_and_compare_stages.py --max-clips 10` | A10G | Tải 10 ảnh đối sánh phân giải cao |
| | **Suy luận toàn bộ tập Test** | `modal run src/infer_and_compare_stages.py --max-clips 18` | A10G | Tải toàn bộ 18 ảnh clips test |

---

## 🎯 Các Độ Đo Đánh Giá (Evaluation Metrics)

Hệ thống đánh giá toàn diện gồm 4 nhóm độ đo:
1. **Binary Classification Metrics**: Overall Accuracy (ACC), True Positive Rate (TPR/Sensitivity), False Positive Rate (FPR/FAR), True Negative Rate (TNR/Specificity), False Negative Rate (FNR/Miss Rate), Precision (PPV), False Discovery Rate (FDR), F1-Score, Matthews Correlation Coefficient (MCC).
2. **Localization Metrics**: Localization True Positive Rate (LTPR), Localization Positive Predictive Value (LPPV), Mean Similarity $S$ (Sørensen-Dice regional similarity).
3. **Pixel Semantic Segmentation Metrics**: mIoU, $\text{IoU}_{\text{fire}}$, $\text{IoU}_{\text{smoke}}$, $\text{Dice}_{\text{fire}}$, $\text{Dice}_{\text{smoke}}$.
4. **Throughput & Speed**: Latency (ms/frame) và FPS (Frames Per Second).
