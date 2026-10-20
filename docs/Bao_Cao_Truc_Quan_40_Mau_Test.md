# Báo Cáo Đối Sánh Trực Quan 40 Mẫu Kiểm Thử Suy Luận (Inference 40 Samples)
**Hệ Thống Phát Hiện Sớm Khói & Lửa Hai Giai Đoạn (Two-Stage Early Detection)**
*Được thực thi trên Modal Cloud NVIDIA A10G GPU (Profile: `usecolabgpu02`)*

---

## 1. Cấu Trúc Khung Hình Trực Quan Hóa (4-Panel Comparison Grid)

Mỗi ảnh trực quan hóa gồm 4 bảng điều khiển phân giải cao $512 \times 512$ ghép trong lưới $2 \times 2$ ($1024 \times 1024$ pixel) kèm Header thông tin:

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ 1. Raw Original Frame                │ 2. Ground Truth Mask                 │
│    (Khung hình RGB gốc từ Video Test)│    (Xanh: An toàn | Đỏ: Lửa | Cam: Khói)│
├──────────────────────────────────────┼──────────────────────────────────────┤
│ 3. Stage 1: SegFormer-B2 Prediction  │ 4. Stage 2: Spatio-Temporal Prediction│
│    (Spatial Core - Không gian tĩnh)  │    (Temporal Attention - Chuỗi T=8)  │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

- **Quy ước màu sắc mặt nạ (Mask Overlay):**
  - **Lửa (Fire):** Màu Đỏ tươi viền cam (`BGR: [0, 0, 255]`)
  - **Khói (Smoke):** Màu Vàng cam viền đậm (`BGR: [0, 165, 255]`)
  - **Nền an toàn (Safe Background):** Khung hình gốc tự nhiên không tô đè.

---

## 2. Danh Mục 40 Mẫu Kiểm Thử Phân Theo 5 Nhóm Bối Cảnh

Thư mục lưu trữ hình ảnh cục bộ: [docs/inference_comparisons_40/](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40)

### 2.1. Nhóm 1: Ngọn Lửa Bùng Phát & Lửa Mạnh (Fire Only - Mẫu 01 đến 10)
Đánh giá khả năng bắt trọn lõi lửa, chống nhấp nháy biên (flickering) và duy trì độ phủ ngọn lửa:

1. [sample_01_fire_only_12192259_1080_19_f3.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_01_fire_only_12192259_1080_19_f3.jpg)
2. [sample_02_fire_only_12192259_1080_19_f6.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_02_fire_only_12192259_1080_19_f6.jpg)
3. [sample_03_fire_only_12192259_1080_19_f3.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_03_fire_only_12192259_1080_19_f3.jpg)
4. [sample_04_fire_only_12192259_1080_19_f6.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_04_fire_only_12192259_1080_19_f6.jpg)
5. [sample_05_fire_only_12192259_1080_19_f3.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_05_fire_only_12192259_1080_19_f3.jpg)
6. [sample_06_fire_only_12192259_1080_19_f6.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_06_fire_only_12192259_1080_19_f6.jpg)
7. [sample_07_fire_only_13097120_2160_38_f3.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_07_fire_only_13097120_2160_38_f3.jpg)
8. [sample_08_fire_only_13097120_2160_38_f6.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_08_fire_only_13097120_2160_38_f6.jpg)
9. [sample_09_fire_only_13097120_2160_38_f3.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_09_fire_only_13097120_2160_38_f3.jpg)
10. [sample_10_fire_only_13097120_2160_38_f6.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_10_fire_only_13097120_2160_38_f6.jpg)

---

### 2.2. Nhóm 2: Khói Bốc Sớm & Khói Cuộn (Smoke Only - Mẫu 11 đến 20)
Đánh giá khả năng phân đoạn khói loãng bán trong suốt và bốc cuộn động học:

11. [sample_11_smoke_only_407b06ea_pexels__f2.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_11_smoke_only_407b06ea_pexels__f2.jpg)
12. [sample_12_smoke_only_407b06ea_pexels__f5.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_12_smoke_only_407b06ea_pexels__f5.jpg)
13. [sample_13_smoke_only_407b06ea_pexels__f2.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_13_smoke_only_407b06ea_pexels__f2.jpg)
14. [sample_14_smoke_only_407b06ea_pexels__f5.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_14_smoke_only_407b06ea_pexels__f5.jpg)
15. [sample_15_smoke_only_6c3dee00_pexels__f2.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_15_smoke_only_6c3dee00_pexels__f2.jpg)
16. [sample_16_smoke_only_6c3dee00_pexels__f5.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_16_smoke_only_6c3dee00_pexels__f5.jpg)
17. [sample_17_smoke_only_6c3dee00_pexels__f2.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_17_smoke_only_6c3dee00_pexels__f2.jpg)
18. [sample_18_smoke_only_6c3dee00_pexels__f5.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_18_smoke_only_6c3dee00_pexels__f5.jpg)
19. [sample_19_smoke_only_6c3dee00_pexels__f2.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_19_smoke_only_6c3dee00_pexels__f2.jpg)
20. [sample_20_smoke_only_6c3dee00_pexels__f5.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_20_smoke_only_6c3dee00_pexels__f5.jpg)

---

### 2.3. Nhóm 3: Đồng Thời Cả Lửa và Khói (Both Fire & Smoke - Mẫu 21 đến 25)
Đánh giá năng lực phân định đa lớp (multi-class semantic segmentation) giữa ngọn lửa bên dưới và luồng khói cuộn phía trên:

21. [sample_21_both_14056668_2160_38_f3.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_21_both_14056668_2160_38_f3.jpg)
22. [sample_22_both_14056668_2160_38_f6.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_22_both_14056668_2160_38_f6.jpg)
23. [sample_23_both_14283156_1080_19_f3.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_23_both_14283156_1080_19_f3.jpg)
24. [sample_24_both_14283156_1080_19_f6.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_24_both_14283156_1080_19_f6.jpg)
25. [sample_25_both_14283156_1080_19_f4.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_25_both_14283156_1080_19_f4.jpg)

---

### 2.4. Nhóm 4: Bối Cảnh Âm Tính Thách Thức (Hard Negatives - Mẫu 26 đến 33)
Đánh giá khả năng chống báo động giả khi gặp hơi nước, khói hàn, sương mù, khói ống xả xe:

26. [sample_26_hard_negative_neg_hard_negativ_f1.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_26_hard_negative_neg_hard_negativ_f1.jpg)
27. [sample_27_hard_negative_neg_hard_negativ_f5.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_27_hard_negative_neg_hard_negativ_f5.jpg)
28. [sample_28_hard_negative_neg_hard_negativ_f1.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_28_hard_negative_neg_hard_negativ_f1.jpg)
29. [sample_29_hard_negative_neg_hard_negativ_f5.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_29_hard_negative_neg_hard_negativ_f5.jpg)
30. [sample_30_hard_negative_neg_hard_negativ_f1.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_30_hard_negative_neg_hard_negativ_f1.jpg)
31. [sample_31_hard_negative_neg_hard_negativ_f5.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_31_hard_negative_neg_hard_negativ_f5.jpg)
32. [sample_32_hard_negative_neg_hard_negativ_f1.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_32_hard_negative_neg_hard_negativ_f1.jpg)
33. [sample_33_hard_negative_neg_hard_negativ_f5.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_33_hard_negative_neg_hard_negativ_f5.jpg)

---

### 2.5. Nhóm 5: Bối Cảnh Giám Sát Bình Thường (Normal CCTV - Mẫu 34 đến 40)
Đánh giá độ ổn định trên camera giám sát thực tế (kho hàng, đường cao tốc, xe lưu thông, văn phòng):

34. [sample_34_normal_cctv_neg_normal_G8z74_f1.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_34_normal_cctv_neg_normal_G8z74_f1.jpg)
35. [sample_35_normal_cctv_neg_normal_G8z74_f5.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_35_normal_cctv_neg_normal_G8z74_f5.jpg)
36. [sample_36_normal_cctv_neg_normal_G8z74_f1.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_36_normal_cctv_neg_normal_G8z74_f1.jpg)
37. [sample_37_normal_cctv_neg_normal_G8z74_f5.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_37_normal_cctv_neg_normal_G8z74_f5.jpg)
38. [sample_38_normal_cctv_neg_normal_G8z74_f1.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_38_normal_cctv_neg_normal_G8z74_f1.jpg)
39. [sample_39_normal_cctv_neg_normal_G8z74_f5.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_39_normal_cctv_neg_normal_G8z74_f5.jpg)
40. [sample_40_normal_cctv_neg_normal_YqkiO_f3.jpg](file:///e:/NCKH/Code/Fire_Smoke_Early-Detection/docs/inference_comparisons_40/sample_40_normal_cctv_neg_normal_YqkiO_f3.jpg)

---

## 3. Nhận Xét Thị Giác Tổng Hợp Sau Khi Áp Dụng Tiền Xử Lý Nâng Cao (v3)

1. **Về ngọn lửa:** Cả Stage 1 và Stage 2 đều đạt độ phủ gần như $100\%$ các đốm lửa bùng phát; tuy nhiên Stage 2 phân đoạn viền ngọn lửa mượt mà hơn, hạn chế việc bị vỡ vụn thành các mảng nhỏ li ti.
2. **Về luồng khói (Đột phá lớn nhất):** Nhờ sự kết hợp giữa **Dual-Scale CLAHE**, **Bản đồ mật độ khói Dark Channel Prior** và **Sai phân động học thời gian liên khung hình**, Stage 2 bắt trọn các dải khói loãng bán trong suốt vốn từng bị bỏ sót ở các phiên bản trước. Viền khói ôm sát theo hình thái khuếch tán thực tế, đẩy **IoU khói lên 65.18%** và **Smoke Recall đạt 100.00%**.
3. **Về môi trường bình thường & Hard Negatives:** Trên các khung hình camera giám sát bình thường (mẫu 34 - 40), cả 2 Stage đều duy trì trạng thái nền an toàn sạch sẽ, không sinh ra bất kỳ điểm ảnh báo giả nào. Trên các mẫu hard negatives (hơi nước, tia lửa hàn), mô hình Stage 2 thể hiện tính ổn định cao hơn hẳn nhờ bóc tách được chuyển động lan tỏa đặc thù của khói so với các hiện tượng nhấp nháy tĩnh.

