# Thiết Kế Kiến Trúc Dữ Liệu: Event-Driven Tracking Thay Thế Cho Nhãn Video Cố Định (NCKH)

> **Câu hỏi trung tâm:** Khi đã có nhãn Tracking (Bounding Box + Track ID per frame), có cần chia video thành 3 nhãn cố định (`0_normal_noise`, `1_controlled_fire`, `2_uncontrolled_hazard`) nữa không?  
> **Câu trả lời khoa học:** **KHÔNG CẦN THIẾT và KHÔNG NÊN.** Việc chuyển sang cấu trúc dữ liệu diễn tiến thời gian (**Process-based / Event-driven**) sẽ làm tăng tính mới khoa học, tính giải thích được (Explainable AI) và độ chính xác của hệ thống.

---

## 1. So Sánh Hai Hướng Tiếp Cận

| Tiêu chí so sánh | Hướng 1: Nhãn Video Cố Định (3 Classes Cũ) | Hướng 2: Tracking Process & State Engine (Đề xuất Mới) |
| :--- | :--- | :--- |
| **Bản chất gán nhãn** | Gán 1 nhãn duy nhất cho toàn bộ video clip ngắn (3-15s). | Gán nhãn đối tượng: `0: Fire`, `1: Smoke` kèm `Track ID` theo chuỗi thời gian. |
| **Tính chủ quan (Ambiguity)** | 🔴 **Rất cao**: Ranh giới giữa "lửa bếp" (controlled) và "chập điện cháy bếp" (hazard) trong 5s đầu là giống hệt nhau. | 🟢 **Không có**: Chỉ ghi nhận sự vật thực tế (đâu là lửa, đâu là khói, diễn ra ở đâu). |
| **Tính giải thích được (Explainability)** | 🔴 **Hộp đen (Black-box)**: Mô hình báo "Hazard 85%" nhưng không giải thích được lý do tại sao. | 🟢 **Minh bạch**: "Cảnh báo Hazard vì ngọn lửa Track #1 có diện tích tăng 200% trong 3 giây". |
| **Khả năng ứng dụng thực tế** | 🟡 Kém linh hoạt khi áp dụng vào camera RTSP chạy liên tục 24/7. | 🟢 Rất cao: Chạy online mượt mà trên luồng video kéo dài không giới hạn. |

---

## 2. Tại Sao Hướng Tiếp Cận Mới Lại Ưu Việt Hơn Cho NCKH?

### 2.1 Bản chất vật lý của sự cố cháy nổ
Quá trình cháy tự nhiên luôn diễn ra theo chuỗi sự kiện tuyến tính theo thời gian:
$$\text{Bình thường / Nhiễu} \longrightarrow \text{Khói sớm (Early Smoke)} \longrightarrow \text{Bùng lửa (Ignition)} \longrightarrow \text{Cháy lan (Spreading / Hazard)}$$

Nếu bạn cắt 1 đoạn video 5 giây ở giữa giai đoạn "Bùng lửa" và bắt mô hình đoán đó là `Controlled` hay `Hazard` thì mô hình rất dễ đoán sai. Nhưng nếu mô hình **Track được ngọn lửa đó từ $t=0s$ đến $t=5s$** và thấy diện tích của nó tăng vọt, nó sẽ kết luận là `Hazard` với độ chính xác $100\%$.

---

### 2.2 Động học Tracking tự động suy ra trạng thái nguy hiểm (Inferred Hazard State)

Khi bạn đã có thông tin Tracking (`track_id`, $x, y, w, h, t$), cấp độ nguy hiểm sẽ được **SUY LUẬN TỰ ĐỘNG** bằng các chỉ số động học (Kinematic Metrics) thay vì bắt mô hình học vẹt nhãn video:

```
                                  ┌───────────────────────────┐
                                  │   DETECTION & TRACKING    │
                                  │   - Class: Fire / Smoke   │
                                  │   - Track ID: #1, #2...   │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │    STATE ENGINE RULES     │
                                  └─────────────┬─────────────┘
                                                │
          ┌─────────────────────────────────────┼─────────────────────────────────────┐
          │                                     │                                     │
          ▼                                     ▼                                     ▼
  (Không detect / Tích tắc < 2s)       (Track ID đứng yên / Vùng an toàn)    (Diện tích tăng vọt / Lan rộng)
┌──────────────────┐                  ┌──────────────────┐                  ┌──────────────────┐
│   0. NORMAL      │                  │  1. CONTROLLED   │                  │    2. HAZARD     │
│   (Bình thường)  │                  │ (Có kiểm soát)   │                  │ (NGUY HIỂM CAO)  │
└──────────────────┘                  └──────────────────┘                  └──────────────────┘
```

#### Công thức suy luận các trạng thái:

1. **Trạng thái 0: Normal / Noise (Bình thường / Nhiễu)**
   - Không phát hiện được Track ID nào của Lửa/Khói.
   - HOẶC Track ID xuất hiện trong thời gian $t < 2.0\text{s}$ rồi biến mất hoàn toàn (Đặc trưng của hơi nước, khói thuốc lá, hoặc vệt sáng đèn xe).

2. **Trạng thái 1: Controlled Fire / Smoke (Có kiểm soát)**
   - Track ID phát hiện Lửa/Khói nhưng tốc độ tăng trưởng diện tích xấp xỉ 0: 
     $$\frac{\Delta \text{Area}}{\Delta t} \approx 0$$
   - Ngọn lửa đứng yên tại một vị trí cố định (Lửa bếp gas, ngọn nến, lửa hàn kim loại).

3. **Trạng thái 2: Uncontrolled Hazard (Báo động cháy nguy hiểm)**
   - Tốc độ tăng trưởng diện tích của Track ID vượt ngưỡng:
     $$\frac{\Delta \text{Area}}{\Delta t} > 50\% \quad \text{trong } 3\text{ giây}$$
   - Tốc độ di chuyển tâm (Velocity) của Track ID lớn (ngọn lửa đang lan rộng sang vị trí mới).
   - Xuất hiện đồng thời nhiều Track ID Lửa và Khói bùng phát cùng lúc.

---

## 3. Cấu Trúc Dataset Đề Xuất Mới (Process-based Scenario Dataset)

Thay vì chia thư mục dataset thành 3 nhãn video cứng, bạn nên tái cấu trúc dataset thành **3 nhóm kịch bản (Scenarios)** để huấn luyện mô hình Detection/Tracking:

```
dataset_tracking/
├── scenario_1_normal_background/        # 1. Kịch bản nhiễu & sinh hoạt
│   ├── video_steam_vapor_01.mp4         #    (Hơi nước, khói thuốc, đèn xe, bãi xe...)
│   ├── video_welding_noise_02.mp4
│   └── ...
├── scenario_2_controlled_fire/          # 2. Kịch bản cháy an toàn / cố định
│   ├── video_gas_stove_01.mp4           #    (Bếp gas, lò nướng, hàn cắt kim loại...)
│   └── ...
└── scenario_3_fire_evolution/           # 3. Kịch bản diễn tiến cháy thực tế
    ├── video_warehouse_fire_01.mp4      #    (Quá trình: Bình thường -> Khói mỏng -> Lửa -> Bùng phát)
    └── ...
```

---

## 4. Tóm Tắt Đóng Góp Khoa Học Khi Sử Dụng Hướng Tiếp Cận Này

1. **Kiến trúc mô hình hiện đại hơn:** Không bắt mô hình làm 2 nhiệm vụ mâu thuẫn (vừa phân loại video vừa detect), mà tập trung 100% năng lượng mô hình vào bài toán **Object Detection & Tracking**.
2. **Minh bạch và Đánh giá chuẩn xác (Explainable AI):** Tránh được lỗi chủ quan của người gán nhãn khi phân định video nào là "controlled" hay "hazard".
3. **Đóng góp thuật toán State Engine:** Đề xuất một module suy luận logic (Decision State Machine) dựa trên các tham số hình học và thời gian của Track ID — đây là một đóng góp rất hay để trình bày trong bài báo khoa học!
