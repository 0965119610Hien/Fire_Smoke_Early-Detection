# So Sánh Các Chiến Lược Sinh Dữ Liệu Lửa/Khói Bằng Diffusion Model (NCKH)

> **Đặt vấn đề:** Khi sử dụng mô hình Diffusion để sinh dữ liệu tổng hợp (Synthetic Data), liệu việc sinh lửa/khói từ một video **nền trống (Pure Background)** có đạt độ tự nhiên vật lý hay không? Có nên sử dụng **video có sẵn ngọn lửa nhỏ** để kế thừa quy tắc hình học & ánh sáng không? Phương án nào tối ưu hơn cho NCKH?

---

## 1. Phân Tích Kỹ Thuật 2 Phương Pháp Sinh Dữ Liệu

---

### Phương Án A: Sinh Lửa/Khói Từ Video Nền Trống (Pure Background Inpainting)

* **Cơ chế:** Lấy một video bối cảnh trống (kho hàng, phòng khách, bãi xe), vẽ một Bounding Box Mask tại vị trí mong muốn, sau đó dùng Stable Diffusion Inpainting / AnimateDiff để sinh ra Lửa/Khói hoàn toàn từ hư vô (from scratch).

```
Video Nền Trống + Bounding Box Mask ──► Diffusion Inpainting ──► Video Lửa/Khói Tổng Hợp
```

#### 🟡 Đánh giá Độ tự nhiên (Realism):
- **Không gian (Spatial):** Nếu không có ControlNet Depth/Normal, ngọn lửa rất dễ bị sinh **"lơ lửng" giữa không trung** hoặc không bám vào bề mặt vật liệu cháy (sàn gỗ, thùng carton).
- **Ánh sáng (Global Illumination):** Rất khó mô phỏng ánh sáng hắt (Fire Glow). Một ngọn lửa bùng lên trong phòng tối bắt buộc phải chiếu ánh sáng da cam lên các bức tường xung quanh. Inpainting thông thường chỉ thay đổi pixel trong vùng Mask mà quên mất hiệu ứng hắt sáng ra môi trường xung quanh.
- **Thời gian (Temporal Consistency):** Khi sinh video $> 2$ giây, các mô hình Video Diffusion thuần túy dễ bị hiện tượng **Flickering (nhấp nháy)** hoặc ngọn lửa biến dạng không theo quy luật vật lý.

---

### Phương Án B: Biến Đổi / Bùng Phát Từ Video Lửa Nhỏ Có Sẵn (Fire Evolution / Guided Generation)

* **Cơ chế:** Sử dụng video thực tế đã có sẵn đốm lửa nhỏ hoặc vệt khói mỏng (đã có sẵn quy tắc chiếu sáng, góc quay và sự cuộn nhiệt tự nhiên). Sử dụng Diffusion (ControlNet Tile + AnimateDiff / SVD) để **"phát triển" (evolve/expand)** đốm lửa nhỏ đó thành đám cháy bùng phát lớn hơn hoặc khói đặc hơn.

```
Video Lửa Nhỏ Thực Tế (Có ánh sáng hắt & chuyển động gốc)
                        │
                        ▼
      Trích xuất ControlNet (Depth / Tile) & Motion Features
                        │
                        ▼
      Diffusion Guided Expansion (Prompt: "expanding fire, heavy smoke plume")
                        │
                        ▼
      Video Diễn Tiến Bùng Phát Tự Nhiên 100%
```

#### 🟢 Đánh giá Độ tự nhiên (Realism):
- **Vật lý & Hình học (Physics & Geometry):** **RẤT TỰ NHIÊN.** Ngọn lửa gốc đã có sẵn chân lửa, cấu trúc cuộn nhiệt (thermal convection) và ánh sáng hắt vào môi trường. Mô hình Diffusion chỉ đóng vai trò khuếch đại (magnify) các yếu tố này lên.
- **Thời gian (Temporal Consistency):** Kế thừa luồng chuyển động (Motion Vector) từ video thực tế nên video sinh ra giữ được độ mượt mà, không bị giật lag hay biến dạng kỳ dị.

---

## 2. Bảng So Sánh Chi Tiết Ưu & Nhược Điểm

| Tiêu chí so sánh | Phương án A: Sinh từ Nền Trống (Pure Inpainting) | Phương án B: Bùng phát từ Lửa Nhỏ (Guided Evolution) |
| :--- | :--- | :--- |
| **Độ tự nhiên về Ánh sáng & Đổ bóng** | 🔴 Kém (Thiếu hiệu ứng hắt sáng lên môi trường xung quanh Mask). | 🟢 **Rất cao** (Kế thừa hoàn hảo hiệu ứng hắt sáng & phản xạ từ ngọn lửa gốc). |
| **Độ tự nhiên về Chuyển động (Motion)** | 🔴 Dễ bị hiện tượng nhấp nháy (Flickering) qua các frames. | 🟢 **Mượt mà** (Dựa trên luồng chuyển động vật lý có sẵn của video gốc). |
| **Độ chính xác Bounding Box** | 🟢 **100% Tự động** (Do ta chủ động vẽ vị trí Mask trước). | 🟡 **Tự động theo tỷ lệ** (Mask tự động mở rộng theo hệ số phóng đại $k$). |
| **Độ linh hoạt bối cảnh (Domain)** | 🟢 Rất cao (Thả lửa vào bất kỳ video bối cảnh nào). | 🟡 Phụ thuộc vào tập video lửa nhỏ ban đầu. |
| **Giá trị đóng góp NCKH (Novelty)** | ⭐⭐⭐ (Ứng dụng Inpainting thông thường). | ⭐⭐⭐⭐⭐ (**Controllable Fire Evolution Synthesis**). |

---

## 3. Khuyến Nghị: PHƯƠNG ÁN NÀO HAY HƠN CHO NCKH?

> **Kết luận:** **PHƯƠNG ÁN B (Biến đổi / Bùng phát từ Video Lửa Nhỏ) HAY HƠN VÀ ĐƯỢC ĐÁNH GIÁ CAO HƠN RẤT NHIỀU TRONG NCKH!**

### Lý do Khoa học:

1. **Giải quyết đúng bài toán Cảnh báo sớm (Early Warning):** Trong thực tế NCKH, bài toán cốt lõi là nhận diện được ngọn lửa từ khi nó còn nhỏ và **dự đoán/mô phỏng được xu hướng bùng phát của nó**. Phương án B mô phỏng chính xác bản chất vật lý này.
2. **Đảm bảo tính chân thực cho Video Tracking:** Một mô hình Tracking cần theo dõi các đường nét liên tục của đối tượng. Nếu dùng Phương án A, sự nhấp nháy (flickering) của Diffusion sẽ làm hỏng thuật toán Tracking. Phương án B giữ được tính liên tục của Track ID.
3. **Đóng góp Mô hình Hybrid (Hybrid Generative Pipeline):** Bạn có thể trình bày trong bài báo khoa học về một quy trình kết hợp:  
   $$\text{Input Video (Lửa nhỏ)} \xrightarrow{\text{ControlNet-Tile}} \text{Latent Diffusion} \xrightarrow{\text{Motion Guidance}} \text{Synthetic Hazard Video}$$  
   Đây là một bài báo/báo cáo có hàm lượng khoa học cực kỳ ấn tượng.
