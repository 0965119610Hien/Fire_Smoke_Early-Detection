# 📊 Phân Tích Dataset Video — Fire & Smoke Detection

**Đường dẫn:** `d:\zip_dataset\Video\Video\`  
**Ngày phân tích:** 17/07/2026  
**Mục tiêu:** Phát hiện sớm lửa & khói (Early Fire & Smoke Detection)

---

## 🗂️ Tổng Quan Dataset

| Thông số | Giá trị |
|---|---|
| Số nhãn (labels) | **3** |
| Tổng số video clip | **1.001** |
| Tổng dung lượng (Video/) | **~9.69 GB** |
| Tập Test riêng | **20 videos, ~1.14 GB** |

---

## 📁 Cấu Trúc Thư Mục

```
d:\zip_dataset\Video\
├── Video/
│   ├── 0_normal_noise/      → 276 clips  (1.71 GB)
│   ├── 1_controlled_fire/   → 402 clips  (5.95 GB)
│   └── 2_uncontrolled_hazard/ → 323 clips (1.93 GB)
└── test/                    → 20 clips   (1.14 GB)
```

---

## 🏷️ Nhãn 0 — `0_normal_noise` (Bình thường / Nhiễu)

> **Ý nghĩa:** Các video **KHÔNG có lửa/khói thực sự**, nhưng chứa các hiện tượng **dễ gây nhầm lẫn** (false positive) với hệ thống phát hiện. Đây là lớp "âm tính" (negative) của bài toán.

### Thống kê cơ bản

| Chỉ số | Giá trị |
|---|---|
| Tổng số clip | **276** |
| Số video gốc | **141 nguồn** |
| Dung lượng | **1.71 GB** |
| Kích thước TB/clip | **6.35 MB** |
| Kích thước min–max | **0.20 – 39.14 MB** |
| Nguồn thu thập | Pexels (159), YouTube (117) |

### Các loại tình huống trong nhãn 0

| Loại tình huống | Số clip | Mô tả |
|---|---|---|
| 🌫️ **Dust Particles** (hạt bụi) | **177** | Bụi bay trong không khí, hạt lơ lửng — dễ bị nhầm thành khói |
| 📷 **Optical Noise** (nhiễu quang học) | **53** | Nhiễu hình ảnh, hiệu ứng ánh sáng, flare, grain — gây nhầm lẫn thị giác |
| 🚶 **Daily Activity FP** (hoạt động sinh hoạt) | **16** | Sinh hoạt hằng ngày có khả năng gây cảnh báo sai (false positive) |
| 🏠 **Residential Fire** (chứa lửa dân dụng) | **10** | Một số clip lửa dân dụng được xếp vào nhóm normal (có thể để học phân biệt) |
| 💨 **Heavy Smoke** (khói đặc) | **8** | Khói không nguy hiểm hoặc trong bối cảnh kiểm soát |
| 🚨 **Alarm/Warning Lights** (đèn báo động) | **6** | Đèn nhấp nháy, cảnh báo — có thể gây nhầm lẫn |
| 🔥 **Visible Flame** (lửa nhỏ/kiểm soát) | **2** | Lửa nhìn thấy nhưng được kiểm soát |
| ⚡ **Welding Sparks** (tia hàn) | **2** | Tia lửa hàn xì |
| 🚬 **Cigarette Smoke** (khói thuốc) | **2** | Khói từ thuốc lá |

> [!NOTE]
> **Mục tiêu của nhãn này:** Dạy model phân biệt các hiện tượng **trông giống khói/lửa nhưng không nguy hiểm**, giảm thiểu false positive trong thực tế.

---

## 🏷️ Nhãn 1 — `1_controlled_fire` (Lửa/Khói có kiểm soát)

> **Ý nghĩa:** Các video chứa **lửa hoặc khói nhưng ở mức độ kiểm soát được**, không gây nguy hiểm tức thì. Đây là lớp **trung gian** (intermediate/controlled hazard).

### Thống kê cơ bản

| Chỉ số | Giá trị |
|---|---|
| Tổng số clip | **402** |
| Số video gốc | **357 nguồn** |
| Dung lượng | **5.95 GB** |
| Kích thước TB/clip | **15.15 MB** |
| Kích thước min–max | **0.07 – 399.92 MB** |
| Nguồn thu thập | Pexels (228), Other (143), YouTube (31) |

### Các loại tình huống trong nhãn 1

| Loại tình huống | Số clip | Mô tả |
|---|---|---|
| 🎬 **Unknown / Raw** (video thô) | **143** | Nhiều video Pexels HD không có tiền tố category (video số hiệu Pexels dạng `12XXXXXX_...fps.mp4`) — đây là raw footage chứa lửa/khói kiểm soát |
| 🚬 **Cigarette Smoke** (khói thuốc lá) | **78** | Khói từ điếu thuốc — visible nhưng không nguy hiểm |
| ⚡ **Welding Sparks** (tia lửa hàn) | **76** | Hàn xì, cắt kim loại — lửa có kiểm soát, tia sáng mạnh |
| 🚶 **Daily Activity** (hoạt động thường ngày) | **36** | Nấu ăn, sinh hoạt có lửa/khói thông thường |
| 🍳 **Cooking Smoke** (khói nấu ăn) | **32** | Khói từ chảo, bếp, nướng — khói dày nhưng an toàn |
| 💨 **Heavy Smoke** (khói đặc, kiểm soát) | **28** | Khói nhiều nhưng trong môi trường kiểm soát (studio, biểu diễn...) |
| 🌫️ **Dust Particles** (bụi hạt) | **4** | Bụi lớn, sương mù nhẹ |
| 🌁 **Diffused Smoke** (khói khuếch tán) | **3** | Khói lan tỏa nhẹ, không tập trung |
| 🔥 **Visible Flame** (ngọn lửa nhỏ) | **1** | Ngọn lửa nhỏ, kiểm soát |
| 🌅 **Early Smoke** (khói sớm) | **1** | Khói ở giai đoạn đầu, mờ nhạt |

> [!IMPORTANT]
> **Đặc điểm nhãn này:** Chiếm dung lượng lớn nhất (~5.95 GB), có nhiều video độ phân giải cao 4K (3840x2160). Đây là lớp khó nhất vì ranh giới với nhãn 2 không luôn rõ ràng.

---

## 🏷️ Nhãn 2 — `2_uncontrolled_hazard` (Nguy hiểm/Mất kiểm soát)

> **Ý nghĩa:** Các video chứa **lửa hoặc khói THỰC SỰ nguy hiểm, mất kiểm soát** — cháy nhà, khói đen dày đặc từ đám cháy, ngọn lửa lan rộng. Đây là lớp **dương tính nguy hiểm** (critical positive).

### Thống kê cơ bản

| Chỉ số | Giá trị |
|---|---|
| Tổng số clip | **323** |
| Số video gốc | **99 nguồn** |
| Dung lượng | **1.93 GB** |
| Kích thước TB/clip | **6.12 MB** |
| Kích thước min–max | **0.39 – 675.90 MB** |
| Nguồn thu thập | YouTube (243), Other (47), Pexels (33) |

### Các loại tình huống trong nhãn 2

| Loại tình huống | Số clip | Mô tả |
|---|---|---|
| 💨 **Heavy Smoke** (khói đặc nguy hiểm) | **130** | Khói dày, đen, mất kiểm soát từ đám cháy thực |
| 🏠 **Residential Fire** (cháy nhà dân) | **52** | Cháy nhà, tòa nhà dân dụng thực tế |
| 🎬 **Unknown / Raw** (video thô) | **47** | Video thực tế không rõ nguồn gốc (cháy thực tế) |
| 🌁 **Diffused Smoke** (khói lan tràn) | **38** | Khói lan tỏa rộng từ đám cháy lớn |
| 🔥 **Visible Flame** (ngọn lửa lớn) | **35** | Ngọn lửa rõ ràng, dữ dội, cháy thực |
| 🌅 **Early Smoke** (khói đầu đám cháy) | **9** | Khói ở giai đoạn đầu — **rất quan trọng cho early detection** |
| 🚶 **Daily Activity** (bối cảnh dân sinh) | **7** | Hoạt động trong hoặc gần đám cháy thực |
| 🌫️ **Dust Particles** | **5** | Bụi từ đổ nát, sập sàn trong đám cháy |

> [!CAUTION]
> **Đặc điểm nhãn này:** Tỉ lệ clip/nguồn rất cao (323 clips từ 99 nguồn), có nghĩa là mỗi video gốc được cắt thành nhiều đoạn để tối đa hóa dữ liệu. Video thực tế từ YouTube chiếm đa số (75%).

---

## 📊 So Sánh Tổng Hợp Giữa 3 Nhãn

| Tiêu chí | Label 0 (Normal) | Label 1 (Controlled) | Label 2 (Hazard) |
|---|---|---|---|
| Số clip | 276 | **402** | 323 |
| Số nguồn gốc | 141 | **357** | 99 |
| Tỉ lệ clip/nguồn | 1.96 | 1.13 | **3.26** |
| Dung lượng | 1.71 GB | **5.95 GB** | 1.93 GB |
| Size TB/clip | 6.35 MB | **15.15 MB** | 6.12 MB |
| Nguồn chủ yếu | Pexels | Pexels | **YouTube** |
| Đặc trưng chính | Bụi, Nhiễu quang học | Khói thuốc, Tia hàn | Khói đặc, Cháy nhà |

---

## 🔍 Phân Tích Nguồn Thu Thập

| Nguồn | Label 0 | Label 1 | Label 2 | Tổng |
|---|---|---|---|---|
| **Pexels** | 159 | 228 | 33 | **420** |
| **YouTube** | 117 | 31 | 243 | **391** |
| **Other** (Adobe Stock, etc.) | 0 | 143 | 47 | **190** |

> [!TIP]
> **Nhận xét:** Label 2 (nguy hiểm thực) chủ yếu lấy từ YouTube (video thực tế đám cháy), trong khi Label 0 và 1 cân bằng hơn giữa Pexels (stock footage) và YouTube.

---

## 📁 Tập Test

| Thông số | Giá trị |
|---|---|
| Vị trí | `d:\zip_dataset\Video\test\` |
| Số file | **20 video** |
| Dung lượng | **1.14 GB** |
| Đặc điểm | Đa số video lửa cháy thực tế (`fire1_001.mp4` đến `fire12_002.mp4`) |
| File lớn nhất | `fire3_001.mp4` (~319 MB), `fire3_002.mp4` (~182 MB) |

---

## 💡 Nhận Xét & Đề Xuất

### Mất cân bằng dữ liệu
- Label 1 có **nhiều clip nhất** (402) nhưng ranh giới với Label 2 cần chú ý
- Label 2 có **ít nguồn nhất** (99 nguồn) nhưng được cắt nhiều clip nhất (tỉ lệ 3.26)
- Cần xem xét oversampling hoặc augmentation cho Label 0 và 2

### Thách thức phân loại
- **Khói thuốc (Label 1) vs Khói sớm đám cháy (Label 2):** Rất khó phân biệt về mặt thị giác
- **Tia hàn (Label 1) vs Lửa thực (Label 2):** Màu sắc tương đồng
- **Dust Particles (Label 0) vs Early Smoke (Label 2):** Đây là thách thức lớn nhất cho early detection

### Điểm mạnh của dataset
- Đa dạng nguồn thu thập (Pexels, YouTube, stock)
- Phủ nhiều trường hợp edge case (alarm lights, welding sparks trong nhãn Normal)
- Có tập test riêng biệt với video thực tế
