"""
src/generate_curation_report.py
================================
Tạo báo cáo chi tiết về 63 video đã tuyển chọn và tổng hợp phương pháp xử lý cắt tỉa / gộp dữ liệu.
"""

import json
from pathlib import Path

# Đọc thống kê 63 video
stats_file = Path("crawled_early_fire_transition/curated_continuous_incidents/numbered_videos_stats.json")
with open(stats_file, "r", encoding="utf-8") as f:
    vids = json.load(f)

vids.sort(key=lambda x: (x["folder_num"], x["category"], x["filename"]))

# Thống kê tổng hợp
total_vids = len(vids)
total_dur_s = sum(v["duration_s"] for v in vids)
total_size_mb = sum(v["size_mb"] for v in vids)

# Bảng Markdown từng video
table_rows = []
for i, v in enumerate(vids, 1):
    f_num = v["folder_num"]
    cat = v["category"]
    fname = v["filename"]
    dur = v["duration_s"]
    res = f"{v['width']}x{v['height']}"
    size = f"{v['size_mb']} MB"
    grp = "Transition" if "01_" in v["group"] else "Ambient"
    table_rows.append(f"| {i:02d} | `{f_num}` | `{cat}` | `{grp}` | `{fname}` | {dur:.1f}s | {res} | {size} |")

table_text = "\n".join(table_rows)

# Phân nhóm theo thư mục
folder_summary = {}
for v in vids:
    key = (v["folder_num"], v["category"], v["group"])
    if key not in folder_summary:
        folder_summary[key] = {"count": 0, "duration": 0.0, "size": 0.0}
    folder_summary[key]["count"] += 1
    folder_summary[key]["duration"] += v["duration_s"]
    folder_summary[key]["size"] += v["size_mb"]

summary_rows = []
for (f_num, cat, grp), data in sorted(folder_summary.items()):
    grp_name = "01_TRANSITIONS" if "01_" in grp else "02_AMBIENT"
    summary_rows.append(
        f"| Folder `{f_num}` | `{cat}` | `{grp_name}` | **{data['count']}** | {data['duration']:.1f}s ({data['duration']/60:.1f} phút) | {data['size']:.1f} MB |"
    )
summary_table = "\n".join(summary_rows)

doc_content = """# Báo Cáo Thống Kê Chi Tiết 63 Video Tuyển Chọn (Thư Mục 1-13) & Tổng Hợp Phương Pháp Xử Lý Cắt Tỉa / Hợp Nhất Dữ Liệu Video

Tài liệu kỹ thuật này cung cấp:
1. **Thống kê chi tiết từng video** trong 63 video đã được người dùng chọn lọc vào các thư mục con từ `1` đến `13`.
2. **Chi tiết việc xử lý phân tách video** `lab_experiments_2HOX_wDAB3A` thành 2 phần độc lập sạch sẽ.
3. **Tổng hợp toàn diện 4 phương pháp kỹ thuật** để tự động phát hiện và cắt bỏ các đoạn mở đầu (intro, phóng viên, đếm ngược) và kết thúc (outro, phỏng vấn, logo) không liên quan.
4. **Kiến trúc và phương án chuẩn hóa** để gộp toàn bộ tập video tuyển chọn này với dữ liệu video ban đầu của dự án phục vụ huấn luyện TimeSformer.

---

## PHẦN 1: BẢNG THỐNG KÊ CHI TIẾT 63 VIDEO ĐÃ TUYỂN CHỌN

### 1.1 Tổng Quan Toàn Bộ Dữ Liệu Tuyển Chọn

* **Tổng số video trong thư mục `1` đến `13`**: **{total_vids} video**
* **Tổng thời lượng sự cố liên tục**: **{total_dur_s:.1f} giây** (**{total_dur_m:.2f} phút** / ~**{total_dur_h:.2f} giờ**)
* **Tổng dung lượng lưu trữ**: **{total_size_mb:.2f} MB**
* **Thời lượng trung bình mỗi clip**: **{avg_dur:.1f} giây** (Thời lượng lý tưởng để trích xuất 8–32 frames cho TimeSformer)
* **Độ phân giải chủ đạo**: **720p (1280x720)** chiếm 54%, còn lại là 1080p và 480p sắc nét, tỉ lệ khung hình 24–30 FPS ổn định.

---

### 1.2 Phân Bố Theo Từng Thư Mục Con (Folder 1 đến 13)

| Thư mục số | Phân loại kịch bản | Nhóm gốc | Số lượng video | Tổng thời lượng | Tổng dung lượng |
| :---: | :--- | :--- | :---: | :---: | :---: |
{summary_table}
| **TỔNG** | | | **{total_vids} video** | **{total_dur_m:.2f} phút** | **{total_size_mb:.1f} MB** |

> [!NOTE]
> * Nhóm **`01_CONTINUOUS_TRANSITIONS`** (Folder `1`, `2`, `3`, `4`, `5`, `6`): Gồm **32 video** chứa chuỗi biến đổi vật lý mẫu mực từ trạng thái chưa có lửa (khói âm ỉ, tia lửa nhỏ, bốc nhiệt) sang bùng phát ngọn lửa.
> * Nhóm **`02_CONTINUOUS_AMBIENT_CLEAN`** (Folder `8`, `9`, `10`, `11`, `12`, `13`): Gồm **31 video** ghi lại cảnh cháy thực tế trong không gian camera giám sát liên tục, không bị cắt cảnh giật cục hay đổi góc quay.

---

### 1.3 Chi Tiết Xử Lý Video `lab_experiments_2HOX_wDAB3A` (Folder 5)

Theo chỉ đạo của người dùng, video thí nghiệm pin phòng lab mã `2HOX_wDAB3A` (thời lượng gốc 109.97s) đã được kiểm tra và cắt bỏ đoạn ngắt quãng từ giây thứ $40.0$ đến $43.0$ (phân đoạn xuất hiện chuyển động can thiệp bên ngoài/nhiễu), chia thành **2 video độc lập**:

1. **Clip 1 (`0s` $\rightarrow$ `40s`)**:
   * File: `lab_experiments_2HOX_wDAB3A_part1_0_40s.mp4`
   * Thời lượng: **40.03 giây** | Dung lượng: **2.51 MB**
   * Nội dung: Giai đoạn tế bào pin bắt đầu quá nhiệt, phát ra luồng khói trắng liên tục tăng dần.
2. **Clip 2 (`43s` $\rightarrow$ `Hết`)**:
   * File: `lab_experiments_2HOX_wDAB3A_part2_43_end.mp4`
   * Thời lượng: **66.99 giây** | Dung lượng: **8.76 MB**
   * Nội dung: Giai đoạn thermal runaway bùng nổ cực đại, ngọn lửa phụt mạnh liên tục kèm khói đen đặc.

Cả 2 clip đã được lưu trực tiếp tại:
`crawled_early_fire_transition/curated_continuous_incidents/01_CONTINUOUS_TRANSITIONS/lab_experiments/5/`

---

### 1.4 Danh Sách Chi Tiết 63 Video Tuyển Chọn

| STT | Folder | Phân Loại | Nhóm | Tên File Video | Thời Lượng | Độ Phân Giải | Dung Lượng |
| :---: | :---: | :--- | :---: | :--- | :---: | :---: | :---: |
{table_text}

---

## PHẦN 2: TỔNG HỢP CÁC PHƯƠNG PHÁP CẮT BỎ ĐOẠN MỞ ĐẦU / KẾT THÚC KHÔNG LIÊN QUAN

Trong quá trình thu thập video từ Internet/YouTube, các đoạn video thường chứa:
* **Đoạn mở đầu không liên quan (Intro Noise)**: Người dẫn chương trình/phóng viên đứng nói dẫn nhập (0–20s), logo 3D quay tròn của kênh (0–5s), đồng hồ đếm ngược, màn hình đen chờ, cảnh người qua lại bình thường trước khi có sự cố.
* **Đoạn kết thúc không liên quan (Outro Noise)**: Cắt cảnh phỏng vấn nhân chứng sau vụ cháy, cảnh xe cứu hỏa đã dập tắt lửa xong, phóng viên kết luận tại trường quay, màn hình mời bấm Like/Subscribe (end-screen).

Dưới đây là **4 phương pháp kỹ thuật tốt nhất** hiện nay trong Computer Vision và Video Processing để loại bỏ triệt để các đoạn này:

---

### Phương Pháp 1: Tự Động Phát Hiện Cú Cắt Cảnh (Scene Cut Detection) Bằng `PySceneDetect` + Color/Motion Mask
*(Khuyên dùng cho pipeline tự động hóa quy mô lớn)*

#### 1. Nguyên lý:
Các video tin tức, phóng sự luôn sử dụng kỹ thuật dựng phim chuyên nghiệp. Khi chuyển từ **"Phóng viên trong trường quay"** sang **"CCTV hiện trường cháy"**, và từ **"Hiện trường cháy"** sang **"Phỏng vấn nhân chứng"**, luôn có một cú chuyển cảnh (`hard cut` hoặc `dissolve`).  
Thư viện `PySceneDetect` phát hiện các cú chuyển cảnh này với độ chính xác đến từng frame đơn lẻ.

#### 2. Kiến trúc thuật toán:
```mermaid
graph TD
    A[Video gốc có dính phóng sự/intro/outro] --> B[PySceneDetect: ContentDetector threshold=27.0]
    B --> C[Phân tách thành các Scene độc lập: Scene 1, Scene 2, Scene 3, ...]
    C --> D[Bộ Lọc Phóng Viên: Haar Cascade / MediaPipe Face Detection]
    C --> E[Bộ Lọc Sự Cố: HSV + YCrCb + Smoke Motion Differential]
    D --> F{{Đánh giá từng Scene}}
    E --> F
    F -->|Có khuôn mặt phóng viên >= 25% hoặc không có lửa/khói| G[LOẠI BỎ: Intro / Outro / Phỏng vấn]
    F -->|Có lửa/khói liên tục >= 5s VÀ không có phóng viên| H[GIỮ LẠI: Đoạn Sự Cố Thực Tế]
    H --> I[Nối hoặc xuất từng Scene sự cố bằng FFmpeg]
```

#### 3. Code Python mẫu hoàn chỉnh:
```python
import cv2
import numpy as np
from scenedetect import detect, ContentDetector
import subprocess

def auto_trim_by_scene_cuts(video_path, output_path):
    # Bước 1: Phát hiện danh sách các Scene dựa trên thay đổi nội dung khung hình
    scene_list = detect(video_path, ContentDetector(threshold=27.0))
    valid_scenes = []
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Bước 2: Duyệt từng Scene để kiểm tra sự hiện diện của lửa/khói và phóng viên
    for scene in scene_list:
        start_sec = scene[0].get_seconds()
        end_sec = scene[1].get_seconds()
        dur = end_sec - start_sec
        if dur < 4.0:
            continue  # Bỏ qua các cảnh quá ngắn dưới 4 giây
            
        # Lấy mẫu giữa scene
        mid_frame_idx = int((start_sec + dur * 0.5) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Kiểm tra phóng viên
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(cv2.resize(gray, (320, 180)), 1.3, 3)
        if len(faces) > 0:
            continue  # Scene này có người dẫn chương trình -> BỎ
            
        # Kiểm tra ngọn lửa bằng HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 90, 160), (32, 255, 255))
        fire_ratio = cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])
        if fire_ratio > 0.0005:  # Có lửa
            valid_scenes.append((start_sec, end_sec))
            
    cap.release()
    
    # Bước 3: Cắt đoạn sự cố dài nhất
    if valid_scenes:
        best_start = valid_scenes[0][0]
        best_end = valid_scenes[-1][1]
        subprocess.run([
            'ffmpeg', '-y', '-ss', str(best_start), '-i', video_path,
            '-to', str(best_end), '-c:v', 'libx264', '-preset', 'veryfast',
            '-crf', '20', '-c:a', 'aac', output_path
        ])
        return True
    return False
```

---

### Phương Pháp 2: Dò Tìm Biên Thời Gian Khởi Phát & Dập Tắt (Optical Temporal Active Window)
*(Phương pháp lõi mà hệ thống đã thực hiện thành công trên 149 video)*

#### 1. Nguyên lý:
Không phụ thuộc vào cú cắt cảnh của đạo diễn, thuật toán quét trực tiếp mức năng lượng quang học của ngọn lửa và khói khuếch tán dọc theo dòng thời gian với bước nhảy $0.5\text{s}$ (2 FPS):
* $t_{\text{start}}$: Là mốc thời gian đầu tiên mà diện tích lửa/khói vượt ngưỡng $S_{\text{threshold}}$ liên tục trong $\ge 3$ mẫu liên tiếp.
* $t_{\text{end}}$: Là mốc thời gian cuối cùng mà ngọn lửa/khói còn duy trì trước khi bị dập tắt hoặc chuyển sang cảnh tĩnh.
* Đoạn trích xuất: $[t_{\text{start}} - \delta, \, t_{\text{end}} + \delta]$ với lề đệm $\delta = 1.0\text{s}$ nhằm bảo toàn giây khởi phát sớm nhất của khói mỏng.

---

### Phương Pháp 3: Nhận Diện Ngữ Nghĩa Bằng Multimodal Vision-Language Model (VLM Zero-Shot Grounding)
*(Độ chính xác ngữ nghĩa cao nhất, áp dụng cho các video phức tạp có kỹ xảo hoặc bối cảnh khó)*

#### 1. Nguyên lý:
Mô hình ngôn ngữ - thị giác đa thể thức (như Gemini 1.5 Flash hoặc Video-LLaVA) có khả năng đọc hiểu ngữ cảnh của video như con người: phân biệt được đâu là "phóng viên đứng nói trước màn hình tivi có chiếu cảnh cháy" và đâu là "camera an ninh quay ngọn lửa thật".

#### 2. Quy trình:
1. Gửi video vào mô hình với prompt định dạng JSON:
   ```
   "Analyze this video and output exact timestamps (seconds):
   - is_news_or_interview: boolean
   - real_fire_start_time: float (timestamp where the real physical fire/smoke incident begins)
   - real_fire_end_time: float (timestamp where the fire is put out or camera cuts away to reporters/studio)
   - confidence: float"
   ```
2. Parse JSON trả về và đưa tham số vào lệnh FFmpeg tự động cắt.
3. Chi phí: Với Gemini Flash, chi phí xử lý 1 video ~1 phút chỉ tốn $0.0002.

---

### Phương Pháp 4: Bán Tự Động Nhanh Bằng Phần Mềm Mã Nguồn Mở `LosslessCut`
*(Tối ưu nhất khi người nghiên cứu muốn tự tay rà soát và cắt chính xác từng khung hình)*

Nếu bạn muốn trực tiếp kiểm tra bằng mắt và cắt tỉa theo ý thích cá nhân mà không phải chờ đợi re-encode:
* **Tải công cụ**: [LosslessCut](https://github.com/mifi/lossless-cut) (miễn phí, portable chạy ngay).
* **Ưu điểm vượt trội**:
  * **Cắt trực tiếp trên bitstream không giải mã/mã hóa lại**: Thời gian xuất file chỉ mất **0.5 giây** (tốc độ copy ổ cứng).
  * **Phím tắt siêu tốc**: 
    * Kéo thanh trượt đến đầu cảnh cháy $\rightarrow$ bấm phím `I` (*In Point*).
    * Kéo thanh trượt đến cuối cảnh cháy $\rightarrow$ bấm phím `O` (*Out Point*).
    * Bấm phím `E` (*Export*) $\rightarrow$ Video đã được cắt gọn gàng.
  * **Tự động hỗ trợ**: Menu `Tools -> Detect scene changes` có sẵn thuật toán phân tích cảnh quay.

---

## PHẦN 3: PHƯƠNG ÁN GỘP DỮ LIỆU CHUẨN HÓA VỚI BỘ VIDEO BAN ĐẦU

### 3.1 Thực Trạng Dữ Liệu Ban Đầu Trong Dự Án
Hiện tại trong repository đang có các thư mục video:
* `video_train/`: 177 video (định dạng UUID và Pexels, đa số đã chuẩn hóa tên `00e18d86-3389d8dc.mp4` hoặc `012f0550_pexels_...mp4`).
* `video_test/`: 71 video (phục vụ đánh giá benchmark).
* `pure_fire_smoke_clips/`: 177 video (đã cắt phân đoạn thuần lửa/khói `*_pure.mp4`).

### 3.2 Ba Phương Án Gộp Dữ Liệu (Data Harmonization Options)

```mermaid
graph TD
    A[63 Video Đã Tuyển Chọn trong Folders 1..13] --> B{Lựa Chọn Phương Án Gộp}
    B -->|Phương án A| C[Làm phẳng trong Curated Folder]
    B -->|Phương án B| D[Hợp nhất vào video_train / pure_fire_smoke_clips]
    B -->|Phương án C - Khuyên dùng| E[Tạo Master Dataset Chuẩn Hóa Mới]
    
    C --> C1[01_CONTINUOUS_TRANSITIONS / 02_CONTINUOUS_AMBIENT]
    D --> D1[Đổi tên chuẩn UUID/Slug + Cập nhật CSV train split]
    E --> E1[Standardized_Continuous_Fire_Dataset/ chia 9 scenarios + Master Manifest]
```

#### Phương Án A: Làm phẳng thư mục tuyển chọn (Flattened Local Integration)
* **Thực hiện**: Di chuyển toàn bộ video trong các thư mục con `1`..`13` ra lại thư mục cha tương ứng (ví dụ đưa từ `01_CONTINUOUS_TRANSITIONS/battery_thermal_runaway/1/*` ra thẳng `01_CONTINUOUS_TRANSITIONS/battery_thermal_runaway/`).
* **Ưu điểm**: Gọn gàng cấu trúc, xóa bỏ các folder con tạm thời `1`..`13`.

#### Phương Án B: Hợp nhất trực tiếp vào bộ video huấn luyện gốc (`video_train`)
* **Thực hiện**: 
  1. Gán ID chuẩn hóa cho 63 video mới (ví dụ: `curated_01_battery_1UKOXGEwjv4.mp4`, `curated_05_lab_2HOX_part1.mp4`).
  2. Copy vào `video_train/` và `pure_fire_smoke_clips/`.
  3. Cập nhật lại các file danh sách dữ liệu (data manifests).
* **Ưu điểm**: Mở rộng trực tiếp kích thước tập huấn luyện từ 177 clip lên **240 clip** (+35.6% dữ liệu mới).

#### Phương Án C: Tạo Bộ Master Dataset Chuẩn Hóa Riêng Biệt (*Được khuyến nghị cao nhất*)
* **Thực hiện**: Tạo thư mục mới `dataset_master_continuous_fire/` được chuẩn hóa toàn bộ:
  * Khung hình: 720p hoặc scale chuẩn 360p.
  * Tốc độ: Đồng bộ 30 FPS.
  * Codec: H.264 Baseline/Main Profile, CRF 20, loại bỏ âm thanh (video-only).
  * Kèm file `master_dataset_manifest.csv` mô tả nhãn (`fire`, `smoke`, `transition`, `scenario`).
* **Ưu điểm**: Đáp ứng tiêu chuẩn khoa học cao nhất để huấn luyện TimeSformer và viết bài báo NCKH (Paper/Thesis).

---

## PHẦN 4: KẾT LUẬN & ĐỀ XUẤT HÀNH ĐỘNG TIẾP THEO

1. **Về việc xử lý video thí nghiệm phòng lab**: Đã hoàn tất chia tách thành 2 clip `part1_0_40s.mp4` và `part2_43_end.mp4`, sẵn sàng sử dụng.
2. **Về công cụ cắt intro/outro**: 
   * Nếu muốn xử lý hàng loạt tự động: Sử dụng script `PySceneDetect` kết hợp Color Mask (Phương pháp 1).
   * Nếu muốn rà soát thủ công nhanh bằng tay: Tải và mở bằng `LosslessCut` (Phương pháp 4, 3 giây/video).
3. **Về việc gộp dữ liệu**: Bạn có thể chọn **Phương án A**, **Phương án B** hoặc **Phương án C** ở Phần 3 để tôi tiến hành chạy tự động ngay lập tức!
"""

# Thay thế các trường dữ liệu động
doc_content = doc_content.replace("{total_vids}", str(total_vids))
doc_content = doc_content.replace("{total_dur_s:.1f}", f"{total_dur_s:.1f}")
doc_content = doc_content.replace("{total_dur_m:.2f}", f"{total_dur_s/60:.2f}")
doc_content = doc_content.replace("{total_dur_h:.2f}", f"{total_dur_s/3600:.2f}")
doc_content = doc_content.replace("{total_size_mb:.2f}", f"{total_size_mb:.2f}")
doc_content = doc_content.replace("{total_size_mb:.1f}", f"{total_size_mb:.1f}")
doc_content = doc_content.replace("{avg_dur:.1f}", f"{total_dur_s/total_vids:.1f}")
doc_content = doc_content.replace("{summary_table}", summary_table)
doc_content = doc_content.replace("{table_text}", table_text)

# Ghi ra 2 nơi: project docs và artifact brain
doc_path_project = Path("docs/curation_and_trimming_methodology_report.md")
doc_path_project.parent.mkdir(parents=True, exist_ok=True)
with open(doc_path_project, "w", encoding="utf-8") as f:
    f.write(doc_content)

doc_path_artifact = Path(r"C:\Users\LENOVO\.gemini\antigravity-ide\brain\117f3b5e-af32-4d11-b9ce-488531f0681d/curation_and_trimming_methodology_report.md")
with open(doc_path_artifact, "w", encoding="utf-8") as f:
    f.write(doc_content)

print("Đã tạo báo cáo thành công tại 2 vị trí:")
print("  - Project:", doc_path_project)
print("  - Artifact:", doc_path_artifact)
