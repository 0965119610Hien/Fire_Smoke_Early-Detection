# Alamy Video Crawler – Fire/Smoke Detection Test Dataset

## Tổng quan

Script thu thập 300 video từ Alamy cho bộ dataset đánh giá mô hình phát hiện cháy/khói trong môi trường nhà xưởng.

---

## Cài đặt

```bash
# 1. Clone / copy scripts vào thư mục làm việc
mkdir fire_dataset && cd fire_dataset

# 2. Cài thư viện Python
pip install requests tqdm yt-dlp fake-useragent beautifulsoup4

# 3. Cài ffmpeg (dùng cho validate_dataset.py)
# Ubuntu/Debian:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg
# Windows: tải từ https://ffmpeg.org/download.html
```

---

## Cấu trúc file

```
.
├── alamy_video_crawler.py   ← Script crawl chính
├── validate_dataset.py      ← Script kiểm tra chất lượng
├── crawler.log              ← Log file (tự sinh)
└── dataset_test/            ← Output directory
    ├── class_0/             ← Negative (hard negatives)
    │   ├── dust_particles/
    │   ├── steam_vapor/
    │   ├── welding_sparks/
    │   ├── optical_noise/
    │   └── cigarette_smoke/
    ├── class_1/             ← Positive (fire/smoke)
    │   ├── early_smoke/
    │   ├── heavy_smoke/
    │   ├── diffused_smoke/
    │   └── visible_flame/
    ├── manifest.csv         ← Metadata đầy đủ tất cả video
    └── progress.json        ← Checkpoint để resume
```

---

## Cách chạy

### Chạy đầy đủ (crawl + download)
```bash
python alamy_video_crawler.py
```

### Dry run – chỉ thu metadata, không download
```bash
python alamy_video_crawler.py --dry-run
```

### Resume từ lần chạy trước (mặc định bật)
```bash
python alamy_video_crawler.py --resume
```

### Bắt đầu lại từ đầu
```bash
python alamy_video_crawler.py --no-resume
```

### Tùy chỉnh delay giữa requests (tránh bị ban IP)
```bash
python alamy_video_crawler.py --delay-min 2.0 --delay-max 5.0
```

### Tùy chỉnh output directory
```bash
python alamy_video_crawler.py --output /data/fire_test_dataset
```

---

## Validate dataset sau khi crawl

```bash
python validate_dataset.py
# hoặc custom path:
python validate_dataset.py /data/fire_test_dataset
```

Kết quả:
- In bảng thống kê phân bổ theo class/subtype
- Kiểm tra duration thực tế bằng ffprobe
- Loại bỏ duplicate bằng MD5
- Xuất `validation_report.csv`

---

## Phân bổ target

| Class | Subtype            | Target |
|-------|--------------------|--------|
| 1     | early_smoke        | 35     |
| 1     | heavy_smoke        | 40     |
| 1     | diffused_smoke     | 35     |
| 1     | visible_flame      | 40     |
| **1** | **TOTAL**          | **150**|
| 0     | dust_particles     | 30     |
| 0     | steam_vapor        | 30     |
| 0     | welding_sparks     | 25     |
| 0     | optical_noise      | 25     |
| 0     | cigarette_smoke    | 20     |
| **0** | **TOTAL**          | **150**|
|       | **GRAND TOTAL**    | **300**|

---

## Tiêu chí lọc tự động

Script tự động loại bỏ video không đạt:
- **Thời lượng**: ngoài khoảng 5–15 giây
- **Resolution**: ngoài 480p–1080p (kiểm tra qua ffprobe)
- **Keyword exclusion**: phone screen, slow motion, timelapse, zoom in, extreme closeup...
- **File size**: < 50KB (tải lỗi)

---

## Lưu ý quan trọng

### Alamy chỉ cho phép tải preview clip miễn phí
Script download **preview clip** (thường 15–30s, watermarked) từ Alamy. Để có bản full resolution sạch:
- Đăng ký tài khoản Alamy và mua subscription nếu dùng cho mục đích thương mại
- Hoặc dùng preview clip (đủ dùng cho training/evaluation model phát hiện cháy)

### Anti-bot considerations
- Delay ngẫu nhiên 1.5–3.5s giữa mỗi request (cấu hình được)
- Rotate User-Agent (dùng fake-useragent)
- Nếu bị 429/403: tăng delay, dùng VPN, hoặc chia nhỏ session

### yt-dlp fallback
Khi direct download thất bại, script tự động thử lại bằng yt-dlp (hỗ trợ Alamy natively). Đảm bảo `yt-dlp` đã được cài và cập nhật:
```bash
pip install -U yt-dlp
```

---

## manifest.csv – Schema

| Column | Mô tả |
|--------|-------|
| alamy_id | ID nội bộ của Alamy |
| class_label | 0 (negative) hoặc 1 (positive) |
| subtype | Subtype cụ thể (vd: early_smoke, dust_particles) |
| title | Tiêu đề từ Alamy |
| duration_s | Thời lượng theo metadata Alamy |
| search_query | Query đã dùng để tìm |
| page | Trang kết quả |
| local_path | Đường dẫn file local |
| download_status | ok / skip / error / dry_run |
| skip_reason | Lý do bỏ qua (nếu có) |
| preview_url | URL preview video |
| thumbnail_url | URL thumbnail |
