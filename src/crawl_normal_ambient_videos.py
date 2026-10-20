"""
src/crawl_normal_ambient_videos.py
==================================
Crawl các video giám sát (CCTV / Webcam) bình thường KHÔNG CÓ LỬA / KHÓI:
- Kịch bản: Văn phòng, phòng khách, nhà bếp an toàn, giao thông nắng ráo, công viên/rừng cây, bãi đỗ xe.
- Giới hạn thời lượng: 15s - 180s.
- Lưu trữ vào: data/crawled_normal_ambient/
"""

import os
import sys
import json
import subprocess
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = Path(r"E:\NCKH\Code\Fire_Smoke_Early-Detection")
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "crawled_normal_ambient"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NORMAL_QUERIES = [
    "cctv office indoor security camera footage",
    "living room home security camera cctv footage",
    "highway traffic surveillance camera daytime sunny",
    "city intersection pedestrian cctv daytime",
    "warehouse indoor surveillance security camera",
    "parking lot cctv camera daytime cars",
    "park garden trees wind security camera",
    "convenience store indoor cctv surveillance footage"
]

EXCLUDE = [
    "fire", "smoke", "burn", "explosion", "flame", "crash", "accident",
    "news", "interview", "podcast", "vlog", "gameplay", "animation", "review"
]

def search_and_download(query, max_downloads=2):
    print(f"\n🔍 [TÌM KIẾM] Query: '{query}'")
    cmd = [
        "yt-dlp",
        f"ytsearch6:{query}",
        "--dump-json",
        "--default-search", "ytsearch",
        "--no-playlist",
        "--match-filter", "duration >= 15 & duration <= 240",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=60)
        candidates = []
        for line in res.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                title = data.get("title", "").lower()
                if any(x in title for x in EXCLUDE):
                    continue
                candidates.append(data)
            except Exception:
                continue
        
        print(f"  -> Tìm thấy {len(candidates)} ứng viên hợp lệ.")
        downloaded = 0
        for cand in candidates:
            if downloaded >= max_downloads:
                break
            vid_id = cand.get("id")
            title_clean = "".join(c if c.isalnum() else "_" for c in cand.get("title", ""))[:30]
            out_file = OUT_DIR / f"normal_{vid_id}_{title_clean}.mp4"
            if out_file.exists() and out_file.stat().st_size > 500000:
                print(f"  [BỎ QUA] Đã tồn tại: {out_file.name}")
                downloaded += 1
                continue
            
            print(f"  ⬇️ [TẢI VỀ] {cand.get('title')} ({cand.get('duration')}s)")
            dl_cmd = [
                "yt-dlp",
                f"https://www.youtube.com/watch?v={vid_id}",
                "-o", str(out_file),
                "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
                "--merge-output-format", "mp4",
                "--no-playlist",
                "--quiet",
                "--no-warnings",
            ]
            r = subprocess.run(dl_cmd, timeout=120)
            if r.returncode == 0 and out_file.exists():
                print(f"  ✅ Đã tải: {out_file.name} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
                downloaded += 1
            else:
                print(f"  ❌ Lỗi tải video {vid_id}")
    except Exception as e:
        print(f"  ⚠️ Lỗi tìm kiếm: {e}")

def main():
    print("=" * 70)
    print("  BẮT ĐẦU CRAWL VIDEO ÂM TÍNH / BÌNH THƯỜNG (NORMAL AMBIENT VIDEOS)")
    print("=" * 70)
    for q in NORMAL_QUERIES:
        search_and_download(q, max_downloads=1)

if __name__ == "__main__":
    main()
