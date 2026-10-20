"""
src/crawl_early_fire_transition.py
==================================
Crawler Chuyên Biệt Cho Dữ Liệu Cháy Khởi Phát Sớm (Early Transition Sequences)
Mục tiêu NCKH:
  Thu thập các video ghi lại ĐÚNG GIAI ĐOẠN CHUYỂN TIẾP:
    [Chưa có gì / Khói mỏng / Đốm lửa nhỏ] ──> [Bùng phát thành ngọn lửa rõ ràng]
  để bổ sung cho mô hình TimeSformer và State Engine.

6 Nhóm Kịch Bản Săn Tìm (Target Scenarios):
  1. electrical_ignition     : Chập điện ổ cắm, tủ điện, dây dẫn bốc khói rồi tóe lửa.
  2. battery_thermal_runaway : Pin lithium, xe điện xì khói rồi bùng nổ lửa (CCTV).
  3. smoldering_transition   : Tàn thuốc, thùng rác, đống củi âm ỉ khói rồi bùng cháy.
  4. lab_experiments         : Thí nghiệm đốt phòng, đồ đạc từ điểm mồi lửa (NIST, UL).
  5. kitchen_stove           : Nồi chảo dầu bốc khói rồi tự bắt lửa (CCTV bếp).
  6. wildfire_ignition       : Tia lửa bén cỏ khô bốc khói rồi lan thành ngọn lửa.

Tính Năng:
  - Tự động lọc thời lượng: 15s đến 240s (loại bỏ video tin tức dài hàng tiếng).
  - Loại trừ rác: Tin tức bình luận (talking heads), vlog, 3D animation, game.
  - Kiểm định chuyển tiếp (Transition Verifier): Kiểm tra frame đầu ít/chưa có lửa, frame sau lửa bùng lên.
  - Ghi nhận manifest chi tiết: crawled_manifest.csv.
"""

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("crawl_early_transition.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("EarlyFireCrawler")

# ── 1. NGÂN HÀNG TRUY VẤN CÓ CHỦ ĐÍCH (70+ TRUY VẤN - 9 KỊCH BẢN) ──────────────

TRANSITION_SCENARIOS = [
    {
        "category": "electrical_ignition",
        "description": "Chập điện ổ cắm / đường dây / tủ điện bốc khói rồi phát hỏa",
        "queries": [
            "cctv electrical outlet catching fire",
            "security camera electrical socket smoke to flame",
            "wire short circuit smoke catching fire cctv",
            "power strip catching fire caught on camera",
            "electrical panel smoke then fire security camera",
            "breaker box fire starting cctv footage",
            "air conditioner indoor unit catching fire cctv",
            "ceiling fan electrical fire start security camera",
            "transformer explosion fire surveillance video",
            "power supply smoke catches fire cctv",
            "overhead powerline sparking fire security camera",
            "server room fire ignition cctv",
            "lithium charger catching fire cctv",
        ]
    },
    {
        "category": "battery_thermal_runaway",
        "description": "Pin lithium / xe điện xì khói rồi bùng cháy dữ dội",
        "queries": [
            "lithium battery smoke then fire cctv footage",
            "ebike battery smoking then catching fire security camera",
            "scooter battery smoke to explosion cctv",
            "power bank battery catch fire cctv",
            "electric scooter battery catches fire in building cctv",
            "laptop battery catches fire cctv office",
            "phone battery explodes while charging security camera",
            "drone lipo battery fire thermal runaway camera",
            "hoverboard catching fire in house cctv",
            "electric car battery fire charging station cctv",
            "ev battery fire thermal runaway security camera",
            "forklift battery explodes catches fire cctv",
            "vape battery explosion caught on camera",
        ]
    },
    {
        "category": "smoldering_transition",
        "description": "Cháy âm ỉ từ tàn thuốc / rác bốc khói rồi bùng lửa",
        "queries": [
            "smoldering trash can catching fire cctv",
            "cigarette smoldering couch to flame experiment",
            "smoldering pallet catching fire warehouse cctv",
            "garbage can smoke to fire security camera",
            "smoldering fire transition to open flame experiment",
            "cigarette mattress smoldering to flame test",
            "mulch fire starting smoldering surveillance",
            "sawdust smoldering ignition caught on camera",
            "dryer lint catching fire surveillance",
            "smoldering compost pile catches fire camera",
            "ashtray smoke caught on cctv fire",
            "smoldering cardboard warehouse surveillance",
        ]
    },
    {
        "category": "kitchen_stove",
        "description": "Bếp đun quên tắt / dầu ăn bốc khói rồi bốc lửa",
        "queries": [
            "kitchen cooking oil smoke then fire cctv",
            "unattended cooking oil catching fire caught on camera",
            "stove pan smoke to flame security camera",
            "cooking fire starts on stove cctv footage",
            "towel near gas stove catching fire kitchen cctv",
            "microwave catching fire kitchen security camera",
            "toaster oven catching fire kitchen cctv",
            "restaurant kitchen deep fryer catching fire cctv",
            "toaster catch fire cctv security camera",
            "frying pan oil fire caught on home camera",
            "unattended stove pan catches fire camera",
            "grease fire in restaurant kitchen surveillance",
        ]
    },
    {
        "category": "industrial_workplace",
        "description": "Xỉ hàn / máy mài bén vật liệu cháy, rò rỉ hóa chất bắt lửa",
        "queries": [
            "welding sparks catching fire warehouse cctv",
            "grinder sparks ignite cardboard surveillance camera",
            "flammable liquid spill catching fire factory cctv",
            "dust ignition explosion test camera",
            "conveyor belt friction fire factory surveillance",
            "gas cylinder leak catching fire cctv footage",
            "auto repair shop fire start security camera",
            "metal cutting torch sparks fire surveillance",
            "factory furnace fire starting cctv",
            "woodworking workshop dust fire security camera",
            "chemical lab fire start caught on camera",
        ]
    },
    {
        "category": "vehicle_engine_fire",
        "description": "Khoang máy ô tô, xe tải, xe máy bốc khói rồi bùng cháy",
        "queries": [
            "car engine smoke then fire dashcam",
            "car engine bay catching fire gas station cctv",
            "truck wheel brake catching fire surveillance",
            "bus engine fire caught on highway camera",
            "motorcycle catching fire traffic camera",
            "parked car engine starts smoking then flames cctv",
            "semi truck fire starts on highway camera",
            "car fire gas pump surveillance camera",
            "car engine compartment fire caught on dashcam",
            "tractor fire start in field camera",
        ]
    },
    {
        "category": "wildfire_forest_ignition",
        "description": "Cháy rừng khởi phát từ đốm tàn / tia lửa ban đầu",
        "queries": [
            "brush fire starting from spark cctv",
            "grass fire ignition caught on camera",
            "wildfire start security camera timeline",
            "campfire sparks igniting dry leaves camera",
            "lightning strike tree catches fire security camera",
            "power line sparking grass fire roadside camera",
            "bushfire starting smoke caught on look out camera",
            "wildfire ignition caught on fire lookout tower camera",
            "dry grass wildfire early smoke surveillance",
            "prescribed burn ignition time lapse",
        ]
    },
    {
        "category": "lab_experiments",
        "description": "Thí nghiệm khoa học đốt từ tia lửa đến ngọn lửa (NIST/UL)",
        "queries": [
            "room fire burn experiment ignition timeline",
            "wastebasket fire experiment timeline NIST",
            "fire growth experiment ignition to flashover",
            "sofa ignition fire burn experiment cctv",
            "christmas tree fire timeline NIST experiment",
            "curtain ignition test UL fire experiment",
            "room corner fire test timeline video",
            "curtain fabric flame spread test video",
            "mattress flammability test timeline",
            "candle catches curtain on fire experiment",
        ]
    },
    {
        "category": "hard_negatives",
        "description": "Mẫu âm tính (Không cháy): Hơi nước, sương mù, khói lạnh, hàn xì",
        "queries": [
            "thick boiling water steam kitchen cctv",
            "humidifier mist indoor security camera",
            "dense morning fog highway surveillance camera",
            "car exhaust cold morning traffic camera",
            "stadium smoke flare surveillance footage",
            "stage fog machine indoor venue camera",
            "welding sparks dark workshop without fire",
            "cold weather car tailpipe steam camera",
            "commercial kitchen steam ventilation camera",
            "fog rolling into city security camera",
            "snow cannon mist ski resort webcam",
            "construction site dust cloud security camera",
        ]
    }
]

EXCLUDE_KEYWORDS = [
    "vlog", "reaction", "podcast", "commentary", "gameplay", "minecraft",
    "gta", "animation", "3d render", "tutorial how to", "compilation funny",
    "news report anchor", "interview talking head"
]


# ── 2. KIỂM TRA & TÌM KIẾM BẰNG YT-DLP ────────────────────────────────────────

def search_youtube_candidates(query: str, max_results: int = 5) -> List[Dict]:
    """Tìm kiếm danh sách video trên YouTube qua yt-dlp."""
    cmd = [
        "yt-dlp",
        f"ytsearch{max_results}:{query}",
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
                title = data.get("title", "")
                # Lọc tiêu đề chứa từ khóa rác
                title_lower = title.lower()
                if any(ex in title_lower for ex in EXCLUDE_KEYWORDS):
                    continue

                candidates.append({
                    "id": data.get("id"),
                    "title": title,
                    "duration": data.get("duration", 0),
                    "url": f"https://www.youtube.com/watch?v={data.get('id')}",
                    "view_count": data.get("view_count", 0),
                    "query": query,
                })
            except Exception:
                continue
        return candidates
    except Exception as e:
        log.warning(f"Lỗi khi search query '{query}': {e}")
        return []


def download_video(video_url: str, output_path: Path) -> bool:
    """Tải video chất lượng tối đa 720p MP4 (tải nhanh, tối ưu dung lượng)."""
    cmd = [
        "yt-dlp",
        video_url,
        "-o", str(output_path),
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
        return output_path.exists() and output_path.stat().st_size > 50_000
    except Exception as e:
        log.warning(f"Lỗi khi tải video {video_url}: {e}")
        return False


# ── 3. THẨM ĐỊNH GIAI ĐOẠN CHUYỂN TIẾP (TRANSITION VERIFIER) ─────────────────

def verify_transition_dynamics(video_path: Path) -> Tuple[bool, str, Dict]:
    """
    Kiểm tra video có thực sự chứa chuỗi chuyển tiếp không:
      - Frame đầu (5-10% thời lượng): Không được cháy bùng to từ trước (Lửa diện tích nhỏ hoặc chỉ khói).
      - Frame sau (70%-90% thời lượng): Xuất hiện ngọn lửa hoặc đám cháy rõ rệt.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "cannot_open", {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_s = total_frames / fps

    if total_frames < 30:
        cap.release()
        return False, "too_short", {}

    # 1. Đọc frame đầu (khoảng 5-8% video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.08))
    ret_early, frame_early = cap.read()

    # 2. Đọc frame giữa (khoảng 50% video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.50))
    ret_mid, frame_mid = cap.read()

    # 3. Đọc frame cuối (khoảng 85% video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.85))
    ret_late, frame_late = cap.read()

    cap.release()

    if not (ret_early and ret_late):
        return False, "read_error", {}

    def get_fire_pixel_count(img_bgr):
        # Không gian YCrCb: Cr > 148 và R > 155
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        r = img_bgr[:, :, 2]
        b = img_bgr[:, :, 0]
        fire_pts = (cr > 148) & (r > 155) & (r > b + 25)
        return int(fire_pts.sum())

    fire_early = get_fire_pixel_count(frame_early)
    fire_late = get_fire_pixel_count(frame_late)

    h, w = frame_early.shape[:2]
    total_pixels = h * w
    early_ratio = (fire_early / total_pixels) * 100.0
    late_ratio = (fire_late / total_pixels) * 100.0

    growth_ratio = round(late_ratio / max(0.001, early_ratio), 2)
    stats = {
        "duration_s": round(duration_s, 1),
        "early_fire_pct": round(early_ratio, 3),
        "late_fire_pct": round(late_ratio, 3),
        "fire_growth_ratio": growth_ratio,
    }

    # Nếu ngay từ đầu lửa đã chiếm diện tích lớn (> 4%), đây là video đã cháy sẵn
    if early_ratio > 4.0:
        return False, "already_developed_fire", stats

    # Tiêu chí chuyển tiếp:
    # 1. Lúc đầu ít lửa (<0.2%) và lúc sau bùng lên (>0.3%)
    # 2. Hoặc lúc sau lửa tăng gấp 1.4 lần và tối thiểu late_ratio > 0.05%
    is_transition = (early_ratio < 0.2 and late_ratio > 0.3) or (late_ratio > 0.05 and late_ratio >= early_ratio * 1.4)

    if is_transition:
        return True, "valid_transition", stats
    else:
        return False, "static_or_no_flame", stats


# ── 4. CHƯƠNG TRÌNH ĐIỀU PHỐI CRAWLER CHÍNH ───────────────────────────────────

def run_early_transition_crawler(
    output_dir: Path,
    max_per_query: int = 6,
    target_category: Optional[str] = None,
    dry_run: bool = False,
    total_limit: Optional[int] = 350,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "crawled_manifest.csv"
    existing_ids: Set[str] = set()
    if manifest_path.exists():
        try:
            df_old = pd.read_csv(manifest_path)
            if "video_id" in df_old.columns:
                existing_ids = set(df_old["video_id"].dropna().unique())
        except Exception:
            pass

    scenarios = TRANSITION_SCENARIOS
    if target_category:
        scenarios = [s for s in scenarios if s["category"] == target_category]

    existing_downloaded = len(existing_ids)
    limit_str = f"Mục tiêu tổng cộng: {total_limit} video (đã có {existing_downloaded})" if total_limit else "Không giới hạn số lượng"
    log.info(f"Bắt đầu Crawler Khói Mỏng ➔ Lửa Bùng Phát (Early Transition). Số kịch bản: {len(scenarios)} | {limit_str}")
    log.info(f"Chế độ: {'DRY RUN (Chỉ tìm kiếm, không tải)' if dry_run else 'DOWNLOAD (Tải và thẩm định video)'}")

    all_results: List[Dict] = []
    total_downloaded = 0
    stop_crawling = False

    if total_limit and existing_downloaded >= total_limit:
        log.info(f"Đã có đủ {existing_downloaded}/{total_limit} video trong kho. Dừng!")
        return

    for sc in scenarios:
        if stop_crawling:
            break

        cat = sc["category"]
        desc = sc["description"]
        log.info(f"\n========================================================")
        log.info(f"Đang quét Kịch bản: [{cat.upper()}] - {desc}")
        log.info(f"========================================================")

        for q in sc["queries"]:
            if stop_crawling:
                break

            log.info(f"Truy vấn: '{q}'")
            candidates = search_youtube_candidates(q, max_results=max_per_query)
            log.info(f"  -> Tìm thấy {len(candidates)} ứng viên hợp lệ.")

            for cand in candidates:
                if total_limit and (existing_downloaded + total_downloaded) >= total_limit:
                    log.info(f"Đã đạt tổng cộng {total_limit} video. Dừng crawler!")
                    stop_crawling = True
                    break

                v_id = cand["id"]
                if v_id in existing_ids:
                    log.info(f"  [Đã có] Bỏ qua {v_id} ({cand['title'][:40]})")
                    continue

                out_filename = f"{cat}_{v_id}.mp4"
                out_filepath = video_dir / out_filename

                record = {
                    "video_id": v_id,
                    "category": cat,
                    "title": cand["title"],
                    "duration_s": cand["duration"],
                    "url": cand["url"],
                    "search_query": q,
                    "filename": out_filename,
                    "status": "candidate_found",
                    "is_transition_verified": False,
                    "early_fire_pct": 0.0,
                    "late_fire_pct": 0.0,
                    "fire_growth_ratio": 1.0,
                    "transition_reason": "",
                }

                if dry_run:
                    log.info(f"  [DRY-RUN] Ứng viên: {cand['title'][:55]} | {cand['duration']}s")
                    all_results.append(record)
                    existing_ids.add(v_id)
                    total_downloaded += 1
                    continue

                # Tải video thật
                curr_total = existing_downloaded + total_downloaded + 1
                log.info(f"  [Đang tải {curr_total}/{total_limit or '∞'}] {cand['title'][:50]} ({cand['duration']}s)...")
                is_dl_ok = download_video(cand["url"], out_filepath)
                if not is_dl_ok:
                    log.warning(f"  [Lỗi tải] Không thể tải {cand['url']}")
                    record["status"] = "download_error"
                    all_results.append(record)
                    continue

                # Thẩm định động học chuyển tiếp
                if cat == "hard_negatives":
                    is_trans = False
                    reason = "hard_negative_sample"
                    stats = {"early_fire_pct": 0.0, "late_fire_pct": 0.0, "fire_growth_ratio": 1.0}
                else:
                    is_trans, reason, stats = verify_transition_dynamics(out_filepath)

                record["status"] = "downloaded"
                record["is_transition_verified"] = is_trans
                record["early_fire_pct"] = stats.get("early_fire_pct", 0.0)
                record["late_fire_pct"] = stats.get("late_fire_pct", 0.0)
                record["fire_growth_ratio"] = stats.get("fire_growth_ratio", 1.0)
                record["transition_reason"] = reason

                if is_trans:
                    log.info(f"  ⭐ [ĐẠT CHUẨN CHUYỂN TIẾP] Lửa tăng từ {stats.get('early_fire_pct')}% -> {stats.get('late_fire_pct')}% (x{stats.get('fire_growth_ratio')} lần)!")
                elif cat == "hard_negatives":
                    log.info(f"  🛡️ [MẪU ÂM TÍNH] Đã nạp mẫu không cháy: {cand['title'][:50]}")
                else:
                    log.info(f"  ℹ️ [Video thường/Không tăng] {reason} (Lửa ban đầu: {stats.get('early_fire_pct')}%, lúc sau: {stats.get('late_fire_pct')}%)")

                existing_ids.add(v_id)
                all_results.append(record)
                total_downloaded += 1

                # Ghi nhận manifest ngay lập tức
                df_single = pd.DataFrame([record])
                if not manifest_path.exists():
                    df_single.to_csv(manifest_path, index=False, encoding="utf-8")
                else:
                    df_single.to_csv(manifest_path, mode="a", header=False, index=False, encoding="utf-8")

                if total_limit and (existing_downloaded + total_downloaded) >= total_limit:
                    log.info(f"Đã hoàn thành tải đủ {total_limit} video. Kết thúc tác vụ!")
                    stop_crawling = True
                    break

                # Nghỉ ngắn giữa các lần tải
                time.sleep(1.2)

    log.info(f"\n========================================================")
    log.info(f"TỔNG KẾT QUÁ TRÌNH CRAWL:")
    log.info(f"  - Số video trong đợt chạy này: {len(all_results)}")
    log.info(f"  - Đã tải về: {total_downloaded}")
    log.info(f"  - Thư mục lưu: {video_dir}")
    log.info(f"  - Manifest: {manifest_path}")
    log.info(f"========================================================")


def main():
    parser = argparse.ArgumentParser(description="Crawler Video Khói Mỏng ➔ Lửa Bùng Phát (Early Transition)")
    parser.add_argument("--output_dir", type=str, default="crawled_early_fire_transition", help="Thư mục xuất dữ liệu")
    parser.add_argument("--max_per_query", type=int, default=6, help="Số video tối đa cho mỗi truy vấn")
    parser.add_argument("--category", type=str, default=None, help="Chạy riêng 1 kịch bản: electrical_ignition, battery_thermal_runaway, smoldering_transition...")
    parser.add_argument("--dry_run", action="store_true", help="Chỉ tìm kiếm và xuất manifest, không tải video")
    parser.add_argument("--total_limit", type=int, default=350, help="Số lượng video tải tối đa rồi dừng")
    args = parser.parse_args()

    run_early_transition_crawler(
        output_dir=Path(args.output_dir),
        max_per_query=args.max_per_query,
        target_category=args.category,
        dry_run=args.dry_run,
        total_limit=args.total_limit,
    )


if __name__ == "__main__":
    main()


