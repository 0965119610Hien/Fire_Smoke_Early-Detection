"""
=============================================================================
ALAMY VIDEO CRAWLER - Fire/Smoke Detection Test Dataset
=============================================================================
Mục tiêu: Thu thập 300 video từ Alamy theo yêu cầu kỹ thuật dataset đánh giá
  - Class 1 (Positive): 150 video cháy/khói thực tế
  - Class 0 (Negative): 150 video nhiễu hóc búa (hard negatives)

Yêu cầu kỹ thuật video:
  - Thời lượng: 5–15 giây
  - Góc: High-angle/Top-down (CCTV perspective)
  - Độ phân giải: 480p–1080p
  - Raw footage, không có transition/jump cut

Cách dùng:
  pip install requests tqdm yt-dlp beautifulsoup4 fake-useragent
  python alamy_video_crawler.py

Lưu ý: Script này sử dụng Alamy's internal search API.
Alamy cung cấp preview clip miễn phí (~30s) - chỉ download preview clip.
=============================================================================
"""

import os
import re
import json
import time
import random
import logging
import argparse
import hashlib
import csv
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from urllib.parse import quote_plus, urlencode

import requests
from tqdm import tqdm

# ─── Optional imports (graceful fallback) ─────────────────────────────────────
try:
    from fake_useragent import UserAgent
    UA = UserAgent()
    def get_ua():
        return UA.chrome
except ImportError:
    def get_ua():
        agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        ]
        return random.choice(agents)

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("crawler.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("dataset_test")
TARGET_TOTAL = 300
TARGET_PER_CLASS = TARGET_TOTAL // 2   # 150 each
MIN_DURATION_S = 5
MAX_DURATION_S = 15
DELAY_MIN = 1.5    # seconds between requests
DELAY_MAX = 3.5

# Alamy base URL & API endpoint
ALAMY_BASE = "https://www.alamy.com"
ALAMY_SEARCH_API = "https://www.alamy.com/search/imageresults.aspx"
ALAMY_FOOTAGE_SEARCH = "https://www.alamy.com/footage-search-results.aspx"

# ─── Search Query Bank ─────────────────────────────────────────────────────────
# Mỗi query được gán (class_label, sub_type, search_terms, target_count)
#
# Class 1 – POSITIVE (150 videos tổng)
POSITIVE_QUERIES = [
    # Khói khởi phát (Early-stage smoke) – 35 videos
    {"subtype": "early_smoke",  "queries": [
        "electrical socket sparking smoke top view",
        "trash bin smoke starting fire overhead",
        "smoldering wire smoke factory ceiling view",
        "early stage smoke industrial building cctv",
        "thin smoke rising warehouse corner surveillance",
    ], "target": 35},

    # Khói đặc / Hóa chất (Heavy Chemical Smoke) – 40 videos
    {"subtype": "heavy_smoke",  "queries": [
        "black smoke plastic burning factory overhead",
        "thick black smoke fire warehouse cctv",
        "rubber tire burning dark smoke top view",
        "chemical fire heavy smoke industrial building",
        "dense smoke rolling fire plant overhead camera",
        "pallet fire black smoke factory surveillance",
    ], "target": 40},

    # Khói lan tỏa (Diffused Smoke) – 35 videos
    {"subtype": "diffused_smoke", "queries": [
        "white smoke spreading warehouse room surveillance",
        "paper fire smoke filling room overhead camera",
        "fabric cloth fire white smoke top view cctv",
        "smoke filling factory floor overhead view",
        "diffused smoke fire industrial building camera",
    ], "target": 35},

    # Lửa trần (Visible Flame) – 40 videos
    {"subtype": "visible_flame", "queries": [
        "fire flame burning factory warehouse overhead cctv",
        "fire outbreak industrial floor top view camera",
        "flame burning boxes factory surveillance camera",
        "fire visible flame manufacturing plant overhead",
        "open fire burning warehouse floor cctv footage",
    ], "target": 40},
]

# Class 0 – NEGATIVE / HARD NEGATIVES (150 videos tổng)
NEGATIVE_QUERIES = [
    # Bụi xưởng / Hạt li ti – 30 videos
    {"subtype": "dust_particles", "queries": [
        "sawdust flying woodworking factory overhead",
        "cement dust cloud construction overhead camera",
        "workshop dust particles air factory cctv",
        "flour dust cloud industrial plant top view",
    ], "target": 30},

    # Hơi nước / Steam – 30 videos
    {"subtype": "steam_vapor", "queries": [
        "steam pipe industrial factory overhead camera",
        "pressure valve steam release factory top view",
        "cooking steam kitchen industrial overhead cctv",
        "fog machine mist indoor factory surveillance",
        "steam boiler factory overhead view",
    ], "target": 30},

    # Tia lửa hàn (Welding sparks) – 25 videos
    {"subtype": "welding_sparks", "queries": [
        "welding sparks metal workshop overhead camera",
        "grinding sparks factory floor top view",
        "metal cutting sparks industrial overhead cctv",
        "welder sparks flying factory surveillance",
    ], "target": 25},

    # Nhiễu quang học (Optical / Lighting) – 25 videos
    {"subtype": "optical_noise", "queries": [
        "forklift flashing orange light warehouse cctv",
        "warning beacon strobe light factory floor overhead",
        "sunlight reflection factory floor surveillance camera",
        "flickering light industrial warehouse top view",
    ], "target": 25},

    # Khói thuốc lá / vape / nhang – 20 videos
    {"subtype": "cigarette_smoke", "queries": [
        "cigarette smoke person factory area overhead",
        "vape smoke person indoor top view cctv",
        "incense smoke small plume overhead camera",
        "smoking person warehouse cctv surveillance",
    ], "target": 20},
]

# ─── Dataclass cho mỗi video đã crawl ────────────────────────────────────────
@dataclass
class VideoEntry:
    alamy_id: str
    class_label: int          # 0 or 1
    subtype: str
    title: str
    duration_s: float
    preview_url: str
    thumbnail_url: str
    search_query: str
    page: int
    local_path: str = ""
    download_status: str = "pending"  # pending | ok | skip | error
    skip_reason: str = ""

# ─── Session Helper ────────────────────────────────────────────────────────────
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": ALAMY_BASE + "/",
        "Origin": ALAMY_BASE,
        "Cache-Control": "no-cache",
    })
    return s

def random_delay():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

# ─── Alamy Search API ──────────────────────────────────────────────────────────
def search_alamy_footage(session: requests.Session, query: str, page: int = 1,
                          per_page: int = 50) -> dict:
    """
    Gọi Alamy internal API để tìm kiếm footage (video clips).
    Trả về dict JSON từ response hoặc {} nếu thất bại.

    Alamy sử dụng endpoint search với các param:
      qt        = query text
      imgt      = 0 (tất cả), 1 (RF), 2 (RM)
      mediatype = footage
      sortby    = relevant | newest
      p         = page (1-based)
      pn        = per page (max 100)
    """
    session.headers["User-Agent"] = get_ua()

    params = {
        "qt": query,
        "imgt": 0,          # RF + RM
        "sortby": "relevant",
        "mediatype": "footage",
        "p": page,
        "pn": per_page,
        "lic": "2",         # RF only để tránh tính phí
        "imageid": "",
        "from": "",
        "to": "",
        "location": "",
        "fromc": "",
        "toc": "",
        "Contenttype": "1",
        "currentPage": page,
    }

    try:
        r = session.get(ALAMY_SEARCH_API, params=params, timeout=20)
        r.raise_for_status()

        ct = r.headers.get("Content-Type", "")
        if "json" in ct:
            return r.json()

        # Alamy trả về HTML khi không có JSON → parse thủ công
        return _parse_alamy_html_results(r.text, query, page)

    except requests.RequestException as e:
        log.warning(f"Search failed for '{query}' page {page}: {e}")
        return {}


def _parse_alamy_html_results(html: str, query: str, page: int) -> dict:
    """
    Fallback: Parse HTML response từ Alamy search để lấy video metadata.
    Alamy nhúng data vào script tag dạng window.__INITIAL_STATE__ hoặc JSON-LD.
    """
    # Thử trích xuất JSON từ window.__INITIAL_STATE__
    pattern_state = re.compile(
        r'window\.__INITIAL_STATE__\s*=\s*({.*?});\s*</script>',
        re.DOTALL
    )
    m = pattern_state.search(html)
    if m:
        try:
            data = json.loads(m.group(1))
            return _normalize_initial_state(data)
        except json.JSONDecodeError:
            pass

    # Thử trích xuất từ data-track hoặc data-assets attribute
    pattern_assets = re.compile(r'data-assets="([^"]+)"')
    m2 = pattern_assets.search(html)
    if m2:
        try:
            import html as html_mod
            raw = html_mod.unescape(m2.group(1))
            data = json.loads(raw)
            return {"assets": data, "query": query, "page": page}
        except (json.JSONDecodeError, Exception):
            pass

    # Thử trích xuất từ JSON-LD
    pattern_ld = re.compile(
        r'<script type="application/ld\+json">(.*?)</script>',
        re.DOTALL
    )
    items = []
    for m3 in pattern_ld.finditer(html):
        try:
            obj = json.loads(m3.group(1))
            if isinstance(obj, list):
                items.extend(obj)
            elif isinstance(obj, dict):
                items.append(obj)
        except json.JSONDecodeError:
            continue

    if items:
        return {"jsonld": items, "query": query, "page": page}

    log.debug(f"Could not parse Alamy HTML for query='{query}' page={page}")
    return {}


def _normalize_initial_state(state: dict) -> dict:
    """Chuẩn hóa cấu trúc window.__INITIAL_STATE__ về format thống nhất."""
    results = []
    try:
        assets = (state.get("search", {})
                       .get("results", {})
                       .get("assets", []))
        for a in assets:
            results.append({
                "id":           a.get("id", ""),
                "title":        a.get("caption", a.get("title", "")),
                "duration":     a.get("duration", 0),
                "previewUrl":   a.get("previewUrl", a.get("videoUrl", "")),
                "thumbnailUrl": a.get("thumbnailUrl", a.get("thumbnail", "")),
            })
    except Exception:
        pass
    return {"normalizedAssets": results}


def extract_video_entries_from_response(resp: dict, class_label: int,
                                         subtype: str, query: str,
                                         page: int) -> list[VideoEntry]:
    """
    Chuyển đổi raw response từ Alamy thành danh sách VideoEntry.
    Lọc theo tiêu chí: thời lượng, có preview URL.
    """
    entries = []

    # --- Xử lý dạng normalizedAssets ---
    assets = resp.get("normalizedAssets", [])

    # --- Xử lý dạng assets thô ---
    if not assets:
        assets = resp.get("assets", resp.get("data", {}).get("assets", []))

    for asset in assets:
        if not isinstance(asset, dict):
            continue

        asset_id = str(asset.get("id", asset.get("imageId", "")))
        if not asset_id:
            continue

        title = asset.get("title", asset.get("caption", "untitled"))
        duration_raw = asset.get("duration", asset.get("clipDuration", 0))

        # Normalize duration: đôi khi Alamy trả về milliseconds
        try:
            dur_s = float(duration_raw)
            if dur_s > 1000:          # có thể là ms
                dur_s = dur_s / 1000.0
        except (TypeError, ValueError):
            dur_s = 0.0

        # Lọc thời lượng
        if dur_s < MIN_DURATION_S or dur_s > MAX_DURATION_S:
            continue

        preview_url = (asset.get("previewUrl", "") or
                       asset.get("videoUrl", "") or
                       asset.get("streamUrl", ""))
        thumb_url   = (asset.get("thumbnailUrl", "") or
                       asset.get("thumbnail", ""))

        if not preview_url:
            continue

        entries.append(VideoEntry(
            alamy_id=asset_id,
            class_label=class_label,
            subtype=subtype,
            title=title[:120],
            duration_s=dur_s,
            preview_url=preview_url,
            thumbnail_url=thumb_url,
            search_query=query,
            page=page,
        ))

    return entries


# ─── Download Helper ───────────────────────────────────────────────────────────
def download_video(session: requests.Session, entry: VideoEntry,
                   output_dir: Path, dry_run: bool = False) -> bool:
    """
    Download preview video từ URL đã biết.
    Trả về True nếu thành công.
    """
    # Tạo thư mục theo class / subtype
    save_dir = output_dir / f"class_{entry.class_label}" / entry.subtype
    save_dir.mkdir(parents=True, exist_ok=True)

    # Tên file: {subtype}_{alamy_id}.mp4
    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", entry.alamy_id)
    filename = f"{entry.subtype}_{safe_id}.mp4"
    filepath = save_dir / filename
    entry.local_path = str(filepath)

    if filepath.exists() and filepath.stat().st_size > 10_000:
        log.debug(f"Already exists: {filename}")
        entry.download_status = "ok"
        return True

    if dry_run:
        entry.download_status = "dry_run"
        return True

    try:
        session.headers["User-Agent"] = get_ua()
        session.headers["Referer"] = f"{ALAMY_BASE}/stock-video/{entry.alamy_id}.html"

        with session.get(entry.preview_url, stream=True, timeout=30) as r:
            r.raise_for_status()

            content_type = r.headers.get("Content-Type", "")
            if "video" not in content_type and "octet" not in content_type:
                entry.download_status = "skip"
                entry.skip_reason = f"Non-video content-type: {content_type}"
                return False

            total = int(r.headers.get("Content-Length", 0))
            with open(filepath, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

        # Kiểm tra file tải về có hợp lệ không
        file_size = filepath.stat().st_size
        if file_size < 50_000:   # < 50KB → bỏ
            filepath.unlink(missing_ok=True)
            entry.download_status = "skip"
            entry.skip_reason = f"File too small: {file_size} bytes"
            return False

        entry.download_status = "ok"
        log.info(f"✓ Downloaded: {filename} ({file_size/1024:.1f} KB)")
        return True

    except requests.RequestException as e:
        log.warning(f"Download failed [{entry.alamy_id}]: {e}")
        if filepath.exists():
            filepath.unlink(missing_ok=True)
        entry.download_status = "error"
        entry.skip_reason = str(e)
        return False


# ─── yt-dlp Fallback ──────────────────────────────────────────────────────────
def try_ytdlp_download(entry: VideoEntry, output_dir: Path) -> bool:
    """
    Fallback dùng yt-dlp khi direct download thất bại.
    yt-dlp hỗ trợ Alamy natively.
    """
    try:
        import yt_dlp
    except ImportError:
        log.warning("yt-dlp not installed. Run: pip install yt-dlp")
        return False

    save_dir = output_dir / f"class_{entry.class_label}" / entry.subtype
    save_dir.mkdir(parents=True, exist_ok=True)

    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", entry.alamy_id)
    outtmpl = str(save_dir / f"{entry.subtype}_{safe_id}.%(ext)s")

    # Alamy stock video URL
    page_url = f"{ALAMY_BASE}/stock-video/{entry.alamy_id}.html"

    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "mp4[height<=1080][height>=480]/bestvideo[height<=1080]+bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 3,
        "http_headers": {
            "User-Agent": get_ua(),
            "Referer": ALAMY_BASE,
        },
        # Lọc theo thời lượng
        "match_filter": yt_dlp.utils.match_filter_func(
            f"duration >= {MIN_DURATION_S} & duration <= {MAX_DURATION_S}"
        ),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(page_url, download=True)
            if info:
                # Tìm file vừa tải về
                ext = info.get("ext", "mp4")
                filepath = Path(outtmpl.replace("%(ext)s", ext))
                if filepath.exists():
                    entry.local_path = str(filepath)
                    entry.download_status = "ok"
                    log.info(f"✓ yt-dlp: {filepath.name}")
                    return True
    except yt_dlp.utils.DownloadError as e:
        log.debug(f"yt-dlp failed [{entry.alamy_id}]: {e}")

    entry.download_status = "error"
    entry.skip_reason = "yt-dlp failed"
    return False


# ─── Exclusion Filter ─────────────────────────────────────────────────────────
EXCLUSION_KEYWORDS = [
    # Vi phạm tiêu chí loại bỏ
    "phone screen", "mobile screen", "smartphone recording",
    "slow motion", "timelapse", "time lapse", "time-lapse",
    "zoom in", "close-up fire", "extreme closeup",
]

def should_exclude(entry: VideoEntry) -> tuple[bool, str]:
    """Kiểm tra xem video có vi phạm tiêu chí loại bỏ không."""
    title_lower = entry.title.lower()

    for kw in EXCLUSION_KEYWORDS:
        if kw in title_lower:
            return True, f"Excluded keyword: '{kw}'"

    # Kiểm tra thời lượng (đã lọc trước nhưng double-check)
    if entry.duration_s < MIN_DURATION_S or entry.duration_s > MAX_DURATION_S:
        return True, f"Duration {entry.duration_s:.1f}s out of range [{MIN_DURATION_S}, {MAX_DURATION_S}]"

    return False, ""


# ─── Manifest Saver ───────────────────────────────────────────────────────────
def save_manifest(entries: list[VideoEntry], output_dir: Path):
    """Lưu manifest CSV đầy đủ metadata."""
    manifest_path = output_dir / "manifest.csv"

    fieldnames = [
        "alamy_id", "class_label", "subtype", "title",
        "duration_s", "search_query", "page",
        "local_path", "download_status", "skip_reason",
        "preview_url", "thumbnail_url",
    ]

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            row = asdict(e)
            writer.writerow({k: row[k] for k in fieldnames})

    log.info(f"Manifest saved: {manifest_path} ({len(entries)} entries)")


def save_progress_json(entries: list[VideoEntry], output_dir: Path):
    """Lưu progress JSON để có thể resume."""
    progress = {
        "total": len(entries),
        "ok": sum(1 for e in entries if e.download_status == "ok"),
        "pending": sum(1 for e in entries if e.download_status == "pending"),
        "skipped": sum(1 for e in entries if e.download_status in ("skip", "dry_run")),
        "error": sum(1 for e in entries if e.download_status == "error"),
        "by_class": {
            "class_0": sum(1 for e in entries if e.class_label == 0 and e.download_status == "ok"),
            "class_1": sum(1 for e in entries if e.class_label == 1 and e.download_status == "ok"),
        },
        "entries": [asdict(e) for e in entries],
    }

    progress_path = output_dir / "progress.json"
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    return progress


def load_progress_json(output_dir: Path) -> list[VideoEntry]:
    """Load tiến trình đã lưu để resume."""
    progress_path = output_dir / "progress.json"
    if not progress_path.exists():
        return []

    with open(progress_path, encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for row in data.get("entries", []):
        try:
            e = VideoEntry(**row)
            entries.append(e)
        except Exception:
            continue

    log.info(f"Resumed from progress.json: {len(entries)} entries")
    return entries


# ─── Main Crawl Logic ─────────────────────────────────────────────────────────
def crawl_group(session: requests.Session, group: dict, class_label: int,
                existing_ids: set, output_dir: Path, dry_run: bool,
                pbar: tqdm) -> list[VideoEntry]:
    """Crawl một query group (một subtype) cho đến khi đủ target."""
    collected: list[VideoEntry] = []
    subtype = group["subtype"]
    queries = group["queries"]
    target = group["target"]

    per_query_target = max(1, target // len(queries) + 2)  # +2 buffer

    for query in queries:
        if sum(1 for e in collected if e.download_status == "ok") >= target:
            break

        for page in range(1, 6):  # tối đa 5 trang mỗi query
            if sum(1 for e in collected if e.download_status == "ok") >= target:
                break

            log.info(f"[{subtype}] Searching: '{query}' page {page}")
            resp = search_alamy_footage(session, query, page=page, per_page=50)
            random_delay()

            if not resp:
                log.warning(f"Empty response for '{query}' page {page}")
                break

            entries = extract_video_entries_from_response(
                resp, class_label, subtype, query, page
            )
            log.info(f"  → Found {len(entries)} candidates after duration filter")

            for entry in entries:
                if entry.alamy_id in existing_ids:
                    continue

                # Kiểm tra exclusion
                excluded, reason = should_exclude(entry)
                if excluded:
                    entry.download_status = "skip"
                    entry.skip_reason = reason
                    collected.append(entry)
                    existing_ids.add(entry.alamy_id)
                    continue

                # Thử direct download
                ok = download_video(session, entry, output_dir, dry_run)
                if not ok and not dry_run:
                    # Fallback yt-dlp
                    ok = try_ytdlp_download(entry, output_dir)

                collected.append(entry)
                existing_ids.add(entry.alamy_id)

                if ok:
                    pbar.update(1)

                random_delay()

            if not entries:
                break  # Không có kết quả → sang query tiếp

    ok_count = sum(1 for e in collected if e.download_status == "ok")
    log.info(f"[{subtype}] Collected {ok_count}/{target} OK videos")
    return collected


def run_crawler(output_dir: Path = OUTPUT_DIR,
                dry_run: bool = False,
                resume: bool = True):
    """Entry point chính."""
    output_dir.mkdir(parents=True, exist_ok=True)
    session = make_session()

    # Load existing progress
    all_entries: list[VideoEntry] = []
    existing_ids: set = set()

    if resume:
        all_entries = load_progress_json(output_dir)
        existing_ids = {e.alamy_id for e in all_entries}

    ok_count_init = sum(1 for e in all_entries if e.download_status == "ok")
    log.info(f"Starting crawler. Already have {ok_count_init}/{TARGET_TOTAL} OK videos.")

    ok_class0 = sum(1 for e in all_entries if e.class_label == 0 and e.download_status == "ok")
    ok_class1 = sum(1 for e in all_entries if e.class_label == 1 and e.download_status == "ok")
    remaining = TARGET_TOTAL - (ok_class0 + ok_class1)

    with tqdm(total=TARGET_TOTAL, initial=ok_class0 + ok_class1,
              desc="Total videos", unit="vid") as pbar:

        # ── Crawl Class 1 – Positive ──────────────────────────────────────────
        if ok_class1 < TARGET_PER_CLASS:
            log.info(f"=== Crawling Class 1 (Positive). Have {ok_class1}/{TARGET_PER_CLASS} ===")
            for group in POSITIVE_QUERIES:
                ok_so_far = sum(
                    1 for e in all_entries
                    if e.class_label == 1 and e.download_status == "ok"
                )
                if ok_so_far >= TARGET_PER_CLASS:
                    break

                new_entries = crawl_group(
                    session, group, class_label=1,
                    existing_ids=existing_ids,
                    output_dir=output_dir,
                    dry_run=dry_run,
                    pbar=pbar,
                )
                all_entries.extend(new_entries)
                save_progress_json(all_entries, output_dir)
                save_manifest(all_entries, output_dir)

        # ── Crawl Class 0 – Negative ──────────────────────────────────────────
        if ok_class0 < TARGET_PER_CLASS:
            log.info(f"=== Crawling Class 0 (Negative). Have {ok_class0}/{TARGET_PER_CLASS} ===")
            for group in NEGATIVE_QUERIES:
                ok_so_far = sum(
                    1 for e in all_entries
                    if e.class_label == 0 and e.download_status == "ok"
                )
                if ok_so_far >= TARGET_PER_CLASS:
                    break

                new_entries = crawl_group(
                    session, group, class_label=0,
                    existing_ids=existing_ids,
                    output_dir=output_dir,
                    dry_run=dry_run,
                    pbar=pbar,
                )
                all_entries.extend(new_entries)
                save_progress_json(all_entries, output_dir)
                save_manifest(all_entries, output_dir)

    # ── Final Report ──────────────────────────────────────────────────────────
    progress = save_progress_json(all_entries, output_dir)
    save_manifest(all_entries, output_dir)

    print("\n" + "="*60)
    print("  CRAWL COMPLETE – SUMMARY")
    print("="*60)
    print(f"  Total entries:  {progress['total']}")
    print(f"  ✓ Downloaded:   {progress['ok']}")
    print(f"  ✗ Errors:       {progress['error']}")
    print(f"  ⊘ Skipped:      {progress['skipped']}")
    print(f"  Class 0 OK:     {progress['by_class']['class_0']} / {TARGET_PER_CLASS}")
    print(f"  Class 1 OK:     {progress['by_class']['class_1']} / {TARGET_PER_CLASS}")
    print(f"  Output dir:     {output_dir.resolve()}")
    print("="*60)

    if progress['ok'] < TARGET_TOTAL:
        shortfall = TARGET_TOTAL - progress['ok']
        print(f"\n⚠ Còn thiếu {shortfall} videos.")
        print("  Gợi ý:")
        print("  1. Thêm query từ POSITIVE_QUERIES / NEGATIVE_QUERIES")
        print("  2. Tăng số trang (range(1, 6) → range(1, 10))")
        print("  3. Chạy lại với --resume để tiếp tục từ chỗ dở")

    return all_entries


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Alamy Video Crawler – Fire/Smoke Detection Test Dataset"
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chỉ crawl metadata, không download video"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Bắt đầu lại từ đầu, bỏ qua progress.json"
    )
    parser.add_argument(
        "--delay-min", type=float, default=DELAY_MIN,
        help=f"Thời gian chờ tối thiểu giữa requests (giây, default {DELAY_MIN})"
    )
    parser.add_argument(
        "--delay-max", type=float, default=DELAY_MAX,
        help=f"Thời gian chờ tối đa giữa requests (giây, default {DELAY_MAX})"
    )

    args = parser.parse_args()

    DELAY_MIN = args.delay_min
    DELAY_MAX = args.delay_max

    run_crawler(
        output_dir=Path(args.output),
        dry_run=args.dry_run,
        resume=not args.no_resume,
    )