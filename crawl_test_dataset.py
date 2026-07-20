"""
=============================================================================
TEST DATASET CRAWLER  —  Fire & Smoke Detection  (3-class)
=============================================================================
Nhãn & subtype dựa trực tiếp trên phân phối thực tế của train set:

  0_normal_noise       (8 subtypes × 35 = 280 video)
  1_controlled_fire    (7 subtypes × 35 = 245 video)
  2_uncontrolled_hazard(7 subtypes × 35 = 245 video)
  ─────────────────────────────────────────────────
  TOTAL: 770 video  ·  22 subtypes  ·  35 video/subtype

Duration: 20 giây – 5 phút (khác train set 5–15s)
Nguồn   : Pexels API + Pixabay API (miễn phí, không watermark)

CÀI ĐẶT:
  pip install requests tqdm

LẤY API KEY:
  Pexels  → https://www.pexels.com/api/
  Pixabay → https://pixabay.com/api/docs/

CHẠY:
  python crawl_test_dataset.py --pexels-key  oATsNNUxtm2zGHii6jpfHGPXxXUcaO1H9RDxRVve6N7P4BzmNXrPgZjo  --pixabay-key 56437068-9cd3b2f0e69fee4db3646368d --output dataset_test
  # Chỉ crawl 1 nhãn
  python crawl_test_dataset.py --pexels-key K --pixabay-key K --label 0
  python crawl_test_dataset.py --pexels-key K --pixabay-key K --label 1
  python crawl_test_dataset.py --pexels-key K --pixabay-key K --label 2

  # Dry-run (chỉ xem metadata, không download)
  python crawl_test_dataset.py --pexels-key K --pixabay-key K --dry-run

  # Resume sau khi bị gián đoạn
  python crawl_test_dataset.py --pexels-key K --pixabay-key K  # resume mặc định
=============================================================================
"""

import os, re, json, time, random, logging, argparse, csv, hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import requests
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_crawler.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_DUR   = 20       # giây  — test set cần video dài hơn train
MAX_DUR   = 300      # giây  — tối đa 5 phút
MIN_H     = 480
MAX_H     = 1080
MIN_BYTES = 500_000  # 500 KB

DELAY_MIN = 1.5
DELAY_MAX = 3.5

PEXELS_URL  = "https://api.pexels.com/videos/search"
PIXABAY_URL = "https://pixabay.com/api/videos/"

PER_SUBTYPE = 35  # target mỗi subtype

# ── Query Bank — 22 subtypes ───────────────────────────────────────────────────
#
# Thiết kế query dựa trên phân tích phân phối thực tế của train set.
# Mỗi query gồm 3–6 từ, có "anchor" không gian indoor/factory/building
# để tránh bias về cảnh ngoài trời / cinematic.
#
# Format: { "label": int, "subtype": str, "queries": list[str] }
#

ALL_GROUPS = [

    # ═══════════════════════════════════════════════════════════════════════════
    # NHÃN 0 — Normal noise  (8 subtypes × 35 = 280)
    # Không có lửa/khói thực sự — dễ gây false positive
    # ═══════════════════════════════════════════════════════════════════════════

    {
        "label":   0,
        "subtype": "dust_particles",
        # Bụi bay trong không khí — chiếm 64% label 0 trong train set
        # Test cần đa dạng hơn: xưởng gỗ, xi măng, bột mì, bụi sập
        "queries": [
            "sawdust flying woodworking workshop indoor",
            "wood cutting dust cloud workshop interior",
            "cement dust construction indoor room",
            "concrete grinding dust cloud indoor",
            "drywall sanding dust indoor building",
            "flour dust cloud bakery kitchen indoor",
            "powder dust industrial factory air",
            "grain silo dust indoor building",
            "sand blasting dust indoor factory",
            "demolition dust cloud building interior",
            "dust particles sunlight beam warehouse",
            "fiberglass dust insulation indoor work",
        ],
    },

    {
        "label":   0,
        "subtype": "optical_noise",
        # Nhiễu quang học — đèn nhấp nháy, flare, ánh sáng lạ
        # Chiếm 19% label 0 trong train set
        "queries": [
            "forklift orange beacon light warehouse indoor",
            "forklift flashing warning light factory floor",
            "strobe warning light factory indoor ceiling",
            "emergency strobe alarm light indoor hallway",
            "sunlight beam shaft dust indoor warehouse",
            "flickering fluorescent light factory indoor",
            "headlight glare reflection indoor parking",
            "laser light show indoor event room",
            "neon sign flickering indoor dark room",
            "welding arc light flash factory indoor",
        ],
    },

    {
        "label":   0,
        "subtype": "daily_activity_fp",
        # Hoạt động sinh hoạt dễ trigger nhầm — 6% label 0
        # Test cần cover thêm: hơi thở lạnh, máy in, dry ice
        "queries": [
            "cold breath vapor person indoor winter",
            "breath mist cold warehouse factory indoor",
            "dry ice vapor fog indoor event",
            "fog machine mist indoor concert room",
            "printer paper dust exhaust office indoor",
            "person running fast warehouse floor cctv",
            "forklift moving fast warehouse indoor",
            "air conditioner vent condensation indoor",
            "hvac ceiling vent air flow indoor office",
            "shadow moving wall indoor factory cctv",
        ],
    },

    {
        "label":   0,
        "subtype": "cigarette_vape_smoke",
        # Khói thuốc lá / vape / nhang — hiện chỉ có 2 clip trong label 0 train
        # Đây là subtype bị thiếu nghiêm trọng nhất của label 0
        "queries": [
            "person smoking cigarette indoor room close",
            "cigarette smoke exhale indoor office person",
            "vape smoke exhale indoor person room",
            "e-cigarette vapor indoor person exhale",
            "incense stick smoke burning indoor room",
            "joss stick smoke temple indoor close",
            "pipe smoke person indoor room",
            "hookah shisha smoke indoor lounge",
            "cigarette smoke floating indoor dim light",
            "vape cloud person indoor dark room",
        ],
    },

    {
        "label":   0,
        "subtype": "welding_sparks_noise",
        # Tia lửa hàn — chỉ có 2 clip trong label 0 train
        # Khác với welding_sparks_ctrl (label 1): ở đây góc nhìn xa hơn, ít smoke
        "queries": [
            "welding sparks flying workshop indoor",
            "angle grinder sparks metal workshop floor",
            "metal cutting sparks factory indoor",
            "plasma cutter sparks indoor factory",
            "grinder sparks workshop concrete floor",
            "metalwork grinding sparks indoor building",
            "spot welding sparks assembly line indoor",
            "sparks flying metal fabrication indoor",
            "arc welding light flash factory indoor",
            "cutting torch sparks metal indoor workshop",
        ],
    },

    {
        "label":   0,
        "subtype": "steam_vapor",
        # Hơi nước / steam — KHÔNG CÓ trong label 0 train (0 clip)
        # Ưu tiên cao nhất để lấp đầy khoảng trống
        "queries": [
            "steam pipe release industrial factory indoor",
            "pressure valve steam release building indoor",
            "boiler steam cloud factory interior",
            "steam cleaning machine factory floor indoor",
            "industrial autoclave steam release indoor",
            "steam rising large pot kitchen indoor",
            "cooking pot steam cloud kitchen indoor",
            "commercial kitchen steam exhaust indoor",
            "laundry steam press indoor factory",
            "car wash steam indoor facility",
            "hot spring steam indoor spa room",
            "steam iron clothes indoor room person",
        ],
    },

    {
        "label":   0,
        "subtype": "cooking_smoke_normal",
        # Khói nấu ăn bình thường — KHÔNG CÓ trong label 0 train (0 clip)
        # Quan trọng: phân biệt với cooking_smoke_threshold (label 1)
        # Label 0: nấu ăn bình thường, không có nguy cơ
        "queries": [
            "cooking smoke steam kitchen home normal",
            "frying pan steam smoke kitchen normal indoor",
            "wok cooking steam smoke kitchen restaurant",
            "stove cooking normal smoke steam indoor",
            "pot boiling steam kitchen indoor home",
            "chef cooking smoke steam restaurant kitchen",
            "barbecue grill smoke outdoor patio normal",
            "smoker bbq meat smoke outdoor backyard",
            "pizza oven smoke steam indoor restaurant",
            "bread baking steam oven indoor bakery",
        ],
    },

    {
        "label":   0,
        "subtype": "decorative_fire_fp",
        # Lửa trang trí / không nguy hiểm — KHÔNG CÓ trong label 0 train
        # Lửa thật nhưng model phải học KHÔNG báo động
        "queries": [
            "fireplace fire burning indoor living room",
            "wood fireplace flames indoor home cozy",
            "gas fireplace indoor home burning calm",
            "candle flame burning indoor table dark",
            "multiple candles burning indoor ceremony",
            "oil lantern flame burning indoor night",
            "campfire outdoor controlled backyard night",
            "bonfire outdoor gathering night controlled",
            "fire pit outdoor patio burning calm",
            "sparkler handheld night outdoor celebration",
            "birthday candles cake flame indoor",
            "tiki torch flame outdoor garden night",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # NHÃN 1 — Controlled fire  (7 subtypes × 35 = 245)
    # Lửa/khói có kiểm soát — ranh giới khó phân biệt với nhãn 2
    # ═══════════════════════════════════════════════════════════════════════════

    {
        "label":   1,
        "subtype": "early_smoke_onset",
        # Khói khởi phát — chỉ có 1 clip trong label 1 train
        # ĐÂY LÀ SUBTYPE QUAN TRỌNG NHẤT cho early detection
        # Khói mỏng bốc lên từ nguồn nhỏ, chưa lan rộng
        "queries": [
            "electrical outlet smoke starting indoor room",
            "socket sparking smoke indoor wall",
            "smoldering wire smoke indoor ceiling",
            "circuit board smoke burn indoor",
            "appliance overheating smoke thin indoor",
            "trash bin smoke smoldering indoor start",
            "cardboard smoldering smoke corner indoor",
            "paper smoldering smoke thin indoor room",
            "early stage smoke rising indoor building",
            "thin smoke starting fire indoor corner",
            "smoke beginning rising warehouse floor",
            "small smoke source indoor room start",
        ],
    },

    {
        "label":   1,
        "subtype": "cigarette_controlled",
        # Khói thuốc lá kiểm soát — 78 clip trong train (nhiều nhất label 1)
        # Test cần đa dạng bối cảnh: văn phòng, bar, ngoài trời
        "queries": [
            "person smoking cigarette outdoor walking",
            "cigarette smoke person outdoor street",
            "smoking area outdoor people cigarette",
            "man smoking cigarette office building outside",
            "woman smoking cigarette break outdoor",
            "cigarette smoke outdoor wind dispersing",
            "group people smoking outdoor area",
            "cigarette smoke indoor bar lounge",
            "person smoking indoor factory break room",
            "cigarette butt smoke ashtray indoor",
        ],
    },

    {
        "label":   1,
        "subtype": "welding_sparks_ctrl",
        # Tia lửa hàn có kiểm soát — 76 clip trong train label 1
        # Khác label 0: có smoke rõ hơn, công nhân đang làm việc tích cực
        "queries": [
            "welder working sparks smoke factory indoor",
            "welding smoke sparks metal fabrication",
            "mig welding smoke sparks close factory",
            "tig welding smoke arc indoor workshop",
            "welder helmet sparks smoke industrial",
            "metal welding torch flame sparks indoor",
            "pipe welding sparks smoke construction",
            "shipyard welding sparks smoke industrial",
            "auto body welding sparks smoke garage",
            "structural steel welding sparks smoke",
        ],
    },

    {
        "label":   1,
        "subtype": "cooking_smoke_threshold",
        # Khói nấu ăn vượt ngưỡng — 32 clip trong train label 1
        # Khác label 0: nhiều khói hơn, có thể kích hoạt detector
        # Khác label 2: vẫn trong tầm kiểm soát, không có nguy hiểm thực
        "queries": [
            "frying pan smoke heavy kitchen indoor",
            "burnt food smoke pan kitchen indoor",
            "stove fire smoke heavy kitchen indoor",
            "cooking smoke thick kitchen ventilation",
            "deep fryer smoke heavy kitchen indoor",
            "smoking pan overheated kitchen indoor",
            "grill heavy smoke indoor kitchen",
            "wok high heat heavy smoke kitchen",
            "popcorn burnt microwave smoke indoor",
            "toast burning smoke indoor kitchen",
            "smoking oil pan kitchen smoke alarm",
            "overcooked food heavy smoke kitchen",
        ],
    },

    {
        "label":   1,
        "subtype": "heavy_smoke_controlled",
        # Khói đặc trong môi trường kiểm soát — 28 clip train label 1
        # Studio, biểu diễn, thử nghiệm — khói nhiều nhưng KHÔNG nguy hiểm
        "queries": [
            "smoke machine effect indoor studio controlled",
            "fog machine smoke indoor concert controlled",
            "theatrical smoke effect indoor stage",
            "smoke effect indoor photo shoot studio",
            "military smoke grenade training outdoor",
            "smoke bomb color outdoor controlled",
            "fire drill smoke simulation building indoor",
            "smoke test building ventilation indoor",
            "controlled burn outdoor training firefighter",
            "smoke signal outdoor controlled burning",
        ],
    },

    {
        "label":   1,
        "subtype": "small_visible_flame",
        # Ngọn lửa nhỏ kiểm soát — chỉ 1 clip trong train label 1
        # Lửa thấy rõ nhưng còn nhỏ, chưa lan rộng
        "queries": [
            "small fire controlled indoor trash bin",
            "paper burning small flame indoor controlled",
            "small flame burning box indoor",
            "waste bin fire small indoor controlled",
            "small campfire controlled outdoor night",
            "bonfire small controlled outdoor",
            "small fire experiment controlled indoor",
            "candle flame large burning indoor",
            "small fire burning floor indoor controlled",
            "match lighter small flame indoor close",
        ],
    },

    {
        "label":   1,
        "subtype": "daily_activity_fire_ctx",
        # Hoạt động hàng ngày có lửa/khói — 36 clip trong train label 1
        # Nấu ăn thông thường, hút thuốc, hoạt động bình thường có fire/smoke
        "queries": [
            "person cooking indoor stove smoke normal",
            "chef frying food smoke kitchen indoor",
            "person grilling outdoor smoke normal",
            "smoking person walking street outdoor",
            "factory worker near smoke machine indoor",
            "person incense burning indoor room",
            "mechanic welding car garage indoor",
            "farmer burning leaves outdoor field",
            "person campfire outdoor cooking normal",
            "candle lighting indoor ceremony normal",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # NHÃN 2 — Uncontrolled hazard  (7 subtypes × 35 = 245)
    # Lửa/khói thực sự nguy hiểm — mất kiểm soát, cần can thiệp
    # ═══════════════════════════════════════════════════════════════════════════

    {
        "label":   2,
        "subtype": "heavy_smoke_hazard",
        # Khói đặc nguy hiểm — 130/323 = 40% label 2 trong train
        # Khói đen / nâu đậm cuộn nhanh từ đám cháy thực
        "queries": [
            "black smoke fire building hazard",
            "thick black smoke factory fire indoor",
            "heavy dark smoke warehouse fire spreading",
            "dense smoke rolling fire building indoor",
            "dark smoke fire industrial building",
            "heavy black smoke fire structure",
            "thick smoke filling building fire",
            "smoke billowing building fire outdoor",
            "heavy smoke fire residential building",
            "dark brown smoke fire structure collapse",
            "black smoke pouring building windows fire",
            "toxic smoke fire industrial building",
        ],
    },

    {
        "label":   2,
        "subtype": "residential_indoor_fire",
        # Cháy nhà dân — 52 clip trong train label 2
        # Cháy nhà, căn hộ, tòa nhà dân dụng thực tế
        "queries": [
            "house fire indoor room burning real",
            "apartment fire burning indoor real",
            "residential building fire indoor real",
            "home fire burning room indoor real",
            "house fire room flame burning indoor",
            "apartment building fire indoor flames",
            "bedroom fire burning house indoor real",
            "living room fire burning indoor real",
            "kitchen fire out of control house",
            "residential fire smoke flame indoor real",
            "townhouse fire burning indoor real",
            "condo apartment fire indoor smoke flame",
        ],
    },

    {
        "label":   2,
        "subtype": "diffused_smoke_spreading",
        # Khói lan tràn rộng — 38 clip trong train label 2
        # Khói đã lan ra nhiều phòng / khu vực
        "queries": [
            "smoke spreading hallway building fire real",
            "smoke filling corridor building fire",
            "smoke spreading multiple rooms fire real",
            "smoke cloud filling building interior fire",
            "smoke spreading floor building fire real",
            "white gray smoke filling room fire",
            "smoke spreading warehouse fire indoor",
            "smoke diffusing building hallway fire",
            "smoke filled room building fire real",
            "smoke spreading stairwell building fire",
        ],
    },

    {
        "label":   2,
        "subtype": "large_visible_flame",
        # Ngọn lửa lớn / dữ dội — 35 clip trong train label 2
        # Lửa rõ ràng mất kiểm soát, bùng phát mạnh
        "queries": [
            "large fire flame building indoor real",
            "fire flame burning warehouse floor real",
            "fire outbreak flame factory indoor real",
            "intense fire flame burning building",
            "fire spreading flame building indoor real",
            "large fire burning structure real",
            "fire flame engulfing room building",
            "fire burning intense building real",
            "raging fire flame indoor building real",
            "fire spreading floor factory indoor real",
            "large flame burning industrial building",
            "fire engulfing building floor real",
        ],
    },

    {
        "label":   2,
        "subtype": "early_smoke_real_fire",
        # Khói đầu của đám cháy thực — chỉ 9 clip trong train label 2
        # QUAN TRỌNG NHẤT cho early detection — cần ưu tiên crawl
        # Khác label 1 early_smoke: đây là khói của đám cháy THỰC bắt đầu
        "queries": [
            "fire starting smoke indoor building real",
            "smoke first stage real fire building",
            "fire beginning smoke indoor structure real",
            "early fire smoke real building indoor",
            "smoke appearing fire start indoor real",
            "initial smoke fire indoor building real",
            "fire ignition smoke indoor building",
            "smoke rising fire beginning indoor real",
            "fire starting indoor building smoke real",
            "early stage real fire smoke indoor building",
        ],
    },

    {
        "label":   2,
        "subtype": "toxic_material_smoke",
        # Khói hóa chất / vật liệu độc hại — không có subtype rõ trong train
        # Phái sinh từ heavy_smoke: nhựa, cao su, hóa chất cháy
        "queries": [
            "plastic burning toxic smoke indoor fire",
            "rubber tire burning black smoke fire",
            "chemical fire toxic smoke indoor",
            "PVC burning toxic smoke building fire",
            "paint thinner fire toxic smoke indoor",
            "electronic equipment fire toxic smoke",
            "cable wire burning toxic smoke building",
            "foam mattress burning toxic smoke fire",
            "solvent fire toxic smoke indoor building",
            "industrial chemical fire smoke toxic",
        ],
    },

    {
        "label":   2,
        "subtype": "daily_ctx_real_fire",
        # Bối cảnh sinh hoạt trong / gần đám cháy thực — 7 clip train label 2
        # Người chạy trốn, sơ tán, cứu hỏa trong cảnh cháy thực
        "queries": [
            "people evacuating building fire real",
            "firefighter entering burning building fire",
            "people running fire smoke building real",
            "evacuation fire building smoke real",
            "firefighter smoke fire building indoor",
            "emergency fire building people outside real",
            "fire truck building fire real outdoor",
            "person escaping fire building indoor real",
            "crowd evacuating smoke fire building",
            "firefighter battling building fire real",
        ],
    },
]

# ── Exclude keywords — CHỈ lỗi kỹ thuật video ────────────────────────────────
EXCLUDE_KW = [
    "slow motion", "slowmo", "slomo",
    "timelapse", "time lapse", "time-lapse", "hyperlapse",
    "drone", "aerial",
    "phone screen", "mobile screen", "screen recording",
    "animation", "animated", "cartoon", "cgi",
    "3d render", "visual effect", "vfx", "green screen",
]

# ── Data model ─────────────────────────────────────────────────────────────────
@dataclass
class VideoEntry:
    vid_id:       str
    source:       str        # pexels | pixabay
    label:        int        # 0 | 1 | 2
    subtype:      str
    title:        str
    duration_s:   float
    width:        int
    height:       int
    download_url: str
    page_url:     str
    search_query: str
    local_path:   str  = ""
    status:       str  = "pending"  # pending|ok|skip|error
    skip_reason:  str  = ""

# ── Helpers ────────────────────────────────────────────────────────────────────
def rdelay():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

def md5_bytes(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def title_ok(title: str) -> tuple[bool, str]:
    t = title.lower()
    for kw in EXCLUDE_KW:
        if kw in t:
            return False, f"excluded kw: '{kw}'"
    return True, ""

def pick_best_pexels(files: list[dict]) -> Optional[dict]:
    candidates = []
    for f in files:
        h = f.get("height", 0)
        if h < MIN_H or h > MAX_H:
            continue
        link = f.get("link", "")
        if not link:
            continue
        candidates.append((h, f.get("width", 0), link))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    h, w, link = candidates[0]
    return {"height": h, "width": w, "link": link}

def pick_best_pixabay(videos: dict) -> Optional[dict]:
    for key in ["large", "medium", "small"]:
        v = videos.get(key, {})
        h = v.get("height", 0)
        if v.get("url") and MIN_H <= h <= MAX_H:
            return {"height": h, "width": v.get("width", 0), "link": v["url"]}
    return None

# ── Pexels search ──────────────────────────────────────────────────────────────
def search_pexels(api_key: str, query: str, page: int = 1,
                  per_page: int = 40) -> list[dict]:
    headers = {"Authorization": api_key}
    params  = {
        "query":        query,
        "per_page":     per_page,
        "page":         page,
        "min_duration": MIN_DUR,
        "max_duration": MAX_DUR,
    }
    try:
        r = requests.get(PEXELS_URL, headers=headers, params=params, timeout=20)
        if r.status_code == 429:
            log.warning("Pexels rate limit — sleep 60s")
            time.sleep(60)
            return []
        r.raise_for_status()
        return r.json().get("videos", [])
    except requests.RequestException as e:
        log.warning(f"[Pexels] '{query}' p{page}: {e}")
        return []

def pexels_to_entry(v: dict, label: int, subtype: str,
                    query: str) -> Optional[VideoEntry]:
    dur = float(v.get("duration", 0))
    if not (MIN_DUR <= dur <= MAX_DUR):
        return None
    best = pick_best_pexels(v.get("video_files", []))
    if not best:
        return None
    title = v.get("url", "").split("/")[-2].replace("-", " ")
    ok, reason = title_ok(title)
    e = VideoEntry(
        vid_id=str(v["id"]), source="pexels",
        label=label, subtype=subtype,
        title=title[:120], duration_s=dur,
        width=best["width"], height=best["height"],
        download_url=best["link"], page_url=v.get("url", ""),
        search_query=query,
    )
    if not ok:
        e.status = "skip"
        e.skip_reason = reason
    return e

# ── Pixabay search ─────────────────────────────────────────────────────────────
def search_pixabay(api_key: str, query: str, page: int = 1,
                   per_page: int = 50) -> list[dict]:
    params = {
        "key":          api_key,
        "q":            query,
        "per_page":     per_page,
        "page":         page,
        "min_duration": MIN_DUR,
        "max_duration": MAX_DUR,
        "video_type":   "film",
        "lang":         "en",
    }
    try:
        r = requests.get(PIXABAY_URL, params=params, timeout=20)
        if r.status_code == 429:
            log.warning("Pixabay rate limit — sleep 60s")
            time.sleep(60)
            return []
        r.raise_for_status()
        return r.json().get("hits", [])
    except requests.RequestException as e:
        log.warning(f"[Pixabay] '{query}' p{page}: {e}")
        return []

def pixabay_to_entry(v: dict, label: int, subtype: str,
                     query: str) -> Optional[VideoEntry]:
    dur = float(v.get("duration", 0))
    if not (MIN_DUR <= dur <= MAX_DUR):
        return None
    best = pick_best_pixabay(v.get("videos", {}))
    if not best:
        return None
    title = v.get("tags", "untitled")
    ok, reason = title_ok(title)
    e = VideoEntry(
        vid_id=str(v["id"]), source="pixabay",
        label=label, subtype=subtype,
        title=title[:120], duration_s=dur,
        width=best["width"], height=best["height"],
        download_url=best["link"],
        page_url=f"https://pixabay.com/videos/{v['id']}/",
        search_query=query,
    )
    if not ok:
        e.status = "skip"
        e.skip_reason = reason
    return e

# ── Download ───────────────────────────────────────────────────────────────────
def download_entry(entry: VideoEntry, out_dir: Path) -> bool:
    label_name = {0: "0_normal_noise",
                  1: "1_controlled_fire",
                  2: "2_uncontrolled_hazard"}[entry.label]
    save_dir  = out_dir / label_name / entry.subtype
    save_dir.mkdir(parents=True, exist_ok=True)

    safe_id  = re.sub(r"[^a-zA-Z0-9_\-]", "_", entry.vid_id)
    filename = f"{entry.source}_{entry.subtype}_{safe_id}.mp4"
    filepath = save_dir / filename
    entry.local_path = str(filepath)

    if filepath.exists() and filepath.stat().st_size >= MIN_BYTES:
        entry.status = "ok"
        return True

    try:
        r = requests.get(entry.download_url, stream=True, timeout=60)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "")
        if "video" not in ct and "octet-stream" not in ct:
            entry.status = "skip"
            entry.skip_reason = f"bad content-type: {ct}"
            return False

        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=131072):
                if chunk:
                    f.write(chunk)

        size = filepath.stat().st_size
        if size < MIN_BYTES:
            filepath.unlink(missing_ok=True)
            entry.status = "skip"
            entry.skip_reason = f"too small ({size}B)"
            return False

        entry.status = "ok"
        log.info(f"✓ {filename}  {entry.duration_s:.0f}s  "
                 f"{entry.width}×{entry.height}  {size//1024}KB")
        return True

    except requests.RequestException as e:
        if filepath.exists():
            filepath.unlink(missing_ok=True)
        entry.status = "error"
        entry.skip_reason = str(e)
        log.debug(f"✗ Download error [{entry.vid_id}]: {e}")
        return False

# ── Persist ────────────────────────────────────────────────────────────────────
def save_all(entries: list[VideoEntry], out_dir: Path):
    # Progress JSON
    breakdown = {}
    for e in entries:
        key = f"label{e.label}_{e.subtype}"
        breakdown.setdefault(key, {"ok": 0, "skip": 0, "error": 0, "pending": 0})
        breakdown[key][e.status] = breakdown[key].get(e.status, 0) + 1

    data = {
        "total":   len(entries),
        "ok":      sum(1 for e in entries if e.status == "ok"),
        "label_0": sum(1 for e in entries if e.label == 0 and e.status == "ok"),
        "label_1": sum(1 for e in entries if e.label == 1 and e.status == "ok"),
        "label_2": sum(1 for e in entries if e.label == 2 and e.status == "ok"),
        "breakdown": breakdown,
        "entries": [asdict(e) for e in entries],
    }
    with open(out_dir / "test_progress.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Manifest CSV
    fields = ["vid_id", "source", "label", "subtype", "title",
              "duration_s", "width", "height", "local_path",
              "status", "skip_reason", "download_url", "page_url",
              "search_query"]
    with open(out_dir / "test_manifest.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for e in entries:
            w.writerow({k: asdict(e)[k] for k in fields})


def load_progress(out_dir: Path) -> tuple[list[VideoEntry], set[str]]:
    pf = out_dir / "test_progress.json"
    if not pf.exists():
        return [], set()
    with open(pf, encoding="utf-8") as f:
        data = json.load(f)
    entries = []
    for row in data.get("entries", []):
        try:
            entries.append(VideoEntry(**row))
        except Exception:
            pass
    ids = {e.vid_id for e in entries}
    log.info(f"Resume: {len(entries)} entries, {data.get('ok', 0)} OK  "
             f"(L0={data.get('label_0',0)}, "
             f"L1={data.get('label_1',0)}, "
             f"L2={data.get('label_2',0)})")
    return entries, ids

# ── Group crawler ──────────────────────────────────────────────────────────────
def crawl_group(pexels_key: str, pixabay_key: str,
                group: dict, existing_ids: set,
                out_dir: Path, dry_run: bool,
                pbar: tqdm) -> list[VideoEntry]:

    label   = group["label"]
    subtype = group["subtype"]
    queries = group["queries"]
    new_entries: list[VideoEntry] = []

    def ok_count():
        return sum(1 for e in new_entries if e.status == "ok")

    for query in queries:
        if ok_count() >= PER_SUBTYPE:
            break

        # ── Pexels ────────────────────────────────────────────────────────────
        for page in range(1, 10):
            if ok_count() >= PER_SUBTYPE:
                break
            log.info(f"[L{label}][{subtype}][Pexels] '{query}' p{page}")
            hits = search_pexels(pexels_key, query, page=page)
            rdelay()

            for v in hits:
                eid = str(v.get("id", ""))
                if not eid or eid in existing_ids:
                    continue
                entry = pexels_to_entry(v, label, subtype, query)
                if entry is None:
                    continue
                existing_ids.add(eid)

                if entry.status == "skip":
                    new_entries.append(entry)
                    continue

                if not dry_run:
                    ok = download_entry(entry, out_dir)
                else:
                    entry.status = "dry_run"
                    ok = True

                new_entries.append(entry)
                if ok:
                    pbar.update(1)
                rdelay()

            if not hits:
                break

        if ok_count() >= PER_SUBTYPE:
            break

        # ── Pixabay ───────────────────────────────────────────────────────────
        for page in range(1, 10):
            if ok_count() >= PER_SUBTYPE:
                break
            log.info(f"[L{label}][{subtype}][Pixabay] '{query}' p{page}")
            hits = search_pixabay(pixabay_key, query, page=page)
            rdelay()

            for v in hits:
                eid = str(v.get("id", ""))
                if not eid or eid in existing_ids:
                    continue
                entry = pixabay_to_entry(v, label, subtype, query)
                if entry is None:
                    continue
                existing_ids.add(eid)

                if entry.status == "skip":
                    new_entries.append(entry)
                    continue

                if not dry_run:
                    ok = download_entry(entry, out_dir)
                else:
                    entry.status = "dry_run"
                    ok = True

                new_entries.append(entry)
                if ok:
                    pbar.update(1)
                rdelay()

            if not hits:
                break

    final_ok = ok_count()
    log.info(f"[L{label}][{subtype}] {final_ok}/{PER_SUBTYPE} OK")
    if final_ok < PER_SUBTYPE:
        log.warning(f"  ⚠ Thiếu {PER_SUBTYPE - final_ok} — "
                    "thêm query hoặc tăng số trang")
    return new_entries

# ── Main ───────────────────────────────────────────────────────────────────────
def run(pexels_key: str, pixabay_key: str, out_dir: Path,
        resume: bool = True, dry_run: bool = False,
        only_label: Optional[int] = None):

    out_dir.mkdir(parents=True, exist_ok=True)

    all_entries, existing_ids = (load_progress(out_dir)
                                  if resume else ([], set()))

    # Tính target
    groups = [g for g in ALL_GROUPS
              if only_label is None or g["label"] == only_label]
    total_target = len(groups) * PER_SUBTYPE

    already_ok = sum(
        1 for e in all_entries
        if e.status == "ok"
        and (only_label is None or e.label == only_label)
    )

    with tqdm(total=total_target, initial=already_ok,
              desc="Test videos OK", unit="vid", ncols=80) as pbar:

        for group in groups:
            subtype = group["subtype"]
            label   = group["label"]

            done = sum(
                1 for e in all_entries
                if e.label == label
                and e.subtype == subtype
                and e.status == "ok"
            )
            if done >= PER_SUBTYPE:
                log.info(f"[L{label}][{subtype}] already complete ({done})")
                continue

            adj = {**group, "target": PER_SUBTYPE - done}
            new = crawl_group(
                pexels_key, pixabay_key,
                adj, existing_ids=existing_ids,
                out_dir=out_dir, dry_run=dry_run,
                pbar=pbar,
            )
            all_entries.extend(new)
            save_all(all_entries, out_dir)

    save_all(all_entries, out_dir)
    _print_report(all_entries, out_dir, dry_run)

# ── Report ─────────────────────────────────────────────────────────────────────
def _print_report(entries: list[VideoEntry], out_dir: Path, dry_run: bool):
    from collections import defaultdict
    ok_map = defaultdict(int)
    for e in entries:
        if e.status in ("ok", "dry_run"):
            ok_map[(e.label, e.subtype)] += 1

    L0 = sum(v for (l, _), v in ok_map.items() if l == 0)
    L1 = sum(v for (l, _), v in ok_map.items() if l == 1)
    L2 = sum(v for (l, _), v in ok_map.items() if l == 2)

    print("\n" + "="*65)
    print(f"  TEST DATASET CRAWL {'[DRY RUN] ' if dry_run else ''}COMPLETE")
    print("="*65)
    print(f"  Total OK : {L0+L1+L2}")
    print(f"  Label 0  : {L0} / 280")
    print(f"  Label 1  : {L1} / 245")
    print(f"  Label 2  : {L2} / 245")

    LABEL_NAMES = {0: "0_normal_noise",
                   1: "1_controlled_fire",
                   2: "2_uncontrolled_hazard"}

    for lbl in [0, 1, 2]:
        print(f"\n  {LABEL_NAMES[lbl]}:")
        for g in ALL_GROUPS:
            if g["label"] != lbl:
                continue
            cnt = ok_map.get((lbl, g["subtype"]), 0)
            bar = "█" * min(cnt, 35)
            gap = "░" * (35 - min(cnt, 35))
            pct = cnt / PER_SUBTYPE * 100
            print(f"    {g['subtype']:30s}: {cnt:3d}/{PER_SUBTYPE} "
                  f"({pct:5.1f}%)  {bar}{gap}")

    print(f"\n  Output: {out_dir.resolve()}")
    print("="*65)

    if dry_run:
        print("\n  [DRY RUN] Chưa có file nào được download thật.")
        print("  Chạy lại bỏ --dry-run để download.")

# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Test Dataset Crawler — Fire/Smoke 3-class"
    )
    ap.add_argument("--pexels-key",  required=True,
                    help="Pexels API key")
    ap.add_argument("--pixabay-key", required=True,
                    help="Pixabay API key")
    ap.add_argument("--output", default="test_dataset",
                    help="Output directory (default: test_dataset)")
    ap.add_argument("--label", type=int, choices=[0, 1, 2], default=None,
                    help="Chỉ crawl nhãn cụ thể (0, 1 hoặc 2)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Chỉ tìm metadata, không download")
    ap.add_argument("--no-resume", action="store_true",
                    help="Bắt đầu lại từ đầu")
    ap.add_argument("--delay-min", type=float, default=DELAY_MIN)
    ap.add_argument("--delay-max", type=float, default=DELAY_MAX)

    args = ap.parse_args()
    DELAY_MIN = args.delay_min
    DELAY_MAX = args.delay_max

    run(
        pexels_key  = args.pexels_key,
        pixabay_key = args.pixabay_key,
        out_dir     = Path(args.output),
        resume      = not args.no_resume,
        dry_run     = args.dry_run,
        only_label  = args.label,
    )
