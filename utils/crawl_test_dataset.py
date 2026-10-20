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
    #
    # NGUYÊN TẮC QUERY NHÃN 0:
    #   - Góc nhìn rộng / xa: "wide shot", "full room view", "overhead"
    #   - Không gian cụ thể: warehouse aisle, factory floor, corridor
    #   - Mô tả HÀNH VI vật lý: dust floating, mist rising, sparks flying
    #   - Tuyệt đối KHÔNG dùng: "real", "hazard", "controlled", "cctv"
    # ═══════════════════════════════════════════════════════════════════════════

    {
        "label":   0,
        "subtype": "dust_particles",
        # Chiếm 64% label 0 train → test cần đa dạng bối cảnh, không chỉ xưởng gỗ
        # Anchor: góc nhìn xa thấy toàn bộ không gian để thấy bụi lan tỏa
        "queries": [
            # Xưởng gỗ / vật liệu — góc rộng
            "sawdust cloud floating woodworking shop wide shot",
            "wood workshop dust air wide angle full room",
            "sawmill interior dust floating overhead view",
            # Xi măng / xây dựng — bụi dày đặc trông như khói
            "cement dust cloud construction site interior",
            "concrete cutting dust cloud room wide",
            "drywall sanding dust filled room interior",
            # Bụi công nghiệp trông rất giống khói
            "flour mill dust cloud interior wide angle",
            "grain elevator dust interior floating",
            "powder coating dust cloud factory floor",
            # Bụi trong ánh sáng (dễ nhầm nhất)
            "dust particles sunlight shaft warehouse interior",
            "dust motes floating sunlight indoor factory",
            "dust cloud air industrial building wide",
        ],
    },

    {
        "label":   0,
        "subtype": "optical_noise",
        # Nhiễu ánh sáng — xe nâng, đèn cảnh báo, phản chiếu trên sàn tôn
        # Anchor: thấy cả sàn xưởng / hành lang để hiểu bối cảnh
        "queries": [
            # Xe nâng beacon — màu cam/đỏ dễ nhầm ngọn lửa
            "forklift orange beacon light spinning warehouse floor",
            "forklift yellow warning light factory aisle wide",
            "reach truck flashing light warehouse corridor",
            # Đèn cảnh báo công nghiệp
            "industrial strobe warning light factory ceiling wide",
            "emergency beacon red light flashing hallway",
            "rotating warning beacon industrial floor wide shot",
            # Ánh sáng tự nhiên phản chiếu
            "sunlight glare reflection metal factory floor wide",
            "sunbeam window shaft warehouse interior full room",
            "light reflection corrugated roof warehouse interior",
            # Đèn nhấp nháy / hỏng
            "flickering fluorescent tube factory ceiling wide",
            "fluorescent light malfunction warehouse interior",
        ],
    },

    {
        "label":   0,
        "subtype": "daily_activity_fp",
        # Hoạt động thường ngày tạo nhiễu hình ảnh — góc nhìn xa để thấy toàn cảnh
        "queries": [
            # Hơi thở lạnh — trông như khói mỏng
            "breath vapor cold air warehouse winter wide",
            "worker cold breath mist factory winter interior",
            # Dry ice / fog machine — khói trắng dày
            "dry ice vapor floor indoor event wide shot",
            "fog machine effect indoor stage full room view",
            # Thiết bị văn phòng / in ấn
            "laser printer exhaust steam office interior wide",
            "industrial printer paper dust office wide",
            # Máy lạnh / thông gió
            "air conditioner vent mist condensation indoor wide",
            "hvac vent moisture ceiling office wide shot",
            # Người di chuyển nhanh
            "workers walking factory floor overhead wide",
            "person running warehouse aisle wide angle",
        ],
    },

    {
        "label":   0,
        "subtype": "cigarette_vape_smoke",
        # Khói thuốc lá / vape — cần thấy khói tan trong không gian, không zoom mặt
        # Anchor: thấy người + khói + không gian xung quanh
        "queries": [
            # Hút thuốc trong không gian rộng — khói nhìn thấy rõ
            "person smoking cigarette indoor break room wide",
            "worker smoking factory entrance wide shot",
            "man smoking cigarette office corridor medium shot",
            # Vape / e-cigarette — khói trắng dày
            "person vaping indoors smoke cloud wide",
            "vape exhale smoke indoor lounge medium shot",
            "e-cigarette vapor cloud indoor room wide",
            # Nhang / hương — khói mỏng cuộn
            "incense smoke rising indoor room wide shot",
            "joss stick smoke buddhist temple interior wide",
            "multiple incense sticks smoke indoor medium",
            # Hookah / shisha — khói nhiều
            "hookah smoke indoor lounge wide shot",
        ],
    },

    {
        "label":   0,
        "subtype": "welding_sparks_noise",
        # Tia lửa hàn — cần thấy cả khu vực làm việc, không chỉ zoom điểm hàn
        # Label 0 vs Label 1: label 0 góc rộng hơn, ít khói hàn hơn
        "queries": [
            # Góc rộng thấy toàn xưởng
            "welder sparks flying workshop wide angle full view",
            "metal grinding sparks workshop floor wide shot",
            "angle grinder sparks factory interior wide",
            # Nhiều công nhân / dây chuyền
            "welding sparks production line factory wide",
            "multiple welders sparks flying shipyard interior",
            "auto assembly welding sparks factory floor wide",
            # Tia lửa plasma / cắt kim loại
            "plasma cutting sparks metal shop interior",
            "metal cutting torch sparks workshop wide",
            "spark shower metal fabrication factory wide",
            # Góc nhìn từ xa — tia lửa nhỏ trong không gian lớn
            "welder working far background factory wide",
        ],
    },

    {
        "label":   0,
        "subtype": "steam_vapor",
        # Hơi nước — 0 clip trong train, ưu tiên cao nhất
        # Cần thấy rõ nguồn hơi nước + không gian xung quanh
        "queries": [
            # Công nghiệp — ống hơi, van áp suất
            "steam pipe leak industrial factory interior wide",
            "pressure valve steam release boiler room wide",
            "industrial boiler steam cloud interior wide shot",
            "steam cleaning factory floor wide angle",
            # Bếp công nghiệp — hơi nước dày
            "commercial kitchen steam rising wide shot",
            "large cooking pot steam cloud kitchen interior",
            "restaurant kitchen steam exhaust wide view",
            "steam rising food industrial kitchen wide",
            # Giặt / hơi nước dân dụng
            "laundry steam press garment factory wide",
            "steam rising washing machine laundry room",
            # Spa / tắm hơi
            "sauna steam door opening interior wide",
            "hot spring steam indoor facility wide shot",
        ],
    },

    {
        "label":   0,
        "subtype": "cooking_smoke_normal",
        # Khói bếp bình thường — 0 clip trong train
        # Phân biệt với label 1: khói mỏng, nấu ăn kiểm soát tốt, không nguy hiểm
        "queries": [
            # Bếp gia đình — khói nhẹ bình thường
            "cooking frying pan light smoke kitchen wide",
            "home kitchen stove steam smoke normal wide",
            "wok stir fry light smoke kitchen interior",
            # Nhà hàng / bếp thương mại — khói vừa
            "restaurant kitchen cooking smoke medium wide",
            "chef cooking wok smoke restaurant kitchen wide",
            "commercial kitchen grill smoke wide shot",
            # BBQ / nướng ngoài trời
            "backyard barbecue grill smoke outdoor wide",
            "bbq smoker meat smoking outdoor wide shot",
            "charcoal grill smoke backyard wide angle",
            # Bánh / nướng — hơi + khói nhẹ
            "pizza oven smoke steam bakery interior wide",
            "bread oven steam smoke bakery wide",
        ],
    },

    {
        "label":   0,
        "subtype": "decorative_fire_fp",
        # Lửa trang trí / sinh hoạt — 0 clip trong train, model phải học không báo động
        # Quan trọng: thấy cả bối cảnh để rõ đây là lửa an toàn
        "queries": [
            # Lò sưởi — lửa trong không gian sinh hoạt
            "fireplace burning living room interior wide",
            "wood fireplace fire cozy home interior wide",
            "stone fireplace fire burning interior wide shot",
            # Nến — lửa nhỏ trong không gian
            "candles burning indoor ceremony room wide",
            "candlelit dinner restaurant interior wide shot",
            "many candles burning church interior wide",
            # Lửa ngoài trời kiểm soát
            "campfire burning outdoor backyard night wide",
            "fire pit patio outdoor evening wide shot",
            "bonfire outdoor gathering people wide angle",
            # Lửa trang trí sự kiện
            "torch flame outdoor event ceremony wide",
            "fire performer event outdoor night wide",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # NHÃN 1 — Controlled fire  (7 subtypes × 35 = 245)
    #
    # NGUYÊN TẮC QUERY NHÃN 1:
    #   - Thấy rõ nguồn khói/lửa + không gian + bối cảnh con người làm chủ
    #   - Medium shot: thấy rõ sự kiện nhưng không zoom sát
    #   - Dùng từ hành vi: smoldering, rising, drifting (không phải spreading/filling)
    #   - Tránh: từ ngụ ý nguy hiểm cao như "engulfing", "raging", "out of control"
    # ═══════════════════════════════════════════════════════════════════════════

    {
        "label":   1,
        "subtype": "early_smoke_onset",
        # Khói khởi phát — 1 clip trong train, QUAN TRỌNG NHẤT cho early detection
        # Cần thấy: nguồn khói nhỏ + không gian sạch xung quanh (chưa lan)
        "queries": [
            # Thiết bị điện cháy ngầm — nguồn khói nhỏ trong phòng sạch
            "electrical outlet sparking smoke thin indoor wide",
            "power strip smoke smoldering office room wide",
            "overheated appliance smoke rising indoor medium",
            # Thùng rác / vật liệu bốc cháy ngầm
            "trash bin smoldering smoke rising indoor wide",
            "cardboard box smoldering smoke corner room",
            "paper smoldering smoke thin rising indoor",
            # Máy móc thiết bị quá nhiệt
            "overheating motor smoke rising factory floor wide",
            "machine overheating smoke thin rising indoor",
            # Khói mỏng trong không gian sạch
            "thin smoke column rising empty warehouse wide",
            "smoke rising ceiling indoor empty room wide",
            "small smoke source indoor room medium shot",
            "smoke wisp rising industrial floor wide angle",
        ],
    },

    {
        "label":   1,
        "subtype": "cigarette_controlled",
        # Khói thuốc lá kiểm soát — 78 clip train, cần đa dạng bối cảnh
        # Khác label 0: label 1 có thể thấy người hút rõ hơn, khói dày hơn
        "queries": [
            # Trong không gian kín — khói tích tụ rõ hơn
            "person smoking cigarette bar interior medium shot",
            "man smoking indoor lounge smoke visible wide",
            "smoking room interior cigarette smoke wide",
            # Nơi làm việc / nghỉ giải lao
            "worker smoking break factory exterior medium",
            "employees smoking area industrial site wide",
            "construction worker smoking site medium shot",
            # Ngoài trời — khói bay theo gió
            "person smoking outdoor street smoke drifting",
            "smoking outdoors wind cigarette smoke medium",
            "group smoking outdoor area smoke visible wide",
            # Khói thuốc tích tụ trong phòng
            "cigarette smoke accumulating dim room interior",
        ],
    },

    {
        "label":   1,
        "subtype": "welding_sparks_ctrl",
        # Tia lửa hàn có kiểm soát — 76 clip train
        # Khác label 0: có khói hàn rõ, công nhân tích cực làm việc
        "queries": [
            # Hàn MIG/TIG — khói + tia lửa thấy rõ
            "mig welding sparks smoke metal workshop medium",
            "tig welder arc sparks smoke factory medium shot",
            "welder helmet sparks smoke visible factory wide",
            # Cắt kim loại — khói + tia lửa mạnh
            "angle grinder metal sparks smoke workshop wide",
            "metal cutting sparks heavy smoke indoor",
            "plasma cutter sparks smoke industrial wide",
            # Hàn đường ống / kết cấu
            "pipe welder sparks smoke construction medium",
            "structural steel welding sparks smoke site",
            # Dây chuyền sản xuất
            "automotive welding robot sparks factory wide",
            "welding line sparks smoke production factory",
        ],
    },

    {
        "label":   1,
        "subtype": "cooking_smoke_threshold",
        # Khói bếp vượt ngưỡng — 32 clip train
        # Khác label 0: khói dày hơn rõ rệt, có thể kích hoạt smoke detector
        # Khác label 2: người nấu vẫn kiểm soát được, không có nguy hiểm thực
        "queries": [
            # Chảo / nồi bốc khói dày — vẫn có người kiểm soát
            "frying pan heavy smoke burnt food kitchen wide",
            "pan smoking overheated kitchen smoke thick wide",
            "burnt food smoke thick pan stove wide shot",
            # Bếp dầu / chiên rán — khói nhiều
            "deep fryer smoking oil kitchen interior wide",
            "commercial fryer heavy smoke kitchen wide",
            "wok high heat heavy smoke kitchen interior",
            # Nướng / BBQ khói dày trong nhà
            "indoor grill heavy smoke kitchen wide shot",
            "charbroil grill smoke thick indoor kitchen",
            # Lò vi sóng / bánh mì cháy
            "microwave smoke burnt food kitchen interior",
            "toaster oven smoke burnt kitchen wide",
            "smoke rising oven kitchen door open wide",
        ],
    },

    {
        "label":   1,
        "subtype": "heavy_smoke_controlled",
        # Khói đặc trong môi trường kiểm soát — 28 clip train
        # Nhiều khói nhưng không nguy hiểm: biểu diễn, huấn luyện, studio
        "queries": [
            # Biểu diễn / sân khấu — khói có chủ đích
            "stage fog machine heavy smoke concert indoor wide",
            "theatrical smoke effect stage performance wide",
            "smoke machine effect music concert indoor wide",
            # Studio / chụp ảnh — khói kiểm soát
            "smoke effect photography studio indoor wide",
            "atmospheric haze studio smoke indoor wide shot",
            # Huấn luyện phòng cháy chữa cháy
            "fire drill smoke simulation building interior wide",
            "firefighter training smoke building exercise wide",
            "smoke training room firefighter interior wide",
            # Khói màu ngoài trời có kiểm soát
            "colored smoke bomb outdoor controlled wide",
            "military smoke grenade training exercise wide",
        ],
    },

    {
        "label":   1,
        "subtype": "small_visible_flame",
        # Ngọn lửa nhỏ kiểm soát — 1 clip train, cần bổ sung nhiều
        # Lửa thấy rõ nhưng nhỏ, có người gần đó, chưa lan
        "queries": [
            # Lửa nhỏ trong thùng rác / vật liệu — vẫn kiểm soát
            "small fire trash bin indoor room medium shot",
            "waste bin fire small flame indoor wide",
            "paper burning small flame floor indoor medium",
            # Lửa thử nghiệm / kiểm soát
            "controlled burn small flame outdoor safety",
            "fire test small flame laboratory indoor wide",
            # Lửa lò / bếp nhỏ thấy ngọn
            "gas stove flame visible kitchen medium shot",
            "open fire small pit outdoor medium wide",
            # Đuốc / lửa cầm tay
            "hand torch small flame indoor industrial",
            "blowtorch small flame metalwork indoor medium",
            "fire starter small flame outdoor medium shot",
        ],
    },

    {
        "label":   1,
        "subtype": "daily_activity_fire_ctx",
        # Sinh hoạt thường ngày có lửa/khói — 36 clip train
        # Nấu ăn, hút thuốc, hoạt động thường ngày — người kiểm soát tốt
        "queries": [
            # Nấu ăn tại nhà / nhà hàng
            "person cooking stove smoke light kitchen wide",
            "chef cooking restaurant kitchen smoke normal wide",
            "home cooking frying steam smoke kitchen medium",
            # Người hút thuốc trong bối cảnh làm việc
            "construction worker smoking site break wide",
            "factory worker smoking outdoor break medium",
            # Lò / bếp ngoài trời
            "person tending outdoor fire pit backyard wide",
            "farmer field burning controlled outdoor wide",
            "person bonfire outdoor evening medium shot",
            # Thợ hàn / thợ cơ khí
            "mechanic welding car repair garage medium",
            "worker welding fence outdoor medium wide",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # NHÃN 2 — Uncontrolled hazard  (7 subtypes × 35 = 245)
    #
    # NGUYÊN TẮC QUERY NHÃN 2:
    #   - Thấy rõ đám cháy + khói + không gian bị ảnh hưởng (không zoom sát lửa)
    #   - Dùng từ hành vi nguy hiểm: spreading, engulfing, billowing, pouring out
    #   - Bối cảnh cụ thể: building on fire, structure fire, warehouse fire
    #   - Tránh: "real" (không có tác dụng trên API), "hazard", "dangerous"
    #   - Ưu tiên: video từ khoảng cách trung bình thấy cả tòa nhà + khói
    # ═══════════════════════════════════════════════════════════════════════════

    {
        "label":   2,
        "subtype": "heavy_smoke_hazard",
        # Khói đặc nguy hiểm — 40% label 2 train, khói đen/nâu cuộn nhanh
        # Cần thấy: khói dày che khuất tầm nhìn + bối cảnh công trình
        "queries": [
            # Khói đen từ nhà xưởng / kho
            "black smoke billowing warehouse building fire",
            "thick black smoke pouring industrial building",
            "heavy dark smoke factory structure fire wide",
            # Khói che khuất tầm nhìn trong phòng
            "dense smoke filling warehouse interior fire",
            "thick smoke room visibility zero fire",
            "smoke filling corridor building fire interior",
            # Khói từ đám cháy lớn nhìn từ xa
            "building fire black smoke exterior wide shot",
            "structure fire heavy smoke outdoor wide angle",
            "smoke column rising burning building wide",
            # Khói dày trong không gian kín
            "apartment fire smoke hallway dense interior",
            "smoke filled stairwell building fire interior",
            "heavy smoke industrial fire building interior",
        ],
    },

    {
        "label":   2,
        "subtype": "residential_indoor_fire",
        # Cháy nhà dân — 52 clip train, cháy nhà / căn hộ thực tế
        # Cần thấy: ngọn lửa + khói + phòng / căn hộ — không zoom sát lửa
        "queries": [
            # Cháy phòng / ngôi nhà nhìn từ ngoài vào
            "house fire flames exterior wide shot night",
            "residential building fire wide exterior angle",
            "apartment building fire exterior smoke flames wide",
            # Cháy bên trong thấy qua cửa sổ
            "room fire flames visible window exterior wide",
            "house fire window flames smoke exterior medium",
            # Cháy nội thất nhìn từ cửa
            "living room fire burning furniture wide interior",
            "bedroom fire flames spreading room interior",
            # Cháy bếp mất kiểm soát
            "kitchen fire out of control flames interior",
            "kitchen fire grease flames spreading interior wide",
            # Cháy nhà nhìn toàn cảnh
            "house burning fire smoke residential wide shot",
            "home fire night exterior firefighters wide",
        ],
    },

    {
        "label":   2,
        "subtype": "diffused_smoke_spreading",
        # Khói lan tràn — 38 clip train, khói đã lan ra khỏi nguồn ban đầu
        # Cần thấy: khói bao phủ không gian rộng, không còn thấy rõ nguồn
        "queries": [
            # Khói lan hành lang / cầu thang
            "smoke spreading hallway corridor building fire interior",
            "smoke filled corridor stairwell building fire wide",
            "smoke drifting hallway apartment building fire",
            # Khói trần nhà / toàn phòng
            "smoke layer ceiling room building fire interior wide",
            "smoke filling entire room building fire wide shot",
            "white gray smoke room floor ceiling fire interior",
            # Khói lan nhiều phòng
            "smoke seeping under door room building fire",
            "smoke spreading multiple floors building fire wide",
            "office building smoke spreading interior fire wide",
            # Khói từ đám cháy lan ra ngoài
            "smoke billowing building windows exterior fire wide",
            "smoke pouring building openings fire exterior",
        ],
    },

    {
        "label":   2,
        "subtype": "large_visible_flame",
        # Ngọn lửa lớn — 35 clip train, lửa rõ ràng mất kiểm soát
        # Cần thấy: ngọn lửa cao + không gian bị đốt — không chỉ zoom lửa
        "queries": [
            # Cháy nhà xưởng / kho
            "warehouse fire large flames interior wide shot",
            "factory fire flames burning structure interior wide",
            "industrial building fire large flames wide angle",
            # Cháy ngoài trời tòa nhà
            "building fire large flames exterior wide shot",
            "structure fire flames exterior wide angle night",
            # Cháy xe / thiết bị
            "car fire large flames parking lot wide",
            "vehicle fire engulfing flames outdoor wide",
            # Cháy bãi gỗ / vật liệu
            "lumber yard fire large flames burning wide",
            "storage yard fire flames spreading wide shot",
            # Cháy chung cư / cao tầng
            "apartment fire flames upper floor exterior wide",
            "high rise building fire flames exterior wide",
        ],
    },

    {
        "label":   2,
        "subtype": "early_smoke_real_fire",
        # Khói đầu đám cháy thực — 9 clip train, QUAN TRỌNG NHẤT early detection
        # Giống label 1 early_smoke nhưng: bối cảnh nguy hiểm hơn, lan nhanh hơn
        # Cần thấy: khói đang tăng dần + dấu hiệu nguy hiểm trong frame
        "queries": [
            # Khói bắt đầu từ thiết bị điện / máy móc
            "electrical fire smoke starting building interior wide",
            "fuse box fire smoke beginning room wide",
            "server room fire smoke beginning interior wide",
            # Khói đầu từ vật liệu cháy nguy hiểm
            "chemical storage smoke beginning building interior",
            "paint storage fire smoke starting warehouse wide",
            # Khói tăng nhanh — dấu hiệu đám cháy lớn sắp bùng
            "smoke increasing rapidly indoor building wide",
            "smoke growing thicker room building fire start",
            # Phát hiện khói sớm trong không gian lớn
            "early smoke detection warehouse interior wide",
            "smoke alarm triggered building smoke early wide",
            "fire beginning smoke visible ceiling wide indoor",
        ],
    },

    {
        "label":   2,
        "subtype": "toxic_material_smoke",
        # Khói hóa chất / vật liệu độc — phái sinh từ heavy_smoke
        # Đặc trưng: khói đen đậm / màu lạ + mùi hôi (không thể hiện qua video nhưng visual đặc trưng)
        "queries": [
            # Cháy nhựa / polymer — khói đen đặc trưng
            "plastic burning black smoke fire industrial wide",
            "plastic warehouse fire black smoke billowing wide",
            "burning polymer material black smoke factory",
            # Cháy cao su — khói đen dày nhất
            "rubber tire fire black smoke outdoor wide",
            "tire fire black smoke burning wide shot",
            # Cháy thiết bị điện tử
            "electrical equipment fire black smoke indoor wide",
            "server rack fire smoke electronics room wide",
            "transformer fire black smoke outdoor wide",
            # Cháy hóa chất
            "chemical fire colored smoke outdoor facility wide",
            "solvent fire black smoke building exterior wide",
        ],
    },

    {
        "label":   2,
        "subtype": "daily_ctx_real_fire",
        # Bối cảnh người dân trong / gần đám cháy thực — 7 clip train
        # Cần thấy: người + đám cháy + bối cảnh nguy hiểm rõ ràng
        "queries": [
            # Sơ tán khẩn cấp
            "people evacuating building fire smoke exterior wide",
            "crowd evacuation building fire outdoor wide shot",
            "building evacuation fire smoke people running wide",
            # Lính cứu hỏa đang chiến đấu
            "firefighters battling building fire exterior wide",
            "fire truck building fire hose water wide shot",
            "firefighter entering burning building smoke wide",
            # Người xem / người xung quanh
            "bystanders watching building fire exterior wide",
            "people watching house fire outdoor night wide",
            # Cứu hộ trong đám cháy
            "rescue operation building fire smoke exterior wide",
            "firefighter rescue burning building exterior wide",
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