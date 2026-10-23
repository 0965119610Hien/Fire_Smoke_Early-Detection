"""Attach an error note + fix instruction to every image exported by export_relabel_batches.py.

    python scripts/annotate_relabel_notes.py

Writes, for data/relabel_2026-10-22/:
  mapping.csv                   + columns `loi` (what is wrong) and `cach_sua` (how to fix)
  <group>/GHI_CHU.csv           file name | loi | cach_sua for that folder
  <group>/_ref*/<name>_ref.jpg  regenerated: note banner on top of image | current label overlay
Existing _ref folder names (e.g. a renamed `_ref_A1`) are kept.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "relabel_2026-10-22"
FONT = "C:/Windows/Fonts/arial.ttf"

KEEP = ("Không có khói / lửa (cận cảnh, chữ, logo)", "Giữ RỖNG — không cần sửa")
STEAM_POT = ("Hơi nước bốc từ chảo / nồi đang gán KHÓI", "Hơi nước → nền; chỗ lẫn khói thật mà không chắc → 255. Giữ LỬA (kể cả lửa xanh bếp gas)")
COOLING = ("Hơi bốc từ dàn quạt / tháp giải nhiệt đang gán KHÓI", "Hơi nước → nền")
C1 = ("Khí thải trắng từ ống khói / nhà xưởng, không thấy nguồn cháy",
      "Chờ quy tắc C1 — đề xuất 255; nếu xác định là hơi nước → nền, khói xám → khói")
C2 = ("Khói nhân tạo từ máy thử đầu báo (smoke / fog generator) đang gán KHÓI",
      "Chờ quy tắc C2 — đề xuất giữ KHÓI, vẽ sát vùng khói")
C3 = ("Người hút thuốc / vape trên CCTV, đám trắng đang gán KHÓI",
      "Chờ quy tắc C3 — đề xuất: vape → nền, thuốc lá → khói, không phân biệt được → 255")
WELD_ARC = ("Hồ quang hàn đang gán LỬA; cột khói hàn phía trên gán KHÓI (khói rất mờ)",
            "Chờ quy tắc C4 — đề xuất hồ quang + tia lửa → nền; khói hàn chỉ gán chỗ thấy rõ, còn lại 255")
WELD_SMOKE = ("Cột khói hàn rất mờ đang gán KHÓI, vẽ rộng hơn vùng thấy được",
              "Chờ quy tắc C4 — khói hàn chỉ gán chỗ thấy rõ, còn lại 255")
EXHAUST_DARK = ("Khói xả xe xám / xanh / đen (có muội) nhưng nhãn RỖNG", "Gán KHÓI cho vệt khói (phần chữ / hình vẽ giữ nền)")
EXHAUST_WHITE = ("Khí xả màu trắng nhưng nhãn RỖNG", "Trắng (hơi nước ngưng tụ) → giữ nền; đặc và xám → khói; không chắc → 255")
FLARE = ("Pháo sáng đang cháy + khói nhưng nhãn RỖNG", "Gán LỬA cho ngọn lửa pháo sáng, KHÓI cho vệt khói")
GRENADE = ("Lựu đạn khói đang phun khói nhưng nhãn RỖNG", "Gán KHÓI (kể cả khói màu)")
LD_FLASH = ("Cả vùng chói trắng ở cửa (ánh sáng vụ nổ) đang gán LỬA",
            "Chỉ gán LỬA phần ngọn lửa thấy rõ, vùng chói → 255")
LD_SMOKE = ("Phòng đang đầy khói nhưng nhãn chỉ có LỬA", "Thêm KHÓI cho vùng mù khói (có thể phủ gần cả khung), giữ lửa")

# key: "<video substring>@<frame>" (frame "*" = every frame of that video) or a static file name prefix
NOTES = {
    # ---- A1
    "14112316_1440_2560_60fps_pure@*": STEAM_POT,
    "316af409_pexels_heavy_smoke_18105251@*": STEAM_POT,
    **{f"battery_thermal_runaway_ld00r@{f:06d}": LD_FLASH for f in (125, 150)},
    "battery_thermal_runaway_ld00r@000175": (LD_FLASH[0], LD_FLASH[1] + "; giữ KHÓI phía trên"),
    **{f"battery_thermal_runaway_ld00r@{f:06d}": LD_SMOKE for f in range(200, 551, 25)},
    "electrical_ignition_qRoT41qla9I@000450": ("Lửa thấy rõ dưới lớp khói nhưng chỉ gán KHÓI", "Thêm LỬA cho ngọn lửa, khói giữ xung quanh"),
    "12218199_3114_1440_30fps_pure@000240": ("Lửa rõ dưới ống tre nhưng mask RỖNG", "Gán LỬA"),
    "battery_thermal_runaway_krHU3jlR4Pc@000425": ("Khối lửa lớn chỉ gán một dải mỏng, phần lửa bị trùm bởi KHÓI",
                                                   "Gán LỬA cho cả khối lửa; KHÓI chỉ phần xám phía trên"),
    "36f3eac1_pexels_cigarette_smoke_5075501@000125": ("KHÓI vẽ thành khối chữ nhật dọc điếu thuốc, ảnh không thấy khói",
                                                       "Xoá khói; giữ LỬA ở đầu thuốc"),
    # ---- A2
    **{f"neg_hard_negatives_iaZ3E1mD96Q@{f}": KEEP for f in ("000154", "000328", "001024", "001198", "002242", "002416")},
    **{f"neg_hard_negatives_iaZ3E1mD96Q@{f}": FLARE for f in ("000502", "000676", "000850", "001372", "001894")},
    **{f"neg_hard_negatives_iaZ3E1mD96Q@{f}": GRENADE for f in ("001546", "001720", "002068", "002764")},
    "neg_hard_negatives_iaZ3E1mD96Q@002590": ("Pháo nổ (thunderflash), gần như không thấy khói", "Có vệt khói rõ → KHÓI, không thì giữ RỖNG"),
    **{f"neg_hard_negatives_BvTBuLESrJ0@{f}": KEEP for f in ("000088", "000187", "000286", "000385", "000484", "000583", "000682")},
    **{f"neg_hard_negatives_BvTBuLESrJ0@{f}": ("Cột khói thử nóng giữa sân vận động nhưng nhãn RỖNG "
                                                "(viền xanh từ frame 1276 là của camera Fike trong video, KHÔNG phải nhãn)",
                                                "Gán KHÓI cho cột khói")
       for f in ("000781", "000880", "000979", "001078", "001177", "001276", "001375", "001474", "001573")},
    **{f"neg_hard_negatives_erqtaaYXWzY@{f}": KEEP for f in ("000922", "001241", "002511", "004027", "005075")},
    **{f"neg_hard_negatives_erqtaaYXWzY@{f}": EXHAUST_DARK
       for f in ("000601", "002823", "003155", "003457", "003791", "004421", "004566")},
    **{f"neg_hard_negatives_erqtaaYXWzY@{f}": EXHAUST_WHITE for f in ("000284", "001564", "001879", "002096")},
    # ---- A3
    "000122_4-Image-No-4_jpg": ("Hơi xả từ van đường ống đang gán KHÓI", "Hơi nước → nền"),
    "000106": ("Vệt trắng lớn từ tháp giải nhiệt (bên phải) đang gán KHÓI",
               "Tháp giải nhiệt → nền; 2 vệt nhỏ trên ống khói theo quy tắc C1"),
    "000170": ("Vệt trắng lớn từ tháp giải nhiệt (bên phải) đang gán KHÓI",
               "Tháp giải nhiệt → nền; 2 vệt nhỏ trên ống khói theo quy tắc C1"),
    "000135": COOLING, "000254": COOLING, "000506": COOLING, "000863": COOLING,
    # ---- B
    "12393269_1440_2560_30fps_pure@*": ("LỬA trong lò vẽ thành hình chữ nhật", "Vẽ lại sát viền ngọn lửa"),
    "electrical_ignition_VimYV-zgYo0@000930": ("KHÓI trùm lên dòng chữ chèn \"Smoke Detected!!\"", "Chữ chèn → 255; khói vẽ sát"),
    "36f3eac1_pexels_cigarette_smoke_5075501@000150": ("Khói thuốc vẽ trùm sang cửa sổ / nền", "Vẽ sát vùng khói thấy được, mép quá mờ → 255"),
    "28f4ec5d_pexels_cigarette_smoke_8139860@000120": ("Khói thuốc vẽ trùm sang cửa sổ / nền", "Vẽ sát vùng khói thấy được, mép quá mờ → 255"),
    "000464": ("KHÓI vẽ theo từng ô kính, viền thô", "Vẽ lại sát vùng khói; chỗ không chắc → 255"),
    # ---- C
    "6c3dee00_pexels_heavy_smoke_6618036@*": ("Đám trắng rất lớn bốc từ mái nhà xưởng (drone), không thấy nguồn cháy, đang gán KHÓI", C1[1]),
    "9a06358f_pexels_heavy_smoke_3568453@*": ("Ống khói đang nhả vệt trắng nhưng nhãn RỖNG", C1[1]),
    "000021_test_429_jpg": C1, "001085_train_1910_jpg": C1, "000453": C1,
    **{p: C2 for p in ("000358", "000755", "000141", "000507", "000618", "000945_20test740_jpg", "001029", "001229_20test488_jpg",
                       "000940_25test167_jpg", "001073", "000496", "000739", "001315_1test394_jpg", "000509", "001082")},
    "001303": ("Đám trắng lớn trước mặt người trên CCTV (nhiều khả năng vape) đang gán KHÓI", C3[1]),
    "001257_d77040d3": C3, "001252_15c0eafe": C3, "000337": C3,
    "05ca4b9b_pexels_welding_sparks_34771762@000128": ("Hồ quang hàn đang gán LỬA; KHÓI trùm cả người thợ hàn", WELD_ARC[1]),
    "0aefc692_pexels_welding_sparks_4941457@000075": ("KHÓI trùm lên mặt nạ hàn và người thợ", "Vẽ sát khói hàn thấy được, còn lại 255"),
    "000460": ("Đám trắng trong đêm, không rõ khói hay sương", "Xem kỹ ảnh gốc; không chắc → 255"),
}


def lookup(r):
    if r.source == "video":
        frame = r.frame.replace("raw_", "")
        for key in (f"@{frame}", "@*"):
            hits = [v for k, v in NOTES.items() if k.endswith(key) and k.split("@")[0] in r.video]
            if hits:
                return hits[0]
        if "05ca4b9b" in r.video:  # remaining welding frames: arc labelled as fire or not
            return WELD_ARC if r.category == "both" else WELD_SMOKE
    else:
        hits = [v for k, v in NOTES.items() if "@" not in k and r.frame.startswith(k if not k.isdigit() else k + "_")]
        if hits:
            return hits[0]
    raise KeyError(f"no note for {r.upload_name}")


def wrap(draw, text, font, width):
    lines, cur = [], ""
    for w in text.split():
        t = f"{cur} {w}".strip()
        if draw.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def ref_image(r, loi, sua):
    img = cv2.imread(str(ROOT / r.image_path))
    mask = cv2.imread(str(ROOT / r.mask_path), cv2.IMREAD_GRAYSCALE)
    if img.shape[:2] != mask.shape:
        mask = cv2.resize(mask, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    ov = img.copy()
    ov[mask == 1] = (0, 0, 255)
    ov[mask == 2] = (255, 255, 0)
    out = cv2.addWeighted(img, 0.55, ov, 0.45, 0)
    for cls, col in ((1, (0, 0, 255)), (2, (255, 255, 0))):
        cnts, _ = cv2.findContours((mask == cls).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, col, 2)
    pair = np.hstack([img, out])
    if pair.shape[1] < 1400:
        s = 1400 / pair.shape[1]
        pair = cv2.resize(pair, (1400, int(pair.shape[0] * s)))
    w = pair.shape[1]
    font = ImageFont.truetype(FONT, max(20, w // 60))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = [(f"[{r.split}] {r.upload_name}", (200, 200, 200))]
    lines += [(t, (255, 120, 120)) for t in wrap(probe, "LỖI: " + loi, font, w - 20)]
    lines += [(t, (140, 255, 140)) for t in wrap(probe, "SỬA: " + sua, font, w - 20)]
    lh = int(font.size * 1.35)
    banner = Image.new("RGB", (w, lh * len(lines) + 12), (0, 0, 0))
    d = ImageDraw.Draw(banner)
    for i, (t, col) in enumerate(lines):
        d.text((10, 6 + i * lh), t, font=font, fill=col)
    return np.vstack([cv2.cvtColor(np.array(banner), cv2.COLOR_RGB2BGR), pair])


def main():
    m = pd.read_csv(OUT / "mapping.csv", encoding="utf-8-sig")
    notes = [lookup(r) for r in m.itertuples()]
    m["loi"], m["cach_sua"] = [n[0] for n in notes], [n[1] for n in notes]
    m.to_csv(OUT / "mapping.csv", index=False, encoding="utf-8-sig")
    for group, g in m.groupby("group"):
        gdir = OUT / group
        g[["upload_name", "split", "loi", "cach_sua"]].to_csv(gdir / "GHI_CHU.csv", index=False, encoding="utf-8-sig")
        refs = sorted(p for p in gdir.iterdir() if p.is_dir() and p.name.startswith("_ref"))
        ref_dir = refs[0] if refs else gdir / "_ref"
        ref_dir.mkdir(exist_ok=True)
        for r in g.itertuples():
            cv2.imwrite(str(ref_dir / (Path(r.upload_name).stem + "_ref.jpg")), ref_image(r, r.loi, r.cach_sua),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(m.groupby("group").size().to_string())
    print((m.cach_sua == KEEP[1]).sum(), "images need no change (kept for context)")


if __name__ == "__main__":
    main()
