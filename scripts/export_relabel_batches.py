"""Copy the frames / static images listed in docs/reports/Danh_sach_sua_nhan_2026-10-22.md into one folder per
group, ready to upload to Roboflow for relabeling.

    python scripts/export_relabel_batches.py

Output: data/relabel_2026-10-22/<group>/            images to upload (flat)
        data/relabel_2026-10-22/<group>/_ref/       current label overlay (red = fire, cyan = smoke), reference only
        data/relabel_2026-10-22/mapping.csv         uploaded name -> original image / mask, split, group
Video frames are named <video>__<frame>.jpg so a Roboflow export can be mapped back by prefix.
"""
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "data" / "dataset_video_unified"
STATIC_DIR = ROOT / "data" / "Fire_Smoke.v4i.png-mask-semantic" / "train"
SPLITS = ROOT / "data" / "data_splits"
OUT = ROOT / "data" / "relabel_2026-10-22"

ALL = "all"
# group -> list of (video substring, frames) ; frames = ALL or list of 6-digit indices
VIDEO_GROUPS = {
    "A1_video_sai_nhan": [
        ("14112316_1440_2560_60fps_pure", ALL),
        ("316af409_pexels_heavy_smoke_18105251", ["000030", "000060", "000090", "000240", "000270", "000300", "000330", "000360"]),
        ("battery_thermal_runaway_ld00r", [f"{i:06d}" for i in range(125, 551, 25)]),
        ("electrical_ignition_qRoT41qla9I", ["000450"]),
        ("12218199_3114_1440_30fps_pure", ["000240"]),
        ("battery_thermal_runaway_krHU3jlR4Pc", ["000425"]),
        ("36f3eac1_pexels_cigarette_smoke_5075501", ["000125"]),
    ],
    "A2_video_am_tinh_co_khoi_that": [
        ("neg_hard_negatives_iaZ3E1mD96Q", ALL),
        ("neg_hard_negatives_BvTBuLESrJ0", ALL),
        ("neg_hard_negatives_erqtaaYXWzY", ALL),
    ],
    "B_nhan_tho": [
        ("12393269_1440_2560_30fps_pure", ALL),
        ("electrical_ignition_VimYV-zgYo0", ["000930"]),
        ("36f3eac1_pexels_cigarette_smoke_5075501", ["000150"]),
        ("28f4ec5d_pexels_cigarette_smoke_8139860", ["000120"]),
    ],
    "C1_ong_khoi_cong_nghiep": [
        ("6c3dee00_pexels_heavy_smoke_6618036", ALL),
        ("9a06358f_pexels_heavy_smoke_3568453", ALL),
    ],
    "C4_han_ho_quang": [
        ("05ca4b9b_pexels_welding_sparks_34771762", ALL),
        ("0aefc692_pexels_welding_sparks_4941457", ["000075"]),
    ],
}
# group -> static file name prefixes (the 6-digit index alone is not unique for some images)
STATIC_GROUPS = {
    "A3_anh_tinh_hoi_nuoc": ["000122_4-Image-No-4_jpg", "000106", "000170", "000135", "000254", "000506", "000863"],
    "B_nhan_tho": ["000464"],
    "C1_ong_khoi_cong_nghiep": ["000021_test_429_jpg", "001085_train_1910_jpg", "000453"],
    "C2_khoi_nhan_tao": ["000358", "000755", "000141", "000507", "000618", "000945_20test740_jpg", "001029", "001229_20test488_jpg",
                         "000940_25test167_jpg", "001073", "000496", "000739", "001315_1test394_jpg", "000509", "001082"],
    "C3_vape_hut_thuoc_cctv": ["001303", "001257_d77040d3", "001252_15c0eafe", "000337"],
    "C5_khong_ro": ["000460"],
}


def overlay(img_path, mask_path):
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img.shape[:2] != mask.shape:
        mask = cv2.resize(mask, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    ov = img.copy()
    ov[mask == 1] = (0, 0, 255)
    ov[mask == 2] = (255, 255, 0)
    out = cv2.addWeighted(img, 0.55, ov, 0.45, 0)
    for cls, col in ((1, (0, 0, 255)), (2, (255, 255, 0))):
        cnts, _ = cv2.findContours((mask == cls).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, col, 2)
    return np.hstack([img, out])


def export(group, name, img_path, mask_path, rows, **meta):
    d = OUT / group
    (d / "_ref").mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, d / name)
    cv2.imwrite(str(d / "_ref" / (Path(name).stem + "_ref.jpg")), overlay(img_path, mask_path), [cv2.IMWRITE_JPEG_QUALITY, 85])
    rows.append({"group": group, "upload_name": name, "image_path": str(img_path.relative_to(ROOT)),
                 "mask_path": str(mask_path.relative_to(ROOT)), **meta})


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    rows = []
    vm = pd.read_csv(SPLITS / "video_manifest_with_splits.csv", encoding="utf-8-sig")
    for group, items in VIDEO_GROUPS.items():
        for key, frames in items:
            g = vm[vm.video_name.str.contains(key, regex=False)]
            assert g.video_name.nunique() == 1, f"{key}: {g.video_name.unique()}"
            if frames != ALL:
                g = g[g.frame_name.isin([f"raw_{f}" for f in frames])]
                assert len(g) == len(frames), f"{key}: missing frames"
            for r in g.sort_values("frame_index").itertuples():
                export(group, f"{r.video_name}__{r.frame_name}.jpg", VIDEO_DIR / r.image_path, VIDEO_DIR / r.mask_path,
                       rows, source="video", split=r.split, video=r.video_name, frame=r.frame_name, category=r.category)

    sm = pd.read_csv(SPLITS / "static_v4i_manifest_with_splits.csv", encoding="utf-8-sig")
    for group, prefixes in STATIC_GROUPS.items():
        for p in prefixes:
            hit = sm[sm.filename.str.startswith(p if not p.isdigit() else p + "_")]
            assert len(hit) == 1, f"{p}: {len(hit)} matches"
            r = hit.iloc[0]
            export(group, r.filename, STATIC_DIR / r.filename, STATIC_DIR / r.mask_filename, rows,
                   source="static", split=r.split, video="", frame=r.filename, category=r.category)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mapping.csv", index=False, encoding="utf-8-sig")
    print(df.groupby(["group", "source"]).size().to_string())
    print(f"total {len(df)} -> {OUT}")


if __name__ == "__main__":
    main()
