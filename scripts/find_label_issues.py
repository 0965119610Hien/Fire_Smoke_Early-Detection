"""Flag video frames / static images whose labels are likely wrong under the labeling policy
(steam / mist = background, any flame incl. blue gas flame = fire, combustion/exhaust smoke = smoke).

Heuristics (candidates only — every flag needs a visual check):
  steam_like   : a large share of the smoke pixels is bright and unsaturated (white plume -> likely steam)
  box_mask     : a big connected component fills > 93% of its bounding box (box-shaped annotation)
  temporal     : within a video, a class disappears / its area jumps > 5x between neighbouring labeled frames
  neg_video    : every frame of a hard-negative video (content must be checked against the policy)

Outputs data/manual_check_unified/label_issue_candidates.csv and contact sheets
(image | overlay, red = fire, cyan = smoke) in data/manual_check_unified/label_issue_sheets/.

    python scripts/find_label_issues.py
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "data" / "dataset_video_unified"
STATIC_DIR = ROOT / "data" / "Fire_Smoke.v4i.png-mask-semantic" / "train"
SPLITS = ROOT / "data" / "data_splits"
OUT_DIR = ROOT / "data" / "manual_check_unified"

WHITE_V, WHITE_S = 0.75, 0.15     # HSV thresholds (0..1) for "white" pixels
STEAM_FRAC = 0.5                  # flag if >= this share of smoke pixels is white
BOX_FILL, BOX_MIN_AREA = 0.93, 0.01
JUMP = 5.0


def load(img_path, mask_path):
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img.shape[:2] != mask.shape:
        img = cv2.resize(img, mask.shape[::-1])
    return img, mask


def frame_flags(img, mask):
    flags, info = [], {}
    smoke = mask == 2
    if smoke.sum() > 200:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        white = (hsv[..., 2] > WHITE_V) & (hsv[..., 1] < WHITE_S)
        frac = float(white[smoke].mean())
        info["smoke_white_frac"] = round(frac, 3)
        if frac >= STEAM_FRAC:
            flags.append("steam_like")
    h, w = mask.shape
    for cls, name in ((1, "fire"), (2, "smoke")):
        n, _, stats, _ = cv2.connectedComponentsWithStats((mask == cls).astype(np.uint8))
        for x, y, bw, bh, area in stats[1:]:
            if area >= BOX_MIN_AREA * h * w and area / (bw * bh) > BOX_FILL:
                flags.append(f"box_mask_{name}")
                break
    return flags, info


def overlay(img, mask):
    ov = img.copy()
    ov[mask == 1] = (0, 0, 255)
    ov[mask == 2] = (255, 255, 0)
    out = cv2.addWeighted(img, 0.55, ov, 0.45, 0)
    for cls, col in ((1, (0, 0, 255)), (2, (255, 255, 0))):
        cnts, _ = cv2.findContours((mask == cls).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, col, 2)
    return out


def tile(img, mask, label, height=300):
    pair = np.hstack([img, overlay(img, mask)])
    s = height / pair.shape[0]
    pair = cv2.resize(pair, (int(pair.shape[1] * s), height))
    cv2.rectangle(pair, (0, 0), (pair.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(pair, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return pair


def save_sheet(tiles, path, per_row=2):
    rows = []
    for i in range(0, len(tiles), per_row):
        row = tiles[i:i + per_row]
        width = sum(t.shape[1] for t in tiles[:per_row]) if len(tiles) >= per_row else sum(t.shape[1] for t in row)
        r = np.hstack(row)
        if r.shape[1] < width:
            r = np.hstack([r, np.zeros((r.shape[0], width - r.shape[1], 3), np.uint8)])
        rows.append(r)
    wmax = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((r.shape[0], wmax - r.shape[1], 3), np.uint8)]) for r in rows]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    rows = []
    vm = pd.read_csv(SPLITS / "video_manifest_with_splits.csv", encoding="utf-8-sig")
    for video, g in vm.groupby("video_name"):
        g = g.sort_values("frame_index").reset_index(drop=True)
        areas = {"fire": g.fire_pixels.values, "smoke": g.smoke_pixels.values}
        for i, r in g.iterrows():
            img, mask = load(VIDEO_DIR / r.image_path, VIDEO_DIR / r.mask_path)
            flags, info = frame_flags(img, mask)
            if video.startswith("neg_hard"):
                flags.append("neg_video")
            for name, a in areas.items():
                prev_, next_ = (a[i - 1] if i > 0 else None), (a[i + 1] if i + 1 < len(a) else None)
                if prev_ is not None and next_ is not None and prev_ > 0 and next_ > 0 and a[i] == 0:
                    flags.append(f"temporal_missing_{name}")
                elif prev_ is not None and min(a[i], prev_) > 0 and max(a[i], prev_) / min(a[i], prev_) > JUMP:
                    flags.append(f"temporal_jump_{name}")
            if flags:
                rows.append({"source": "video", "split": r.split, "fold": r.fold, "video": video,
                             "frame": r.frame_name, "category": r.category, "flags": ";".join(flags),
                             "image_path": str(VIDEO_DIR / r.image_path), "mask_path": str(VIDEO_DIR / r.mask_path), **info})

    sm = pd.read_csv(SPLITS / "static_v4i_manifest_with_splits.csv", encoding="utf-8-sig")
    for r in sm.itertuples():
        img, mask = load(STATIC_DIR / r.filename, STATIC_DIR / r.mask_filename)
        flags, info = frame_flags(img, mask)
        if flags:
            rows.append({"source": "static", "split": r.split, "fold": "", "video": "", "frame": r.filename,
                         "category": r.category, "flags": ";".join(flags),
                         "image_path": str(STATIC_DIR / r.filename), "mask_path": str(STATIC_DIR / r.mask_filename), **info})

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "label_issue_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"{len(df)} candidates")
    print(df["flags"].str.split(";").explode().str.replace(r"_(fire|smoke)$", "", regex=True).value_counts().to_string())

    sheets = OUT_DIR / "label_issue_sheets"
    sheets.mkdir(exist_ok=True)
    for f in sheets.glob("*.jpg"):
        f.unlink()
    key = df["flags"].str.split(";").str[0].str.replace(r"_(fire|smoke)$", "", regex=True)
    for (src, flag), g in df.groupby([df.source, key]):
        for k in range(0, len(g), 8):
            tiles = []
            for r in g.iloc[k:k + 8].itertuples():
                img, mask = load(r.image_path, r.mask_path)
                name = f"{r.video[:28]}/{r.frame}" if src == "video" else r.frame[:40]
                tiles.append(tile(img, mask, f"[{r.split}] {name} {r.flags}"))
            save_sheet(tiles, sheets / f"{src}_{flag}_{k // 8:03d}.jpg")


if __name__ == "__main__":
    main()
