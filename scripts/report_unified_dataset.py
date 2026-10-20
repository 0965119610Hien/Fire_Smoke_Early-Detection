"""
scripts/report_unified_dataset.py
=================================
Detailed statistics of the unified dataset (scripts/build_unified_dataset.py)
-> docs/Unified_Dataset_Statistics.md

Usage:
  python scripts/report_unified_dataset.py
"""

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLITS_DIR = DATA / "data_splits"
VIDEO_ROOT = DATA / "dataset_video_unified"
STATIC_ROOT = DATA / "Fire_Smoke.v4i.png-mask-semantic" / "train"
OUT = ROOT / "docs" / "Unified_Dataset_Statistics.md"
SPLITS = ("train", "val", "test")
CATS = ("fire_only", "smoke_only", "both", "empty")
ROUND_NAMES = {0: "Âm tính", 1: "Round 1", 2: "Round 2", 3: "Round 3 (mới)"}
MIN_REGION = 16  # px, same as the evaluation's object-level threshold


def read_csv(p):
    return pd.read_csv(p, encoding="utf-8-sig")


def md_table(df, index=True, fmt=None):
    df = df.copy()
    if fmt:
        for c, f in fmt.items():
            if c in df.columns:
                df[c] = df[c].map(f)
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, *vals in df.itertuples(name=None):  # itertuples keeps per-column dtypes (no int -> float)
        cells = ([str(idx)] if index else []) + [str(v) for v in vals]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def with_total(df):
    df = df.copy()
    df.loc["Tổng"] = df.sum(numeric_only=True)
    return df.astype(int)


def pct_cols(df, cols):
    tot = df[cols].sum(1).replace(0, 1)
    return df[cols].apply(lambda c: (100 * c / tot).round(1).astype(str) + "%")


def video_kind(v):
    return "neg_hard" if v.startswith("neg_hard") else ("neg_normal" if v.startswith("neg_normal") else "positive")


def topic(v):
    if v.startswith("neg_hard"):
        return "âm tính khó (hard negative)"
    if v.startswith("neg_normal"):
        return "âm tính thường (CCTV/cảnh thường)"
    m = re.search(r"pexels_(.+?)_\d+", v)
    if m:
        return m.group(1).replace("_", " ")
    for key in ("electrical_ignition", "battery_thermal_runaway"):
        if v.startswith(key):
            return key.replace("_", " ")
    if re.match(r"\d{7,}", v):
        return "pexels (không ghi chủ đề)"
    return "khác"


def regions(mask, cls):
    n, _, st, _ = cv2.connectedComponentsWithStats((mask == cls).astype(np.uint8), connectivity=8)
    return [int(a) for a in st[1:, cv2.CC_STAT_AREA] if a >= MIN_REGION]


def mask_object_stats(paths):
    """Per mask: number of fire / smoke regions (>= MIN_REGION px)."""
    out = []
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        out.append((len(regions(m, 1)), len(regions(m, 2))))
    return out


def coverage_summary(cov):
    cov = np.asarray(cov)
    if not len(cov):
        return {"n": 0}
    return {"n": len(cov), "mean": cov.mean(), "median": np.median(cov), "p10": np.percentile(cov, 10),
            "p90": np.percentile(cov, 90), "<1%": 100 * (cov < 1).mean(), "<0.1%": 100 * (cov < 0.1).mean()}


def coverage_table(df, area):
    rows = {}
    for s in SPLITS + ("Tổng",):
        d = df if s == "Tổng" else df[df["split"] == s]
        for cls, col in (("Lửa", "fire_pixels"), ("Khói", "smoke_pixels")):
            pos = d[d[col] > 0]
            rows[(s, cls)] = coverage_summary(100 * pos[col] / area[pos.index])
    t = pd.DataFrame(rows).T
    t.index = [f"{a} — {b}" for a, b in t.index]
    t.index.name = "Split — lớp"
    t["n"] = t["n"].astype(int)
    f2 = lambda x: f"{x:.2f}"
    f1 = lambda x: f"{x:.1f}%"
    return md_table(t.rename(columns={"n": "Số frame", "mean": "TB %", "median": "Trung vị %", "p10": "P10 %",
                                      "p90": "P90 %", "<1%": "Frame <1%", "<0.1%": "Frame <0.1%"}),
                    fmt={"TB %": f2, "Trung vị %": f2, "P10 %": f2, "P90 %": f2, "Frame <1%": f1, "Frame <0.1%": f1})


FEATURE_NAMES = {
    "frames": "Frame / ảnh", "fire": "Có lửa", "smoke": "Có khói", "fire_only": "fire_only", "smoke_only": "smoke_only",
    "both": "both", "empty": "Rỗng", "fire_area": "Tổng diện tích lửa", "smoke_area": "Tổng diện tích khói",
    "fire_S": "Lửa nhỏ (<0.3%)", "fire_M": "Lửa vừa (0.3–2%)", "fire_L": "Lửa lớn (≥2%)",
    "smoke_S": "Khói nhỏ (<3%)", "smoke_M": "Khói vừa (3–15%)", "smoke_L": "Khói lớn (≥15%)",
}


def fold_section(man, static, report):
    """Share of every balance target per part (test + K folds)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_unified_dataset import area_features

    K = int(report["folds"])
    parts = [("test", -1)] + [(f"fold{k}", k) for k in range(K)]
    target = {"test": 100 * report["test_ratio"], **{f"fold{k}": 100 * (1 - report["test_ratio"]) / K for k in range(K)}}

    def share_table(df):
        area = df["width"] * df["height"]
        fcov, scov = 100 * df["fire_pixels"] / area, 100 * df["smoke_pixels"] / area
        X = pd.DataFrame({p: area_features(df[df["fold"] == k], fcov[df["fold"] == k], scov[df["fold"] == k])
                          for p, k in parts})
        sh = 100 * X.div(X.sum(1).replace(0, 1), axis=0)
        t = pd.DataFrame({p: [f"{X.loc[f, p]:.0f} ({sh.loc[f, p]:.1f}%)" if "area" not in f else f"{sh.loc[f, p]:.1f}%"
                              for f in X.index] for p, _ in parts}, index=[FEATURE_NAMES[f] for f in X.index])
        dev = (sh.sub(pd.Series(target), axis=1)).abs()
        t["Lệch tối đa"] = [f"{v:.1f}" for v in dev.max(1)]
        t.index.name = "Đại lượng"
        return t

    L = ["## 1b. Chia K-fold (MILP)\n",
         f"{report.get('split_method', '')}. Test cố định {100 * report['test_ratio']:.0f}% + {K} fold "
         f"(mỗi fold ≈ {100 * (1 - report['test_ratio']) / K:.0f}%). Fold k làm val trong lần train thứ k, "
         "các fold còn lại làm train; test giống nhau cho mọi fold. File: `data/data_splits/folds/fold<k>/`; "
         "`data/data_splits/` = bố cục fold 0.\n",
         f"Video: {report['video_solver']}. Ảnh tĩnh: {report['static']['solver']}.\n"]
    vk = man.groupby("video_name").agg(fold=("fold", "first"))
    vk["kind"] = [video_kind(v) for v in vk.index]
    kt = pd.crosstab(vk["kind"], vk["fold"]).rename(columns=lambda k: "test" if k == -1 else f"fold{k}")
    kt = kt[[p for p, _ in parts]]
    kt.index.name = "Video"
    L += ["### Số video mỗi phần\n", md_table(kt) + "\n",
          "### Video: số frame và tỉ lệ của mỗi đại lượng (mục tiêu: "
          + ", ".join(f"{p} {v:.0f}%" for p, v in list(target.items())[:2]) + ")\n",
          md_table(share_table(man)) + "\n",
          "### Ảnh tĩnh\n",
          md_table(share_table(static.rename(columns={"filename": "frame_name"}))) + "\n",
          "- *Lệch tối đa* tính bằng điểm %. Một phần độ lệch là không tránh được, ví dụ `bucket11` một mình chứa ~30% số "
          "frame khói nhỏ; ảnh tĩnh rỗng là frame của video âm tính nên bắt buộc đi theo fold của video đó.\n"]
    return "\n".join(L)


def main():
    man = read_csv(SPLITS_DIR / "video_manifest_with_splits.csv")
    report = json.loads((SPLITS_DIR / "unified_build_report.json").read_text(encoding="utf-8"))
    static = pd.concat([read_csv(SPLITS_DIR / f"static_v4i_{s}_manifest.csv") for s in SPLITS], ignore_index=True)
    sparse = {s: read_csv(SPLITS_DIR / f"{s}_clips.csv") for s in SPLITS}
    dense = {s: read_csv(SPLITS_DIR / f"dense_{s}_clips.csv") for s in SPLITS}

    man["kind"] = man["video_name"].map(video_kind)
    man["topic"] = man["video_name"].map(topic)
    man["area"] = man["width"] * man["height"]
    print("Đếm vùng lửa/khói trên mask video...")
    obj = mask_object_stats(VIDEO_ROOT / p for p in man["mask_path"])
    man["fire_regions"] = [a for a, _ in obj]
    man["smoke_regions"] = [b for _, b in obj]
    print("Đếm vùng lửa/khói trên mask ảnh tĩnh...")
    sobj = mask_object_stats(STATIC_ROOT / p for p in static["mask_filename"])
    static["fire_regions"] = [a for a, _ in sobj]
    static["smoke_regions"] = [b for _, b in sobj]
    static["area"] = static["width"] * static["height"]

    vids = man.groupby("video_name").agg(split=("split", "first"), round=("round", "first"), kind=("kind", "first"),
                                         topic=("topic", "first"), frames=("frame_name", "size"),
                                         fire=("has_fire", "sum"), smoke=("has_smoke", "sum"),
                                         w=("width", "first"), h=("height", "first"))
    vids["empty"] = man[man["category"] == "empty"].groupby("video_name").size().reindex(vids.index, fill_value=0)
    vinfo = report["videos"]
    vids["fps"] = [vinfo[v]["fps"] for v in vids.index]
    vids["raw"] = [f"{vinfo[v]['size_raw'][0]}x{vinfo[v]['size_raw'][1]}" for v in vids.index]
    vids["saved"] = [vinfo[v]["frames_saved"] for v in vids.index]

    L = []
    w = L.append
    w("# Thống kê bộ dữ liệu hợp nhất (unified)\n")
    w(f"Sinh tự động bởi `scripts/report_unified_dataset.py` lúc {datetime.now():%Y-%m-%d %H:%M}, "
      f"từ bản build `{report['built']}` của `scripts/build_unified_dataset.py`.\n")
    w("Quy ước: **frame có nhãn** = frame được giám sát khi train/đánh giá. **Rỗng** = frame có nhãn nhưng không có lửa/khói. "
      "Một frame có thể có cả lửa và khói (`both`), nên cột *Có lửa* + *Có khói* lớn hơn số frame.\n")

    # ---------------------------------------------------------------- 1. overview
    w("## 1. Tổng quan\n")
    ov = pd.DataFrame({
        "Video": [vids["split"].eq(s).sum() for s in SPLITS],
        "Frame video có nhãn": [man["split"].eq(s).sum() for s in SPLITS],
        "Clip sparse": [len(sparse[s]) for s in SPLITS],
        "Clip dense": [len(dense[s]) for s in SPLITS],
        "Ảnh tĩnh": [static["split"].eq(s).sum() for s in SPLITS],
    }, index=list(SPLITS))
    ov.index.name = "Split"
    ov = with_total(ov)
    tot = ov.loc["Tổng"]
    for c in ["Video", "Frame video có nhãn", "Ảnh tĩnh"]:
        ov[c] = [f"{v} ({100 * v / tot[c]:.1f}%)" if i != "Tổng" else str(v) for i, v in zip(ov.index, ov[c])]
    w(md_table(ov) + "\n")
    n_files = sum(1 for _ in VIDEO_ROOT.rglob("*") if _.is_file())
    size_gb = sum(p.stat().st_size for p in VIDEO_ROOT.rglob("*") if p.is_file()) / 1024 ** 3
    w(f"- Thư mục `data/dataset_video_unified/`: {n_files:,} file, {size_gb:.2f} GB "
      f"({len(man)} frame có nhãn + {int(vids['saved'].sum()) - len(man):,} frame ngữ cảnh 10 FPS, mask đi kèm frame có nhãn).")
    w(f"- Chia: test {100 * report['test_ratio']:.0f}% + {report['folds']} fold (xem 1b); bố cục mặc định val = fold 0. "
      f"Video cùng nguồn luôn nằm cùng split "
      f"(nhóm nhiều video: {list(report['groups'].values())}).")
    w(f"- Frame `auto_bg` cũ bị loại khỏi nhãn (chỉ còn là ngữ cảnh): {report['dropped_auto_bg']}.\n")

    # ---------------------------------------------------------------- 1b. folds
    if "fold" in man.columns:
        w(fold_section(man, read_csv(SPLITS_DIR / "static_v4i_folds.csv"), report))

    # ---------------------------------------------------------------- 2. video
    w("## 2. Dữ liệu video (bố cục mặc định: val = fold 0)\n")
    w("### 2.1 Theo loại frame (category)\n")
    t = man.pivot_table(index="split", columns="category", values="frame_name", aggfunc="size", fill_value=0)
    t = t.reindex(index=list(SPLITS), columns=list(CATS), fill_value=0)
    t.index.name = "Split"
    t = with_total(t)
    t["Tổng frame"] = t[list(CATS)].sum(1)
    t["Có lửa"] = [man[man["split"] == s]["has_fire"].sum() if s != "Tổng" else man["has_fire"].sum() for s in t.index]
    t["Có khói"] = [man[man["split"] == s]["has_smoke"].sum() if s != "Tổng" else man["has_smoke"].sum() for s in t.index]
    w(md_table(t) + "\n")
    w("Tỉ lệ trong từng split:\n")
    pc = pct_cols(t, list(CATS))
    w(md_table(pc) + "\n")
    w("Tỉ lệ mỗi loại được chia vào train / val / test (mục tiêu ≈ 68 / 17 / 15 với 5 fold):\n")
    feats = {"Frame": man.groupby("split").size(), "Có lửa": man.groupby("split")["has_fire"].sum(),
             "Có khói": man.groupby("split")["has_smoke"].sum(),
             "Rỗng": man[man["category"] == "empty"].groupby("split").size()}
    for k in CATS:
        feats[k] = man[man["category"] == k].groupby("split").size()
    for k, lab in (("positive", "Video dương tính"), ("neg_hard", "Video âm tính khó"), ("neg_normal", "Video âm tính thường")):
        feats[lab] = vids[vids["kind"] == k].groupby("split").size()
    dist = pd.DataFrame(feats).T.reindex(columns=list(SPLITS), fill_value=0).fillna(0).astype(int)
    dist["Tổng"] = dist.sum(1)
    for s in SPLITS:
        dist[s] = [f"{v} ({100 * v / max(t_, 1):.1f}%)" for v, t_ in zip(dist[s], dist["Tổng"])]
    dist.index.name = "Đại lượng"
    w(md_table(dist) + "\n")

    w("### 2.2 Theo nguồn gán nhãn (round)\n")
    r = man.pivot_table(index="round", columns="split", values="frame_name", aggfunc="size", fill_value=0)
    r = r.reindex(columns=list(SPLITS), fill_value=0)
    r["Video"] = vids.groupby("round").size()
    r["Frame"] = r[list(SPLITS)].sum(1)
    for k in CATS:
        r[k] = man[man["category"] == k].groupby("round").size().reindex(r.index, fill_value=0)
    r.index = [ROUND_NAMES[i] for i in r.index]
    r.index.name = "Nguồn"
    w(md_table(with_total(r)) + "\n")
    w("- Round 1: `fire_smoke_video.v3i` (frame gốc). Round 2: `Frames_video.v1i` (28 video). "
      "Round 3: 304 ảnh mới trong `Frames_video.v2i` (11 video). Âm tính: 32 video không có lửa/khói (mask toàn 0).\n")

    w("### 2.3 Theo loại video và chủ đề\n")
    k = vids.pivot_table(index="kind", columns="split", values="frames", aggfunc="size", fill_value=0)
    k = k.reindex(columns=list(SPLITS), fill_value=0)
    k["Tổng video"] = k.sum(1)
    k["Frame"] = vids.groupby("kind")["frames"].sum()
    k.index.name = "Loại video"
    w(md_table(with_total(k)) + "\n")
    tp = vids.groupby("topic").agg(Video=("frames", "size"), Frame=("frames", "sum"), Lửa=("fire", "sum"),
                                   Khói=("smoke", "sum"),
                                   train=("split", lambda s: (s == "train").sum()),
                                   val=("split", lambda s: (s == "val").sum()),
                                   test=("split", lambda s: (s == "test").sum()))
    tp = tp.sort_values("Frame", ascending=False)
    tp.index.name = "Chủ đề (theo tên file)"
    w(md_table(tp) + "\n")

    w("### 2.4 Số frame có nhãn mỗi video\n")
    fpv = []
    for s in SPLITS + ("Tổng",):
        d = vids if s == "Tổng" else vids[vids["split"] == s]
        fpv.append({"Split": s, "Video": len(d), "Min": d["frames"].min(), "Trung vị": int(d["frames"].median()),
                    "TB": round(d["frames"].mean(), 1), "Max": d["frames"].max(),
                    "Video < 8 frame (clip sparse bị pad)": int((d["frames"] < 8).sum())})
    w(md_table(pd.DataFrame(fpv).set_index("Split")) + "\n")
    top = vids.sort_values("frames", ascending=False).head(5)
    w("Video nhiều frame nhất: " + ", ".join(f"`{v}` ({n}, {s})" for v, n, s in zip(top.index, top["frames"], top["split"])) + ".\n")

    w("### 2.5 Độ phân giải và FPS\n")
    res = man.assign(res=man["width"].astype(str) + "x" + man["height"].astype(str))
    rt = res.pivot_table(index="res", columns="split", values="frame_name", aggfunc="size", fill_value=0)
    rt = rt.reindex(columns=list(SPLITS), fill_value=0)
    rt["Tổng"] = rt.sum(1)
    rt = rt.sort_values("Tổng", ascending=False)
    rt.index.name = "Kích thước lưu (W×H)"
    w(md_table(rt) + "\n")
    orient = Counter("ngang" if a > b else ("dọc" if a < b else "vuông") for a, b in zip(vids["w"], vids["h"]))
    w(f"- Hướng khung hình (theo video): {dict(orient)}.")
    rawc = Counter(vids["raw"])
    w(f"- Độ phân giải video gốc phổ biến: {dict(rawc.most_common(6))}.")
    fps = Counter(int(round(f)) for f in vids["fps"])
    w(f"- FPS video gốc: {dict(sorted(fps.items()))}. Bước lấy frame ngữ cảnh = round(FPS/10).\n")

    w("### 2.6 Diện tích lửa / khói trên frame (% diện tích frame, chỉ tính frame có lớp đó)\n")
    w(coverage_table(man, man["area"]) + "\n")
    w("### 2.7 Số vùng (connected component ≥ 16 px) mỗi frame\n")
    ob = []
    for s in SPLITS + ("Tổng",):
        d = man if s == "Tổng" else man[man["split"] == s]
        f, sm = d[d["has_fire"] == 1], d[d["has_smoke"] == 1]
        ob.append({"Split": s, "Vùng lửa": int(d["fire_regions"].sum()), "TB vùng lửa / frame có lửa": round(f["fire_regions"].mean(), 2),
                   "Frame ≥3 vùng lửa": int((d["fire_regions"] >= 3).sum()),
                   "Vùng khói": int(d["smoke_regions"].sum()), "TB vùng khói / frame có khói": round(sm["smoke_regions"].mean(), 2)})
    w(md_table(pd.DataFrame(ob).set_index("Split")) + "\n")
    w("### 2.8 Pixel theo lớp (toàn bộ frame có nhãn)\n")
    px = []
    for s in SPLITS + ("Tổng",):
        d = man if s == "Tổng" else man[man["split"] == s]
        a, fi, sm = d["area"].sum(), d["fire_pixels"].sum(), d["smoke_pixels"].sum()
        px.append({"Split": s, "Nền %": f"{100 * (a - fi - sm) / a:.2f}", "Lửa %": f"{100 * fi / a:.2f}",
                   "Khói %": f"{100 * sm / a:.2f}", "Tỉ lệ nền : lửa : khói": f"{(a - fi - sm) / max(fi, 1):.0f} : 1 : {sm / max(fi, 1):.1f}"})
    w(md_table(pd.DataFrame(px).set_index("Split")) + "\n")

    w("### 2.9 Clip\n")
    cl = []
    for s in SPLITS:
        sp, de = sparse[s], dense[s]
        cl.append({"Split": s, "Sparse: clip": len(sp),
                   "Sparse: clip pad (video ngắn)": int(sp["is_padded"].astype(str).str.lower().eq("true").sum()),
                   "Dense: clip": len(de),
                   "Dense: clip chạm đầu video": int(de["is_padded"].astype(str).str.lower().eq("true").sum())})
    w(md_table(pd.DataFrame(cl).set_index("Split")) + "\n")
    w("- **Sparse**: T = 8 frame có nhãn liên tiếp (1–2 FPS), stride 2 (train) / 4 (val, test); khoảng trống > 2× nhịp "
      "thường của video thì tách đoạn; mọi frame có nhãn nằm trong ≥ 1 clip.")
    w("- **Dense**: mỗi frame có nhãn là frame cuối của một clip T = 8; 7 frame trước lấy ở 10 FPS từ video gốc "
      "(không có nhãn → mask 255). *Chạm đầu video* = frame có nhãn quá sát đầu video nên các frame đầu clip bị lặp.\n")

    # ---------------------------------------------------------------- 3. static
    w("## 3. Ảnh tĩnh (Stage 1, `Fire_Smoke.v4i`)\n")
    st = static.pivot_table(index="split", columns="category", values="filename", aggfunc="size", fill_value=0)
    st = st.reindex(index=list(SPLITS), columns=list(CATS), fill_value=0)
    st.index.name = "Split"
    st = with_total(st)
    st["Tổng ảnh"] = st[list(CATS)].sum(1)
    w(md_table(st) + "\n")
    w(md_table(pct_cols(st, list(CATS))) + "\n")
    sres = Counter(f"{a}x{b}" for a, b in zip(static["width"], static["height"]))
    w(f"- Kích thước ảnh phổ biến: {dict(sres.most_common(8))}. "
      f"Ảnh vuông 640×640 / 416×416 nhiều khả năng đã bị Roboflow kéo giãn (không còn ảnh gốc).")
    w(f"- Ảnh tĩnh được chia theo nhóm ảnh gần trùng (phash Hamming ≤ 4 + cùng ảnh nguồn Roboflow); nhóm giống frame "
      f"video val/test bị ép vào đúng split đó.\n")
    w("Diện tích lửa / khói (% diện tích ảnh, chỉ tính ảnh có lớp đó):\n")
    w(coverage_table(static, static["area"]) + "\n")

    # ---------------------------------------------------------------- 4. per video
    w("## 4. Danh sách video\n")
    for s in SPLITS:
        d = vids[vids["split"] == s].sort_values(["kind", "round"], ascending=[False, True])
        tab = pd.DataFrame({"Video": [f"`{v}`" for v in d.index], "Nguồn": [ROUND_NAMES[x] for x in d["round"]],
                            "Chủ đề": d["topic"].values, "Frame": d["frames"].values, "Lửa": d["fire"].values,
                            "Khói": d["smoke"].values, "Rỗng": d["empty"].values, "Gốc": d["raw"].values,
                            "FPS": [f"{x:.0f}" for x in d["fps"]]})
        w(f"### {s} — {len(d)} video, {int(d['frames'].sum())} frame\n")
        w(md_table(tab, index=False) + "\n")

    # ---------------------------------------------------------------- 5. notes
    w("## 5. Lưu ý khi dùng\n")
    w(f"- Val ({vids['split'].eq('val').sum()} video) và test ({vids['split'].eq('test').sum()} video) vẫn nhỏ: "
      "kết quả đánh giá có khoảng tin cậy rộng, nên báo cáo kèm bootstrap CI theo video.")
    w("- Split hiện tại khác split ngày 2026-10-19, nên số liệu test cũ không so sánh trực tiếp được.")
    w("- Với K-fold: chọn siêu tham số theo trung bình val của các fold; Stage 2 fold k phải dùng Stage 1 fold k "
      "(`segformer_spatial_best_fold<k>.pth`) vì ảnh tĩnh giống frame video đã được xếp cùng fold.")
    w("- Lửa chiếm rất ít pixel (xem 2.6, 2.8): phần lớn frame có lửa < 1% diện tích → cần class weight / Dice và metric object-level.")
    weld = vids[vids["topic"].str.contains("welding")]
    if len(weld):
        w(f"- {len(weld)} video `welding sparks` (tia hàn) được gán nhãn lửa/khói "
          f"({int(weld['fire'].sum())} frame lửa, {int(weld['smoke'].sum())} frame khói). "
          "Nếu muốn dùng chúng làm mẫu âm tính khó thì cần xem lại nhãn.")
    rr = man[man["round"].isin([2, 3])]
    w(f"- Frame chưa được export từ Roboflow chỉ làm ngữ cảnh, nên video round 2/3 gần như không có frame rỗng được giám sát "
      f"({int((rr['category'] == 'empty').sum())}/{len(rr)} frame). Frame rỗng chủ yếu đến từ 32 video âm tính.")
    few = vids[(vids["kind"] == "positive") & (vids["fire"] + vids["smoke"] == 0)]
    if len(few):
        w(f"- {len(few)} video dương tính không có frame lửa/khói nào được gán nhãn (chỉ có frame rỗng): "
          + ", ".join(f"`{v}` ({n} frame, {s})" for v, n, s in zip(few.index, few["frames"], few["split"]))
          + ". Nên kiểm tra lại nhãn trên Roboflow.")
    w("- *Chủ đề* lấy từ từ khóa lúc crawl (tên file), không phải từ nhãn: ví dụ một số video `heavy smoke` / `cooking smoke` "
      "chỉ có nhãn lửa.")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"💾 {OUT}")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
