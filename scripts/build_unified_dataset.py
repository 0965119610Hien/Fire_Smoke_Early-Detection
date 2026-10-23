"""
scripts/build_unified_dataset.py
================================
Gathers ALL labeled video data into one dataset with one preprocessing
convention, then re-splits videos (and static images) evenly.

Sources of labeled video frames
  - round 1 + negatives : data/dataset_video_prepared (raw frames), located in
                          the source video by build_dense_clips.py (anchor_raw_idx)
  - round 2 + round 3   : Roboflow export Frames_video.v2i (640x640, stretched,
                          CLAHE'd at extraction). Each export image is mapped back to
                          frames_for_annotation/<video>/frame_XXXXX.jpg (round 2 via
                          v1_matched_mapping.csv, new frames via pixel matching) and
                          to the source-video frame via `frame_orig`.

One storage convention for every frame (no enhancement baked in)
  - decoded from the ORIGINAL video, aspect ratio preserved, long side <= 960,
    JPEG q95: data/dataset_video_unified/<video>/raw_<idx>.jpg
  - mask resized with nearest neighbour to the same size: raw_<idx>_mask.png
  - CLAHE / dehaze / motion / letterbox happen only at load time (fire_core).

Label policy
  - Only frames that carry a label (Roboflow export, or an `empty`-type frame of a
    negative video) are supervised. Frames that were extracted but never exported
    (old `auto_bg`) are NOT assumed to be background: they are context only.

Split (MILP minimax, scipy / HiGHS)
  - Videos of one source (same pexels / YouTube id, chunks of one recording) form a group.
  - Groups go to a fixed TEST part (default 15%) + K cross-validation folds (default 5) so that,
    in every part, the share of labeled frames, fire / smoke / empty frames, fire / smoke AREA
    (sum of % coverage) and small / medium / large fire / smoke frames is as close as possible
    to the target (the largest deviation is minimised). Numbers of positive / hard-negative /
    normal-negative videos are hard bounds (floor / ceil of the target, >= 1 per part).
  - Static images get the same parts. A static group that looks like a frame of a video is
    forced into that video's part, so in every fold Stage 1 never trains on val / test frames.
    The video split already balances these linked static images (empty / with fire-smoke).

Outputs (data/data_splits/, previous files backed up to archive/<ts>/)
  data_splits/            default layout: test + val = fold 0 + train = folds 1..K-1
  data_splits/folds/fold<k>/  same files with val = fold k
    video_manifest_with_splits.csv, {split}_manifest.csv, {split}_clips.csv (sparse),
    dense_{split}_clips.csv (7 context frames at 10 FPS before each labeled frame),
    static_v4i_{split}_manifest.csv
  video_split_assignment.csv, static_v4i_folds.csv, unified_build_report.json

Usage:
  python scripts/build_unified_dataset.py                  # full build (decode + split)
  python scripts/build_unified_dataset.py --resplit-only   # only redo the split
"""

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLITS_DIR = DATA / "data_splits"
# frozen inputs of the old pipelines (frame names / raw indices / Roboflow mapping of rounds 1-2)
LEGACY = SPLITS_DIR / "legacy_sources"
PREPARED = DATA / "dataset_video_prepared"
ANNOT = DATA / "frames_for_annotation"
RF_EXPORT = DATA / "Frames_video.v2i.png-mask-semantic" / "train"
RAW_DIR = DATA / "all_videos_raw"
OUT = DATA / "dataset_video_unified"
SPLITS = ("train", "val", "test")
MATCH_SIZE = (64, 64)
MATCH_MAX_DIFF = 3.0      # mean abs diff (0-255) of a correct export <-> annotation-frame match
MATCH_MIN_RATIO = 5.0     # second-best / best

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def read_csv(p):
    return pd.read_csv(p, encoding="utf-8-sig")


def imread_u(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread that also works with non-ASCII Windows paths."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, flags) if buf.size else None


def imwrite_u(path, img, params=()):
    ok, buf = cv2.imencode(Path(path).suffix, img, list(params))
    if not ok:
        raise IOError(f"Không ghi được {path}")
    buf.tofile(str(path))


def category(has_fire, has_smoke):
    return "both" if has_fire and has_smoke else ("fire_only" if has_fire else ("smoke_only" if has_smoke else "empty"))


# -----------------------------------------------------------------------------
# 1. Labeled anchors per video
# -----------------------------------------------------------------------------
def match_new_exports(known):
    """Map Roboflow export images not in `known` to (video, frame_XXXXX.jpg) by pixel matching."""
    files = sorted(p.name for p in RF_EXPORT.glob("*.jpg") if p.name not in known)
    if not files:
        return pd.DataFrame(columns=["rf_img", "rf_mask", "orig_name", "video_name", "diff", "ratio"])
    vids = sorted(p.name for p in ANNOT.iterdir() if p.is_dir())
    cache = {}

    def small(v, name):
        if (v, name) not in cache:
            p = ANNOT / v / name
            im = imread_u(p) if p.exists() else None
            cache[(v, name)] = None if im is None else cv2.resize(im, MATCH_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
        return cache[(v, name)]

    rows = []
    for f in files:
        name = re.match(r"(frame_\d+)_jpg", f).group(1) + ".jpg"
        q = cv2.resize(imread_u(RF_EXPORT / f), MATCH_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
        ds = sorted((float(np.abs(q - t).mean()), v) for v in vids if (t := small(v, name)) is not None)
        best, second = ds[0], (ds[1] if len(ds) > 1 else (1e9, None))
        rows.append({"rf_img": f, "rf_mask": f.replace(".jpg", "_mask.png"), "orig_name": name,
                     "video_name": best[1], "diff": best[0], "ratio": second[0] / max(best[0], 1e-6)})
    df = pd.DataFrame(rows)
    bad = df[(df["diff"] > MATCH_MAX_DIFF) | (df["ratio"] < MATCH_MIN_RATIO)]
    if len(bad):
        raise RuntimeError(f"{len(bad)} ảnh export ghép không chắc chắn:\n{bad.head(20).to_string()}")
    return df


def roboflow_anchors():
    """Rounds 2-3: one anchor per exported image, raw index from the annotation manifest."""
    v1 = read_csv(LEGACY / "v1_matched_mapping.csv")
    v1["round"] = 2
    new = match_new_exports(set(v1["rf_img"]))
    new["round"] = 3
    m = pd.concat([v1[["rf_img", "rf_mask", "orig_name", "video_name", "round"]],
                   new[["rf_img", "rf_mask", "orig_name", "video_name", "round"]]], ignore_index=True)
    exported = {p.name for p in RF_EXPORT.glob("*.jpg")}
    missing = exported - set(m["rf_img"])
    if missing:
        raise RuntimeError(f"{len(missing)} ảnh export chưa được ghép: {sorted(missing)[:5]}")
    dup = m.duplicated(["video_name", "orig_name"])
    if dup.any():
        raise RuntimeError(f"Trùng frame trong export:\n{m[dup].to_string()}")

    annot = read_csv(ANNOT / "manifest.csv")
    orig = {(r.video_stem, int(r.frame_idx)): (int(r.frame_orig), int(r.width), int(r.height))
            for r in annot.itertuples()}
    rows = []
    for r in m.itertuples():
        fi = int(re.search(r"(\d+)", r.orig_name).group(1))
        raw_idx, w, h = orig[(r.video_name, fi)]
        rows.append({"video_name": r.video_name, "round": r.round, "raw_idx": raw_idx,
                     "mask_src": str(RF_EXPORT / r.rf_mask), "src_size": (w, h), "src_note": f"{r.rf_img}"})
    return pd.DataFrame(rows), len(new)


def prepared_anchors(skip_videos):
    """Round 1 + negative videos from dataset_video_prepared (auto_bg frames dropped)."""
    man = read_csv(LEGACY / "prepared_video_manifest_with_splits.csv")
    man = man[~man["video_name"].isin(skip_videos)]
    dense = pd.concat([read_csv(LEGACY / f"dense_{s}_clips.csv") for s in SPLITS], ignore_index=True)
    raw_of = {(r.video_name, r.anchor_frame): int(r.anchor_raw_idx) for r in dense.itertuples()}
    rows, dropped = [], Counter()
    for r in man.itertuples():
        if r.mask_type == "auto_bg":
            dropped[r.video_name] += 1
            continue
        rows.append({"video_name": r.video_name, "round": 0 if str(r.video_name).startswith("neg_") else 1,
                     "raw_idx": raw_of[(r.video_name, r.frame_name)], "mask_src": str(PREPARED / r.mask_path),
                     "src_size": None, "src_note": r.frame_name})
    return pd.DataFrame(rows), dropped


def source_videos(videos):
    report = json.loads((LEGACY / "dense_build_report.json").read_text(encoding="utf-8"))["videos"]
    raw = {p.stem: p for p in RAW_DIR.iterdir() if p.is_file()}
    src = {}
    for v in videos:
        if v in raw:
            src[v] = raw[v]
        elif v in report and "source" in report[v]:
            src[v] = ROOT / report[v]["source"]
        else:
            raise RuntimeError(f"Không tìm thấy video gốc của {v}")
    return src


# -----------------------------------------------------------------------------
# 2. Grouping + split
# -----------------------------------------------------------------------------
def source_ids(video_name):
    """Identifiers of the underlying recording: pexels id, YouTube id or the de-chunked name."""
    s = re.sub(r"_chunk_\d+$", "", video_name)
    s = re.sub(r"_pure$", "", s)
    ids = {"name:" + s}
    m = re.search(r"pexels_.*?_(\d{5,})$", s) or re.match(r"(\d{7,})[_-]", s)
    if m:
        ids.add("pexels:" + m.group(1))
    m = re.search(r"(?:negatives|normal|runaway|ignition)_([A-Za-z0-9_-]{11})", s)
    if m:
        ids.add("yt:" + m.group(1))
    return ids


def group_videos(videos):
    owner, parent = {}, {v: v for v in videos}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for v in videos:
        for i in source_ids(v):
            if i in owner:
                parent[find(v)] = find(owner[i])
            else:
                owner[i] = v
    return {v: find(v) for v in videos}


def apply_label_overrides_to(man):
    """Manual relabels (data/label_overrides.csv, scripts/apply_label_overrides.py) on top of the
    freshly decoded masks, so a full rebuild keeps them; label columns of `man` are recomputed."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from apply_label_overrides import apply_overrides

    ov, n_changed, video_stats, _ = apply_overrides()
    keys = man["mask_path"]
    for i in man.index[keys.isin(video_stats)]:
        fire, smoke, _ = video_stats[keys[i]]
        man.loc[i, ["has_fire", "has_smoke", "fire_pixels", "smoke_pixels", "category"]] = [
            int(fire > 0), int(smoke > 0), fire, smoke, category(fire > 0, smoke > 0)]
    print(f"    Override nhãn: {len(ov)} dòng, {n_changed} mask được ghi lại, "
          f"{int(keys.isin(video_stats).sum())} frame video cập nhật")
    return man


def video_kinds(man):
    """video_name -> kind, from the labels (not the name): a video with any fire / smoke frame is
    positive; an all-empty video is neg_normal if named so, otherwise neg_hard (e.g. a `neg_hard_*`
    video, or a former positive whose smoke turned out to be steam / welding)."""
    pos = (man["category"] != "empty").groupby(man["video_name"]).any()
    return {v: "positive" if p else ("neg_normal" if v.startswith("neg_normal") else "neg_hard")
            for v, p in pos.items()}


FIRE_BINS = (0.3, 2.0)     # fire coverage (% of frame): small < 0.3 <= medium < 2.0 <= large
SMOKE_BINS = (3.0, 15.0)   # smoke coverage (% of frame)
KINDS = ("positive", "neg_hard", "neg_normal")


def area_features(df, fcov, scov):
    """Balance targets of a set of labeled frames / images: counts, total coverage, coverage bins."""
    f, s = fcov[df["has_fire"] == 1], scov[df["has_smoke"] == 1]
    return {
        "frames": len(df), "fire": len(f), "smoke": len(s),
        "fire_only": int((df["category"] == "fire_only").sum()), "smoke_only": int((df["category"] == "smoke_only").sum()),
        "both": int((df["category"] == "both").sum()), "empty": int((df["category"] == "empty").sum()),
        "fire_area": float(f.sum()), "smoke_area": float(s.sum()),
        "fire_S": int((f < FIRE_BINS[0]).sum()), "fire_M": int(((f >= FIRE_BINS[0]) & (f < FIRE_BINS[1])).sum()),
        "fire_L": int((f >= FIRE_BINS[1]).sum()),
        "smoke_S": int((s < SMOKE_BINS[0]).sum()), "smoke_M": int(((s >= SMOKE_BINS[0]) & (s < SMOKE_BINS[1])).sum()),
        "smoke_L": int((s >= SMOKE_BINS[1]).sum()),
    }


def milp_assign(X, ratios, counts=None, fixed=None, time_limit=180):
    """Assign groups (rows of X) to bins so every column's share is close to `ratios`.

    Minimises the largest AVOIDABLE |share - ratio| over all bins x columns (tie-break: their
    sum), with HiGHS via scipy.optimize.milp. Deviation that no assignment can avoid (one
    group bigger than a bin's target, forced groups) is subtracted before taking the max.
    counts: (n, K) per-group counts that get HARD bounds floor/ceil(ratio * total) per bin,
            and at least 1 per bin (e.g. number of videos of each kind).
    fixed:  {row: bin} groups forced into a bin.
    Returns (bins array, max deviation in share points, solver status message).
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    X = np.asarray(X, dtype=np.float64)
    keep = X.sum(0) > 0
    X = X[:, keep]
    n, F = X.shape
    B = len(ratios)
    r = np.asarray(ratios, dtype=np.float64)
    tot = X.sum(0)
    nx, nd = n * B, B * F
    nv = nx + nd + 1
    xi = lambda g, b: g * B + b
    di = lambda b, f: nx + b * F + f

    rows, cols, vals, lo, hi = [], [], [], [], []
    k = 0

    def add(entries, l, h):
        nonlocal k
        for c, v in entries:
            rows.append(k), cols.append(c), vals.append(v)
        lo.append(l), hi.append(h)
        k += 1

    for g in range(n):
        add([(xi(g, b), 1.0) for b in range(B)], 1, 1)
    # Unavoidable deviation, which must not set the minimax level for every other column:
    # a single group holding more than any bin's target share, and the forced groups.
    lump = np.clip((X / tot).max(0) - r.max(), 0, None)
    forced = np.zeros((B, F))
    for g, b in (fixed or {}).items():
        forced[b] += X[g] / tot
    floor = np.maximum(lump[None, :], np.clip(forced - r[:, None], 0, None))
    for b in range(B):
        for f in range(F):
            share = [(xi(g, b), X[g, f] / tot[f]) for g in range(n) if X[g, f]]
            add(share + [(di(b, f), -1.0)], -np.inf, r[b])
            add(share + [(di(b, f), 1.0)], r[b], np.inf)
            add([(di(b, f), 1.0), (nv - 1, -1.0)], -np.inf, floor[b, f])  # z = largest AVOIDABLE deviation
    if counts is not None:
        C = np.asarray(counts, dtype=np.float64)
        for j in range(C.shape[1]):
            total = C[:, j].sum()
            if not total:
                continue
            for b in range(B):
                add([(xi(g, b), C[g, j]) for g in range(n) if C[g, j]],
                    max(1, np.floor(r[b] * total)), max(1, np.ceil(r[b] * total)))

    c = np.zeros(nv)
    c[-1] = 1.0
    c[nx:nx + nd] = 1e-3
    lb, ub = np.zeros(nv), np.r_[np.ones(nx), np.full(nd + 1, np.inf)]
    for g, b in (fixed or {}).items():
        ub[xi(g, 0):xi(g, 0) + B] = 0
        lb[xi(g, b)] = ub[xi(g, b)] = 1
    A = coo_matrix((vals, (rows, cols)), shape=(k, nv)).tocsr()
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.r_[np.ones(nx), np.zeros(nd + 1)],
               bounds=Bounds(lb, ub), options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        raise RuntimeError(f"MILP không tìm được lời giải: {res.message}")
    assign = res.x[:nx].reshape(n, B).argmax(1)
    S = np.stack([X[assign == b].sum(0) for b in range(B)])
    maxdev = float(np.abs(100 * S / tot - 100 * r[:, None]).max())
    avoidable = float(np.clip(np.abs(S / tot - r[:, None]) - floor, 0, None).max() * 100)
    return assign, maxdev, f"lệch tránh được tối đa {avoidable:.2f} điểm % | {res.message}"


def bin_name(b, n_folds):
    return "test" if b == n_folds else f"fold{b}"


def link_static(static, video_man, group_of):
    """Static groups (near duplicates: phash Hamming <= 4, same Roboflow source image) and the
    video source group each one looks like (Hamming <= 6 to one of its frames), if any."""
    from audit_data import roboflow_source_name
    from fix_data import STATIC_DUP_HAMMING, STATIC_ROOT, STATIC_VIDEO_HAMMING, UnionFind, phash_bits

    n = len(static)
    sb = phash_bits([STATIC_ROOT / f for f in static["filename"]])
    vb = phash_bits([OUT / p for p in video_man["image_path"]])
    vgroup = video_man["video_name"].map(group_of).to_numpy()
    uf = UnionFind(n)
    for i in range(n):
        d = (sb[i] != sb[i + 1:]).sum(axis=1)
        for o in np.nonzero(d <= STATIC_DUP_HAMMING)[0]:
            uf.union(i, i + 1 + int(o))
    by_src = defaultdict(list)
    for i, fn in enumerate(static["filename"]):
        by_src[roboflow_source_name(fn)].append(i)
    for idx in by_src.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
    root = [uf.find(i) for i in range(n)]
    gidx = {g: k for k, g in enumerate(sorted(set(root)))}
    sgroup = np.array([gidx[x] for x in root])
    hits = defaultdict(Counter)          # static group -> Counter(video source group)
    for i in range(n):
        for vg in vgroup[(vb != sb[i]).sum(axis=1) <= STATIC_VIDEO_HAMMING]:
            hits[int(sgroup[i])][vg] += 1
    return sgroup, dict(hits)


def split_videos(man, group_of, n_folds, test_ratio, time_limit, static=None, link=None):
    """Video groups -> bin (0..n_folds-1 = CV folds, n_folds = test).

    The static images that will follow a video (see split_static) are balance targets too
    (`static_empty`, `static_pos`), so the forced static images also end up evenly spread."""
    area = man["width"] * man["height"]
    fcov, scov = 100 * man["fire_pixels"] / area, 100 * man["smoke_pixels"] / area
    man = man.assign(group=man["video_name"].map(group_of), kind=man["video_name"].map(video_kinds(man)))
    groups = sorted(man["group"].unique())
    X = pd.DataFrame({g: area_features(d, fcov[d.index], scov[d.index]) for g, d in man.groupby("group")}).T.reindex(groups)
    if link is not None:
        sgroup, hits = link
        empty = (static["category"] == "empty").to_numpy()
        X["static_empty"] = 0
        X["static_pos"] = 0
        for g, cnt in hits.items():
            vg = cnt.most_common(1)[0][0]
            X.loc[vg, "static_empty"] += int(empty[sgroup == g].sum())
            X.loc[vg, "static_pos"] += int((~empty[sgroup == g]).sum())
    vk = man.groupby("video_name")[["group", "kind"]].first()
    counts = pd.crosstab(vk["group"], vk["kind"]).reindex(index=groups, columns=list(KINDS), fill_value=0)
    ratios = [(1 - test_ratio) / n_folds] * n_folds + [test_ratio]
    assign, maxdev, msg = milp_assign(X.to_numpy(), ratios, counts=counts.to_numpy(), time_limit=time_limit)
    return dict(zip(groups, assign)), X, maxdev, msg


def split_static(static, link, group_bin, n_folds, test_ratio, time_limit):
    """Static images -> same bins as the videos. A static group that looks like a frame of a
    video is forced into that video's bin, so that in every fold Stage 1 never trains on
    frames of the validation / test videos."""
    sgroup, hits = link
    n_groups = int(sgroup.max()) + 1
    fixed, conflicts = {}, []
    for g, cnt in hits.items():
        bins = Counter()
        for vg, k in cnt.items():
            bins[int(group_bin[vg])] += k
        b = n_folds if n_folds in bins else bins.most_common(1)[0][0]
        fixed[g] = b
        if len(bins) > 1:
            conflicts.append({"group_images": static["filename"][sgroup == g].tolist()[:5],
                              "video_bins": {bin_name(k, n_folds): v for k, v in bins.items()}, "chosen": bin_name(b, n_folds)})

    total = static["width"] * static["height"]
    fcov, scov = 100 * static["fire_pixels"] / total, 100 * static["smoke_pixels"] / total
    st = static.assign(g=sgroup)
    X = pd.DataFrame([area_features(d, fcov[d.index], scov[d.index]) for _, d in st.groupby("g")])
    ratios = [(1 - test_ratio) / n_folds] * n_folds + [test_ratio]
    assign, maxdev, msg = milp_assign(X.to_numpy(), ratios, fixed=fixed, time_limit=time_limit)
    return assign[sgroup], {"groups": n_groups, "multi_image_groups": int((np.bincount(sgroup) > 1).sum()),
                            "forced_by_video": len(fixed), "conflicts": conflicts,
                            "max_dev_points": maxdev, "solver": msg}


# -----------------------------------------------------------------------------
# 3. Decode frames
# -----------------------------------------------------------------------------
def target_size(w, h, max_side):
    s = min(1.0, max_side / max(w, h))
    return int(round(w * s)), int(round(h * s))


def build_video(vid, anchors, src, fps_ctx, T, max_side):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tw, th = target_size(W, H, max_side)
    step = max(1, int(round(fps / fps_ctx)))
    anchors = anchors.sort_values("raw_idx").reset_index(drop=True)
    anchor_idx = set(anchors["raw_idx"].astype(int))
    ctx = {max(0, j - step * t) for j in anchor_idx for t in range(1, T)}
    need = anchor_idx | ctx

    vdir = OUT / vid
    vdir.mkdir(parents=True, exist_ok=True)
    f, got = 0, set()
    last = max(need)
    while f <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if f in need:
            if fr.shape[1::-1] != (W, H):
                raise RuntimeError(f"{vid}: kích thước frame {f} khác metadata")
            imwrite_u(vdir / f"raw_{f:06d}.jpg", cv2.resize(fr, (tw, th), interpolation=cv2.INTER_AREA),
                      [cv2.IMWRITE_JPEG_QUALITY, 95])
            got.add(f)
        f += 1
    cap.release()
    if anchor_idx - got:
        raise RuntimeError(f"{vid}: không giải mã được frame {sorted(anchor_idx - got)[:5]} (video có {n_frames})")

    records, aspect_err = [], []
    for r in anchors.itertuples():
        mask = imread_u(r.mask_src, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(r.mask_src)
        vals = set(np.unique(mask).tolist())
        if not vals <= {0, 1, 2}:
            raise ValueError(f"{r.mask_src}: giá trị mask {sorted(vals)}")
        if r.src_size is None:  # round 1 / negatives: mask is at the prepared frame's aspect
            aspect_err.append(abs(mask.shape[1] / mask.shape[0] - W / H) / (W / H))
        mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        imwrite_u(vdir / f"raw_{r.raw_idx:06d}_mask.png", mask)
        fire, smoke = int((mask == 1).sum()), int((mask == 2).sum())
        records.append({
            "video_name": vid, "frame_index": int(r.raw_idx), "frame_name": f"raw_{r.raw_idx:06d}",
            "image_path": f"{vid}/raw_{r.raw_idx:06d}.jpg", "mask_path": f"{vid}/raw_{r.raw_idx:06d}_mask.png",
            "round": int(r.round), "has_fire": int(fire > 0), "has_smoke": int(smoke > 0),
            "fire_pixels": fire, "smoke_pixels": smoke, "category": category(fire > 0, smoke > 0),
            "width": tw, "height": th, "source_ref": r.src_note,
        })
    info = {"source": str(Path(src).relative_to(ROOT)), "fps": fps, "size_raw": [W, H], "size": [tw, th],
            "ctx_step": step, "anchors": len(anchors), "frames_saved": len(got),
            "max_aspect_err": float(max(aspect_err)) if aspect_err else None}
    return pd.DataFrame(records), info


# -----------------------------------------------------------------------------
# 4. Clips
# -----------------------------------------------------------------------------
def dense_clips(man, step_of, T):
    out = []
    for vid, g in man.groupby("video_name"):
        lab = dict(zip(g["frame_index"], g["frame_name"]))
        step = step_of[vid]
        for r in g.sort_values("frame_index").itertuples():
            seq = [max(0, r.frame_index - step * (T - 1 - t)) for t in range(T)]
            out.append({
                "video_name": vid, "anchor_frame": r.frame_name, "anchor_raw_idx": r.frame_index,
                "start_idx": seq[0], "end_idx": seq[-1], "clip_length": T,
                "frame_names": ";".join(lab.get(j, f"ctx_raw_{j:06d}") for j in seq),
                "image_paths": ";".join(f"{vid}/raw_{j:06d}.jpg" for j in seq),
                "mask_paths": ";".join(f"{vid}/raw_{j:06d}_mask.png" if j in lab else "" for j in seq),
                "has_fire": r.has_fire, "has_smoke": r.has_smoke, "category": r.category,
                "split": r.split, "is_padded": seq[0] == seq[1],
            })
    return pd.DataFrame(out)


def sparse_clips(man, T, strides):
    """Clips of consecutive labeled frames; a gap > 2x the video's usual spacing starts a new run."""
    from create_split_manifests_and_clips import generate_clips_for_video
    out = []
    for vid, g in man.groupby("video_name"):
        g = g.sort_values("frame_index").reset_index(drop=True)
        gaps = np.diff(g["frame_index"].to_numpy())
        typical = np.median(gaps) if len(gaps) else 1
        breaks = np.nonzero(gaps > 2 * typical)[0] + 1
        for run in np.split(np.arange(len(g)), breaks):
            sub = g.iloc[run]
            for c in generate_clips_for_video(sub, clip_len=T, stride=strides[sub["split"].iloc[0]]):
                c["split"] = sub["split"].iloc[0]
                out.append(c)
    return pd.DataFrame(out)


def split_of_fold(fold_col, val_fold):
    return np.where(fold_col == -1, "test", np.where(fold_col == val_fold, "val", "train"))


def write_split_files(dest, man, static, val_fold, step_of, T):
    """All manifests / clips of one train-val-test layout (val = `val_fold`) into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    man = man.assign(split=split_of_fold(man["fold"].to_numpy(), val_fold))
    static = static.assign(split=split_of_fold(static["fold"].to_numpy(), val_fold))
    man.to_csv(dest / "video_manifest_with_splits.csv", index=False, encoding="utf-8-sig")
    dense = dense_clips(man, step_of, T)
    sparse = sparse_clips(man, T, {"train": 2, "val": 4, "test": 4})
    for s in SPLITS:
        man[man["split"] == s].to_csv(dest / f"{s}_manifest.csv", index=False, encoding="utf-8-sig")
        sparse[sparse["split"] == s].drop(columns=["split"]).to_csv(dest / f"{s}_clips.csv", index=False, encoding="utf-8-sig")
        dense[dense["split"] == s].drop(columns=["split"]).to_csv(dest / f"dense_{s}_clips.csv", index=False, encoding="utf-8-sig")
        static[static["split"] == s].to_csv(dest / f"static_v4i_{s}_manifest.csv", index=False, encoding="utf-8-sig")
    return {"sparse_clips": sparse["split"].value_counts().to_dict(), "dense_clips": dense["split"].value_counts().to_dict()}


def bin_table(df, bins, n_folds, kind_col=None):
    """Per-bin summary: counts, share of fire / smoke area, median coverage."""
    area = df["width"] * df["height"]
    fcov, scov = 100 * df["fire_pixels"] / area, 100 * df["smoke_pixels"] / area
    rows = {}
    for b in list(range(n_folds)) + [-1]:
        m = bins == b
        d = df[m]
        r = {}
        if kind_col:
            for k in KINDS:
                r[k] = int(d.loc[d[kind_col] == k, "video_name"].nunique())
        r.update({"frames": len(d), "fire": int(d["has_fire"].sum()), "smoke": int(d["has_smoke"].sum()),
                  "empty": int((d["category"] == "empty").sum()),
                  "fire_area%": round(100 * fcov[m].sum() / fcov.sum(), 1),
                  "smoke_area%": round(100 * scov[m].sum() / scov.sum(), 1),
                  "fire_med%": round(float(fcov[m & (df["has_fire"] == 1)].median()), 2),
                  "smoke_med%": round(float(scov[m & (df["has_smoke"] == 1)].median()), 2)})
        rows["test" if b == -1 else f"fold{b}"] = r
    t = pd.DataFrame(rows).T
    for c in ("frames", "fire", "smoke", "empty"):
        t[c] = [f"{int(v)} ({100 * v / t[c].sum():.1f}%)" for v in t[c]]
    return t


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resplit-only", action="store_true",
                    help="keep data/dataset_video_unified, only redo the split (no decoding)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--time-limit", type=float, default=600, help="seconds per MILP")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=960)
    args = ap.parse_args()
    K = args.folds
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.resplit_only:
        man = read_csv(SPLITS_DIR / "video_manifest_with_splits.csv")
        man = man.drop(columns=[c for c in ("split", "fold", "source_group") if c in man.columns])
        prev = json.loads((SPLITS_DIR / "unified_build_report.json").read_text(encoding="utf-8"))
        infos, dropped = prev["videos"], prev["dropped_auto_bg"]
        videos = sorted(man["video_name"].unique())
        print(f"[1] Dùng lại {OUT}: {len(man)} frame có nhãn / {len(videos)} video")
    else:
        rf, n_new = roboflow_anchors()
        prep, dropped = prepared_anchors(set(rf["video_name"]))
        anchors = pd.concat([prep, rf], ignore_index=True)
        dup = anchors.duplicated(["video_name", "raw_idx"])
        if dup.any():
            raise RuntimeError(f"Hai nhãn trỏ cùng một frame gốc:\n{anchors[dup].head().to_string()}")
        videos = sorted(anchors["video_name"].unique())
        src = source_videos(videos)
        print(f"[1] {len(anchors)} frame có nhãn / {len(videos)} video "
              f"(round1 {int((anchors['round'] == 1).sum())}, round2 {int((anchors['round'] == 2).sum())}, "
              f"round3 {int((anchors['round'] == 3).sum())} [{n_new} ảnh export mới], negative {int((anchors['round'] == 0).sum())}); "
              f"bỏ {sum(dropped.values())} frame auto_bg (chỉ còn làm ngữ cảnh)")
        if OUT.exists():
            shutil.rmtree(OUT)
        parts, infos = [], {}
        for n, vid in enumerate(videos, 1):
            recs, info = build_video(vid, anchors[anchors["video_name"] == vid], src[vid], args.fps, args.T, args.max_side)
            parts.append(recs)
            infos[vid] = info
            warn = "  ⚠️ lệch tỉ lệ" if (info["max_aspect_err"] or 0) > 0.02 else ""
            print(f"    [{n}/{len(videos)}] {vid[:50]:50s} {info['anchors']:3d} nhãn, {info['frames_saved']:4d} frame, "
                  f"{info['size'][0]}x{info['size'][1]}{warn}", flush=True)
        man = pd.concat(parts, ignore_index=True)
        man = apply_label_overrides_to(man)

    # 2. static <-> video links, then videos -> test + K folds (MILP)
    group_of = group_videos(videos)
    man["source_group"] = man["video_name"].map(group_of)
    static = pd.concat([read_csv(SPLITS_DIR / f"static_v4i_{s}_manifest.csv") for s in SPLITS], ignore_index=True)
    static = static.drop(columns=[c for c in ("split", "fold") if c in static.columns]).reset_index(drop=True)
    print("[2] phash ảnh tĩnh + frame video...", flush=True)
    link = link_static(static, man, group_of)
    gbin, _, vdev, vmsg = split_videos(man, group_of, K, args.test_ratio, args.time_limit, static=static, link=link)
    to_fold = lambda b: -1 if b == K else int(b)
    video_bin = {v: int(gbin[group_of[v]]) for v in videos}
    man["fold"] = man["video_name"].map(lambda v: to_fold(video_bin[v]))
    multi = defaultdict(list)
    for v in videos:
        multi[group_of[v]].append(v)
    multi = {g: vs for g, vs in multi.items() if len(vs) > 1}
    print(f"    Video: {len(set(group_of.values()))} nhóm nguồn -> test + {K} fold | lệch tối đa {vdev:.2f} điểm % | {vmsg}")
    kinds = video_kinds(man)
    man["kind"] = man["video_name"].map(kinds)
    print(bin_table(man, man["fold"].to_numpy(), K, kind_col="kind").to_string())
    man = man.drop(columns=["kind"])

    # 3. static images -> same bins
    sbin, sinfo = split_static(static, link, gbin, K, args.test_ratio, args.time_limit)
    static["fold"] = [to_fold(b) for b in sbin]
    print(f"[3] Ảnh tĩnh: {sinfo['groups']} nhóm, {sinfo['forced_by_video']} nhóm ép theo video, "
          f"{len(sinfo['conflicts'])} nhóm giống video ở nhiều fold | lệch tối đa {sinfo['max_dev_points']:.2f} điểm % | {sinfo['solver']}")
    print(bin_table(static, static["fold"].to_numpy(), K).to_string())

    # 4. write: data_splits/ = fold 0 as val (default), data_splits/folds/fold<k>/ = fold k as val
    archive = SPLITS_DIR / "archive" / ts
    archive.mkdir(parents=True, exist_ok=True)
    for p in SPLITS_DIR.iterdir():
        if p.is_file():
            shutil.copy2(p, archive / p.name)
    folds_dir = SPLITS_DIR / "folds"
    if folds_dir.exists():
        shutil.copytree(folds_dir, archive / "folds")
        shutil.rmtree(folds_dir)
    print(f"[4] Sao lưu data_splits -> {archive}")

    step_of = {v: infos[v]["ctx_step"] for v in videos}
    clips = {"default(fold0)": write_split_files(SPLITS_DIR, man, static, 0, step_of, args.T)}
    for k in range(K):
        clips[f"fold{k}"] = write_split_files(folds_dir / f"fold{k}", man, static, k, step_of, args.T)
    assign = (man.groupby("video_name")
              .agg(fold=("fold", "first"), source_group=("source_group", "first"), round=("round", "first"),
                   frames=("frame_name", "size"), fire=("has_fire", "sum"), smoke=("has_smoke", "sum"))
              .reset_index())
    assign["kind"] = assign["video_name"].map(kinds)
    assign.to_csv(SPLITS_DIR / "video_split_assignment.csv", index=False, encoding="utf-8-sig")
    static.to_csv(SPLITS_DIR / "static_v4i_folds.csv", index=False, encoding="utf-8-sig")

    summary = {
        "built": ts, "args": vars(args), "folds": K, "test_ratio": args.test_ratio,
        "split_method": "MILP minimax (scipy/HiGHS): test + K folds by source group; video kinds as hard count bounds",
        "video_max_dev_points": vdev, "video_solver": vmsg, "static": sinfo, "clips": clips,
        "dropped_auto_bg": dict(dropped), "groups": multi, "videos": infos,
    }
    (SPLITS_DIR / "unified_build_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                                          encoding="utf-8")
    print(f"[5] Clip: {clips}")
    print(f"✅ Xong. Mặc định (data_splits/) = val fold0; từng fold: {folds_dir}/fold<k>/")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
