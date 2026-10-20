"""
scripts/fix_data.py
===================
Fixes the data problems found by scripts/audit_data.py, in place, after backing
everything up. Safe to re-run (each step is idempotent).

1. Backup: every file in data/data_splits/ is copied to data/data_splits/archive/<ts>/.
2. Un-stretch video frames: round-2 videos were exported as 640x640 frames
   squeezed from their original aspect ratio (no padding). Frame + mask are
   resized back to the original aspect (long side 640); originals are copied to
   data/archive/stretched_frames_<ts>/ first.
3. Source-level video leakage: chunks of one recording are moved to one split.
4. Pixel statistics in the video manifest are recomputed from the masks.
5. Per-split manifests and T=8 clips are regenerated (with a tail-anchored clip so
   every frame is covered) via src/create_split_manifests_and_clips.py.
6. Static images are re-split by GROUP (phash Hamming <= 4 + same Roboflow
   source image), stratified by category, keeping the previous split ratios.
   Groups that look like frames of a video are forced into that video's split.

Usage:
  python scripts/fix_data.py            # apply
  python scripts/fix_data.py --dry-run  # report only
"""

import argparse
import random
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
VIDEO_ROOT = DATA / "dataset_video_prepared"
STATIC_ROOT = DATA / "Fire_Smoke.v4i.png-mask-semantic" / "train"
ANNOT_MANIFEST = DATA / "frames_for_annotation" / "manifest.csv"
RAW_VIDEO_DIRS = [DATA / "all_videos_raw", DATA / "video_train", DATA / "video_test"]
VIDEO_MANIFEST = SPLITS_DIR / "prepared_video_manifest_with_splits.csv"
SPLITS = ("train", "val", "test")
SEED = 42
STATIC_DUP_HAMMING = 4
STATIC_VIDEO_HAMMING = 6

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from audit_data import roboflow_source_name, source_video_name  # noqa: E402


def read_csv(p):
    return pd.read_csv(p, encoding="utf-8-sig")


def write_csv(df, p, dry):
    if not dry:
        df.to_csv(p, index=False, encoding="utf-8-sig")


# -----------------------------------------------------------------------------
# 1. Backup
# -----------------------------------------------------------------------------
def backup_splits(ts, dry):
    dest = SPLITS_DIR / "archive" / ts
    files = [p for p in SPLITS_DIR.iterdir() if p.is_file()]
    if not dry:
        dest.mkdir(parents=True, exist_ok=True)
        for p in files:
            shutil.copy2(p, dest / p.name)
    print(f"[1] Backup {len(files)} file data_splits -> {dest}")


# -----------------------------------------------------------------------------
# 2. Un-stretch frames
# -----------------------------------------------------------------------------
def original_sizes():
    sizes = {}
    if ANNOT_MANIFEST.exists():
        a = read_csv(ANNOT_MANIFEST)
        for vid, g in a.groupby("video_stem"):
            sizes[vid] = (int(g["width"].iloc[0]), int(g["height"].iloc[0]))
    return sizes


def raw_video_size(video_name):
    stem = source_video_name(video_name)
    for d in RAW_VIDEO_DIRS:
        for p in d.glob(f"{stem}*.mp4"):
            cap = cv2.VideoCapture(str(p))
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w and h:
                return w, h
    return None


def target_size(orig_w, orig_h, long_side):
    if orig_w >= orig_h:
        return long_side, int(round(long_side * orig_h / orig_w))
    return int(round(long_side * orig_w / orig_h)), long_side


def unstretch_frames(manifest, ts, dry):
    sizes = original_sizes()
    archive = DATA / "archive" / f"stretched_frames_{ts}"
    fixed, skipped = [], []
    for vid, g in manifest.groupby("video_name"):
        first = cv2.imread(str(VIDEO_ROOT / g["image_path"].iloc[0]))
        h, w = first.shape[:2]
        if (w, h) != (640, 640):
            continue
        orig = sizes.get(vid) or raw_video_size(vid)
        if orig is None:
            skipped.append((vid, "không tìm thấy kích thước gốc"))
            continue
        if orig[0] == orig[1]:
            continue
        tw, th = target_size(orig[0], orig[1], 640)
        fixed.append((vid, f"{orig[0]}x{orig[1]}", f"{tw}x{th}", len(g)))
        if dry:
            continue
        shutil.copytree(VIDEO_ROOT / vid, archive / vid, dirs_exist_ok=True)
        for _, row in g.iterrows():
            ip, mp = VIDEO_ROOT / row["image_path"], VIDEO_ROOT / row["mask_path"]
            img = cv2.imread(str(ip))
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if img.shape[:2] != (640, 640) or mask.shape[:2] != (640, 640):
                raise RuntimeError(f"{ip}: kích thước không đồng nhất trong video {vid}")
            cv2.imwrite(str(ip), cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA))
            cv2.imwrite(str(mp), cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST))
    print(f"[2] Un-stretch: {len(fixed)} video (backup -> {archive})")
    for f in fixed:
        print(f"    {f[0]}: gốc {f[1]} -> {f[2]} ({f[3]} frame)")
    for s in skipped:
        print(f"    ⚠️ BỎ QUA {s[0]}: {s[1]}")
    return fixed, skipped


# -----------------------------------------------------------------------------
# 3. Source-level video leakage
# -----------------------------------------------------------------------------
def fix_source_leakage(manifest):
    vid_split = manifest.groupby("video_name")["split"].first()
    moves = {}
    groups = defaultdict(list)
    for v, s in vid_split.items():
        groups[source_video_name(v)].append((v, s))
    for src, items in groups.items():
        splits = {s for _, s in items}
        if len(splits) < 2:
            continue
        # keep the split holding most frames; ties -> train > val > test
        frames = Counter()
        for v, s in items:
            frames[s] += int((manifest["video_name"] == v).sum())
        keep = max(splits, key=lambda s: (frames[s], -SPLITS.index(s)))
        for v, s in items:
            if s != keep:
                moves[v] = (s, keep)
    for v, (old, new) in moves.items():
        manifest.loc[manifest["video_name"] == v, "split"] = new
    print(f"[3] Rò rỉ theo video gốc: chuyển {len(moves)} video")
    for v, (old, new) in moves.items():
        print(f"    {v}: {old} -> {new}")
    return moves


# -----------------------------------------------------------------------------
# 4. Recompute pixel statistics
# -----------------------------------------------------------------------------
def recompute_pixels(manifest):
    changed = 0
    for i, row in manifest.iterrows():
        m = cv2.imread(str(VIDEO_ROOT / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        fire, smoke = int((m == 1).sum()), int((m == 2).sum())
        hf, hs = int(fire > 0), int(smoke > 0)
        cat = "both" if hf and hs else ("fire_only" if hf else ("smoke_only" if hs else "empty"))
        new = {"fire_pixels": fire, "smoke_pixels": smoke, "has_fire": hf, "has_smoke": hs, "category": cat}
        if any(row[k] != v for k, v in new.items()):
            changed += 1
        for k, v in new.items():
            manifest.at[i, k] = v
    print(f"[4] Tính lại pixel: {changed} dòng manifest thay đổi")


# -----------------------------------------------------------------------------
# 6. Static re-split by group
# -----------------------------------------------------------------------------
def phash_bits(paths):
    import imagehash
    from PIL import Image
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(int(str(imagehash.phash(im.convert("RGB"))), 16))
    return np.unpackbits(np.array(out, dtype=np.uint64).view(np.uint8).reshape(-1, 8), axis=1)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def resplit_static(video_manifest, dry, video_root=VIDEO_ROOT, ratios=None):
    """ratios=None keeps the current split proportions."""
    parts = [read_csv(SPLITS_DIR / f"static_v4i_{s}_manifest.csv") for s in SPLITS]
    df = pd.concat(parts, ignore_index=True)
    old_split = df["split"].tolist()
    n = len(df)
    ratios = ratios or {s: (df["split"] == s).mean() for s in SPLITS}

    print(f"[6] Static: phash {n} ảnh tĩnh + {len(video_manifest)} frame video...")
    sb = phash_bits([STATIC_ROOT / f for f in df["filename"]])
    vb = phash_bits([Path(video_root) / p for p in video_manifest["image_path"]])

    uf = UnionFind(n)
    for i in range(n):
        d = (sb[i] != sb[i + 1:]).sum(axis=1)
        for o in np.nonzero(d <= STATIC_DUP_HAMMING)[0]:
            uf.union(i, i + 1 + int(o))
    by_src = defaultdict(list)
    for i, fn in enumerate(df["filename"]):
        by_src[roboflow_source_name(fn)].append(i)
    for idx in by_src.values():
        for j in idx[1:]:
            uf.union(idx[0], j)

    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    # Stage 1 must never train on frames of a video val/test video: a static group
    # that looks like such a frame goes to that split (test wins over val).
    # Matches with video-TRAIN frames are harmless and leave the group free.
    vsplit = video_manifest["split"].values
    forced = {}
    for i in range(n):
        d = (vb != sb[i]).sum(axis=1)
        hits = set(vsplit[d <= STATIC_VIDEO_HAMMING]) - {"train"}
        if hits:
            g = uf.find(i)
            forced[g] = forced.get(g, set()) | hits
    forced = {g: ("test" if "test" in s else "val") for g, s in forced.items()}

    cats = df["category"].tolist()
    target = {c: {s: ratios[s] * cnt for s in SPLITS} for c, cnt in Counter(cats).items()}
    count = {c: Counter() for c in target}
    assign = {}

    def add(g, s):
        assign[g] = s
        for i in groups[g]:
            count[cats[i]][s] += 1

    for g, s in forced.items():
        add(g, s)
    rng = random.Random(SEED)
    rest = [g for g in groups if g not in assign]
    rng.shuffle(rest)
    rest.sort(key=lambda g: -len(groups[g]))
    for g in rest:
        gc = Counter(cats[i] for i in groups[g])
        # split with the largest total relative deficit for this group's categories
        def deficit(s):
            return sum(k * (target[c][s] - count[c][s]) / max(target[c][s], 1e-9) for c, k in gc.items())
        add(g, max(SPLITS, key=deficit))

    df["split"] = [assign[uf.find(i)] for i in range(n)]
    moved = sum(a != b for a, b in zip(old_split, df["split"]))
    multi = sum(1 for g in groups.values() if len(g) > 1)
    print(f"    {len(groups)} nhóm ({multi} nhóm >1 ảnh), {len(forced)} nhóm ép theo video; {moved}/{n} ảnh đổi split")
    print("    phân bố mới:\n" + df.groupby(["split", "category"]).size().unstack(fill_value=0).to_string())

    for s in SPLITS:
        write_csv(df[df["split"] == s], SPLITS_DIR / f"static_v4i_{s}_manifest.csv", dry)
    with_splits = SPLITS_DIR / "static_v4i_manifest_with_splits.csv"
    if with_splits.exists():
        w = read_csv(with_splits)
        new_split = dict(zip(df["filename"], df["split"]))
        if "split" in w.columns and "filename" in w.columns:
            w["split"] = w["filename"].map(new_split).fillna(w["split"])
            write_csv(w, with_splits, dry)


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(("DRY RUN — " if dry else "") + f"fix_data {ts}")

    backup_splits(ts, dry)
    manifest = read_csv(VIDEO_MANIFEST)
    unstretch_frames(manifest, ts, dry)
    fix_source_leakage(manifest)
    if not dry:
        recompute_pixels(manifest)
    write_csv(manifest, VIDEO_MANIFEST, dry)

    if not dry:
        import create_split_manifests_and_clips as cs
        print("[5] Sinh lại manifest theo split + clips (có clip neo cuối video)...")
        cs.main()

    resplit_static(manifest, dry)
    print("✅ Xong. Chạy lại: python scripts/audit_data.py")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
