"""
scripts/build_dense_clips.py
============================
Builds the DENSE temporal dataset for Stage 2: every labeled frame of
data/dataset_video_prepared becomes the LAST frame of a T-frame clip whose
first T-1 frames are context frames sampled at `--fps` (default 10) right
before it, decoded from the original video.

- Context frames have no label (mask path left empty -> IGNORE_INDEX in the
  dataset), unless that exact raw frame is itself a labeled frame.
- The labeled frame is located in the source video by thumbnail matching
  (+ a robust linear time fit), then verified by PSNR at full resolution.
- All frames of a video are stored at the same size (long side <= --max-side)
  as JPEG; masks are resized with nearest neighbour.

Outputs:
  data/dataset_video_dense/<video>/raw_<idx>.jpg, raw_<idx>_mask.png
  data/data_splits/dense_{train,val,test}_clips.csv
  data/data_splits/dense_build_report.json

Usage:
  python scripts/build_dense_clips.py --videos 3     # quick check on a few videos
  python scripts/build_dense_clips.py                # all videos
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PREPARED = DATA / "dataset_video_prepared"
SPLITS_DIR = DATA / "data_splits"
MANIFEST = SPLITS_DIR / "prepared_video_manifest_with_splits.csv"
THUMB = (24, 24)
PSNR_OK = 24.0


# -----------------------------------------------------------------------------
# Locate the source video of every prepared video
# -----------------------------------------------------------------------------
def source_stem(video_name):
    s = re.sub(r"_chunk_\d+$", "", video_name)
    return re.sub(r"_pure$", "", s)


def build_index():
    idx = defaultdict(list)
    for p in glob.glob(str(DATA / "**" / "*.mp4"), recursive=True):
        if "dataset_video_" in p:
            continue
        idx[Path(p).stem.lower()].append(p)
    return idx


def locate(video_name, idx):
    s = source_stem(video_name).lower()
    if s in idx:
        return idx[s][0]
    for prefix in ("neg_hard_negatives_", "neg_normal_"):
        if s.startswith(prefix):
            core = s[len("neg_"):]  # 'hard_negatives_<id>' / 'normal_<id>_<title...>'
            hits = [p for k, ps in idx.items() if k.startswith(core[:40]) or core.startswith(k[:40]) for p in ps]
            if hits:
                return hits[0]
    hits = [p for k, ps in idx.items() if s in k for p in ps]
    return hits[0] if hits else None


# -----------------------------------------------------------------------------
# Matching
# -----------------------------------------------------------------------------
def thumb(img):
    return cv2.resize(img, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)


def decode_thumbs(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    thumbs = []
    size = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        size = fr.shape[1::-1]
        thumbs.append(thumb(fr))
    cap.release()
    return np.stack(thumbs), float(fps), size


def robust_line(i, j, tol=3):
    """Fit j = a + b*i by RANSAC over pairs. Returns (a, b, inlier_mask) or None."""
    best = None
    n = len(i)
    if n < 3:
        return None
    for p in range(n):
        for q in range(p + 1, n):
            if i[q] == i[p]:
                continue
            b = (j[q] - j[p]) / (i[q] - i[p])
            if b <= 0:
                continue
            a = j[p] - b * i[p]
            inl = np.abs(a + b * i - j) <= tol
            if best is None or inl.sum() > best[2].sum():
                best = (a, b, inl)
    if best is None:
        return None
    a, b, inl = best
    A = np.vstack([i[inl], np.ones(inl.sum())]).T
    b, a = np.linalg.lstsq(A, j[inl], rcond=None)[0]
    return a, b, inl


def match_frames(prep_thumbs, raw_thumbs):
    """Index of the raw frame for each prepared frame (in order)."""
    d = ((prep_thumbs[:, None] - raw_thumbs[None]) ** 2).mean(axis=(2, 3, 4))  # (n_prep, n_raw)
    arg = d.argmin(axis=1)
    i = np.arange(len(arg), dtype=np.float64)
    fit = robust_line(i, arg.astype(np.float64))
    chosen = arg.copy()
    method = "argmin"
    if fit is not None and fit[2].mean() >= 0.6:
        a, b, _ = fit
        pred = np.clip(np.round(a + b * i).astype(int), 0, raw_thumbs.shape[0] - 1)
        # keep the fitted position when it looks (almost) as good as the best match
        good = d[np.arange(len(arg)), pred] <= 1.5 * d[np.arange(len(arg)), arg] + 1.0
        chosen = np.where(good, pred, arg)
        method = f"line(b={b:.2f}, inliers={fit[2].mean():.0%})"
    return chosen, method, d


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def target_size(w, h, max_side):
    s = min(1.0, max_side / max(w, h))
    return int(round(w * s)), int(round(h * s))


# -----------------------------------------------------------------------------
def process_video(vid, rows, src, out_root, fps_ctx, T, max_side, annot_orig=None):
    rows = rows.sort_values("frame_index").reset_index(drop=True)
    prep_imgs = [cv2.imread(str(PREPARED / p)) for p in rows["image_path"]]
    raw_thumbs, fps, (W, H) = decode_thumbs(src)
    chosen, method, _ = match_frames(np.stack([thumb(x) for x in prep_imgs]), raw_thumbs)
    annot_agree = None
    if annot_orig is not None:
        # round-2 videos: the extraction manifest records the exact source frame
        truth = np.array([annot_orig.get(int(fi), -1) for fi in rows["frame_index"]])
        if (truth >= 0).all():
            annot_agree = float(np.mean(np.abs(truth - chosen) <= 1))
            chosen, method = truth, f"annotation_manifest (thumbnail match agrees {annot_agree:.0%})"
    step = max(1, int(round(fps / fps_ctx)))
    tw, th = target_size(W, H, max_side)

    # raw indices to save: anchors + context
    anchor_of = {int(j): k for k, j in enumerate(chosen)}
    need = set()
    clips_idx = []
    for k, j in enumerate(chosen):
        seq = [max(0, int(j) - step * (T - 1 - t)) for t in range(T)]
        clips_idx.append(seq)
        need.update(seq)

    vdir = out_root / vid
    vdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(src)
    f = 0
    psnrs = []
    while need and f <= max(need):
        ok, fr = cap.read()
        if not ok:
            break
        if f in need:
            img = cv2.resize(fr, (tw, th), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(vdir / f"raw_{f:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if f in anchor_of:
                k = anchor_of[f]
                prep = cv2.resize(prep_imgs[k], (tw, th), interpolation=cv2.INTER_AREA)
                psnrs.append(psnr(prep, img))
                mask = cv2.imread(str(PREPARED / rows.loc[k, "mask_path"]), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(vdir / f"raw_{f:06d}_mask.png"), mask)
        f += 1
    cap.release()

    records = []
    for k, seq in enumerate(clips_idx):
        r = rows.loc[k]
        names, imgs, masks = [], [], []
        for t, j in enumerate(seq):
            # a labeled frame keeps its own name wherever it appears (unique key for evaluation)
            names.append(rows.loc[anchor_of[j], "frame_name"] if j in anchor_of else f"ctx_raw_{j:06d}")
            imgs.append(f"{vid}/raw_{j:06d}.jpg")
            masks.append(f"{vid}/raw_{j:06d}_mask.png" if j in anchor_of else "")
        records.append({
            "video_name": vid, "anchor_frame": r["frame_name"], "anchor_raw_idx": int(seq[-1]),
            "start_idx": int(seq[0]), "end_idx": int(seq[-1]), "clip_length": T,
            "frame_names": ";".join(names), "image_paths": ";".join(imgs), "mask_paths": ";".join(masks),
            "has_fire": int(r["has_fire"]), "has_smoke": int(r["has_smoke"]), "category": r["category"],
            "split": r["split"], "is_padded": seq[0] == seq[1],
        })
    report = {
        "source": os.path.relpath(src, ROOT), "fps": fps, "ctx_step": step, "size": [tw, th],
        "anchors": len(rows), "method": method, "raw_frames": int(raw_thumbs.shape[0]),
        "psnr_median": float(np.median(psnrs)) if psnrs else None,
        "psnr_min": float(np.min(psnrs)) if psnrs else None,
        "monotonic": bool(np.all(np.diff(chosen) > 0)),
        "pre_clahe_stretched_source": annot_orig is not None,
        "annotation_agreement": annot_agree,
        "duplicate_anchor_raw_idx": int(len(chosen) - len(set(chosen.tolist()))),
    }
    return records, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=960)
    ap.add_argument("--videos", type=int, default=0, help="only the first N videos (debug)")
    ap.add_argument("--out", default=str(DATA / "dataset_video_dense"))
    args = ap.parse_args()

    manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    idx = build_index()
    annot = pd.read_csv(DATA / "frames_for_annotation" / "manifest.csv", encoding="utf-8-sig")
    annot_map = {v: dict(zip(g["frame_idx"].astype(int), g["frame_orig"].astype(int)))
                 for v, g in annot.groupby("video_stem")}
    vids = sorted(manifest["video_name"].unique())
    if args.videos:
        vids = vids[: args.videos]
    out_root = Path(args.out)
    all_records, reports = [], {}
    for n, vid in enumerate(vids, 1):
        src = locate(vid, idx)
        if src is None:
            reports[vid] = {"error": "source video not found"}
            print(f"[{n}/{len(vids)}] ❌ {vid}: không tìm thấy video gốc", flush=True)
            continue
        recs, rep = process_video(vid, manifest[manifest["video_name"] == vid], src, out_root,
                                  args.fps, args.T, args.max_side, annot_map.get(vid))
        all_records += recs
        reports[vid] = rep
        ok = rep["monotonic"] and ((rep["psnr_median"] or 0) >= PSNR_OK or (rep["annotation_agreement"] or 0) >= 0.9)
        flag = "" if ok else "  ⚠️"
        print(f"[{n}/{len(vids)}] {vid[:45]:45s} anchors={rep['anchors']:3d} fps={rep['fps']:.1f} step={rep['ctx_step']} "
              f"PSNR med={rep['psnr_median']:.1f} min={rep['psnr_min']:.1f} {rep['method']} mono={rep['monotonic']}{flag}",
              flush=True)

    df = pd.DataFrame(all_records)
    if not args.videos:
        for split in ("train", "val", "test"):
            df[df["split"] == split].drop(columns=["split"]).to_csv(
                SPLITS_DIR / f"dense_{split}_clips.csv", index=False, encoding="utf-8-sig")
        (SPLITS_DIR / "dense_build_report.json").write_text(
            json.dumps({"args": vars(args), "videos": reports}, indent=2, ensure_ascii=False), encoding="utf-8")
        print({s: int((df["split"] == s).sum()) for s in ("train", "val", "test")}, "clips")
    bad = [v for v, r in reports.items() if "error" in r or not r["monotonic"]
           or ((r["psnr_median"] or 0) < PSNR_OK and (r["annotation_agreement"] or 0) < 0.9)]
    print(f"⚠️ cần xem lại: {len(bad)} video {bad[:20]}")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
