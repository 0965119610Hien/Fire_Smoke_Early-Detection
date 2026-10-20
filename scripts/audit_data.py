"""
scripts/audit_data.py
=====================
Data audit before (re)training. Writes docs/reports/data_audit_<date>.json and
prints a summary. Checks:

1. Video leakage: video ids of train/val/test (frame manifests + clip files) are disjoint.
2. Static leakage: perceptual hash (phash) of every static image; cross-split pairs
   with Hamming distance <= 4 (+ identical Roboflow source names across splits).
3. Mask values: every static + video mask only contains {0, 1, 2}.
4. Frame sizes in dataset_video_prepared (min / max / distribution).
5. Pixel statistics per video split, recomputed from the masks, incl. the
   BG_pixels >= empty_frames x H x W sanity constraint and manifest consistency.
6. Clips: clips / frames per video and inferred stride.
7. Illustration images in docs/inference_comparisons*/: split of each video.

Local (CPU):
  python scripts/audit_data.py
Modal (reads the volume, writes the report locally):
  modal run scripts/audit_data.py
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")


def _read_csv(path):
    import pandas as pd
    return pd.read_csv(path, encoding="utf-8-sig")


# -----------------------------------------------------------------------------
# 1. Video leakage
# -----------------------------------------------------------------------------
def check_video_leakage(splits_dir):
    ids = {}
    for split in SPLITS:
        s = set()
        for name in (f"{split}_manifest.csv", f"{split}_clips.csv"):
            p = splits_dir / name
            if p.exists():
                s |= set(_read_csv(p)["video_name"].astype(str))
        ids[split] = s
    overlaps = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            inter = sorted(ids[a] & ids[b])
            if inter:
                overlaps[f"{a}&{b}"] = inter

    # Chunks of the same source video (e.g. '<name>_chunk_02' / '<name>_chunk_03') in different splits
    source_splits = defaultdict(lambda: defaultdict(set))
    for split, vids in ids.items():
        for v in vids:
            source_splits[source_video_name(v)][split].add(v)
    source_overlaps = {src: {s: sorted(v) for s, v in d.items()}
                       for src, d in source_splits.items() if len(d) > 1}
    return {"num_videos": {k: len(v) for k, v in ids.items()}, "overlaps": overlaps,
            "source_video_overlaps": source_overlaps, "ok": not overlaps and not source_overlaps}


def source_video_name(video_name):
    """Strip chunk / suffix markers so chunks of one recording map to one source."""
    s = re.sub(r"_chunk_\d+$", "", video_name)
    return re.sub(r"_pure$", "", s)


# -----------------------------------------------------------------------------
# 2. Static leakage
# -----------------------------------------------------------------------------
def roboflow_source_name(filename):
    """'000123_some_name_jpg.rf.<hash>.jpg' -> 'some_name_jpg' (augmented copies share it)."""
    stem = re.sub(r"\.rf\.[0-9a-f]+.*$", "", filename)
    return re.sub(r"^\d{6}_", "", stem)


def check_static_leakage(splits_dir, static_root, max_hamming=4):
    import numpy as np
    import imagehash
    from PIL import Image

    records = []
    for split in SPLITS:
        df = _read_csv(splits_dir / f"static_v4i_{split}_manifest.csv")
        for _, row in df.iterrows():
            records.append((split, row["filename"], row.get("category", "")))

    hashes, missing = [], []
    for split, fn, cat in records:
        p = static_root / fn
        try:
            with Image.open(p) as im:
                h = imagehash.phash(im.convert("RGB"))
            hashes.append(int(str(h), 16))
        except Exception:
            missing.append(fn)
            hashes.append(None)

    keep = [i for i, h in enumerate(hashes) if h is not None]
    hv = np.array([hashes[i] for i in keep], dtype=np.uint64)
    split_arr = np.array([records[i][0] for i in keep])
    bits = np.unpackbits(hv.view(np.uint8).reshape(-1, 8), axis=1)  # (N, 64)

    pairs = []
    for a in range(len(keep)):
        d = (bits[a] != bits[a + 1:]).sum(axis=1)
        for off in np.nonzero(d <= max_hamming)[0]:
            b = a + 1 + int(off)
            if split_arr[a] != split_arr[b]:
                ra, rb = records[keep[a]], records[keep[b]]
                pairs.append({"a": ra[1], "split_a": ra[0], "b": rb[1], "split_b": rb[0], "hamming": int(d[off])})

    by_pair = Counter(f"{p['split_a']}&{p['split_b']}" for p in pairs)

    # Same Roboflow source image in different splits
    src_splits = defaultdict(set)
    for split, fn, _ in records:
        src_splits[roboflow_source_name(fn)].add(split)
    shared_sources = {k: sorted(v) for k, v in src_splits.items() if len(v) > 1}

    # val vs test category counts (reported identical in docs)
    cat_counts = {}
    for split in SPLITS:
        df = _read_csv(splits_dir / f"static_v4i_{split}_manifest.csv")
        cat_counts[split] = df["category"].value_counts().to_dict()

    return {
        "num_images": len(records),
        "unreadable": missing,
        "max_hamming": max_hamming,
        "cross_split_near_duplicates": len(pairs),
        "near_duplicates_by_split_pair": dict(by_pair),
        "pairs": pairs,
        "shared_roboflow_sources": len(shared_sources),
        "shared_roboflow_source_examples": dict(list(shared_sources.items())[:50]),
        "category_counts": cat_counts,
        "ok": not pairs and not shared_sources,
    }


# -----------------------------------------------------------------------------
# 3 + 4 + 5. Masks, frame sizes, pixel statistics
# -----------------------------------------------------------------------------
def check_masks_and_frames(splits_dir, static_root, video_root):
    import cv2
    import numpy as np

    bad_masks, unreadable = [], []

    def mask_values(p):
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            unreadable.append(str(p))
            return None
        if m.ndim == 3:
            bad_masks.append({"path": str(p), "issue": f"{m.shape[2]} channels"})
            m = m[..., 0]
        vals = set(np.unique(m).tolist())
        if not vals <= {0, 1, 2}:
            bad_masks.append({"path": str(p), "values": sorted(vals)[:10]})
        return m

    # static masks
    n_static = 0
    for split in SPLITS:
        for fn in _read_csv(splits_dir / f"static_v4i_{split}_manifest.csv")["mask_filename"]:
            mask_values(static_root / fn)
            n_static += 1

    # video masks + frame sizes + pixel stats
    manifest = _read_csv(splits_dir / "prepared_video_manifest_with_splits.csv")
    sizes = Counter()
    size_mismatch = []
    stats = {}
    manifest_mismatch = []
    for split in SPLITS:
        df = manifest[manifest["split"] == split]
        acc = {"frames": 0, "empty_frames": 0, "total_pixels": 0, "bg_pixels": 0, "fire_pixels": 0,
               "smoke_pixels": 0, "empty_frame_pixels": 0, "videos": int(df["video_name"].nunique())}
        for _, row in df.iterrows():
            img = cv2.imread(str(video_root / row["image_path"]))
            m = mask_values(video_root / row["mask_path"])
            if img is None:
                unreadable.append(str(video_root / row["image_path"]))
                continue
            h, w = img.shape[:2]
            sizes[f"{w}x{h}"] += 1
            if m is None:
                continue
            if m.shape[:2] != (h, w):
                size_mismatch.append(row["mask_path"])
            bg, fire, smoke = int((m == 0).sum()), int((m == 1).sum()), int((m == 2).sum())
            acc["frames"] += 1
            acc["total_pixels"] += m.size
            acc["bg_pixels"] += bg
            acc["fire_pixels"] += fire
            acc["smoke_pixels"] += smoke
            if fire == 0 and smoke == 0:
                acc["empty_frames"] += 1
                acc["empty_frame_pixels"] += m.size
            if fire != int(row["fire_pixels"]) or smoke != int(row["smoke_pixels"]):
                manifest_mismatch.append({"mask": row["mask_path"], "manifest": [int(row["fire_pixels"]), int(row["smoke_pixels"])],
                                          "actual": [fire, smoke]})
        tp = max(acc["total_pixels"], 1)
        acc.update({
            "bg_pct": 100 * acc["bg_pixels"] / tp, "fire_pct": 100 * acc["fire_pixels"] / tp,
            "smoke_pct": 100 * acc["smoke_pixels"] / tp,
            "empty_frame_pct": 100 * acc["empty_frames"] / max(acc["frames"], 1),
            "constraint_bg_ge_empty_pixels": acc["bg_pixels"] >= acc["empty_frame_pixels"],
            "min_bg_pct_implied_by_empty_frames": 100 * acc["empty_frame_pixels"] / tp,
        })
        stats[split] = acc

    areas = sorted(((int(k.split("x")[0]) * int(k.split("x")[1]), k) for k in sizes))
    return {
        "masks": {"static_checked": n_static, "video_checked": int(len(manifest)), "bad": bad_masks,
                  "unreadable": unreadable, "mask_image_size_mismatch": size_mismatch, "ok": not bad_masks and not unreadable},
        "frame_sizes": {"distribution": dict(sizes.most_common()),
                        "min": areas[0][1] if areas else None, "max": areas[-1][1] if areas else None,
                        "all_512x512": set(sizes) == {"512x512"}},
        "pixel_stats": stats,
        "manifest_pixel_mismatch": {"count": len(manifest_mismatch), "examples": manifest_mismatch[:20]},
    }


# -----------------------------------------------------------------------------
# 6. Clips
# -----------------------------------------------------------------------------
def check_clips(splits_dir):
    out = {}
    manifest = _read_csv(splits_dir / "prepared_video_manifest_with_splits.csv")
    frames_per_video = manifest.groupby("video_name").size().to_dict()
    for split in SPLITS:
        df = _read_csv(splits_dir / f"{split}_clips.csv")
        split_frames = manifest[manifest["split"] == split]
        covered = set()
        for _, row in df.iterrows():
            covered |= {(row["video_name"], n) for n in str(row["frame_names"]).split(";")}
        all_frames = set(zip(split_frames["video_name"], split_frames["frame_name"]))
        uncovered = sorted(all_frames - covered)
        videos_without_clips = sorted(set(split_frames["video_name"]) - set(df["video_name"]))
        clip_frames_not_in_manifest = sorted(covered - all_frames)
        per_video = []
        strides = Counter()
        for vid, g in df.groupby("video_name"):
            starts = sorted(g["start_idx"].astype(int))
            for a, b in zip(starts, starts[1:]):
                strides[b - a] += 1
            per_video.append({"video": vid, "frames": int(frames_per_video.get(vid, 0)), "clips": len(g),
                              "padded": bool(g["is_padded"].astype(str).str.lower().eq("true").any())})
        T = int(df["clip_length"].iloc[0]) if len(df) else 0
        stride = strides.most_common(1)[0][0] if strides else None
        explanation = None
        if stride:
            explanation = (f"{len(df)} clips from {df['video_name'].nunique()} videos: each video with n>=T frames "
                           f"gives floor((n-{T})/{stride})+1 clips (stride {stride}), +1 tail-anchored clip when "
                           f"(n-{T}) % {stride} != 0; videos with n<{T} give 1 padded clip.")
        out[split] = {"clips": len(df), "videos": int(df["video_name"].nunique()), "T": T,
                      "inferred_stride": stride, "stride_histogram": dict(strides),
                      "explanation": explanation,
                      "manifest_videos_without_clips": videos_without_clips,
                      "manifest_frames_not_in_any_clip": len(uncovered),
                      "uncovered_examples": uncovered[:20],
                      "clip_frames_not_in_manifest": len(clip_frames_not_in_manifest),
                      "in_sync_with_manifest": not videos_without_clips and not clip_frames_not_in_manifest,
                      "per_video": per_video}
    return out


# -----------------------------------------------------------------------------
# 7. Illustration images
# -----------------------------------------------------------------------------
ILLUSTRATION_PATTERN = re.compile(r"^(?:comparison_clip|sample_)\d+_(?P<rest>.+)_f\d+\.\w+$")


def match_truncated_video(rest, video_names):
    """File names hold '<optional category>_<truncated video name>'. Return the
    longest '_'-aligned suffix of `rest` that prefixes a known video, and its videos."""
    candidates = [0] + [i + 1 for i, ch in enumerate(rest) if ch == "_"]
    for start in candidates:  # earliest start = longest suffix
        prefix = rest[start:]
        vids = [v for v in video_names if v.startswith(prefix)]
        if vids:
            return prefix, vids
    return None, []


def check_illustrations(splits_dir, docs_dir):
    manifest = _read_csv(splits_dir / "prepared_video_manifest_with_splits.csv")
    vid_split = manifest.groupby("video_name")["split"].first().to_dict()
    rows = []
    for d in sorted(docs_dir.glob("inference_comparisons*")):
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            m = ILLUSTRATION_PATTERN.match(f.name)
            prefix, vids = match_truncated_video(m.group("rest"), vid_split) if m else (None, [])
            matches = sorted({vid_split[v] for v in vids})
            rows.append({"file": f"{d.name}/{f.name}", "video_prefix": prefix, "videos": vids, "splits": matches})
    count = Counter(",".join(r["splits"]) or "unknown" for r in rows)
    return {"files": rows, "split_counts": dict(count),
            "non_test": [r["file"] for r in rows if r["splits"] != ["test"]]}


# -----------------------------------------------------------------------------
def run_audit(splits_dir, static_root, video_root, docs_dir=None, skip_phash=False):
    splits_dir, static_root, video_root = Path(splits_dir), Path(static_root), Path(video_root)
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "paths": {"splits_dir": str(splits_dir), "static_root": str(static_root), "video_root": str(video_root)}}
    print("[1/7] Video leakage..."); report["video_leakage"] = check_video_leakage(splits_dir)
    if skip_phash:
        report["static_leakage"] = {"skipped": True}
    else:
        print("[2/7] Static leakage (phash)..."); report["static_leakage"] = check_static_leakage(splits_dir, static_root)
    print("[3-5/7] Masks, frame sizes, pixel stats..."); report.update(check_masks_and_frames(splits_dir, static_root, video_root))
    print("[6/7] Clips..."); report["clips"] = check_clips(splits_dir)
    if docs_dir is not None and Path(docs_dir).exists():
        print("[7/7] Illustration images..."); report["illustrations"] = check_illustrations(splits_dir, Path(docs_dir))
    return report


def print_summary(r):
    print("\n" + "=" * 70)
    print("  DATA AUDIT SUMMARY")
    print("=" * 70)
    vl = r["video_leakage"]
    print(f"1. Video leakage: {'OK' if vl['ok'] else 'LEAK'} | videos {vl['num_videos']} | "
          f"same id: {vl['overlaps'] or 'none'} | same source (chunks): {vl['source_video_overlaps'] or 'none'}")
    sl = r["static_leakage"]
    if not sl.get("skipped"):
        print(f"2. Static leakage: {sl['cross_split_near_duplicates']} cross-split pairs (hamming<={sl['max_hamming']}) "
              f"{sl['near_duplicates_by_split_pair']} | shared Roboflow sources: {sl['shared_roboflow_sources']}")
        print(f"   category counts: {sl['category_counts']}")
    mk = r["masks"]
    print(f"3. Masks: {'OK' if mk['ok'] else 'PROBLEM'} | bad={len(mk['bad'])} unreadable={len(mk['unreadable'])} "
          f"size-mismatch={len(mk['mask_image_size_mismatch'])}")
    fs = r["frame_sizes"]
    print(f"4. Frame sizes: min={fs['min']} max={fs['max']} all_512x512={fs['all_512x512']} | top: "
          f"{dict(list(fs['distribution'].items())[:6])}")
    print("5. Pixel stats (video):")
    for s, a in r["pixel_stats"].items():
        print(f"   {s:5s}: {a['videos']} videos, {a['frames']} frames, empty {a['empty_frames']} ({a['empty_frame_pct']:.1f}%) | "
              f"BG {a['bg_pct']:.2f}% Fire {a['fire_pct']:.2f}% Smoke {a['smoke_pct']:.2f}% | "
              f"BG>=empty-frame pixels: {a['constraint_bg_ge_empty_pixels']} (min BG implied {a['min_bg_pct_implied_by_empty_frames']:.2f}%)")
    print(f"   manifest pixel-count mismatches: {r['manifest_pixel_mismatch']['count']}")
    print("6. Clips:")
    for s, c in r["clips"].items():
        print(f"   {s:5s}: {c['explanation']}")
        if not c["in_sync_with_manifest"]:
            print(f"          ⚠️ clips lệch manifest: videos không có clip = {c['manifest_videos_without_clips']}, "
                  f"frame của clip không có trong manifest = {c['clip_frames_not_in_manifest']}")
        print(f"          frame manifest không nằm trong clip nào: {c['manifest_frames_not_in_any_clip']}")
    if "illustrations" in r:
        il = r["illustrations"]
        print(f"7. Illustrations: {il['split_counts']} | non-test files: {len(il['non_test'])}")
    print("=" * 70)


def write_report(report, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"data_audit_{datetime.now().strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"💾 Report: {out}")
    return out


# -----------------------------------------------------------------------------
# Modal (optional)
# -----------------------------------------------------------------------------
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("fire-smoke-data-audit")
    volume = modal.Volume.from_name("fire_smoke_dataset_v2")
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "opencv-python-headless", "numpy<2.0.0", "pandas", "Pillow", "ImageHash")

    @app.function(image=image, volumes={"/data": volume}, timeout=3600, memory=8192)
    def audit_remote(skip_phash: bool = False):
        return run_audit("/data/data_splits", "/data/Fire_Smoke.v4i/train", "/data/dataset_video_prepared",
                         docs_dir=None, skip_phash=skip_phash)

    @app.local_entrypoint()
    def main(skip_phash: bool = False):
        report = audit_remote.remote(skip_phash=skip_phash)
        # illustration images only exist locally
        report["illustrations"] = check_illustrations(ROOT / "data" / "data_splits", ROOT / "docs")
        print_summary(report)
        write_report(report, ROOT / "docs" / "reports")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", default=str(ROOT / "data" / "data_splits"))
    ap.add_argument("--static-root", default=str(ROOT / "data" / "Fire_Smoke.v4i.png-mask-semantic" / "train"))
    ap.add_argument("--video-root", default=str(ROOT / "data" / "dataset_video_prepared"))
    ap.add_argument("--docs-dir", default=str(ROOT / "docs"))
    ap.add_argument("--out-dir", default=str(ROOT / "docs" / "reports"))
    ap.add_argument("--skip-phash", action="store_true")
    args = ap.parse_args()
    rep = run_audit(args.splits_dir, args.static_root, args.video_root, args.docs_dir, args.skip_phash)
    print_summary(rep)
    write_report(rep, args.out_dir)
