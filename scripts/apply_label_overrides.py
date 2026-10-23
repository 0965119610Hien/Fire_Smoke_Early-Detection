"""Apply manual label decisions that are not (yet) in the main Roboflow exports.

    python scripts/apply_label_overrides.py            # apply
    python scripts/apply_label_overrides.py --dry-run  # only report

Reads data/label_overrides.csv (mask_path, action, source, rule, note):
  action = empty      -> mask set to all background (0); fire pixels are refused (error).
  action = empty_all  -> same, but fire pixels are dropped too (e.g. welding arc is not a flame).
  action = replace    -> mask copied from `source` (a relabel export mask); same size, values {0,1,2}.
Original masks are backed up once to data/label_overrides_backup/<same relative path>.
Manifests under data/data_splits (top level + folds/*) get has_fire / has_smoke / pixel
counts / ratios / category recomputed from the masks for the affected frames / images.

Re-run after every `build_unified_dataset.py` full build or Roboflow re-export, since
those regenerate the masks from the exports.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OVERRIDES = DATA / "label_overrides.csv"
BACKUP = DATA / "label_overrides_backup"
SPLITS_DIR = DATA / "data_splits"
VIDEO_ROOT = DATA / "dataset_video_unified"


def category(has_fire, has_smoke):
    return "both" if has_fire and has_smoke else ("fire_only" if has_fire else ("smoke_only" if has_smoke else "empty"))


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    return mask


def target_mask(rel, action, source):
    """The mask this override should produce."""
    mask = read_mask(ROOT / rel)
    if action == "empty":
        if (mask == 1).any():
            raise ValueError(f"{rel} has fire pixels; 'empty' would drop them (use empty_all)")
        return np.zeros_like(mask)
    if action == "empty_all":
        return np.zeros_like(mask)
    if action == "replace":
        new = read_mask(ROOT / source)
        if new.shape != mask.shape:
            raise ValueError(f"{source} shape {new.shape} != {rel} shape {mask.shape}")
        if not set(np.unique(new).tolist()) <= {0, 1, 2}:
            raise ValueError(f"{source} has values outside {{0,1,2}}")
        return new.astype(mask.dtype)
    raise ValueError(f"Unknown action {action!r} for {rel}")


def apply_mask(rel, action, source, dry):
    path = ROOT / rel
    new = target_mask(rel, action, source)
    changed = not np.array_equal(read_mask(path), new)
    if changed and not dry:
        bak = BACKUP / rel
        if not bak.exists():
            bak.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, bak)
        cv2.imwrite(str(path), new)
    return changed, new


def refresh_manifests(video_stats, static_stats, dry):
    """Recompute the label columns of rows whose mask was overridden.

    video_stats: {"<video>/<frame>_mask.png": (fire_px, smoke_px, total_px)}, static_stats keyed by mask file name.
    """
    files = list(SPLITS_DIR.glob("*.csv")) + list((SPLITS_DIR / "folds").glob("fold*/*.csv"))
    touched = {}
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        if "mask_path" in df.columns and "has_smoke" in df.columns:
            keys, stats = df["mask_path"], video_stats
        elif "mask_paths" in df.columns and "anchor_frame" in df.columns:
            keys, stats = df["video_name"] + "/" + df["anchor_frame"] + "_mask.png", video_stats
        elif "mask_filename" in df.columns:
            keys, stats = df["mask_filename"], static_stats
        else:
            continue
        hit = keys.isin(stats)
        if not hit.any():
            continue
        for i in df.index[hit]:
            fire, smoke, total = stats[keys[i]]
            vals = {"has_fire": int(fire > 0), "has_smoke": int(smoke > 0),
                    "fire_pixels": fire, "smoke_pixels": smoke,
                    "fire_ratio": fire / total * 100, "smoke_ratio": smoke / total * 100,
                    "category": category(fire > 0, smoke > 0)}
            for col, v in vals.items():
                if col in df.columns:
                    df.at[i, col] = v
        touched[str(f.relative_to(ROOT))] = int(hit.sum())
        if not dry:
            df.to_csv(f, index=False, encoding="utf-8-sig")
    return touched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ov = pd.read_csv(OVERRIDES, encoding="utf-8-sig")
    if "source" not in ov.columns:
        ov["source"] = ""
    ov["source"] = ov["source"].fillna("")
    video_stats, static_stats, n_changed = {}, {}, 0
    for r in ov.itertuples():
        rel = r.mask_path.replace("\\", "/")
        changed, mask = apply_mask(rel, r.action, r.source, args.dry_run)
        n_changed += changed
        stat = (int((mask == 1).sum()), int((mask == 2).sum()), int(mask.size))
        p = Path(rel)
        if VIDEO_ROOT in (ROOT / p).parents:
            video_stats[f"{p.parent.name}/{p.name}"] = stat
        else:
            static_stats[p.name] = stat
    touched = refresh_manifests(video_stats, static_stats, args.dry_run)

    tag = "[dry-run] " if args.dry_run else ""
    print(f"{tag}{len(ov)} overrides {ov['action'].value_counts().to_dict()}, {n_changed} masks changed "
          f"({len(ov) - n_changed} already matched)")
    for f, n in touched.items():
        print(f"{tag}  {f}: {n} rows")


if __name__ == "__main__":
    main()
