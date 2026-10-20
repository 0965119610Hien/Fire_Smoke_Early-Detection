"""
scripts/sync_fixed_data_to_modal.py
===================================
Uploads the data fixed by scripts/fix_data.py to the Modal volume
`fire_smoke_dataset_v2` (overwriting the old versions):

- data/data_splits/*.csv, *.json          -> /data_splits/
- data/dataset_video_prepared/<video>/    -> /dataset_video_prepared/<video>/
  for every video whose frames were un-stretched (listed in the newest
  data/archive/stretched_frames_* backup).

Static images (/Fire_Smoke.v4i/train) are unchanged: only their manifests moved.

Usage (needs `pip install modal` + `modal token new`):
  python scripts/sync_fixed_data_to_modal.py --dry-run
  python scripts/sync_fixed_data_to_modal.py
Then verify on the volume:
  modal run scripts/audit_data.py
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VOLUME_NAME = "fire_smoke_dataset_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dense", action="store_true",
                    help="upload data/dataset_video_dense + dense_*_clips.csv (scripts/build_dense_clips.py)")
    ap.add_argument("--unified", action="store_true",
                    help="upload data/dataset_video_unified + every data_splits csv/json (scripts/build_unified_dataset.py)")
    args = ap.parse_args()

    if args.unified:
        unified = DATA / "dataset_video_unified"
        files = sorted(p for p in (DATA / "data_splits").iterdir() if p.is_file() and p.suffix in (".csv", ".json"))
        n = sum(1 for _ in unified.rglob("*") if _.is_file())
        folds = DATA / "data_splits" / "folds"
        n_folds = sum(1 for _ in folds.rglob("*.csv")) if folds.exists() else 0
        print(f"Volume: {VOLUME_NAME}\n- {n} file -> /dataset_video_unified/\n- {len(files)} file -> /data_splits/"
              f"\n- {n_folds} file -> /data_splits/folds/")
        if args.dry_run:
            return
        import modal
        vol = modal.Volume.from_name(VOLUME_NAME)
        with vol.batch_upload(force=True) as batch:
            batch.put_directory(str(unified), "/dataset_video_unified")
            for p in files:
                batch.put_file(str(p), f"/data_splits/{p.name}")
            if n_folds:
                batch.put_directory(str(folds), "/data_splits/folds")
        print("✅ Upload unified xong. Kiểm tra trên volume: modal run scripts/audit_data.py")
        return

    if args.dense:
        dense = DATA / "dataset_video_dense"
        files = sorted((DATA / "data_splits").glob("dense_*"))
        n = sum(1 for _ in dense.rglob("*") if _.is_file())
        print(f"Volume: {VOLUME_NAME}\n- {n} file -> /dataset_video_dense/\n- {[p.name for p in files]} -> /data_splits/")
        if args.dry_run:
            return
        import modal
        vol = modal.Volume.from_name(VOLUME_NAME)
        with vol.batch_upload(force=True) as batch:
            batch.put_directory(str(dense), "/dataset_video_dense")
            for p in files:
                batch.put_file(str(p), f"/data_splits/{p.name}")
        print("✅ Upload dense xong.")
        return

    split_files = sorted(p for p in (DATA / "data_splits").iterdir()
                         if p.is_file() and p.suffix in (".csv", ".json"))
    backups = sorted((DATA / "archive").glob("stretched_frames_*"))
    videos = sorted(p.name for p in backups[-1].iterdir() if p.is_dir()) if backups else []

    print(f"Volume: {VOLUME_NAME}")
    print(f"- {len(split_files)} file data_splits -> /data_splits/")
    print(f"- {len(videos)} video đã un-stretch -> /dataset_video_prepared/<video>/")
    for v in videos:
        print(f"    {v}")
    if args.dry_run:
        return

    import modal
    vol = modal.Volume.from_name(VOLUME_NAME)
    with vol.batch_upload(force=True) as batch:
        for p in split_files:
            batch.put_file(str(p), f"/data_splits/{p.name}")
        for v in videos:
            batch.put_directory(str(DATA / "dataset_video_prepared" / v), f"/dataset_video_prepared/{v}")
    print("✅ Upload xong. Kiểm tra trên volume: modal run scripts/audit_data.py")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
