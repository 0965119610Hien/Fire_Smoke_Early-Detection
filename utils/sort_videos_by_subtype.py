"""
sort_videos_by_subtype.py
=========================
Sap xep video trong Video/Video/<label>/ vao cac subfolder theo subtype,
khop voi cau truc subtype cua dataset_test.

Chien luoc:
  - Doc keyword tu ten file (pattern: ..._<keyword>_...)
  - Map keyword -> subtype folder tuong ung
  - File khong nhan dang duoc -> folder "_unclassified"
  - Che do DRY_RUN=True de preview truoc khi thuc su di chuyen
"""

import shutil
from pathlib import Path
from collections import defaultdict

# --- CAU HINH -----------------------------------------------------------
VIDEO_ROOT = Path("e:/NCKH/files/Data/Video/Video")
DRY_RUN = True   # Dat False de thuc su di chuyen file
# ------------------------------------------------------------------------

# --- MAPPING: keyword trong ten file -> subtype folder (theo tung label) --
#
# Thu tu QUAN TRONG: keyword cu the hon (dai hon) phai dung truoc.
# Script se dung keyword dau tien khop voi ten file.
#
LABEL_SUBTYPE_MAP = {
    "0_normal_noise": [
        ("optical_noise",        "optical_noise"),
        ("steam_vapor",          "steam_vapor"),
        ("cooking_smoke",        "cooking_smoke_normal"),
        ("cigarette_smoke",      "cigarette_vape_smoke"),
        ("cigarette_vape",       "cigarette_vape_smoke"),
        ("dust_particles",       "dust_particles"),
        ("decorative_fire",      "decorative_fire_fp"),
        ("alarm_warning_lights", "decorative_fire_fp"),
        ("welding_sparks",       "welding_sparks_noise"),
        ("daily_activity_fp",    "daily_activity_fp"),
        # Files bi relabel tu label 1/2 sang 0
        ("heavy_smoke",          "_unclassified"),
        ("residential_fire",     "_unclassified"),
        ("visible_flame",        "_unclassified"),
        ("early_smoke",          "_unclassified"),
        ("diffused_smoke",       "_unclassified"),
    ],
    "1_controlled_fire": [
        ("early_smoke",          "early_smoke_onset"),
        ("heavy_smoke",          "heavy_smoke_controlled"),
        ("welding_sparks",       "welding_sparks_ctrl"),
        ("cooking_smoke",        "cooking_smoke_threshold"),
        ("cigarette_smoke",      "cigarette_controlled"),
        ("cigarette_vape",       "cigarette_controlled"),
        ("visible_flame",        "small_visible_flame"),
        ("daily_activity_fp",    "daily_activity_fire_ctx"),
        ("diffused_smoke",       "daily_activity_fire_ctx"),
        ("dust_particles",       "_unclassified"),
    ],
    "2_uncontrolled_hazard": [
        ("residential_fire",     "residential_indoor_fire"),
        ("visible_flame",        "large_visible_flame"),
        ("early_smoke",          "early_smoke_real_fire"),
        ("diffused_smoke",       "diffused_smoke_spreading"),
        ("heavy_smoke",          "heavy_smoke_hazard"),
        ("daily_activity_fp",    "daily_ctx_real_fire"),
        ("dust_particles",       "early_smoke_real_fire"),
        ("toxic",                "toxic_material_smoke"),
    ],
}


def find_subtype(filename: str, label: str) -> str:
    """Tra ve subtype folder dua vao keyword trong ten file."""
    name_lower = filename.lower()
    mappings = LABEL_SUBTYPE_MAP.get(label, [])
    for keyword, subtype in mappings:
        if keyword in name_lower:
            return subtype
    return "_unclassified"


def main():
    stats = defaultdict(lambda: defaultdict(int))
    unclassified_files = []
    moves = []

    for label in ["0_normal_noise", "1_controlled_fire", "2_uncontrolled_hazard"]:
        label_dir = VIDEO_ROOT / label
        if not label_dir.exists():
            print(f"[WARN] Thu muc khong ton tai: {label_dir}")
            continue

        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
        # Chi lay file truc tiep trong label_dir (khong de quy vao subfolder)
        videos = [f for f in label_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in video_exts]

        for vid in videos:
            subtype = find_subtype(vid.name, label)
            stats[label][subtype] += 1
            dst_path = label_dir / subtype / vid.name
            moves.append((vid, dst_path))
            if subtype == "_unclassified":
                unclassified_files.append((label, vid.name))

    # --- In bao cao preview -------------------------------------------------
    sep = "=" * 70
    print(f"\n{sep}")
    print("  PREVIEW SAP XEP VIDEO -> SUBTYPE FOLDER")
    mode_str = "DRY RUN (chi xem, chua di chuyen)" if DRY_RUN else "*** THUC SU DI CHUYEN ***"
    print(f"  MODE: {mode_str}")
    print(sep)

    total_videos = 0
    for label in ["0_normal_noise", "1_controlled_fire", "2_uncontrolled_hazard"]:
        label_total = sum(stats[label].values())
        total_videos += label_total
        print(f"\n[{label}]  (tong: {label_total} videos)")
        for subtype in sorted(stats[label].keys()):
            cnt = stats[label][subtype]
            prefix = "  [!]" if subtype == "_unclassified" else "     "
            print(f"  {prefix} -> {subtype:<35} {cnt:>4} videos")

    print(f"\n{'-' * 70}")
    print(f"  Tong cong: {total_videos} videos se duoc sap xep")

    if unclassified_files:
        print(f"\n[!] {len(unclassified_files)} file khong nhan dang duoc keyword:")
        for lbl, fname in unclassified_files[:30]:
            print(f"    [{lbl}] {fname}")
        if len(unclassified_files) > 30:
            print(f"    ... va {len(unclassified_files) - 30} file nua")

    if DRY_RUN:
        print("\n[DRY RUN] Khong co file nao bi di chuyen.")
        print("De thuc su di chuyen: dat DRY_RUN = False roi chay lai.\n")
        return

    # --- Thuc su di chuyen --------------------------------------------------
    print("\nBat dau di chuyen files...")
    moved_count = 0
    error_count = 0

    for src, dst in moves:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            final_dst = dst
            if final_dst.exists():
                stem = dst.stem
                suffix = dst.suffix
                i = 2
                while final_dst.exists():
                    final_dst = dst.parent / f"{stem}_{i}{suffix}"
                    i += 1
            shutil.move(str(src), str(final_dst))
            moved_count += 1
        except Exception as e:
            print(f"  [ERROR] {src.name}: {e}")
            error_count += 1

    print(f"\nHoan thanh!")
    print(f"  Di chuyen thanh cong: {moved_count} files")
    if error_count:
        print(f"  Loi: {error_count} files")


if __name__ == "__main__":
    main()
