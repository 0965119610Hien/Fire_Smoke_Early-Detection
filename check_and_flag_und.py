"""
check_and_flag_und.py
=====================
Quet toan bo video trong Video/Video/<label>/<subtype>/ va flag
nhung truong hop "phan van" (keyword trong ten file mau thuan voi vi tri)
vao folder und.

Cac truong hop bi flag la "phan van":
  1. Tat ca file trong bat ky folder _unclassified nao
  2. File co keyword trong ten file MAU THUAN voi label dang chua no
  3. File khong co keyword nao nhan dang duoc (pexels ID thuan, UUID, "video XXXX")
  4. Truong hop bien gioi L1<->L2: diffused_smoke trong L1, daily_activity_fp trong L2
"""

import re
import shutil
from pathlib import Path
from collections import defaultdict

# --- CAU HINH -----------------------------------------------------------
VIDEO_ROOT  = Path("e:/NCKH/files/Data/Video/Video")
UND_DIR     = VIDEO_ROOT / "und"   # Thu muc dau ra cho cac file phan van
DRY_RUN     = True   # Dat False de thuc su di chuyen
# ------------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

# --- DINH NGHIA KEYWORD VA LABEL CUA CHUNG ------------------------------

# Keyword nao thuoc label nao (theo dinh nghia chinh thuc du an)
KEYWORD_LABEL = {
    # Label 0 keywords
    "optical_noise":        0,
    "steam_vapor":          0,
    "cooking_smoke":        0,   # trong L0 la normal
    "cigarette_smoke":      0,
    "cigarette_vape":       0,
    "dust_particles":       0,
    "decorative_fire":      0,
    "alarm_warning_lights": 0,
    "welding_sparks":       None,  # co ca L0 va L1 -> can kiem tra subtype
    "daily_activity_fp":    None,  # co ca L0, L1, L2 -> can kiem tra subtype

    # Label 1 keywords
    "early_smoke":          None,  # co ca L1 va L2 -> bien gioi
    "heavy_smoke":          None,  # co ca L1 va L2 -> can kiem tra

    # Label 2 keywords
    "residential_fire":     2,
    "visible_flame":        None,  # co ca L1 (small) va L2 (large)
    "diffused_smoke":       None,  # bien gioi L1/L2
    "toxic":                2,
}

# Cac truong hop MAU THUAN ro rang (keyword xu huong label X nhung dang o label Y)
CLEAR_MISMATCH = [
    # (keyword_in_name, current_label_int, ly_do)
    ("residential_fire",  0, "residential_fire la L2, nhung dang o L0"),
    ("residential_fire",  1, "residential_fire la L2, nhung dang o L1"),
    ("visible_flame",     0, "visible_flame (neu o L0) la FP, nen check lai"),
    ("dust_particles",    2, "dust_particles la L0, nhung dang o L2"),
    ("toxic",             0, "toxic material la L2, nhung dang o L0"),
    ("toxic",             1, "toxic material la L2, nhung dang o L1"),
]

# Truong hop BIEN GIOI (khong ro rang, can dua vao und)
BORDERLINE = [
    # (keyword, current_label, subtype_folder, ly_do)
    ("diffused_smoke",    1,  None, "diffused_smoke thuoc bien gioi L1/L2"),
    ("daily_activity_fp", 2,  None, "daily_activity_fp trong L2 - co the la L0/L1"),
    ("heavy_smoke",       0,  None, "heavy_smoke trong L0 - da relabel, nen xem lai"),
    ("early_smoke",       0,  None, "early_smoke trong L0 - da relabel, nen xem lai"),
    ("visible_flame",     0,  None, "visible_flame trong L0 - da relabel, nen xem lai"),
]


def get_label_int(label_folder: str) -> int:
    if label_folder.startswith("0_"): return 0
    if label_folder.startswith("1_"): return 1
    if label_folder.startswith("2_"): return 2
    return -1


def has_keyword(filename: str, keyword: str) -> bool:
    return keyword.lower() in filename.lower()


def is_no_keyword_file(filename: str) -> bool:
    """
    File khong co keyword nhan dang duoc:
    - Pexels ID thuan: 12109024_1920_1080_30fps.mp4
    - UUID dang: 00e18d86-3389d8dc.mp4
    - "video (XXXX).mp4"
    - Ten chi co so va fps
    - Ten file dat biet khac
    """
    stem = Path(filename).stem.lower()
    patterns = [
        r'^\d{6,}[_-][\d_fx]+fps?$',       # pexels ID + resolution
        r'^[\da-f]{8}-[\da-f]{8}$',           # UUID 8-8
        r'^video \(\d+\)$',                   # "video (XXXX)"
        r'^\d{6,}$',                          # so thuan
        r'^[\da-f]{8}$',                      # hash thuan
        r'^[\da-f]{8}\.\d{8}$',               # hash.hash
        r'^\d+_\d+_\d+_\d+fps$',             # ID_W_H_fps
        r'^\d+[_-].*fps$',                    # ID-uhd_WxH_fps
        r'^weihnachtsbaum',                   # ten phim duc
        r'^what type of',                     # ten phim tieng anh
        r'^2016 april',
        r'^240_f_',                            # adobe stock ID
        r'^8f08a68c$',
        r'^8f\.mp4$',
    ]
    for pat in patterns:
        if re.match(pat, stem):
            return True
    # Them: file co ten la so va resolution don gian
    if re.match(r'^\d{6,}[_\-].*\d+fps', stem):
        return True
    if re.match(r'^\d{8}[_\-]\d{8}', stem):
        return True
    return False


def scan_and_flag():
    flags = []        # (src_path, reason)
    stats = defaultdict(int)

    for label_dir in sorted(VIDEO_ROOT.iterdir()):
        if not label_dir.is_dir():
            continue
        label_name = label_dir.name
        if not (label_name.startswith("0_") or label_name.startswith("1_") or label_name.startswith("2_")):
            continue
        label_int = get_label_int(label_name)

        for subtype_dir in sorted(label_dir.iterdir()):
            if not subtype_dir.is_dir():
                continue
            subtype_name = subtype_dir.name

            for vid in sorted(subtype_dir.iterdir()):
                if not vid.is_file() or vid.suffix.lower() not in VIDEO_EXTS:
                    continue

                stats["total"] += 1
                reason = None

                # --- RULE 1: Da nam trong _unclassified ---
                if subtype_name == "_unclassified":
                    reason = f"Da trong _unclassified (khong nhan dang keyword)"

                # --- RULE 2: File khong co keyword nao ---
                elif is_no_keyword_file(vid.name):
                    reason = f"Ten file khong co keyword nhan dang (pexels ID / UUID / 'video XXXX')"

                # --- RULE 3: MAU THUAN ro rang ---
                if reason is None:
                    for kw, expected_fail_label, lyrdo in CLEAR_MISMATCH:
                        if has_keyword(vid.name, kw) and label_int == expected_fail_label:
                            reason = lyrdo
                            break

                # --- RULE 4: BIEN GIOI ---
                if reason is None:
                    for kw, bl_label, bl_subtype, lyrdo in BORDERLINE:
                        if has_keyword(vid.name, kw) and label_int == bl_label:
                            if bl_subtype is None or subtype_name == bl_subtype:
                                reason = lyrdo
                                break

                if reason:
                    flags.append((vid, label_name, subtype_name, reason))
                    stats["flagged"] += 1

    return flags, stats


def main():
    flags, stats = scan_and_flag()

    # --- In bao cao ---
    sep = "=" * 72
    print(f"\n{sep}")
    print("  KIEM TRA VIDEO - FLAG CHUYEN VAO FOLDER 'und'")
    mode_str = "DRY RUN (chi xem)" if DRY_RUN else "*** THUC SU DI CHUYEN ***"
    print(f"  MODE: {mode_str}")
    print(sep)

    # Nhom theo label
    by_label = defaultdict(list)
    for vid, label, subtype, reason in flags:
        by_label[label].append((vid, subtype, reason))

    for label in sorted(by_label.keys()):
        items = by_label[label]
        print(f"\n[{label}]  ({len(items)} files bi flag)")

        # Nhom theo ly do
        by_reason = defaultdict(list)
        for vid, subtype, reason in items:
            by_reason[reason[:60]].append((vid.name, subtype))

        for reason, vids in sorted(by_reason.items()):
            print(f"\n  [{reason}]")
            for fname, sub in sorted(vids)[:15]:
                print(f"    <- {sub}/{fname}")
            if len(vids) > 15:
                print(f"    ... va {len(vids)-15} file nua")

    print(f"\n{'-' * 72}")
    print(f"  Tong video quet: {stats['total']}")
    print(f"  Tong bi flag   : {stats['flagged']}")
    pct = stats['flagged'] / max(stats['total'], 1) * 100
    print(f"  Ti le flag     : {pct:.1f}%")

    if DRY_RUN:
        print("\n[DRY RUN] Khong co file nao bi di chuyen.")
        print("Dat DRY_RUN = False roi chay lai de thuc su di chuyen.\n")
        return

    # --- Di chuyen vao und ---
    print(f"\nTao thu muc und va di chuyen {stats['flagged']} files...")
    moved = 0
    errors = 0

    for vid, label, subtype, reason in flags:
        # Dat vao und/<label>/<subtype>/<filename>
        dst_dir = UND_DIR / label / subtype
        dst = dst_dir / vid.name

        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            # Tranh trung ten
            if dst.exists():
                stem, sfx = dst.stem, dst.suffix
                i = 2
                while dst.exists():
                    dst = dst_dir / f"{stem}_{i}{sfx}"
                    i += 1
            shutil.move(str(vid), str(dst))
            moved += 1
        except Exception as e:
            print(f"  [ERROR] {vid.name}: {e}")
            errors += 1

    # Xoa folder rong sau khi chuyen
    for label_dir in sorted(VIDEO_ROOT.iterdir()):
        if label_dir.is_dir() and label_dir.name.startswith(("0_", "1_", "2_")):
            for subtype_dir in sorted(label_dir.iterdir()):
                if subtype_dir.is_dir():
                    try:
                        remaining = list(subtype_dir.iterdir())
                        if not remaining:
                            subtype_dir.rmdir()
                    except Exception:
                        pass

    print(f"\nHoan thanh!")
    print(f"  Da di chuyen: {moved} files")
    print(f"  Luu tai     : {UND_DIR}")
    if errors:
        print(f"  Loi: {errors} files")


if __name__ == "__main__":
    main()
