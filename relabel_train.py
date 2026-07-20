"""
relabel_train.py
================
Di chuyen files trong train set (Video/Video/) theo logic dung:
  - cigarette_smoke  (78 files): label 1 -> label 0
  - cooking_smoke    (32 files): label 1 -> label 0

Ket qua duoc ghi vao relabel_log.json de co the rollback.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

BASE      = Path('d:/zip_dataset/Video/Video')
LABEL_0   = BASE / '0_normal_noise'
LABEL_1   = BASE / '1_controlled_fire'
LABEL_2   = BASE / '2_uncontrolled_hazard'
LOG_FILE  = Path('d:/zip_dataset/relabel_log.json')

# ── Xac dinh file can di chuyen ────────────────────────────────────────────
def get_files_to_move():
    moves = []
    for f in sorted(LABEL_1.glob('*.mp4')):
        name = f.name
        if 'cigarette_smoke' in name or 'cooking_smoke' in name:
            moves.append({
                'file':     name,
                'from':     str(f),
                'to':       str(LABEL_0 / name),
                'subtype':  'cigarette_smoke' if 'cigarette_smoke' in name else 'cooking_smoke',
            })
    return moves


def count_dir(d: Path) -> int:
    return len(list(d.glob('*.mp4')))


def run():
    moves = get_files_to_move()

    if not moves:
        print('Khong co file nao can di chuyen.')
        return

    # ── Preview ────────────────────────────────────────────────────────────
    cig_files  = [m for m in moves if m['subtype'] == 'cigarette_smoke']
    cook_files = [m for m in moves if m['subtype'] == 'cooking_smoke']
    print('=' * 70)
    print(f'RELABEL TRAIN SET — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)
    print(f'  cigarette_smoke : {len(cig_files)} files')
    print(f'  cooking_smoke   : {len(cook_files)} files')
    print(f'  TONG            : {len(moves)} files')
    print()
    print(f'Truoc khi di chuyen:')
    print(f'  label 0: {count_dir(LABEL_0)} files')
    print(f'  label 1: {count_dir(LABEL_1)} files')
    print(f'  label 2: {count_dir(LABEL_2)} files')
    print()

    # ── Ghi log truoc khi thuc hien ────────────────────────────────────────
    log = {
        'timestamp': datetime.now().isoformat(),
        'description': 'Relabel: cigarette_smoke + cooking_smoke from label1 -> label0',
        'moves': moves,
        'before': {
            'label_0': count_dir(LABEL_0),
            'label_1': count_dir(LABEL_1),
            'label_2': count_dir(LABEL_2),
        }
    }

    # ── Thuc hien di chuyen ────────────────────────────────────────────────
    moved = 0
    errors = []

    for item in moves:
        src = Path(item['from'])
        dst = Path(item['to'])

        if not src.exists():
            errors.append(f'KHONG TIM THAY: {src.name}')
            continue
        if dst.exists():
            errors.append(f'FILE TON TAI O DICH: {dst.name}')
            continue

        try:
            shutil.move(str(src), str(dst))
            moved += 1
            print(f'  [OK] {item["subtype"]:20s} -> 0_normal_noise / {src.name}')
        except Exception as e:
            errors.append(f'LOI khi di chuyen {src.name}: {e}')

    # ── Ket qua ────────────────────────────────────────────────────────────
    log['after'] = {
        'label_0': count_dir(LABEL_0),
        'label_1': count_dir(LABEL_1),
        'label_2': count_dir(LABEL_2),
    }
    log['moved_count'] = moved
    log['error_count'] = len(errors)
    log['errors'] = errors

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print()
    print('=' * 70)
    print(f'HOAN THANH: {moved}/{len(moves)} files da di chuyen')
    if errors:
        print(f'  LOI: {len(errors)}')
        for e in errors:
            print(f'    - {e}')
    print()
    print(f'Sau khi di chuyen:')
    print(f'  label 0: {log["after"]["label_0"]} files  (truoc: {log["before"]["label_0"]})')
    print(f'  label 1: {log["after"]["label_1"]} files  (truoc: {log["before"]["label_1"]})')
    print(f'  label 2: {log["after"]["label_2"]} files  (khong doi)')
    print()
    print(f'Log da luu: {LOG_FILE}')
    print('De rollback: chay python relabel_rollback.py')


if __name__ == '__main__':
    run()
