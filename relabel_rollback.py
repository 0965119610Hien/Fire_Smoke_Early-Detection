"""
relabel_rollback.py
===================
Rollback cac thay doi tu relabel_train.py dua vao relabel_log.json.
"""

import json
import shutil
from pathlib import Path

LOG_FILE = Path('d:/zip_dataset/relabel_log.json')

def rollback():
    if not LOG_FILE.exists():
        print('Khong tim thay relabel_log.json — khong co gi de rollback.')
        return

    with open(LOG_FILE, encoding='utf-8') as f:
        log = json.load(f)

    moves = log.get('moves', [])
    print(f'Rollback {len(moves)} files...')

    restored = 0
    errors = []
    for item in moves:
        src = Path(item['to'])   # file hien tai o label 0
        dst = Path(item['from']) # tra ve label 1

        if not src.exists():
            errors.append(f'KHONG TIM THAY: {src.name}')
            continue
        if dst.exists():
            errors.append(f'FILE DA TON TAI O DICH: {dst.name}')
            continue
        try:
            shutil.move(str(src), str(dst))
            restored += 1
            print(f'  [OK] Restored: {src.name}')
        except Exception as e:
            errors.append(f'LOI: {src.name}: {e}')

    print(f'Hoan thanh rollback: {restored}/{len(moves)} files.')
    if errors:
        for e in errors:
            print(f'  LOI: {e}')

if __name__ == '__main__':
    rollback()
