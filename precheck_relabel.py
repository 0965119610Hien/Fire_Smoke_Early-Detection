"""
Pre-check: dem so file can di chuyen giua cac label trong train set
"""
from pathlib import Path

base = Path('d:/zip_dataset/Video/Video')
l0 = base / '0_normal_noise'
l1 = base / '1_controlled_fire'
l2 = base / '2_uncontrolled_hazard'

cigarette = [f for f in l1.glob('*.mp4') if 'cigarette_smoke' in f.name]
cooking   = [f for f in l1.glob('*.mp4') if 'cooking_smoke' in f.name]
welding   = [f for f in l1.glob('*.mp4') if 'welding_sparks' in f.name]
daily     = [f for f in l1.glob('*.mp4') if 'daily_activity_fp' in f.name]

print(f'=== Files trong label 1 can xem xet ===')
print(f'cigarette_smoke  : {len(cigarette)} files -> de xuat chuyen sang label 0')
print(f'cooking_smoke    : {len(cooking)} files -> de xuat chuyen sang label 0')
print(f'welding_sparks   : {len(welding)} files -> giu label 1 (can visual check)')
print(f'daily_activity_fp: {len(daily)} files -> giu label 1')
print()

cig0  = [f for f in l0.glob('*.mp4') if 'cigarette' in f.name]
cook0 = [f for f in l0.glob('*.mp4') if 'cooking' in f.name]
weld0 = [f for f in l0.glob('*.mp4') if 'welding' in f.name]
print(f'Label 0 hien co:')
print(f'  cigarette: {len(cig0)} files')
print(f'  cooking  : {len(cook0)} files')
print(f'  welding  : {len(weld0)} files')

total0 = len(list(l0.glob('*.mp4')))
total1 = len(list(l1.glob('*.mp4')))
total2 = len(list(l2.glob('*.mp4')))
print()
print(f'Hien tai:')
print(f'  label 0: {total0} files')
print(f'  label 1: {total1} files')
print(f'  label 2: {total2} files')
print()
move_count = len(cigarette) + len(cooking)
print(f'Sau khi relabel:')
print(f'  label 0: {total0 + move_count} files (+{move_count})')
print(f'  label 1: {total1 - move_count} files (-{move_count})')
print(f'  label 2: {total2} files (khong doi)')
