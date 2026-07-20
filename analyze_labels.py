"""
Phân tích mismatch nhãn giữa train set (Video/Video) và test set (dataset_test)
"""
import os
from pathlib import Path
from collections import defaultdict

train_base = Path('d:/zip_dataset/Video/Video')
test_base  = Path('d:/zip_dataset/dataset_test')


def extract_subtype(filename: str) -> str:
    name = filename.replace('cleaned_', '').replace('.mp4', '')
    for src in ['pexels_', 'yt_']:
        if src in name:
            after = name.split(src, 1)[1]
            parts = []
            for p in after.split('_'):
                if p.isdigit() and len(p) >= 5:
                    break
                if len(p) > 2 and p[:1].isdigit():
                    break
                parts.append(p)
            return '_'.join(parts).rstrip('_')
    return '[other]'


# ── Thu thập subtype từ train set ──────────────────────────────────────────
train_labels = {}
for label_dir in sorted(train_base.iterdir()):
    if not label_dir.is_dir():
        continue
    subtypes = defaultdict(int)
    for f in label_dir.glob('*.mp4'):
        st = extract_subtype(f.name)
        subtypes[st] += 1
    train_labels[label_dir.name] = dict(subtypes)

# ── In train summary ────────────────────────────────────────────────────────
print('=' * 80)
print('TRAIN SET (Video/Video) - Phân bố subtype theo label')
print('=' * 80)
for label, subtypes in sorted(train_labels.items()):
    total = sum(subtypes.values())
    print(f'\n[{label}] ({total} videos):')
    for st, cnt in sorted(subtypes.items(), key=lambda x: -x[1]):
        if '[other]' not in st:
            print(f'  {st}: {cnt}')

# ── In test summary ─────────────────────────────────────────────────────────
print('\n\n' + '=' * 80)
print('TEST SET (dataset_test) - Phân bố subtype theo label')
print('=' * 80)

test_label_subtypes_map = {}
for label_dir in sorted(test_base.iterdir()):
    if not label_dir.is_dir():
        continue
    subtypes = {}
    for subtype_dir in sorted(label_dir.iterdir()):
        if subtype_dir.is_dir():
            cnt = len(list(subtype_dir.glob('*.mp4')))
            subtypes[subtype_dir.name] = cnt
    test_label_subtypes_map[label_dir.name] = subtypes
    total = sum(subtypes.values())
    print(f'\n[{label_dir.name}] ({total} videos):')
    for st, cnt in sorted(subtypes.items(), key=lambda x: -x[1]):
        print(f'  {st}: {cnt}')

# ── Mapping subtype → label trong train ────────────────────────────────────
subtype_to_train_label = defaultdict(list)
for lbl, subtypes in train_labels.items():
    for st in subtypes:
        subtype_to_train_label[st].append(lbl)

# ── Phân tích mismatch ─────────────────────────────────────────────────────
print('\n\n' + '=' * 80)
print('PHAN TICH MISMATCH - subtype test co noi dung giong train nhung khac label')
print('=' * 80)

# Các subtype test và label tương ứng
test_label_subtypes = {
    '0_normal_noise': [
        'cigarette_vape_smoke', 'cooking_smoke_normal', 'daily_activity_fp',
        'decorative_fire_fp', 'dust_particles', 'optical_noise',
        'steam_vapor', 'welding_sparks_noise',
    ],
    '1_controlled_fire': [
        'cigarette_controlled', 'cooking_smoke_threshold', 'daily_activity_fire_ctx',
        'early_smoke_onset', 'heavy_smoke_controlled', 'small_visible_flame',
        'welding_sparks_ctrl',
    ],
    '2_uncontrolled_hazard': [
        'daily_ctx_real_fire', 'diffused_smoke_spreading', 'early_smoke_real_fire',
        'heavy_smoke_hazard', 'large_visible_flame', 'residential_indoor_fire',
        'toxic_material_smoke',
    ],
}

# Map keyword → train label (chỉ giữ các subtype thực sự rõ ràng)
# Đây là bảng so sánh thủ công dựa trên ngữ nghĩa
MISMATCH_TABLE = [
    # (test_label, test_subtype, expected_label, reason)
    # ── NHÃN 0: có subtype lẽ ra phải ở nhãn khác? ──
    ('0_normal_noise', 'cigarette_vape_smoke',
     '1_controlled_fire',
     'Train label 1 co 78 clip cigarette_smoke; test dat o label 0 la khong nhat quan'),

    ('0_normal_noise', 'decorative_fire_fp',
     '?',
     'MISMATCH NGUY HIEM: decorative_fire_fp (lửa thật: fireplace, candle) trong label 0 '
     'nhưng train KHONG CO loai nay. Fireplace co the bi classify la 2_uncontrolled_hazard'),

    ('0_normal_noise', 'cooking_smoke_normal',
     '?',
     'Train label 1 co cooking_smoke (32 clips). Test goi "cooking_smoke_normal" o label 0 '
     'nhung ranh gioi voi cooking_smoke_threshold (label 1) rat mo'),

    ('0_normal_noise', 'welding_sparks_noise',
     '1_controlled_fire',
     'Train label 1 co welding_sparks (76 clips). welding_sparks_noise o label 0 ma '
     'train dat welding_sparks o label 1 - DAY LA MISMATCH CHINH'),

    # ── NHÃN 1 ──
    ('1_controlled_fire', 'cigarette_controlled',
     '0_normal_noise',
     'cigarette_smoke o train nam o label 1 (78 clips). Test dong y dat o label 1 - OK'),

    ('1_controlled_fire', 'daily_activity_fire_ctx',
     '?',
     'Subtype nay KHONG CO trong train. Ten am chi "real fire context" - co the '
     'nen la label 2 tuy noi dung'),

    ('1_controlled_fire', 'heavy_smoke_controlled',
     '2_uncontrolled_hazard',
     'Train label 2 co heavy_smoke (rat nhieu clip). '
     'heavy_smoke_controlled o label 1 nhung train heavy_smoke o label 2 - MISMATCH'),

    ('1_controlled_fire', 'small_visible_flame',
     '?',
     'Train label 1 co visible_flame (1 clip: ac283487_pexels_early_smoke). '
     'Label 2 co visible_flame nhieu hon. Co the can xem xet lai'),

    # ── NHÃN 2 ──
    ('2_uncontrolled_hazard', 'early_smoke_real_fire',
     '1_controlled_fire',
     'Train label 2 co early_smoke (be69c337, e4d9fb64). '
     'Test dat early_smoke_real_fire o label 2 - Phu hop neu la khoi tu chay that'),

    ('2_uncontrolled_hazard', 'daily_ctx_real_fire',
     '?',
     'Train label 2 co daily_activity_fp (8f94bda2 clips). '
     'Test dat daily_ctx_real_fire o label 2 - Phu hop'),
]

found_issues = []
for test_label, test_subtype, expected, reason in MISMATCH_TABLE:
    test_dir = test_base / test_label / test_subtype
    if not test_dir.exists():
        continue
    cnt = len(list(test_dir.glob('*.mp4')))
    if cnt == 0:
        continue

    severity = 'CRITICAL' if expected not in ['?', test_label] else 'WARNING'
    found_issues.append((severity, test_label, test_subtype, cnt, expected, reason))

print()
for severity, test_label, test_subtype, cnt, expected, reason in sorted(found_issues, key=lambda x: x[0]):
    icon = '[!!!]' if severity == 'CRITICAL' else '[?] '
    print(f'{icon} {test_label}/{test_subtype} ({cnt} videos)')
    print(f'     Du kien: {expected}')
    print(f'     Ly do  : {reason}')
    print()

# ── Tìm file cụ thể bị sai ─────────────────────────────────────────────────
print('=' * 80)
print('VIDEO BI SAI NHAN RO RANG - tim theo ten file')
print('=' * 80)
print()

# Các subtype test da co noi dung ro rang sai trong train
critical_checks = [
    # (test_folder, keyword_in_filename_suggests_wrong_label, train_label_of_that_keyword)
    ('0_normal_noise', 'heavy_smoke', '2_uncontrolled_hazard'),
    ('0_normal_noise', 'residential_fire', '2_uncontrolled_hazard'),
    ('0_normal_noise', 'visible_flame', '2_uncontrolled_hazard'),
    ('0_normal_noise', 'welding_sparks', '1_controlled_fire'),
    ('1_controlled_fire', 'dust_particles', '0_normal_noise'),
    ('1_controlled_fire', 'daily_activity_fp', '0_normal_noise'),
    ('2_uncontrolled_hazard', 'dust_particles', '0_normal_noise'),
    ('2_uncontrolled_hazard', 'daily_activity_fp', '0_normal_noise'),
]

found_wrong = []
for test_label, keyword, train_label in critical_checks:
    test_dir = test_base / test_label
    for subtype_dir in test_dir.iterdir():
        if not subtype_dir.is_dir():
            continue
        for f in subtype_dir.glob('*.mp4'):
            if keyword in f.name:
                found_wrong.append({
                    'file': f.name,
                    'location': f'{test_label}/{subtype_dir.name}/',
                    'keyword': keyword,
                    'correct_label': train_label,
                })

if found_wrong:
    print(f'Tim thay {len(found_wrong)} file co ten cho thay sai nhan:')
    for w in found_wrong[:50]:
        print(f'  [{w["location"]}] {w["file"]}')
        print(f'       Keyword "{w["keyword"]}" -> Nen o [{w["correct_label"]}]')
    if len(found_wrong) > 50:
        print(f'  ... va {len(found_wrong) - 50} file khac')
else:
    print('Khong tim thay file sai nhan ro rang qua keyword.')

print()
print('XONG.')
