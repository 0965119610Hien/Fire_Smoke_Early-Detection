"""
dataset_stats.py
================
Thong ke toan bo train set va test set:
- So luong video theo label / subtype
- Duration (min/max/mean/median)
- Resolution (W x H)
- FPS
- File size
Ket qua ghi ra dataset_stats.json va in ra stdout.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics

import av

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_BASE = Path('d:/zip_dataset/Video/Video')
TEST_BASE  = Path('d:/zip_dataset/dataset_test')
OUT_JSON   = Path('d:/zip_dataset/dataset_stats.json')

# ── Helper ─────────────────────────────────────────────────────────────────
def get_video_meta(path: Path) -> dict:
    """Lay metadata video qua PyAV. Tra ve None neu loi."""
    try:
        with av.open(str(path)) as container:
            stream = next(
                (s for s in container.streams if s.type == 'video'), None
            )
            if stream is None:
                return None
            duration = float(container.duration) / 1_000_000 if container.duration else None
            fps_rat  = stream.average_rate
            fps      = float(fps_rat) if fps_rat else None
            w, h     = stream.width, stream.height
            return {
                'duration_s': round(duration, 2) if duration else None,
                'width':  w,
                'height': h,
                'fps':    round(fps, 2) if fps else None,
                'size_mb': round(path.stat().st_size / 1_048_576, 2),
            }
    except Exception:
        return None


def stats_of(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {}
    return {
        'count':  len(vals),
        'min':    round(min(vals), 2),
        'max':    round(max(vals), 2),
        'mean':   round(statistics.mean(vals), 2),
        'median': round(statistics.median(vals), 2),
        'stdev':  round(statistics.stdev(vals), 2) if len(vals) > 1 else 0,
    }


def scan_flat_dir(label_dir: Path) -> dict:
    """Quet thu muc flat (train style): label_dir/*.mp4"""
    metas = []
    for f in sorted(label_dir.glob('*.mp4')):
        m = get_video_meta(f)
        if m:
            metas.append(m)
    return summarize(metas)


def scan_subtype_dir(label_dir: Path) -> dict:
    """Quet thu muc co subfolder subtype (test style)."""
    result = {'subtypes': {}, 'all': []}
    for sub in sorted(label_dir.iterdir()):
        if not sub.is_dir():
            continue
        metas = []
        for f in sorted(sub.glob('*.mp4')):
            m = get_video_meta(f)
            if m:
                metas.append(m)
        result['subtypes'][sub.name] = summarize(metas)
        result['all'].extend(metas)
    result['all'] = summarize(result['all'])
    return result


def summarize(metas: list) -> dict:
    if not metas:
        return {'count': 0}
    return {
        'count':      len(metas),
        'duration_s': stats_of([m['duration_s'] for m in metas]),
        'size_mb':    stats_of([m['size_mb']    for m in metas]),
        'fps':        stats_of([m['fps']         for m in metas]),
        'resolutions': sorted(
            {f'{m["width"]}x{m["height"]}' for m in metas if m.get('width')}
        ),
        'res_dominant': dominant(
            [f'{m["width"]}x{m["height"]}' for m in metas if m.get('width')]
        ),
        'total_duration_min': round(
            sum(m['duration_s'] for m in metas if m['duration_s']) / 60, 1
        ),
        'total_size_gb': round(
            sum(m['size_mb'] for m in metas) / 1024, 3
        ),
    }


def dominant(lst):
    if not lst:
        return None
    from collections import Counter
    c = Counter(lst)
    top = c.most_common(5)
    return {k: v for k, v in top}


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    result = {'train': {}, 'test': {}}

    # ── TRAIN ─────────────────────────────────────────────────────────────
    print('Scanning TRAIN set...', flush=True)
    all_train = []
    for label_dir in sorted(TRAIN_BASE.iterdir()):
        if not label_dir.is_dir():
            continue
        print(f'  [{label_dir.name}]...', flush=True)
        metas = []
        for f in sorted(label_dir.glob('*.mp4')):
            m = get_video_meta(f)
            if m:
                metas.append(m)
            else:
                print(f'    WARN: {f.name}', flush=True)
        result['train'][label_dir.name] = summarize(metas)
        all_train.extend(metas)
    result['train']['_total'] = summarize(all_train)

    # ── TEST ──────────────────────────────────────────────────────────────
    print('Scanning TEST set...', flush=True)
    all_test = []
    for label_dir in sorted(TEST_BASE.iterdir()):
        if not label_dir.is_dir():
            continue
        print(f'  [{label_dir.name}]...', flush=True)
        subtype_data = {}
        all_label = []
        for sub in sorted(label_dir.iterdir()):
            if not sub.is_dir():
                continue
            print(f'    [{sub.name}]...', flush=True)
            metas = []
            for f in sorted(sub.glob('*.mp4')):
                m = get_video_meta(f)
                if m:
                    metas.append(m)
            subtype_data[sub.name] = summarize(metas)
            all_label.extend(metas)
        result['test'][label_dir.name] = {
            'subtypes': subtype_data,
            'all': summarize(all_label),
        }
        all_test.extend(all_label)
    result['test']['_total'] = summarize(all_test)

    # ── Save ──────────────────────────────────────────────────────────────
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\nDa luu: {OUT_JSON}')

    # ── Print summary ──────────────────────────────────────────────────────
    print_summary(result)


def print_summary(result):
    BOLD  = ''
    RESET = ''

    def pstat(name, s):
        if not s or 'count' not in s or s['count'] == 0:
            print(f'    {name}: (no data)')
            return
        dur = s.get('duration_s', {})
        sz  = s.get('size_mb', {})
        print(f'    {name}: {s["count"]} videos | '
              f'dur {dur.get("min","-")}-{dur.get("max","-")}s '
              f'(mean {dur.get("mean","-")}s) | '
              f'size {sz.get("min","-")}-{sz.get("max","-")} MB | '
              f'fps mean {s.get("fps",{}).get("mean","-")} | '
              f'total {s.get("total_duration_min","-")} min / {s.get("total_size_gb","-")} GB')

    print()
    print('=' * 80)
    print('TONG HOP TRAIN SET')
    print('=' * 80)
    for lbl, s in result['train'].items():
        if lbl == '_total':
            continue
        print(f'  [{lbl}]')
        pstat('', s)
    print(f'  [TOTAL]')
    pstat('', result['train']['_total'])

    print()
    print('=' * 80)
    print('TONG HOP TEST SET')
    print('=' * 80)
    for lbl, data in result['test'].items():
        if lbl == '_total':
            continue
        print(f'  [{lbl}]')
        if 'subtypes' in data:
            for sub, s in data['subtypes'].items():
                pstat(f'    {sub}', s)
            pstat('  LABEL TOTAL', data.get('all', {}))
        else:
            pstat('', data)
    print(f'  [TOTAL]')
    pstat('', result['test']['_total'])


if __name__ == '__main__':
    main()
