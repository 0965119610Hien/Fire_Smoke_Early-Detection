import json

with open('dataset_stats.json', encoding='utf-8') as f:
    d = json.load(f)

train = d['train']
test  = d['test']

print('=== TRAIN - Per label summary ===')
for lbl in ['0_normal_noise','1_controlled_fire','2_uncontrolled_hazard']:
    s   = train[lbl]
    dur = s.get('duration_s', {})
    fps = s.get('fps', {})
    sz  = s.get('size_mb', {})
    rd  = s.get('res_dominant', {})
    print(f'[{lbl}]')
    print(f'  count          : {s["count"]}')
    print(f'  duration (s)   : min={dur.get("min")} max={dur.get("max")} mean={dur.get("mean")} stdev={dur.get("stdev")}')
    print(f'  size (MB)      : min={sz.get("min")} max={sz.get("max")} mean={sz.get("mean")}')
    print(f'  fps mean       : {fps.get("mean")} stdev={fps.get("stdev")}')
    print(f'  total dur (min): {s.get("total_duration_min")}')
    print(f'  total size (GB): {s.get("total_size_gb")}')
    print(f'  top resolutions: {list(rd.items())[:5]}')
    print()

tot = train['_total']
print('=== TRAIN TOTAL ===')
print(f'  count: {tot["count"]}  dur_mean={tot["duration_s"]["mean"]}s  total={tot["total_duration_min"]}min / {tot["total_size_gb"]}GB')
print()

print('=== TEST - Per label summary ===')
for lbl in ['0_normal_noise','1_controlled_fire','2_uncontrolled_hazard']:
    ld = test[lbl]['all']
    dur = ld.get('duration_s', {})
    fps = ld.get('fps', {})
    sz  = ld.get('size_mb', {})
    rd  = ld.get('res_dominant', {})
    print(f'[{lbl}]')
    print(f'  count          : {ld["count"]}')
    print(f'  duration (s)   : min={dur.get("min")} max={dur.get("max")} mean={dur.get("mean")} stdev={dur.get("stdev")}')
    print(f'  size (MB)      : min={sz.get("min")} max={sz.get("max")} mean={sz.get("mean")}')
    print(f'  fps mean       : {fps.get("mean")} stdev={fps.get("stdev")}')
    print(f'  total dur (min): {ld.get("total_duration_min")}')
    print(f'  total size (GB): {ld.get("total_size_gb")}')
    print(f'  top resolutions: {list(rd.items())[:5]}')
    print()
    print('  Subtypes:')
    for sub, sv in test[lbl]['subtypes'].items():
        sd = sv.get('duration_s', {})
        print(f'    {sub}: {sv["count"]} videos, dur mean={sd.get("mean")}s ({sd.get("min")}-{sd.get("max")}s)')
    print()

tot2 = test['_total']
print('=== TEST TOTAL ===')
print(f'  count: {tot2["count"]}  dur_mean={tot2["duration_s"]["mean"]}s  total={tot2["total_duration_min"]}min / {tot2["total_size_gb"]}GB')
