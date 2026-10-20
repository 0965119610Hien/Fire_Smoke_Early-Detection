import pandas as pd
from pathlib import Path

df = pd.read_csv('crawled_early_fire_transition/crawled_manifest.csv')
vids_dir = Path('crawled_early_fire_transition/videos')

# Filter videos that have duration > 30s so they have significant intro/outro
long_vids = df[df['duration_s'] >= 35]

print("Candidate long videos with news/cctv/test:")
for idx, r in long_vids.head(15).iterrows():
    f = vids_dir / r['filename']
    if f.exists():
        clean_title = "".join(c for c in str(r['title']) if ord(c) < 128)[:60]
        print(f"ID: {r['video_id']:12s} Cat: {r['category']:24s} Dur: {r['duration_s']:3d}s Title: {clean_title}")
