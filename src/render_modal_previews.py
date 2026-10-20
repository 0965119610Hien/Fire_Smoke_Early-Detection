"""
render_modal_previews.py
========================
Tự động ghép 20 video MP4 mẫu từ dữ liệu gán nhãn SegFormer-B2 trên Modal Volume.
"""
import modal

app = modal.App("render-segformer-previews")
vol = modal.Volume.from_name("fire_smoke_dataset")

modal_img = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("opencv-python-headless", "numpy<2.0.0")
)

@app.function(image=modal_img, volumes={"/data": vol}, timeout=600)
def render_samples_on_modal():
    import cv2
    from pathlib import Path
    
    labels_dir = Path("/data/annotation_labels")
    chunk_dirs = sorted([d for d in labels_dir.glob("*") if d.is_dir()])
    
    video_map = {}
    for d in chunk_dirs:
        stem = d.name.split("_chunk_")[0]
        if stem not in video_map:
            video_map[stem] = []
        video_map[stem].append(d)
        
    selected_videos = list(video_map.keys())[:20]
    out_dir = Path("/data/sample_video_previews")
    out_dir.mkdir(exist_ok=True)
    
    results = []
    for vid_idx, vid_stem in enumerate(selected_videos, 1):
        chunks = video_map[vid_stem]
        all_vis = []
        for c_dir in chunks:
            all_vis.extend(sorted(c_dir.glob("*_vis.jpg")))
        if not all_vis:
            continue
        first_img = cv2.imread(str(all_vis[0]))
        if first_img is None:
            continue
        h, w = first_img.shape[:2]
        
        out_mp4 = out_dir / f"{vid_idx:02d}_{vid_stem}_annotated.mp4"
        writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (w, h))
        for vf in all_vis:
            img = cv2.imread(str(vf))
            if img is not None:
                writer.write(img)
        writer.release()
        results.append(out_mp4.name)
    vol.commit()
    return len(results)

@app.local_entrypoint()
def main():
    n = render_samples_on_modal.remote()
    print(f"[OK] Rendered {n} sample MP4s from SegFormer-B2 trained labels!")
