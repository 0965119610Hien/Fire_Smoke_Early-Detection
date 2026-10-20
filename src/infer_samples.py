"""
src/infer_samples.py
====================
Visual check of a fold's models on videos they never trained on: every test video +
some val videos of the fold, a few labeled frames per video spread over the video.

Each output image (one per frame): Original | Ground truth | Stage 1 | Stage 2 (full),
red = fire, cyan = smoke (argmax prediction), un-letterboxed back to the frame size,
with per-frame IoU in the header. Stage 2 sees the dense clip (7 context frames at
10 FPS) ending at that frame, exactly as in training / evaluation.

Run on Modal (GPU), images are downloaded to data/inference_samples_fold<k>/:
  modal run src/infer_samples.py --fold 0
  modal run src/infer_samples.py --fold 0 --val-videos 13 --frames-per-video 4
"""

import csv
import io
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIRE_BGR, SMOKE_BGR = (0, 0, 255), (255, 200, 0)
TILE_W = 480


def pick_samples(test_df, val_df, val_videos, frames_per_video, seed=0):
    """[(split, row index in its df)]: every test video + `val_videos` val videos
    (positive first, then hard / normal negatives), `frames_per_video` anchors each."""
    rng = np.random.default_rng(seed)
    kind = lambda v: 2 if v.startswith("neg_normal") else (1 if v.startswith("neg_hard") else 0)
    vv = sorted(val_df["video_name"].unique(), key=lambda v: (kind(v), v))
    n_neg = max(1, val_videos // 4)
    pos = [v for v in vv if kind(v) == 0]
    neg = [v for v in vv if kind(v) > 0]
    chosen_val = list(rng.choice(pos, size=min(len(pos), val_videos - n_neg), replace=False)) + \
        list(rng.choice(neg, size=min(len(neg), n_neg), replace=False))
    out = []
    for split, df, vids in (("test", test_df, sorted(test_df["video_name"].unique())), ("val", val_df, sorted(chosen_val))):
        for v in vids:
            rows = df[df["video_name"] == v].sort_values("anchor_raw_idx")
            idx = np.unique(np.linspace(0, len(rows) - 1, min(frames_per_video, len(rows))).round().astype(int))
            out += [(split, rows.index[i]) for i in idx]
    return out


def unletterbox(prob, h, w, size=512):
    """(C, size, size) model-resolution probabilities -> (C, h, w) on the original frame."""
    import cv2
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    crop = prob[:, top:top + nh, left:left + nw]
    return np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in crop])


def iou(pred, gt, c):
    p, g = pred == c, gt == c
    union = (p | g).sum()
    return float((p & g).sum() / union) if union else float("nan")


def overlay(img, mask, title):
    import cv2
    out = img.copy()
    color = img.copy()
    color[mask == 1] = FIRE_BGR
    color[mask == 2] = SMOKE_BGR
    out = cv2.addWeighted(out, 0.55, color, 0.45, 0)
    for c, col in ((1, FIRE_BGR), (2, SMOKE_BGR)):
        cnts, _ = cv2.findContours((mask == c).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, col, 2)
    th = int(round(TILE_W * img.shape[0] / img.shape[1]))
    out = cv2.resize(out, (TILE_W, th), interpolation=cv2.INTER_AREA)
    head = np.full((44, TILE_W, 3), 30, np.uint8)
    for k, line in enumerate(title.split("\n")[:2]):
        cv2.putText(head, line, (6, 17 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, out])


def fmt(x):
    return "-" if np.isnan(x) else f"{100 * x:.0f}%"


def run(data_root, splits_dir, stage1_ckpt, stage2_ckpt, val_videos=13, frames_per_video=4):
    import cv2
    import pandas as pd
    import torch
    import fire_core as fc
    from evaluate_test import load_models, predict_probs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, cfg, meta = load_models(stage1_ckpt, stage2_ckpt, device)
    dfs = {s: pd.read_csv(Path(splits_dir) / f"dense_{s}_clips.csv") for s in ("test", "val")}
    ds = {s: {"stage1": fc.VideoClipDataset(Path(splits_dir) / f"dense_{s}_clips.csv", data_root, use_motion=False),
              "stage2": fc.VideoClipDataset(Path(splits_dir) / f"dense_{s}_clips.csv", data_root,
                                            use_motion=cfg.get("use_motion", True), letterbox=cfg.get("letterbox", True))}
          for s in dfs}
    samples = pick_samples(dfs["test"], dfs["val"], val_videos, frames_per_video)
    print(f"{len(samples)} frame / {len({(s, dfs[s].loc[i, 'video_name']) for s, i in samples})} video", flush=True)

    files, rows = {}, []
    for n, (split, i) in enumerate(samples, 1):
        r = dfs[split].loc[i]
        img = fc.read_image(Path(data_root) / r["image_paths"].split(";")[-1])
        gt = fc.read_mask(Path(data_root) / r["mask_paths"].split(";")[-1])
        h, w = img.shape[:2]
        preds = {}
        for name, key in (("stage1_framewise", "stage1"), ("stage2_full", "stage2")):
            clip, _ = ds[split][key][i]
            prob = predict_probs(name, models[name], clip, device)[-1]
            preds[name] = unletterbox(prob, h, w).argmax(0).astype(np.uint8)
        rec = {"split": split, "video": r["video_name"], "frame": r["anchor_frame"], "category": r["category"]}
        tiles = [overlay(img, np.zeros_like(gt), f"{split} | {r['video_name'][:44]}\n{r['anchor_frame']} | {w}x{h}"),
                 overlay(img, gt, f"Ground truth ({r['category']})\nfire {int((gt == 1).sum())} px | smoke {int((gt == 2).sum())} px")]
        for name, label in (("stage1_framewise", "Stage 1"), ("stage2_full", "Stage 2 (temporal)")):
            p = preds[name]
            fi, si = iou(p, gt, 1), iou(p, gt, 2)
            rec[f"{name}_iou_fire"], rec[f"{name}_iou_smoke"] = fi, si
            tiles.append(overlay(img, p, f"{label}\nIoU fire {fmt(fi)} | smoke {fmt(si)}"))
        panel = np.hstack(tiles)
        ok, buf = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rel = f"{split}/{r['video_name']}/{r['anchor_frame']}.jpg"
        files[rel] = buf.tobytes()
        rec["image"] = rel
        rows.append(rec)
        print(f"  [{n}/{len(samples)}] {rel}", flush=True)
    return {"files": files, "rows": rows, "checkpoints": {k: {"run_id": v["run_id"], "epoch": v["epoch"]} for k, v in meta.items()}}


def save_local(res, out_dir):
    out_dir = Path(out_dir)
    for rel, data in res["files"].items():
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(res["rows"][0].keys()))
        w.writeheader()
        w.writerows(res["rows"])
    print(f"💾 {len(res['files'])} ảnh -> {out_dir} (checkpoints: {res['checkpoints']})")


try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("fire-smoke-infer-samples")
    volume = modal.Volume.from_name("fire_smoke_dataset_v2")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch==2.2.2", "torchvision==0.17.2", "transformers==4.40.0", "opencv-python-headless",
                     "numpy<2.0.0", "pandas", "Pillow")
        .add_local_file(SRC / "fire_core.py", "/root/fire_core.py")
        .add_local_file(SRC / "evaluate_test.py", "/root/evaluate_test.py")
        .add_local_file(SRC / "metrics_calculator.py", "/root/metrics_calculator.py")
    )

    @app.function(image=image, volumes={"/data": volume}, gpu="A10G", timeout=3600, memory=16384)
    def infer_remote(fold: int, stage1_ckpt: str, stage2_ckpt: str, val_videos: int, frames_per_video: int):
        splits_dir = "/data/data_splits" + (f"/folds/fold{fold}" if fold >= 0 else "")
        return run("/data/dataset_video_unified", splits_dir, stage1_ckpt, stage2_ckpt, val_videos, frames_per_video)

    @app.local_entrypoint()
    def main(fold: int = 0, val_videos: int = 13, frames_per_video: int = 4, stage1_ckpt: str = "", stage2_ckpt: str = ""):
        tag = f"_fold{fold}" if fold >= 0 else ""
        stage1_ckpt = stage1_ckpt or f"/data/checkpoints/segformer_spatial_best{tag}.pth"
        stage2_ckpt = stage2_ckpt or f"/data/checkpoints/timesformer_dense{tag}_spatiotemporal_best.pth"
        res = infer_remote.remote(fold, stage1_ckpt, stage2_ckpt, val_videos, frames_per_video)
        save_local(res, ROOT / "data" / f"inference_samples{tag}")
