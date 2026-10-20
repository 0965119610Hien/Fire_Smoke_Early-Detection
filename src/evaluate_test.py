"""
src/evaluate_test.py
====================
Canonical test-set evaluation + ablation. Produces ONE consistent result file
(docs/reports/eval_<run_id>.json + .md) used to write the report.

Configurations (same video test set):
- stage1_framewise : Stage 1 checkpoint, every frame processed independently.
- stage2_gamma0    : Stage 2 checkpoint with gamma forced to 0 (effect of the decode-head fine-tune).
- stage2_full      : Stage 2 checkpoint as trained.

Protocol:
- Every unique frame (video, frame) is counted ONCE. For Stage 2 the prediction is
  taken from the clip where the frame is at its LATEST position (causal streaming).
- Pixel level: global 3x3 confusion matrix (pad 255 ignored) -> IoU / Dice / Fg-mIoU / All-mIoU.
- Frame level: frame positive for class c if area(prob_c > tau) >= max(A_min, 1) px
  (at the 512x512 model resolution). (tau, A_min) selected on VAL (recall >= 0.95 with the
  lowest FAR, else best F1) and then FIXED for test.
- Video level: video positive if >= k consecutive positive frames (k selected on val).
- Object level: LTPR / LPPV / S after removing predicted regions smaller than A_min.
- Bootstrap 95% CI (1000 resamples over VIDEOS).
- Per-video table, speed (CPU preprocessing / GPU per clip / streaming with feature cache).

Local CPU debug:
  python src/evaluate_test.py --limit 2
Full run on Modal:
  modal run src/evaluate_test.py
"""

import argparse
import copy
import json
import re
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TAUS = (0.3, 0.4, 0.5, 0.6, 0.7)
A_MINS = (0, 50, 200, 500, 1000, 2000)
KS = (1, 2, 3)
FG = {1: "fire", 2: "smoke"}
CONFIG_NAMES = ("stage1_framewise", "stage2_gamma0", "stage2_full")
TARGET_RECALL = 0.95
N_BOOT = 1000


def local_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        return f"{commit}-dirty" if dirty else commit
    except Exception:
        return "unknown"


# =============================================================================
# Pure helpers (numpy only)
# =============================================================================
def frame_index_of(frame_name):
    digits = re.findall(r"\d+", str(frame_name))
    return int(digits[-1]) if digits else 0


def assign_latest_positions(clips_df):
    """Map every unique frame (video, frame_name) to the (clip_idx, position) where it
    appears LATEST in its clip. Unlabeled context frames (dense clips) are skipped.
    Returns {clip_idx: [positions to keep]}."""
    from fire_core import clip_frame_keys, clip_labeled_positions
    best = {}
    for ci in range(len(clips_df)):
        row = clips_df.iloc[ci]
        keys = clip_frame_keys(row)
        for pos in clip_labeled_positions(row):
            key = keys[pos]
            if key not in best or pos >= best[key][1]:
                best[key] = (ci, pos)
    needed = defaultdict(set)
    for ci, pos in best.values():
        needed[ci].add(pos)
    return {ci: sorted(p) for ci, p in needed.items()}


def binary_counts(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    return {"TP": int((y_true & y_pred).sum()), "FP": int((~y_true & y_pred).sum()),
            "TN": int((~y_true & ~y_pred).sum()), "FN": int((y_true & ~y_pred).sum())}


def rates(c):
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    nan = float("nan")
    recall = tp / (tp + fn) if tp + fn else nan
    far = fp / (fp + tn) if fp + tn else nan
    prec = tp / (tp + fp) if tp + fp else nan
    f1 = 2 * prec * recall / (prec + recall) if tp else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    total = tp + fp + tn + fn
    return {**c, "Recall": recall, "FAR": far, "Precision": prec, "F1": f1, "MCC": float(mcc),
            "BaseRate": (tp + fn) / total if total else nan, "N": total}


def choose_operating_point(cands):
    """cands: [(params, rates)]. Recall >= TARGET_RECALL with lowest FAR, else best F1."""
    ok = [c for c in cands if not np.isnan(c[1]["Recall"]) and c[1]["Recall"] >= TARGET_RECALL]
    if ok:
        return min(ok, key=lambda c: (np.nan_to_num(c[1]["FAR"], nan=1.0), -c[1]["Recall"]))
    return max(cands, key=lambda c: (c[1]["F1"], -np.nan_to_num(c[1]["FAR"], nan=1.0)))


def frame_pred(rec, c, tau_i, a_min):
    area = rec["area"][c][tau_i]
    return area > 0 and area >= a_min


def max_run(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def video_level(records, c, tau_i, a_min, k):
    by_vid = defaultdict(list)
    for r in records:
        by_vid[r["video"]].append(r)
    y_true, y_pred = [], []
    for vid, rs in by_vid.items():
        rs = sorted(rs, key=lambda r: r["idx"])
        y_true.append(any(r["gt"][c] for r in rs))
        y_pred.append(max_run([frame_pred(r, c, tau_i, a_min) for r in rs]) >= k)
    return rates(binary_counts(y_true, y_pred))


def select_thresholds(val_records):
    """Per class: (tau, A_min) at frame level, then k at video level, all on VAL."""
    sel = {}
    for c in FG:
        cands = []
        for ti, tau in enumerate(TAUS):
            for a in A_MINS:
                y_true = [r["gt"][c] for r in val_records]
                y_pred = [frame_pred(r, c, ti, a) for r in val_records]
                cands.append(((ti, a), rates(binary_counts(y_true, y_pred))))
        (ti, a), fr = choose_operating_point(cands)
        kc = [((k,), video_level(val_records, c, ti, a, k)) for k in KS]
        (k,), vr = choose_operating_point(kc)
        sel[c] = {"tau": TAUS[ti], "tau_index": ti, "A_min": a, "k": k, "val_frame": fr, "val_video": vr}
    return sel


def nan_percentiles(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if not len(x):
        return [float("nan"), float("nan")]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def bootstrap(records, sel, n_boot=N_BOOT, seed=0):
    from fire_core import iou_dice_from_cm
    vids = sorted({r["video"] for r in records})
    per_vid = {}
    for v in vids:
        rs = [r for r in records if r["video"] == v]
        counts = {}
        for c, s in sel.items():
            counts[c] = binary_counts([r["gt"][c] for r in rs], [frame_pred(r, c, s["tau_index"], s["A_min"]) for r in rs])
        per_vid[v] = (sum(r["cm"] for r in rs), counts)

    rng = np.random.default_rng(seed)
    out = defaultdict(list)
    for _ in range(n_boot):
        sample = rng.choice(len(vids), size=len(vids), replace=True)
        cm = sum(per_vid[vids[i]][0] for i in sample)
        m = iou_dice_from_cm(cm)
        out["fg_miou"].append(m["fg_miou"])
        out["iou_fire"].append(m["iou_fire"])
        out["iou_smoke"].append(m["iou_smoke"])
        for c, name in FG.items():
            tot = {k: sum(per_vid[vids[i]][1][c][k] for i in sample) for k in ("TP", "FP", "TN", "FN")}
            r = rates(tot)
            out[f"{name}_frame_recall"].append(r["Recall"])
            out[f"{name}_frame_far"].append(r["FAR"])
    return {k: nan_percentiles(v) for k, v in out.items()}


# =============================================================================
# Model loading / inference
# =============================================================================
def load_models(stage1_ckpt, stage2_ckpt, device):
    import torch
    import fire_core as fc

    s1 = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    stage1 = fc.build_segformer_b2(pretrained=False)
    stage1.load_state_dict(s1["model_state_dict"])
    stage1.to(device).eval()

    s2 = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
    cfg = s2.get("config")
    legacy = cfg is None
    if legacy:
        # Pre-fix checkpoints: bidirectional temporal module, no pos_embed, plain resize.
        cfg = {"causal": False, "temporal_hidden": 128, "num_heads": 4, "max_T": 16,
               "letterbox": False, "use_motion": True, "T": 8, "preprocess_version": "legacy"}
        print("⚠️ Checkpoint Stage 2 cũ (không có config): đánh giá ở chế độ legacy "
              "(causal=False, pos_embed=0, resize thẳng). Kết quả CHỈ để tham khảo.")
    base = fc.build_segformer_b2(pretrained=False)
    stage2 = fc.SpatioTemporalSegFormer(base, temporal_hidden=cfg["temporal_hidden"], num_heads=cfg["num_heads"],
                                        max_T=cfg.get("max_T", 16), causal=cfg["causal"], verbose=False)
    fc.load_stage2_state_dict(stage2, s2["model_state_dict"], legacy_ok=legacy)
    stage2.to(device).eval()
    gamma0 = copy.deepcopy(stage2)
    with torch.no_grad():
        gamma0.temporal_refinement.gamma.zero_()

    meta = {
        "stage1": {"path": str(stage1_ckpt), "run_id": s1.get("run_id", "legacy"), "epoch": s1.get("epoch"),
                   "config": s1.get("config")},
        "stage2": {"path": str(stage2_ckpt), "run_id": s2.get("run_id", "legacy"), "epoch": s2.get("epoch"),
                   "gamma": float(stage2.temporal_refinement.gamma.item()), "config": s2.get("config"),
                   "legacy": legacy, "eval_model_config": cfg},
    }
    return {"stage1_framewise": stage1, "stage2_gamma0": gamma0, "stage2_full": stage2}, cfg, meta


def predict_probs(name, model, clip, device):
    """clip: (T, 3, H, W) -> probs (T, 3, H, W) float32 numpy."""
    import torch
    import torch.nn.functional as F
    use_amp = device.type == "cuda"
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        x = clip.to(device)
        if name == "stage1_framewise":
            logits = F.interpolate(model(pixel_values=x).logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        else:
            logits = model(x.unsqueeze(0))[0]
    return torch.softmax(logits.float(), dim=1).cpu().numpy()


def frame_record(video, frame_name, prob, gt):
    from fire_core import confusion_matrix_3c, IGNORE_INDEX
    valid = gt != IGNORE_INDEX
    return {
        "video": video, "frame": frame_name, "idx": frame_index_of(frame_name),
        "cm": confusion_matrix_3c(prob.argmax(0), gt),
        "area": {c: [int(((prob[c] > t) & valid).sum()) for t in TAUS] for c in FG},
        "gt": {c: bool((gt == c).any()) for c in FG},
    }


def localization_update(acc, prob, gt, sel):
    import metrics_calculator as mc
    for c, s in sel.items():
        pred_bin = (prob[c] > s["tau"]).astype(np.uint8)
        blobs = mc.extract_connected_components(pred_bin, min_area=max(s["A_min"], 16))
        filtered = np.zeros_like(pred_bin)
        for b in blobs:
            filtered |= b
        r = mc.compute_localization_metrics([filtered * c], [gt], target_class=c, min_area=16)
        a = acc[c]
        a["gt"] += r["Total_GT_Regions"]
        a["det"] += r["Detected_GT_Regions"]
        a["pred"] += r["Total_Pred_Regions"]
        a["correct"] += r["Correct_Pred_Regions"]
        a["sim_sum"] += r["Mean_Similarity_S"] * r["Num_Matches"]
        a["matches"] += r["Num_Matches"]


def run_split(datasets, models, clips_df, device, sel_by_cfg=None, limit=0, desc=""):
    """Evaluate all configs on one split. Returns {cfg: records}, {cfg: localization acc}."""
    needed = assign_latest_positions(clips_df)
    clip_ids = sorted(needed)
    if limit:
        clip_ids = clip_ids[:limit]
    records = {n: [] for n in models}
    loc = {n: {c: defaultdict(float) for c in FG} for n in models}
    t0 = time.time()
    for j, ci in enumerate(clip_ids):
        keys = datasets["stage2"].frame_keys(ci)
        cache = {}
        for name, model in models.items():
            ds_key = "stage1" if name == "stage1_framewise" else "stage2"
            if ds_key not in cache:
                cache[ds_key] = datasets[ds_key][ci]
            clip, masks = cache[ds_key]
            probs = predict_probs(name, model, clip, device)
            for pos in needed[ci]:
                gt = masks[pos].numpy().astype(np.uint8)
                records[name].append(frame_record(keys[pos][0], keys[pos][1], probs[pos], gt))
                if sel_by_cfg is not None:
                    localization_update(loc[name], probs[pos], gt, sel_by_cfg[name])
        print(f"  [{desc}] clip {j + 1}/{len(clip_ids)} ({time.time() - t0:.0f}s)", flush=True)
    return records, loc


# =============================================================================
# Metrics on records
# =============================================================================
def summarize_test(records, sel, loc):
    from fire_core import iou_dice_from_cm
    cm = sum(r["cm"] for r in records)
    pixel = iou_dice_from_cm(cm)
    pixel["confusion_matrix"] = cm.tolist()

    frame, video, objects = {}, {}, {}
    alarm_true, alarm_pred = [], []
    for c, name in FG.items():
        s = sel[c]
        y_true = [r["gt"][c] for r in records]
        y_pred = [frame_pred(r, c, s["tau_index"], s["A_min"]) for r in records]
        frame[name] = rates(binary_counts(y_true, y_pred))
        video[name] = {"k": s["k"], **video_level(records, c, s["tau_index"], s["A_min"], s["k"])}
        a = loc[c]
        objects[name] = {
            "LTPR": a["det"] / a["gt"] if a["gt"] else float("nan"),
            "LPPV": a["correct"] / a["pred"] if a["pred"] else float("nan"),
            "S": a["sim_sum"] / a["matches"] if a["matches"] else float("nan"),
            "gt_regions": int(a["gt"]), "pred_regions": int(a["pred"]), "min_pred_area": max(s["A_min"], 16),
        }
        alarm_true.append(y_true)
        alarm_pred.append(y_pred)
    frame["any"] = rates(binary_counts(np.any(alarm_true, axis=0), np.any(alarm_pred, axis=0)))

    per_video = []
    by_vid = defaultdict(list)
    for r in records:
        by_vid[r["video"]].append(r)
    for vid, rs in sorted(by_vid.items()):
        m = iou_dice_from_cm(sum(r["cm"] for r in rs))
        row = {"video": vid, "frames": len(rs), "iou_bg": m["iou_bg"], "iou_fire": m["iou_fire"],
               "iou_smoke": m["iou_smoke"]}
        for c, name in FG.items():
            s = sel[c]
            cnt = binary_counts([r["gt"][c] for r in rs], [frame_pred(r, c, s["tau_index"], s["A_min"]) for r in rs])
            row[f"{name}_FP_frames"] = cnt["FP"]
            row[f"{name}_FN_frames"] = cnt["FN"]
        per_video.append(row)

    return {"num_frames": len(records), "num_videos": len(by_vid), "pixel": pixel, "frame": frame,
            "video": video, "object": objects, "bootstrap_ci95": bootstrap(records, sel), "per_video": per_video}


# =============================================================================
# Speed
# =============================================================================
def measure_speed(models, dataset, device, T, runs):
    import torch
    import fire_core as fc

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    out = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "amp_fp16": device.type == "cuda", "batch": 1, "T": T, "runs": runs}

    # (a) CPU preprocessing per frame
    row = dataset.df.iloc[0]
    paths = row["image_paths"].split(";")
    times, prev = [], None
    for p in paths[: max(runs, 1)]:
        img = fc.read_image(dataset.data_root / p)
        t = time.perf_counter()
        base = fc.enhance_base(img)
        x = fc.apply_motion(base, prev) if dataset.use_motion else base
        prev = base
        x = fc.letterbox_image_and_mask(x, None, dataset.target_size)[0] if dataset.letterbox else x
        fc.normalize_to_chw(x)
        times.append(time.perf_counter() - t)
    out["preprocess_ms_per_frame_cpu"] = 1000 * float(np.mean(times))
    out["preprocess_source_resolution"] = list(img.shape[:2])

    amp = dict(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")
    H, W = dataset.target_size
    clip = torch.randn(1, T, 3, H, W, device=device)

    def timed(fn, n):
        fn()
        sync()
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            sync()
            ts.append(time.perf_counter() - t)
        return 1000 * float(np.mean(ts))

    with torch.no_grad(), torch.autocast(**amp):
        s1, s2 = models["stage1_framewise"], models["stage2_full"]
        # (b) model time per clip of T frames
        out["stage1_ms_per_clip"] = timed(lambda: s1(pixel_values=clip[0]), runs)
        out["stage2_ms_per_clip"] = timed(lambda: s2(clip), runs)
        # (c) streaming, stride 1: per new frame
        frame = clip[:, 0]
        out["stage1_stream_ms_per_frame"] = timed(lambda: s1(pixel_values=frame), runs)
        cache = deque([s2.encode_frames(clip[0, i:i + 1]) for i in range(T)], maxlen=T)

        def stream_step():
            cache.append(s2.encode_frames(frame))       # encoder + linear_fuse once per frame
            fused = torch.cat(list(cache), dim=0)
            z = s2.temporal_refinement(fused, 1, len(cache))[-1:]  # temporal over the window
            logits = s2.decode_head.classifier(z)                   # classifier for the newest frame only
            return torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        out["stage2_stream_ms_per_frame_cached"] = timed(stream_step, runs)
    out["stage2_stream_fps_cached"] = 1000.0 / out["stage2_stream_ms_per_frame_cached"]
    out["stage1_stream_fps"] = 1000.0 / out["stage1_stream_ms_per_frame"]
    return out


# =============================================================================
# Main evaluation
# =============================================================================
def run_evaluation(data_root, splits_dir, stage1_ckpt, stage2_ckpt, limit=0, speed_runs=20, git_commit="unknown",
                   dense_root=None):
    import torch
    import fire_core as fc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, cfg, meta = load_models(stage1_ckpt, stage2_ckpt, device)
    T = int(cfg.get("T", 8))
    # the clip layout follows the Stage 2 checkpoint; all configs see the same labeled frames
    clip_mode = cfg.get("clip_mode", "sparse")
    prefix = "dense_" if clip_mode != "sparse" else ""
    if clip_mode != "sparse":
        data_root = dense_root or data_root
    print(f"ℹ️ clip_mode = {clip_mode} | data_root = {data_root}")

    def make_datasets(split):
        csv = Path(splits_dir) / f"{prefix}{split}_clips.csv"
        return {
            "stage1": fc.VideoClipDataset(csv, data_root, use_motion=False, letterbox=True),
            "stage2": fc.VideoClipDataset(csv, data_root, use_motion=cfg.get("use_motion", True),
                                          letterbox=cfg.get("letterbox", True)),
        }

    print("▶ VAL: chọn ngưỡng (tau, A_min, k)...")
    val_ds = make_datasets("val")
    val_records, _ = run_split(val_ds, models, val_ds["stage2"].df, device, limit=limit, desc="val")
    selections = {n: select_thresholds(val_records[n]) for n in models}

    print("▶ TEST: đánh giá với ngưỡng cố định...")
    test_ds = make_datasets("test")
    test_records, loc = run_split(test_ds, models, test_ds["stage2"].df, device, sel_by_cfg=selections,
                                  limit=limit, desc="test")

    results = {}
    for n in models:
        results[n] = {
            "selection_on_val": {FG[c]: s for c, s in selections[n].items()},
            "test": summarize_test(test_records[n], selections[n], loc[n]),
        }

    print("▶ Đo tốc độ...")
    speed = measure_speed(models, test_ds["stage2"], device, T, speed_runs)

    run_id = meta["stage2"]["run_id"]
    return {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_git_commit": git_commit,
        "limit_clips": limit,
        "clip_mode": clip_mode,
        "data_root": str(data_root),
        "protocol": {
            "unique_frames": "each (video, frame) counted once; Stage 2 prediction from the clip where the frame is at its latest position",
            "frame_positive": "area(prob_c > tau) >= max(A_min, 1) pixels at 512x512 model resolution, pad excluded",
            "selection": f"on val: recall >= {TARGET_RECALL} with lowest FAR, else max F1",
            "taus": TAUS, "A_mins": A_MINS, "ks": KS, "bootstrap": f"{N_BOOT} resamples over videos",
        },
        "checkpoints": meta,
        "results": results,
        "speed": speed,
    }


# =============================================================================
# Report writing
# =============================================================================
def _pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.2f}"


def _ci(ci):
    return f"[{_pct(ci[0])}, {_pct(ci[1])}]"


def to_markdown(res):
    L = [f"# Evaluation — run `{res['run_id']}` ({res.get('clip_mode', 'sparse')})", "",
         f"- Generated: {res['generated_at']} | eval git: `{res['eval_git_commit']}` | limit clips: {res['limit_clips'] or 'none'}",
         f"- Stage 1: `{res['checkpoints']['stage1']['path']}` (run {res['checkpoints']['stage1']['run_id']}, epoch {res['checkpoints']['stage1']['epoch']})",
         f"- Stage 2: `{res['checkpoints']['stage2']['path']}` (run {res['checkpoints']['stage2']['run_id']}, epoch {res['checkpoints']['stage2']['epoch']}, gamma {res['checkpoints']['stage2']['gamma']:.4f}, legacy={res['checkpoints']['stage2']['legacy']})",
         f"- Protocol: {res['protocol']['unique_frames']}; {res['protocol']['frame_positive']}; thresholds {res['protocol']['selection']}.",
         "", "## Pixel level (test, %)", "",
         "| Config | Frames | Videos | IoU BG | IoU Fire | IoU Smoke | Dice Fire | Dice Smoke | Fg-mIoU [95% CI] | All-mIoU |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in res["results"].items():
        t, p, ci = r["test"], r["test"]["pixel"], r["test"]["bootstrap_ci95"]
        L.append(f"| {n} | {t['num_frames']} | {t['num_videos']} | {_pct(p['iou_bg'])} | {_pct(p['iou_fire'])} {_ci(ci['iou_fire'])} | "
                 f"{_pct(p['iou_smoke'])} {_ci(ci['iou_smoke'])} | {_pct(p['dice_fire'])} | {_pct(p['dice_smoke'])} | "
                 f"{_pct(p['fg_miou'])} {_ci(ci['fg_miou'])} | {_pct(p['all_miou'])} |")

    L += ["", "## Frame level (test; thresholds fixed from val)", "",
          "| Config | Class | tau | A_min | Base rate | TP | FP | TN | FN | Recall [95% CI] | FAR [95% CI] | Precision | F1 | MCC |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in res["results"].items():
        for name in list(FG.values()) + ["any"]:
            f = r["test"]["frame"][name]
            s = r["selection_on_val"].get(name, {})
            ci = r["test"]["bootstrap_ci95"]
            rc = _ci(ci[f"{name}_frame_recall"]) if name != "any" else ""
            fc_ = _ci(ci[f"{name}_frame_far"]) if name != "any" else ""
            L.append(f"| {n} | {name} | {s.get('tau', '-')} | {s.get('A_min', '-')} | {_pct(f['BaseRate'])} | {f['TP']} | {f['FP']} | "
                     f"{f['TN']} | {f['FN']} | {_pct(f['Recall'])} {rc} | {_pct(f['FAR'])} {fc_} | {_pct(f['Precision'])} | "
                     f"{_pct(f['F1'])} | {f['MCC']:.3f} |")

    L += ["", "## Video level (test)", "", "| Config | Class | k | TP | FP | TN | FN | Recall | FAR |", "|---|---|---|---|---|---|---|---|---|"]
    for n, r in res["results"].items():
        for name, v in r["test"]["video"].items():
            L.append(f"| {n} | {name} | {v['k']} | {v['TP']} | {v['FP']} | {v['TN']} | {v['FN']} | {_pct(v['Recall'])} | {_pct(v['FAR'])} |")

    L += ["", "## Object level (test)", "", "| Config | Class | LTPR | LPPV | S | GT regions | Pred regions |", "|---|---|---|---|---|---|---|"]
    for n, r in res["results"].items():
        for name, o in r["test"]["object"].items():
            L.append(f"| {n} | {name} | {_pct(o['LTPR'])} | {_pct(o['LPPV'])} | {_pct(o['S'])} | {o['gt_regions']} | {o['pred_regions']} |")

    sp = res["speed"]
    L += ["", "## Speed", "",
          f"- Device: {sp['device']} | batch {sp['batch']} | T {sp['T']} | AMP fp16: {sp['amp_fp16']} | runs {sp['runs']}",
          f"- (a) CPU preprocessing: {sp['preprocess_ms_per_frame_cpu']:.1f} ms/frame (source {sp['preprocess_source_resolution']})",
          f"- (b) Model per clip: Stage 1 {sp['stage1_ms_per_clip']:.1f} ms | Stage 2 {sp['stage2_ms_per_clip']:.1f} ms",
          f"- (c) Streaming stride 1: Stage 1 {sp['stage1_stream_ms_per_frame']:.1f} ms/frame ({sp['stage1_stream_fps']:.1f} FPS) | "
          f"Stage 2 with feature cache {sp['stage2_stream_ms_per_frame_cached']:.1f} ms/frame ({sp['stage2_stream_fps_cached']:.1f} FPS)",
          "", "## Per video (stage2_full, test)", "",
          "| Video | Frames | IoU BG | IoU Fire | IoU Smoke | Fire FP fr. | Fire FN fr. | Smoke FP fr. | Smoke FN fr. |",
          "|---|---|---|---|---|---|---|---|---|"]
    for row in res["results"]["stage2_full"]["test"]["per_video"]:
        L.append(f"| {row['video']} | {row['frames']} | {_pct(row['iou_bg'])} | {_pct(row['iou_fire'])} | {_pct(row['iou_smoke'])} | "
                 f"{row['fire_FP_frames']} | {row['fire_FN_frames']} | {row['smoke_FP_frames']} | {row['smoke_FN_frames']} |")
    return "\n".join(L) + "\n"


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def write_outputs(res, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "" if res.get("clip_mode", "sparse") == "sparse" else f"_{res['clip_mode']}"
    stem = f"eval_{res['run_id']}{mode}" + (f"_limit{res['limit_clips']}" if res["limit_clips"] else "")
    jp, mp = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    jp.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    mp.write_text(to_markdown(res), encoding="utf-8")
    print(f"💾 {jp}\n💾 {mp}")
    return jp, mp


# =============================================================================
# Modal (optional)
# =============================================================================
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("fire-smoke-evaluate-test")
    volume = modal.Volume.from_name("fire_smoke_dataset_v2")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch==2.2.2", "torchvision==0.17.2", "transformers==4.40.0", "opencv-python-headless",
                     "numpy<2.0.0", "pandas", "Pillow")
        .add_local_file(SRC / "fire_core.py", "/root/fire_core.py")
        .add_local_file(SRC / "metrics_calculator.py", "/root/metrics_calculator.py")
    )

    @app.function(image=image, volumes={"/data": volume}, gpu="A10G", timeout=7200, memory=32768)
    def evaluate_remote(limit: int = 0, stage1_ckpt: str = "/data/checkpoints/segformer_spatial_best.pth",
                        stage2_ckpt: str = "/data/checkpoints/timesformer_spatiotemporal_best.pth",
                        speed_runs: int = 20, git_commit: str = "unknown", fold: int = -1):
        # thresholds are chosen on the val part of the fold; test is the same for every fold
        splits_dir = "/data/data_splits" + (f"/folds/fold{fold}" if fold >= 0 else "")
        res = run_evaluation("/data/dataset_video_unified", splits_dir, stage1_ckpt, stage2_ckpt,
                             limit=limit, speed_runs=speed_runs, git_commit=git_commit)
        res = json.loads(json.dumps(res, default=_json_default))
        # also keep the report on the volume, in case the local client is gone when the job ends
        write_outputs(res, "/data/reports")
        volume.commit()
        return res

    @app.local_entrypoint()
    def main(limit: int = 0, stage1_ckpt: str = "/data/checkpoints/segformer_spatial_best.pth",
             stage2_ckpt: str = "/data/checkpoints/timesformer_spatiotemporal_best.pth", speed_runs: int = 20,
             fold: int = -1):
        # spawn + get: if this machine goes away the job still finishes and saves to /data/reports
        call = evaluate_remote.spawn(limit=limit, stage1_ckpt=stage1_ckpt, stage2_ckpt=stage2_ckpt,
                                     speed_runs=speed_runs, git_commit=local_git_commit(), fold=fold)
        print(f"🚀 Đã gửi job đánh giá (call id: {call.object_id}). Kết quả cũng được lưu ở volume: /reports/")
        write_outputs(call.get(), ROOT / "docs" / "reports")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(ROOT / "data" / "dataset_video_unified"))
    ap.add_argument("--splits-dir", default=str(ROOT / "data" / "data_splits"))
    ap.add_argument("--stage1-ckpt", default=str(ROOT / "checkpoints" / "segformer_spatial_best.pth"))
    ap.add_argument("--stage2-ckpt", default=str(ROOT / "checkpoints" / "checkpoints" / "timesformer_spatiotemporal_best.pth"))
    ap.add_argument("--dense-root", default=None, help="only if dense clips live outside --data-root")
    ap.add_argument("--limit", type=int, default=0, help="number of clips per split (0 = all)")
    ap.add_argument("--speed-runs", type=int, default=3)
    ap.add_argument("--out-dir", default=str(ROOT / "docs" / "reports"))
    args = ap.parse_args()
    result = run_evaluation(args.data_root, args.splits_dir, args.stage1_ckpt, args.stage2_ckpt,
                            limit=args.limit, speed_runs=args.speed_runs, git_commit=local_git_commit(),
                            dense_root=args.dense_root)
    write_outputs(result, args.out_dir)
