"""
src/evaluate_all_metrics.py
============================
Script đánh giá toàn diện mô hình sau khi huấn luyện trên TẤT CẢ CÁC ĐỘ ĐO KHOA HỌC:
1. Binary Classification Metrics (ACC, TPR, FPR, TNR, FNR, PPV, F1, MCC, FDR)
2. Localization Metrics (LTPR, LPPV, Mean Similarity S)
3. Pixel-level Semantic Segmentation Metrics (mIoU, IoU Fire, IoU Smoke, Dice Fire, Dice Smoke)
4. Speed Metrics (Latency ms, FPS trên GPU)
5. Temporal Video Early Detection Metrics (Frame-to-Detect, Persistence) cho Stage 2

Cách chạy trên Modal GPU:
  modal run src/evaluate_all_metrics.py --stage 1
  modal run src/evaluate_all_metrics.py --stage 2
  modal run src/evaluate_all_metrics.py --stage both
"""

import modal
import os
import sys

app = modal.App("fire-smoke-evaluation-suite")
volume = modal.Volume.from_name("fire_smoke_dataset_v2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "transformers==4.40.0",
        "opencv-python-headless",
        "numpy<2.0.0",
        "Pillow",
        "einops",
        "scikit-learn",
        "tqdm",
        "pandas",
        "tabulate",
    )
)

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A100",
    timeout=7200,
    memory=32768,
)
def evaluate_models(stage: str = "both"):
    import time
    import math
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import cv2

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import SegformerForSemanticSegmentation
    from einops import rearrange, repeat
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"📊 KHỞI ĐỘNG HỆ THỐNG ĐÁNH GIÁ METRICS TOÀN DIỆN TRÊN GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Đánh giá Stage: {stage.upper()}")
    print("=" * 80)

    DATA_DIR = Path("/data")
    SPLITS_DIR = DATA_DIR / "data_splits"
    CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

    # =========================================================================
    # CORE METRIC FORMULAS
    # =========================================================================
    def compute_binary_metrics(y_true, y_pred):
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        total = tp + tn + fp + fn
        acc = (tp + tn) / max(total, 1)
        tpr = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        fpr = fp / max(fp + tn, 1) if (fp + tn) > 0 else 0.0
        tnr = tn / max(tn + fp, 1) if (tn + fp) > 0 else 0.0
        fnr = fn / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        ppv = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        fdr = fp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        f1  = (2 * ppv * tpr) / (ppv + tpr) if (ppv + tpr) > 0 else 0.0

        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
        mcc = (mcc_num / mcc_den) if mcc_den > 0 else 0.0

        return {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "ACC": acc, "TPR_Recall": tpr, "FPR_FAR": fpr,
            "TNR_Specificity": tnr, "FNR_MissRate": fnr,
            "PPV_Precision": ppv, "FDR": fdr, "F1": f1, "MCC": mcc
        }

    def extract_connected_blobs(binary_mask, min_area=16):
        binary_mask = (binary_mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        blobs = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                blobs.append((labels == i).astype(np.uint8))
        return blobs

    def compute_loc_metrics(pred_masks, gt_masks, target_class=1, iou_thresh=0.2, min_area=16):
        total_gt = 0
        detected_gt = 0
        total_pred = 0
        correct_pred = 0
        similarities = []

        for p_m, g_m in zip(pred_masks, gt_masks):
            p_blobs = extract_connected_blobs((p_m == target_class), min_area)
            g_blobs = extract_connected_blobs((g_m == target_class), min_area)

            total_pred += len(p_blobs)
            total_gt += len(g_blobs)

            if len(g_blobs) == 0 and len(p_blobs) == 0:
                continue

            matched_p = set()
            for g_blob in g_blobs:
                best_iou = 0.0
                best_sim = 0.0
                best_p_idx = -1
                for p_idx, p_blob in enumerate(p_blobs):
                    inter = np.sum((g_blob == 1) & (p_blob == 1))
                    union = np.sum((g_blob == 1) | (p_blob == 1))
                    if union == 0:
                        continue
                    iou = inter / union
                    sim = (2.0 * inter) / (np.sum(g_blob) + np.sum(p_blob))
                    if iou > best_iou:
                        best_iou = iou
                        best_sim = sim
                        best_p_idx = p_idx

                if best_iou >= iou_thresh and best_p_idx != -1:
                    detected_gt += 1
                    matched_p.add(best_p_idx)
                    similarities.append(best_sim)

            correct_pred += len(matched_p)

        ltpr = (detected_gt / total_gt) if total_gt > 0 else 1.0
        lppv = (correct_pred / total_pred) if total_pred > 0 else (1.0 if total_gt == 0 else 0.0)
        mean_s = float(np.mean(similarities)) if len(similarities) > 0 else 0.0

        return {
            "Total_GT_Regions": total_gt, "Detected_GT_Regions": detected_gt,
            "Total_Pred_Regions": total_pred, "Correct_Pred_Regions": correct_pred,
            "LTPR": ltpr, "LPPV": lppv, "Mean_Similarity_S": mean_s
        }

    def compute_pixel_metrics(pred_masks, gt_masks):
        ious, dices = [], []
        for c in range(3):
            inter_total, union_total, card_total = 0, 0, 0
            for p_m, g_m in zip(pred_masks, gt_masks):
                p = (p_m == c)
                g = (g_m == c)
                inter_total += np.sum(p & g)
                union_total += np.sum(p | g)
                card_total += np.sum(p) + np.sum(g)
            iou = (inter_total + 1e-6) / (union_total + 1e-6)
            dice = (2.0 * inter_total + 1e-6) / (card_total + 1e-6)
            ious.append(float(iou))
            dices.append(float(dice))
        return {
            "IoU_BG": ious[0], "IoU_Fire": ious[1], "IoU_Smoke": ious[2],
            "Dice_BG": dices[0], "Dice_Fire": dices[1], "Dice_Smoke": dices[2],
            "Fg_mIoU": (ious[1] + ious[2]) / 2.0,
            "All_mIoU": sum(ious) / 3.0
        }

    def measure_speed(eval_fn, sample_data, runs=40):
        # Warmup
        for _ in range(5):
            _ = eval_fn(sample_data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            _ = eval_fn(sample_data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        latency_ms = (dt / runs) * 1000.0
        fps = runs / dt
        return {"Latency_ms": latency_ms, "FPS": fps}

    def enhance_smoke_and_fire_v2(img_bgr, prev_frame_bgr=None):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_micro = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
        clahe_macro = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        l_micro = clahe_micro.apply(l).astype(np.float32)
        l_macro = clahe_macro.apply(l).astype(np.float32)
        l_enhanced = 0.65 * l_micro + 0.35 * l_macro

        img_float = img_bgr.astype(np.float32)
        b_ch, g_ch, r_ch = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
        diff_rgb = np.maximum(np.maximum(np.abs(r_ch - g_ch), np.abs(g_ch - b_ch)), np.abs(b_ch - r_ch))
        w_neutral = np.clip(1.0 - (diff_rgb / 35.0), 0.0, 1.0)

        min_rgb = np.minimum(np.minimum(r_ch, g_ch), b_ch)
        kernel_dcp = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dark_channel = cv2.erode(min_rgb, kernel_dcp)
        dark_norm = cv2.normalize(dark_channel, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

        smoke_boost = (w_neutral * dark_norm) * 28.0
        l_final = np.clip(l_enhanced + smoke_boost, 0.0, 255.0).astype(np.uint8)

        enhanced_lab = cv2.merge([l_final, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        if prev_frame_bgr is not None:
            diff_temporal = cv2.absdiff(enhanced_bgr, prev_frame_bgr)
            gray_diff = cv2.cvtColor(diff_temporal, cv2.COLOR_BGR2GRAY)
            motion_mask = cv2.GaussianBlur(gray_diff, (5, 5), 0)
            motion_weight = (motion_mask.astype(np.float32) / 255.0) * 0.15
            enhanced_bgr = np.clip(enhanced_bgr.astype(np.float32) * (1.0 + motion_weight[:, :, None]), 0, 255).astype(np.uint8)

        return enhanced_bgr

    all_results = {}

    # =========================================================================
    # STAGE 1 EVALUATION: SegFormer-B2
    # =========================================================================
    if stage in ["1", "both"]:
        print("\n" + "=" * 80)
        print("🔍 ĐÁNH GIÁ STAGE 1: SEGFORMER-B2 (SPATIAL CORE) TRÊN TẬP TEST")
        print("=" * 80)

        s1_ckpt = CHECKPOINTS_DIR / "segformer_spatial_best.pth"
        test_csv = SPLITS_DIR / "static_v4i_test_manifest.csv"
        v4i_dir = DATA_DIR / "Fire_Smoke.v4i" / "train"

        if not s1_ckpt.exists():
            print(f"⚠️ Chưa tìm thấy checkpoint Stage 1 tại {s1_ckpt}. Hãy chạy train_stage1_segformer.py trước!")
        elif not test_csv.exists():
            print(f"⚠️ Không tìm thấy file test split tại {test_csv}!")
        else:
            print(f"📦 Đang nạp mô hình từ {s1_ckpt}...")
            model_s1 = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b2", num_labels=3, ignore_mismatched_sizes=True
            ).to(device)
            ckpt = torch.load(s1_ckpt, map_location=device)
            model_s1.load_state_dict(ckpt["model_state_dict"])
            model_s1.eval()

            df_test = pd.read_csv(test_csv)
            print(f"📄 Số lượng ảnh Test: {len(df_test)}")

            pred_masks = []
            gt_masks = []
            y_fire_true, y_fire_pred = [], []
            y_smoke_true, y_smoke_pred = [], []
            y_any_true, y_any_pred = [], []

            IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            with torch.no_grad():
                for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Evaluating Stage 1"):
                    img_p = v4i_dir / row["filename"]
                    mask_p = v4i_dir / row["mask_filename"]

                    img = cv2.imread(str(img_p))
                    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
                    if img is None or mask is None:
                        continue

                    img = enhance_smoke_and_fire_v2(img)

                    # Letterbox resize
                    h, w = img.shape[:2]
                    scale = min(512 / w, 512 / h)
                    nw, nh = int(round(w * scale)), int(round(h * scale))
                    img_r = cv2.resize(img, (nw, nh))
                    mask_r = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

                    pad_w, pad_h = 512 - nw, 512 - nh
                    img_pad = cv2.copyMakeBorder(img_r, pad_h//2, pad_h - pad_h//2, pad_w//2, pad_w - pad_w//2, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                    mask_pad = cv2.copyMakeBorder(mask_r, pad_h//2, pad_h - pad_h//2, pad_w//2, pad_w - pad_w//2, cv2.BORDER_CONSTANT, value=0)
                    mask_pad = np.clip(mask_pad, 0, 2).astype(np.uint8)

                    # Chuẩn bị input tensor
                    img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    img_norm = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
                    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

                    outputs = model_s1(pixel_values=tensor)
                    logits = outputs.logits
                    logits_up = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
                    pred = torch.argmax(logits_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                    pred_masks.append(pred)
                    gt_masks.append(mask_pad)

                    # Binary classification labels (ngưỡng diện tích 16 pixels để kích hoạt báo động)
                    gt_has_fire = 1 if np.sum(mask_pad == 1) >= 16 else 0
                    gt_has_smoke = 1 if np.sum(mask_pad == 2) >= 16 else 0
                    pred_has_fire = 1 if np.sum(pred == 1) >= 16 else 0
                    pred_has_smoke = 1 if np.sum(pred == 2) >= 16 else 0

                    y_fire_true.append(gt_has_fire)
                    y_fire_pred.append(pred_has_fire)
                    y_smoke_true.append(gt_has_smoke)
                    y_smoke_pred.append(pred_has_smoke)
                    y_any_true.append(1 if (gt_has_fire or gt_has_smoke) else 0)
                    y_any_pred.append(1 if (pred_has_fire or pred_has_smoke) else 0)

            # Tính toán các chỉ số Stage 1
            fire_cls = compute_binary_metrics(y_fire_true, y_fire_pred)
            smoke_cls = compute_binary_metrics(y_smoke_true, y_smoke_pred)
            any_cls = compute_binary_metrics(y_any_true, y_any_pred)

            fire_loc = compute_loc_metrics(pred_masks, gt_masks, target_class=1)
            smoke_loc = compute_loc_metrics(pred_masks, gt_masks, target_class=2)

            pix_metrics = compute_pixel_metrics(pred_masks, gt_masks)

            # Đo tốc độ FPS
            sample_t = torch.randn(1, 3, 512, 512, device=device)
            def infer_s1(x):
                with torch.no_grad():
                    return model_s1(pixel_values=x)
            speed_s1 = measure_speed(infer_s1, sample_t, runs=40)

            stage1_report = {
                "Classification_Fire": fire_cls,
                "Classification_Smoke": smoke_cls,
                "Classification_Overall": any_cls,
                "Localization_Fire": fire_loc,
                "Localization_Smoke": smoke_loc,
                "Pixel_Segmentation": pix_metrics,
                "Speed": speed_s1
            }
            all_results["Stage1_SegFormer"] = stage1_report

            print("\n" + "-" * 75)
            print("📊 KẾT QUẢ ĐÁNH GIÁ STAGE 1 (SEGFORMER-B2):")
            print(f"  • Overall Alarm ACC : {any_cls['ACC']*100:.2f}% | F1-Score: {any_cls['F1']*100:.2f}% | MCC: {any_cls['MCC']:.4f}")
            print(f"  • Sensitivity (TPR) : Fire={fire_cls['TPR_Recall']*100:.2f}%, Smoke={smoke_cls['TPR_Recall']*100:.2f}%, Any={any_cls['TPR_Recall']*100:.2f}%")
            print(f"  • False Alarm (FAR) : Fire={fire_cls['FPR_FAR']*100:.2f}%, Smoke={smoke_cls['FPR_FAR']*100:.2f}%, Any={any_cls['FPR_FAR']*100:.2f}%")
            print(f"  • Miss Rate (FNR)   : Fire={fire_cls['FNR_MissRate']*100:.2f}%, Smoke={smoke_cls['FNR_MissRate']*100:.2f}%")
            print(f"  • Specificity (TNR) : {any_cls['TNR_Specificity']*100:.2f}%")
            print(f"  • Precision (PPV)   : {any_cls['PPV_Precision']*100:.2f}% | FDR: {any_cls['FDR']*100:.2f}%")
            print(f"  • Localization LTPR : Fire={fire_loc['LTPR']*100:.2f}%, Smoke={smoke_loc['LTPR']*100:.2f}%")
            print(f"  • Localization LPPV : Fire={fire_loc['LPPV']*100:.2f}%, Smoke={smoke_loc['LPPV']*100:.2f}%")
            print(f"  • Mean Similarity S : Fire={fire_loc['Mean_Similarity_S']:.4f}, Smoke={smoke_loc['Mean_Similarity_S']:.4f}")
            print(f"  • Pixel Fg-mIoU     : {pix_metrics['Fg_mIoU']*100:.2f}% (Fire={pix_metrics['IoU_Fire']*100:.2f}%, Smoke={pix_metrics['IoU_Smoke']*100:.2f}%)")
            print(f"  • Inference Speed   : {speed_s1['FPS']:.1f} FPS (Latency: {speed_s1['Latency_ms']:.2f} ms/frame)")
            print("-" * 75)

    # =========================================================================
    # STAGE 2 EVALUATION: TimeSformer Spatio-Temporal
    # =========================================================================
    if stage in ["2", "both"]:
        print("\n" + "=" * 80)
        print("🔍 ĐÁNH GIÁ STAGE 2: TIMESFORMER SPATIO-TEMPORAL TRÊN VIDEO TEST CLIPS")
        print("=" * 80)

        s2_ckpt = CHECKPOINTS_DIR / "timesformer_spatiotemporal_best.pth"
        test_clips_csv = SPLITS_DIR / "test_clips.csv"
        video_dir = DATA_DIR / "dataset_video_prepared"

        if not s2_ckpt.exists():
            print(f"ℹ️ Lưu ý: Checkpoint Stage 2 chưa được tạo tại {s2_ckpt}. Cần chạy train_stage2_timesformer.py để xuất model.")
        elif not test_clips_csv.exists():
            print(f"⚠️ Không tìm thấy file test clips tại {test_clips_csv}!")
        else:
            class TemporalRefinementModule(nn.Module):
                def __init__(self, in_channels=768, hidden_channels=128, num_heads=4, dropout=0.1):
                    super().__init__()
                    self.in_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
                    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
                    self.temp_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
                    self.norm1 = nn.LayerNorm(hidden_channels)
                    self.temp_attn = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=num_heads, dropout=dropout, batch_first=True)
                    self.norm2 = nn.LayerNorm(hidden_channels)
                    self.mlp = nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels * 2),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_channels * 2, hidden_channels),
                        nn.Dropout(dropout),
                    )
                    self.out_proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
                    self.gamma = nn.Parameter(torch.zeros(1))

                def forward(self, z, B, T):
                    BT, C, H, W = z.shape
                    feat = self.in_proj(z)
                    feat_down = self.pool(feat)
                    _, hidden, h_down, w_down = feat_down.shape
                    feat_t = feat_down.view(B, T, hidden, h_down, w_down).permute(0, 3, 4, 2, 1).contiguous().view(B * h_down * w_down, hidden, T)
                    feat_conv = self.temp_conv(feat_t)
                    feat_t = feat_t + feat_conv
                    feat_seq = feat_t.permute(0, 2, 1).contiguous()
                    normed = self.norm1(feat_seq)
                    attn_out, _ = self.temp_attn(normed, normed, normed)
                    feat_seq = feat_seq + attn_out
                    feat_seq = feat_seq + self.mlp(self.norm2(feat_seq))
                    feat_out = feat_seq.view(B, h_down, w_down, T, hidden).permute(0, 3, 4, 1, 2).contiguous().view(BT, hidden, h_down, w_down)
                    feat_up = F.interpolate(feat_out, size=(H, W), mode='bilinear', align_corners=False)
                    delta = self.out_proj(feat_up)
                    return z + self.gamma * delta

            class SpatioTemporalSegFormer(nn.Module):
                def __init__(self, pretrained_segformer):
                    super().__init__()
                    self.segformer = pretrained_segformer.segformer
                    self.decode_head = pretrained_segformer.decode_head
                    self.temporal_refinement = TemporalRefinementModule(
                        in_channels=self.decode_head.linear_fuse.out_channels,
                        hidden_channels=128,
                        num_heads=4,
                        dropout=0.1
                    )

                def forward(self, pixel_values):
                    B, T, C, H, W = pixel_values.shape
                    x_flat = pixel_values.view(B * T, C, H, W)
                    encoder_outputs = self.segformer(x_flat, output_hidden_states=True)
                    hidden_states = encoder_outputs.hidden_states
                    all_hidden_states = ()
                    for h, mlp in zip(hidden_states, self.decode_head.linear_c):
                        h_proj = mlp(h)
                        h_proj = h_proj.permute(0, 2, 1).reshape(B * T, -1, h.shape[2], h.shape[3])
                        h_up = F.interpolate(h_proj, size=hidden_states[0].shape[2:], mode='bilinear', align_corners=False)
                        all_hidden_states += (h_up,)
                    fused = self.decode_head.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
                    fused = self.decode_head.batch_norm(fused)
                    fused = self.decode_head.activation(fused)
                    fused_temporal = self.temporal_refinement(fused, B, T)
                    dropped = self.decode_head.dropout(fused_temporal)
                    logits = self.decode_head.classifier(dropped)
                    logits_up = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
                    return logits_up.view(B, T, 3, H, W)

            class Mlp(nn.Module):
                def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
                    super().__init__()
                    self.fc1 = nn.Linear(in_features, hidden_features or in_features)
                    self.act = nn.GELU()
                    self.fc2 = nn.Linear(hidden_features or in_features, out_features or in_features)
                    self.drop = nn.Dropout(drop)
                def forward(self, x): return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

            class Attention(nn.Module):
                def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
                    super().__init__()
                    inner_dim = dim_head * heads
                    self.heads = heads
                    self.scale = dim_head ** -0.5
                    self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
                    self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
                def forward(self, x):
                    h = self.heads
                    qkv = self.to_qkv(x).chunk(3, dim=-1)
                    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
                    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
                    attn = dots.softmax(dim=-1)
                    out = torch.matmul(attn, v)
                    return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))

            class DividedSpaceTimeBlock(nn.Module):
                def __init__(self, dim, heads=8, dim_head=64, mlp_dim=1024, dropout=0.1):
                    super().__init__()
                    self.time_norm1 = nn.LayerNorm(dim)
                    self.time_attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                    self.space_norm1 = nn.LayerNorm(dim)
                    self.space_attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                    self.norm2 = nn.LayerNorm(dim)
                    self.mlp = Mlp(in_features=dim, hidden_features=mlp_dim, drop=dropout)
                def forward(self, x):
                    b, t, n, d = x.shape
                    x_time = rearrange(x, 'b t n d -> (b n) t d')
                    x_time = x_time + self.time_attn(self.time_norm1(x_time))
                    x = rearrange(x_time, '(b n) t d -> b t n d', b=b, n=n)

                    x_space = rearrange(x, 'b t n d -> (b t) n d')
                    x_space = x_space + self.space_attn(self.space_norm1(x_space))
                    x = rearrange(x_space, '(b t) n d -> b t n d', b=b, t=t)

                    x_mlp = rearrange(x, 'b t n d -> (b t n) d')
                    x_mlp = x_mlp + self.mlp(self.norm2(x_mlp))
                    return rearrange(x_mlp, '(b t n) d -> b t n d', b=b, t=t, n=n)

            class TimeSformerSegmentation(nn.Module):
                def __init__(self, image_size=512, patch_size=16, num_frames=8, num_classes=3, dim=512, depth=6, heads=8, mlp_dim=1024, dropout=0.1):
                    super().__init__()
                    self.image_size = image_size
                    self.grid_size = image_size // patch_size
                    num_patches = self.grid_size ** 2
                    self.to_patch_embedding = nn.Sequential(
                        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
                        nn.Flatten(2)
                    )
                    self.pos_embedding = nn.Parameter(torch.randn(1, 1, num_patches, dim) * 0.02)
                    self.time_embedding = nn.Parameter(torch.randn(1, num_frames, 1, dim) * 0.02)
                    self.layers = nn.ModuleList([DividedSpaceTimeBlock(dim, heads=heads, dim_head=64, mlp_dim=mlp_dim, dropout=dropout) for _ in range(depth)])
                    self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(dim, 256, kernel_size=2, stride=2),
                        nn.BatchNorm2d(256),
                        nn.GELU(),
                        nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                        nn.BatchNorm2d(128),
                        nn.GELU(),
                        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                        nn.BatchNorm2d(64),
                        nn.GELU(),
                        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                        nn.BatchNorm2d(32),
                        nn.GELU(),
                        nn.Conv2d(32, num_classes, kernel_size=1)
                    )
                def forward(self, x):
                    b, t, c, h, w = x.shape
                    x_reshaped = rearrange(x, 'b t c h w -> (b t) c h w')
                    patches = self.to_patch_embedding(x_reshaped).transpose(1, 2)
                    patches = rearrange(patches, '(b t) n d -> b t n d', b=b, t=t)
                    patches = patches + self.pos_embedding
                    patches = patches + self.time_embedding[:, :t, :, :]
                    for layer in self.layers:
                        patches = layer(patches)
                    feat = rearrange(patches, 'b t (h w) d -> (b t) d h w', h=self.grid_size, w=self.grid_size)
                    logits = self.decoder(feat)
                    return rearrange(logits, '(b t) c h w -> b t c h w', b=b, t=t)

            ckpt_s2 = torch.load(s2_ckpt, map_location=device)
            is_spatiotemporal = ckpt_s2.get("architecture") == "SpatioTemporalSegFormer" or any("temporal_refinement" in k for k in ckpt_s2["model_state_dict"].keys())

            if is_spatiotemporal:
                print("⏳ Đang nạp mô hình Stage 2 (Spatio-Temporal SegFormer)...")
                base_s2 = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/mit-b2", num_labels=3, ignore_mismatched_sizes=True
                )
                model_s2 = SpatioTemporalSegFormer(base_s2).to(device)
            else:
                print("⏳ Đang nạp mô hình Stage 2 (Legacy TimeSformer)...")
                model_s2 = TimeSformerSegmentation(
                    image_size=512, patch_size=16, num_frames=8, num_classes=3, dim=512, depth=6, heads=8
                ).to(device)

            model_s2.load_state_dict(ckpt_s2["model_state_dict"])
            model_s2.eval()

            df_clips = pd.read_csv(test_clips_csv)
            print(f"🎬 Số lượng Video Test Clips (T=8): {len(df_clips)}")

            pred_frame_masks = []
            gt_frame_masks = []
            y_fire_t, y_fire_p = [], []
            y_smoke_t, y_smoke_p = [], []
            y_any_t, y_any_p = [], []

            detection_latencies = [] # Số frame từ khi lửa xuất hiện đến khi model kích hoạt báo động

            IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            with torch.no_grad():
                for _, row in tqdm(df_clips.iterrows(), total=len(df_clips), desc="Evaluating Stage 2"):
                    img_rel_paths = row['image_paths'].split(";")
                    mask_rel_paths = row['mask_paths'].split(";")

                    clip_frames = []
                    clip_masks = []
                    prev_enhanced = None

                    for img_rel, mask_rel in zip(img_rel_paths, mask_rel_paths):
                        img_p = video_dir / img_rel
                        mask_p = video_dir / mask_rel

                        img = cv2.imread(str(img_p))
                        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            img = np.zeros((512, 512, 3), dtype=np.uint8)
                        else:
                            if (img.shape[1], img.shape[0]) != (512, 512):
                                img = cv2.resize(img, (512, 512))
                            img = enhance_smoke_and_fire_v2(img, prev_frame_bgr=prev_enhanced)
                            prev_enhanced = img

                        if mask is None:
                            mask = np.zeros((512, 512), dtype=np.uint8)
                        else:
                            if (mask.shape[1], mask.shape[0]) != (512, 512):
                                mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
                            mask = np.clip(mask, 0, 2).astype(np.uint8)

                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        img_norm = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
                        clip_frames.append(img_norm.transpose(2, 0, 1))
                        clip_masks.append(mask)

                    tensor = torch.from_numpy(np.stack(clip_frames, axis=0)).unsqueeze(0).float().to(device) # (1, 8, 3, 512, 512)
                    logits = model_s2(tensor) # (1, 8, 3, 512, 512)
                    preds = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy().astype(np.uint8) # (8, 512, 512)

                    # Đánh giá per-frame và sequence
                    first_gt_fire = None
                    first_pred_fire = None

                    for t in range(8):
                        p_t = preds[t]
                        g_t = clip_masks[t]
                        pred_frame_masks.append(p_t)
                        gt_frame_masks.append(g_t)

                        has_f_gt = 1 if np.sum(g_t == 1) >= 16 else 0
                        has_s_gt = 1 if np.sum(g_t == 2) >= 16 else 0
                        has_f_pr = 1 if np.sum(p_t == 1) >= 16 else 0
                        has_s_pr = 1 if np.sum(p_t == 2) >= 16 else 0

                        y_fire_t.append(has_f_gt)
                        y_fire_p.append(has_f_pr)
                        y_smoke_t.append(has_s_gt)
                        y_smoke_p.append(has_s_pr)
                        y_any_t.append(1 if (has_f_gt or has_s_gt) else 0)
                        y_any_p.append(1 if (has_f_pr or has_s_pr) else 0)

                        if has_f_gt and first_gt_fire is None:
                            first_gt_fire = t
                        if has_f_pr and first_pred_fire is None:
                            first_pred_fire = t

                    if first_gt_fire is not None and first_pred_fire is not None:
                        latency = max(0, first_pred_fire - first_gt_fire)
                        detection_latencies.append(latency)

            # Tính toán chỉ số Stage 2
            fire_cls_s2 = compute_binary_metrics(y_fire_t, y_fire_p)
            smoke_cls_s2 = compute_binary_metrics(y_smoke_t, y_smoke_p)
            any_cls_s2 = compute_binary_metrics(y_any_t, y_any_p)

            fire_loc_s2 = compute_loc_metrics(pred_frame_masks, gt_frame_masks, target_class=1)
            smoke_loc_s2 = compute_loc_metrics(pred_frame_masks, gt_frame_masks, target_class=2)
            pix_s2 = compute_pixel_metrics(pred_frame_masks, gt_frame_masks)

            # Đo tốc độ TimeSformer
            sample_clip = torch.randn(1, 8, 3, 512, 512, device=device)
            def infer_s2(x):
                with torch.no_grad():
                    return model_s2(x)
            speed_s2 = measure_speed(infer_s2, sample_clip, runs=25)
            video_fps = (8.0 * speed_s2["FPS"]) # T=8 frames mỗi clip

            mean_latency_frames = float(np.mean(detection_latencies)) if len(detection_latencies) > 0 else 0.0

            stage2_report = {
                "Classification_Fire": fire_cls_s2,
                "Classification_Smoke": smoke_cls_s2,
                "Classification_Overall": any_cls_s2,
                "Localization_Fire": fire_loc_s2,
                "Localization_Smoke": smoke_loc_s2,
                "Pixel_Segmentation": pix_s2,
                "Speed_Clips_Per_Sec": speed_s2["FPS"],
                "Speed_Video_FPS": video_fps,
                "Latency_Per_Clip_ms": speed_s2["Latency_ms"],
                "Mean_Detection_Latency_Frames": mean_latency_frames
            }
            all_results["Stage2_TimeSformer"] = stage2_report

            print("\n" + "-" * 75)
            print("📊 KẾT QUẢ ĐÁNH GIÁ STAGE 2 (TIMESFORMER):")
            print(f"  • Overall Alarm ACC : {any_cls_s2['ACC']*100:.2f}% | F1-Score: {any_cls_s2['F1']*100:.2f}% | MCC: {any_cls_s2['MCC']:.4f}")
            print(f"  • Sensitivity (TPR) : Fire={fire_cls_s2['TPR_Recall']*100:.2f}%, Smoke={smoke_cls_s2['TPR_Recall']*100:.2f}%, Any={any_cls_s2['TPR_Recall']*100:.2f}%")
            print(f"  • False Alarm (FAR) : Fire={fire_cls_s2['FPR_FAR']*100:.2f}%, Smoke={smoke_cls_s2['FPR_FAR']*100:.2f}%, Any={any_cls_s2['FPR_FAR']*100:.2f}%")
            print(f"  • Miss Rate (FNR)   : Fire={fire_cls_s2['FNR_MissRate']*100:.2f}%, Smoke={smoke_cls_s2['FNR_MissRate']*100:.2f}%")
            print(f"  • Localization LTPR : Fire={fire_loc_s2['LTPR']*100:.2f}%, Smoke={smoke_loc_s2['LTPR']*100:.2f}%")
            print(f"  • Localization LPPV : Fire={fire_loc_s2['LPPV']*100:.2f}%, Smoke={smoke_loc_s2['LPPV']*100:.2f}%")
            print(f"  • Mean Similarity S : Fire={fire_loc_s2['Mean_Similarity_S']:.4f}, Smoke={smoke_loc_s2['Mean_Similarity_S']:.4f}")
            print(f"  • Pixel Fg-mIoU     : {pix_s2['Fg_mIoU']*100:.2f}%")
            print(f"  • Video Throughput  : {video_fps:.1f} Video FPS ({speed_s2['FPS']:.1f} clips/s, {speed_s2['Latency_ms']:.1f} ms/clip)")
            print(f"  • Detection Latency : {mean_latency_frames:.1f} frames (Tốc độ phản ứng cảnh báo sớm)")
            print("-" * 75)

    # Lưu kết quả JSON ra Volume
    report_json = CHECKPOINTS_DIR / "evaluation_metrics_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    volume.commit()
    print(f"\n💾 Đã lưu và commit báo cáo metrics chi tiết vào: {report_json}")
    print("=" * 80)
    return all_results

@app.local_entrypoint()
def main(stage: str = "both"):
    res = evaluate_models.remote(stage=stage)
    print("Done evaluation!")
