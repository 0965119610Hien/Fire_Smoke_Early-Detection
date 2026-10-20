"""
src/evaluate_stage2_local.py
============================
Đánh giá mô hình Stage 2 (SpatioTemporalSegFormer) trên toàn bộ 36 clips trong test_clips.csv
Bao gồm:
- 18 clips dương tính (lửa / khói / cả hai)
- 18 clips âm tính thuần túy (hard negative: hơi nước, khói hàn, sương mù, xe cộ, kho bãi, đường phố)
"""

import os
import sys
import time
import math
import json
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints" / "checkpoints"
if not CHECKPOINTS_DIR.exists():
    CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"

SPLITS_DIR = ROOT_DIR / "data" / "data_splits"
VIDEO_DIR = ROOT_DIR / "data" / "dataset_video_prepared"

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Chạy đánh giá Stage 2 trên thiết bị: {device}")

    ckpt_path = CHECKPOINTS_DIR / "timesformer_spatiotemporal_best.pth"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINTS_DIR / "timesformer_spatiotemporal_latest.pth"
    print(f"📦 Đang nạp checkpoint từ: {ckpt_path}")

    base = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", num_labels=3, ignore_mismatched_sizes=True
    )
    model = SpatioTemporalSegFormer(base).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_clips_csv = SPLITS_DIR / "test_clips.csv"
    df_clips = pd.read_csv(test_clips_csv)
    print(f"🎬 Tổng số clips kiểm thử: {len(df_clips)} clips (8 frames/clip = {len(df_clips)*8} frames)")

    pred_frame_masks = []
    gt_frame_masks = []
    y_fire_t, y_fire_p = [], []
    y_smoke_t, y_smoke_p = [], []
    y_any_t, y_any_p = [], []
    detection_latencies = []

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    with torch.no_grad():
        for _, row in tqdm(df_clips.iterrows(), total=len(df_clips), desc="Evaluating Stage 2"):
            img_rel_paths = row['image_paths'].split(";")
            mask_rel_paths = row['mask_paths'].split(";")

            clip_frames = []
            clip_masks = []

            for img_rel, mask_rel in zip(img_rel_paths, mask_rel_paths):
                img_p = VIDEO_DIR / img_rel
                mask_p = VIDEO_DIR / mask_rel

                img = cv2.imread(str(img_p))
                mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    img = np.zeros((512, 512, 3), dtype=np.uint8)
                else:
                    img = cv2.resize(img, (512, 512))

                if mask is None:
                    mask = np.zeros((512, 512), dtype=np.uint8)
                else:
                    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
                    mask = np.clip(mask, 0, 2).astype(np.uint8)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img_norm = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
                clip_frames.append(img_norm.transpose(2, 0, 1))
                clip_masks.append(mask)

            tensor = torch.from_numpy(np.stack(clip_frames, axis=0)).unsqueeze(0).float().to(device)
            logits = model(tensor)
            preds = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy().astype(np.uint8)

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

    fire_cls_s2 = compute_binary_metrics(y_fire_t, y_fire_p)
    smoke_cls_s2 = compute_binary_metrics(y_smoke_t, y_smoke_p)
    any_cls_s2 = compute_binary_metrics(y_any_t, y_any_p)

    fire_loc_s2 = compute_loc_metrics(pred_frame_masks, gt_frame_masks, target_class=1)
    smoke_loc_s2 = compute_loc_metrics(pred_frame_masks, gt_frame_masks, target_class=2)
    pix_s2 = compute_pixel_metrics(pred_frame_masks, gt_frame_masks)
    mean_latency_frames = float(np.mean(detection_latencies)) if len(detection_latencies) > 0 else 0.0

    stage2_report = {
        "Classification_Fire": fire_cls_s2,
        "Classification_Smoke": smoke_cls_s2,
        "Classification_Overall": any_cls_s2,
        "Localization_Fire": fire_loc_s2,
        "Localization_Smoke": smoke_loc_s2,
        "Pixel_Segmentation": pix_s2,
        "Mean_Detection_Latency_Frames": mean_latency_frames
    }

    print("\n" + "=" * 80)
    print("📊 KẾT QUẢ ĐÁNH GIÁ ĐẦY ĐỦ TRÊN 36 CLIPS (CÓ NEGATIVE DATA):")
    print(f"  • Overall Alarm ACC : {any_cls_s2['ACC']*100:.2f}% | F1-Score: {any_cls_s2['F1']*100:.2f}% | MCC: {any_cls_s2['MCC']:.4f}")
    print(f"  • Sensitivity (TPR) : Fire={fire_cls_s2['TPR_Recall']*100:.2f}%, Smoke={smoke_cls_s2['TPR_Recall']*100:.2f}%, Any={any_cls_s2['TPR_Recall']*100:.2f}%")
    print(f"  • False Alarm (FAR) : Fire={fire_cls_s2['FPR_FAR']*100:.2f}%, Smoke={smoke_cls_s2['FPR_FAR']*100:.2f}%, Any={any_cls_s2['FPR_FAR']*100:.2f}%")
    print(f"  • Specificity (TNR) : Fire={fire_cls_s2['TNR_Specificity']*100:.2f}%, Smoke={smoke_cls_s2['TNR_Specificity']*100:.2f}%, Any={any_cls_s2['TNR_Specificity']*100:.2f}%")
    print(f"  • Precision (PPV)   : Fire={fire_cls_s2['PPV_Precision']*100:.2f}%, Smoke={smoke_cls_s2['PPV_Precision']*100:.2f}%, Any={any_cls_s2['PPV_Precision']*100:.2f}%")
    print(f"  • Miss Rate (FNR)   : Fire={fire_cls_s2['FNR_MissRate']*100:.2f}%, Smoke={smoke_cls_s2['FNR_MissRate']*100:.2f}%")
    print(f"  • Pixel Fg-mIoU     : {pix_s2['Fg_mIoU']*100:.2f}% (Fire: {pix_s2['IoU_Fire']*100:.2f}%, Smoke: {pix_s2['IoU_Smoke']*100:.2f}%, BG: {pix_s2['IoU_BG']*100:.2f}%)")
    print(f"  • Localization LTPR : Fire={fire_loc_s2['LTPR']*100:.2f}%, Smoke={smoke_loc_s2['LTPR']*100:.2f}%")
    print(f"  • Localization LPPV : Fire={fire_loc_s2['LPPV']*100:.2f}%, Smoke={smoke_loc_s2['LPPV']*100:.2f}%")
    print(f"  • Mean Similarity S : Fire={fire_loc_s2['Mean_Similarity_S']:.4f}, Smoke={smoke_loc_s2['Mean_Similarity_S']:.4f}")
    print("=" * 80)

    out_file = ROOT_DIR / "checkpoints" / "stage2_evaluation_36clips.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stage2_report, f, indent=2, ensure_ascii=False)
    print(f"💾 Đã lưu kết quả chi tiết vào: {out_file}")

if __name__ == "__main__":
    main()
