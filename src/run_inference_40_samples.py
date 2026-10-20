"""
src/run_inference_40_samples.py
===============================
Chạy suy luận và trực quan hóa so sánh trực tiếp trên 40 mẫu test đại diện
cho cả 2 Stage:
- Stage 1: SegFormer-B2 (Spatial Core)
- Stage 2: Spatio-Temporal SegFormer (Temporal Attention Module)

40 mẫu bao gồm đầy đủ 5 nhóm bối cảnh:
1. Ngọn lửa bùng phát / Lửa mạnh (Fire) : 10 mẫu
2. Khói cuộn / Khói loãng sớm (Smoke)    : 10 mẫu
3. Cả Lửa và Khói (Both Fire & Smoke)   : 5 mẫu
4. Thách thức Hard-Negatives            : 8 mẫu (hơi nước, khói hàn, khói xe, sương mù)
5. Môi trường Bình thường (Normal CCTV) : 7 mẫu (cao tốc, kho bãi, đường phố, văn phòng)

Kết quả:
- Ảnh 4-Panel Grid (Raw | GT | Stage 1 | Stage 2) chất lượng cao
- Tải trực tiếp về local: docs/inference_comparisons_40/
"""

import modal
import os
import sys

app = modal.App("fire-smoke-infer-40-samples")
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
    )
)

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def run_40_inferences():
    import cv2
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation
    from einops import rearrange
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"🚀 [INFERENCE 40 SAMPLES] So sánh Stage 1 & Stage 2 trên GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    DATA_DIR = Path("/data")
    SPLITS_DIR = DATA_DIR / "data_splits"
    CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
    VIDEO_DIR = DATA_DIR / "dataset_video_prepared"
    OUT_DIR = DATA_DIR / "inference_results_40"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    S1_CKPT = CHECKPOINTS_DIR / "segformer_spatial_best.pth"
    S2_CKPT = CHECKPOINTS_DIR / "timesformer_spatiotemporal_best.pth"
    TEST_CLIPS_CSV = SPLITS_DIR / "test_clips.csv"

    print(f"📦 Đang nạp Stage 1 từ {S1_CKPT.name}...")
    model_s1 = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", num_labels=3, ignore_mismatched_sizes=True
    ).to(device)
    ckpt_s1 = torch.load(S1_CKPT, map_location=device)
    model_s1.load_state_dict(ckpt_s1["model_state_dict"])
    model_s1.eval()

    print(f"📦 Đang nạp Stage 2 từ {S2_CKPT.name}...")
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

    base_s2 = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", num_labels=3, ignore_mismatched_sizes=True
    )
    model_s2 = SpatioTemporalSegFormer(base_s2).to(device)
    ckpt_s2 = torch.load(S2_CKPT, map_location=device)
    model_s2.load_state_dict(ckpt_s2["model_state_dict"])
    model_s2.eval()

    df_clips = pd.read_csv(TEST_CLIPS_CSV)
    print(f"🎬 Tổng số clips kiểm thử: {len(df_clips)} clips.")

    # Lập danh sách 40 mẫu kiểm thử mục tiêu từ 36 clips
    # Mỗi clip có thể chọn 1 hoặc 2 frame đại diện tiêu biểu
    sample_targets = []
    
    # Gom nhóm clips theo category
    cat_clips = {}
    for idx, row in df_clips.iterrows():
        cat = row.get("category", "unknown")
        vid = row["video_name"]
        if "neg_hard_negatives" in vid:
            cat = "hard_negative"
        elif "neg_normal" in vid:
            cat = "normal_cctv"
        cat_clips.setdefault(cat, []).append((idx, row))

    print("📊 Phân bố các nhóm clips trong test_clips.csv:")
    for c, items in cat_clips.items():
        print(f"  - {c}: {len(items)} clips")

    # Thu thập 40 mẫu phân bổ đều
    sample_plan = [
        ("fire_only", 10),
        ("smoke_only", 10),
        ("both", 5),
        ("hard_negative", 8),
        ("normal_cctv", 7),
    ]

    for cat_name, needed in sample_plan:
        items = cat_clips.get(cat_name, [])
        if not items:
            continue
        collected_for_cat = 0
        for clip_idx, row in items:
            # Chọn frame phù hợp
            if cat_name in ["fire_only", "both"]:
                frames_to_pick = [3, 6] if needed - collected_for_cat >= 2 else [4]
            elif cat_name == "smoke_only":
                frames_to_pick = [2, 5] if needed - collected_for_cat >= 2 else [4]
            else:
                frames_to_pick = [1, 5] if needed - collected_for_cat >= 2 else [3]

            for t_pick in frames_to_pick:
                sample_targets.append({
                    "clip_idx": clip_idx,
                    "row": row,
                    "target_frame": t_pick,
                    "category": cat_name
                })
                collected_for_cat += 1
                if collected_for_cat >= needed:
                    break
            if collected_for_cat >= needed:
                break

    # Nếu chưa đủ 40 mẫu, lấy thêm từ danh sách còn lại
    while len(sample_targets) < 40 and len(sample_targets) < len(df_clips) * 2:
        rem_idx = len(sample_targets) % len(df_clips)
        sample_targets.append({
            "clip_idx": rem_idx,
            "row": df_clips.iloc[rem_idx],
            "target_frame": 4,
            "category": "extended"
        })

    sample_targets = sample_targets[:40]
    print(f"🎯 Đã chọn chính xác {len(sample_targets)} mẫu kiểm thử đại diện!")

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    COLOR_FIRE = (0, 0, 255)       # Đỏ tươi
    COLOR_SMOKE = (0, 165, 255)    # Vàng cam

    def overlay_mask(image_bgr, mask, alpha=0.5):
        overlay = image_bgr.copy()
        overlay[mask == 1] = ((1 - alpha) * overlay[mask == 1] + alpha * np.array(COLOR_FIRE)).astype(np.uint8)
        overlay[mask == 2] = ((1 - alpha) * overlay[mask == 2] + alpha * np.array(COLOR_SMOKE)).astype(np.uint8)

        for cls_id, col in [(1, COLOR_FIRE), (2, COLOR_SMOKE)]:
            bin_m = (mask == cls_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, col, 2)
        return overlay

    def add_label(img, text, color=(255, 255, 255), bg_color=(25, 25, 25)):
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 36), bg_color, -1)
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        return img

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

    results_files = {}

    for sample_num, target in enumerate(sample_targets, 1):
        row = target["row"]
        t_inspect = target["target_frame"]
        cat_type = target["category"]
        vid_name = row['video_name']

        img_rel_paths = row['image_paths'].split(";")
        mask_rel_paths = row['mask_paths'].split(";")

        raw_frames, norm_frames, gt_masks = [], [], []
        prev_enhanced = None

        for img_rel, mask_rel in zip(img_rel_paths, mask_rel_paths):
            img_p = VIDEO_DIR / img_rel
            mask_p = VIDEO_DIR / mask_rel

            img = cv2.imread(str(img_p))
            if img is None:
                img = np.zeros((512, 512, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (512, 512))

            mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((512, 512), dtype=np.uint8)
            else:
                mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
                mask = np.clip(mask, 0, 2).astype(np.uint8)

            raw_frames.append(img.copy())
            gt_masks.append(mask)

            img_enhanced = enhance_smoke_and_fire_v2(img, prev_frame_bgr=prev_enhanced)
            prev_enhanced = img_enhanced

            img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_norm = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
            norm_frames.append(img_norm.transpose(2, 0, 1))

        # 1. Inference Stage 1
        t_input = torch.from_numpy(norm_frames[t_inspect]).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out_s1 = model_s1(pixel_values=t_input)
            logits_s1 = out_s1.logits
            logits_up = F.interpolate(logits_s1, size=(512, 512), mode="bilinear", align_corners=False)
            p1_m = torch.argmax(logits_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # 2. Inference Stage 2
        clip_tensor = torch.from_numpy(np.stack(norm_frames, axis=0)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out_s2 = model_s2(clip_tensor)
            preds_s2 = torch.argmax(out_s2, dim=2).squeeze(0).cpu().numpy().astype(np.uint8)
            p2_m = preds_s2[t_inspect]

        raw_rgb = raw_frames[t_inspect]
        gt_m = gt_masks[t_inspect]

        def get_f_s_iou(pred, gt):
            def iou_c(c):
                inter = np.sum((pred == c) & (gt == c))
                union = np.sum((pred == c) | (gt == c))
                if np.sum(gt == c) == 0:
                    return 100.0 if np.sum(pred == c) == 0 else 0.0
                return (inter / union * 100) if union > 0 else 0.0
            return iou_c(1), iou_c(2)

        f_iou_s1, s_iou_s1 = get_f_s_iou(p1_m, gt_m)
        f_iou_s2, s_iou_s2 = get_f_s_iou(p2_m, gt_m)

        # Trạng thái báo động
        has_fire_gt = np.sum(gt_m == 1) > 20
        has_smoke_gt = np.sum(gt_m == 2) > 20
        has_fire_s1 = np.sum(p1_m == 1) > 20
        has_smoke_s1 = np.sum(p1_m == 2) > 20
        has_fire_s2 = np.sum(p2_m == 1) > 20
        has_smoke_s2 = np.sum(p2_m == 2) > 20

        # Panel 1: Raw
        p1 = add_label(raw_rgb.copy(), f"1. Raw Frame [t={t_inspect}]", color=(255, 255, 255))

        # Panel 2: GT
        gt_status = "NORMAL / SAFE"
        if has_fire_gt and has_smoke_gt:
            gt_status = "GT: FIRE + SMOKE"
        elif has_fire_gt:
            gt_status = "GT: FIRE ONLY"
        elif has_smoke_gt:
            gt_status = "GT: SMOKE ONLY"
        p2 = overlay_mask(raw_rgb.copy(), gt_m)
        p2 = add_label(p2, f"2. {gt_status}", color=(0, 255, 0) if not (has_fire_gt or has_smoke_gt) else (0, 200, 255))

        # Panel 3: Stage 1
        s1_status = "S1: NORMAL"
        if has_fire_s1 and has_smoke_s1: s1_status = "S1: FIRE + SMOKE"
        elif has_fire_s1: s1_status = "S1: FIRE ALARM"
        elif has_smoke_s1: s1_status = "S1: SMOKE ALARM"
        p3 = overlay_mask(raw_rgb.copy(), p1_m)
        p3 = add_label(p3, f"3. {s1_status} (F:{f_iou_s1:.0f}% S:{s_iou_s1:.0f}%)", color=(255, 220, 80))

        # Panel 4: Stage 2
        s2_status = "S2: NORMAL"
        if has_fire_s2 and has_smoke_s2: s2_status = "S2: FIRE + SMOKE"
        elif has_fire_s2: s2_status = "S2: FIRE ALARM"
        elif has_smoke_s2: s2_status = "S2: SMOKE ALARM"
        p4 = overlay_mask(raw_rgb.copy(), p2_m)
        p4 = add_label(p4, f"4. {s2_status} (F:{f_iou_s2:.0f}% S:{s_iou_s2:.0f}%)", color=(50, 255, 255))

        # Ghép lưới 2x2
        top_row = np.hstack([p1, p2])
        bottom_row = np.hstack([p3, p4])
        grid_2x2 = np.vstack([top_row, bottom_row])

        # Header
        header = np.zeros((65, 1024, 3), dtype=np.uint8)
        cv2.putText(
            header,
            f"SAMPLE #{sample_num:02d}/40 | Cat: {cat_type.upper()} | {vid_name[:35]}",
            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2
        )
        sub_text = f"GT: {gt_status}  -->  Stage 1: {s1_status}  -->  Stage 2: {s2_status}"
        cv2.putText(
            header,
            sub_text,
            (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1
        )
        final_img = np.vstack([header, grid_2x2])

        out_filename = f"sample_{sample_num:02d}_{cat_type}_{vid_name[:16]}_f{t_inspect}.jpg"
        out_filepath = OUT_DIR / out_filename
        cv2.imwrite(str(out_filepath), final_img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"  [{sample_num:02d}/40] Đã xuất: {out_filename}")

        with open(out_filepath, "rb") as f:
            results_files[out_filename] = f.read()

    volume.commit()
    print("\n" + "=" * 80)
    print(f"🎉 SUY LUẬN XONG TOÀN BỘ 40 MẪU! Đã lưu {len(results_files)} ảnh lên Volume.")
    print("=" * 80)
    return results_files

@app.local_entrypoint()
def main():
    from pathlib import Path
    print("🚀 Bắt đầu gọi hàm suy luận 40 mẫu test trên Modal GPU...")
    image_dict = run_40_inferences.remote()

    local_dir = Path("docs/inference_comparisons_40")
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Đang lưu {len(image_dict)} ảnh so sánh về thư mục cục bộ: {local_dir}")
    for fname, img_bytes in image_dict.items():
        local_p = local_dir / fname
        with open(local_p, "wb") as f:
            f.write(img_bytes)
        print(f"  💾 Đã lưu: {fname}")

    print("\n" + "=" * 80)
    print(f"✨ HOÀN TẤT! Toàn bộ 40 ảnh đã lưu tại: {local_dir}")
    print("=" * 80)
