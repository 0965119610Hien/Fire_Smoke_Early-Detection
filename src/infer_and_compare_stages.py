"""
src/infer_and_compare_stages.py
================================
Chạy suy luận (Inference) so sánh trực tiếp 2 mô hình từ 2 Stage trên tập Test:
1. Stage 1: SegFormer-B2 (Spatial Core - xử lý từng frame đơn lẻ)
2. Stage 2: TimeSformer (Spatio-Temporal - xử lý chuỗi T=8 frames không-thời gian)

Tạo ảnh trực quan hóa 4 khung hình (4-Panel Comparison Grid):
- Panel 1: Original Raw Frame (RGB)
- Panel 2: Ground Truth Annotation (Đỏ: Lửa, Vàng cam: Khói)
- Panel 3: Stage 1 SegFormer Prediction (Spatial Core)
- Panel 4: Stage 2 TimeSformer Prediction (Spatio-Temporal)

Kết quả ảnh so sánh sẽ được:
- Lưu trên Modal Volume: /data/inference_results/
- Tự động tải về máy cục bộ: docs/inference_comparisons/ để xem trực tiếp trong IDE!

Cách chạy trên Modal GPU:
  modal run src/infer_and_compare_stages.py
"""

import modal
import os
import sys

app = modal.App("fire-smoke-infer-compare")
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
def run_inference_comparison(max_clips: int = 5):
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
    print(f"🚀 [INFERENCE COMPARISON] So sánh Stage 1 & Stage 2 trên GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    DATA_DIR = Path("/data")
    SPLITS_DIR = DATA_DIR / "data_splits"
    CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
    VIDEO_DIR = DATA_DIR / "dataset_video_prepared"
    OUT_DIR = DATA_DIR / "inference_results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    S1_CKPT = CHECKPOINTS_DIR / "segformer_spatial_best.pth"
    S2_CKPT = CHECKPOINTS_DIR / "timesformer_spatiotemporal_best.pth"
    TEST_CLIPS_CSV = SPLITS_DIR / "test_clips.csv"

    assert S1_CKPT.exists(), f"Không tìm thấy checkpoint Stage 1 tại {S1_CKPT}"
    assert S2_CKPT.exists(), f"Không tìm thấy checkpoint Stage 2 tại {S2_CKPT}"
    assert TEST_CLIPS_CSV.exists(), f"Không tìm thấy {TEST_CLIPS_CSV}"

    # 1. NẠP MÔ HÌNH STAGE 1: SEGFORMER-B2
    print("⏳ Đang nạp mô hình Stage 1 (SegFormer-B2)...")
    model_s1 = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", num_labels=3, ignore_mismatched_sizes=True
    ).to(device)
    ckpt_s1 = torch.load(S1_CKPT, map_location=device)
    model_s1.load_state_dict(ckpt_s1["model_state_dict"])
    model_s1.eval()

    # 2. ĐỊNH NGHĨA & NẠP MÔ HÌNH STAGE 2
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

    ckpt_s2 = torch.load(S2_CKPT, map_location=device)
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

    # 3. ĐỌC TẬP TEST CLIPS
    df_clips = pd.read_csv(TEST_CLIPS_CSV)
    num_to_run = min(max_clips, len(df_clips))
    print(f"🎬 Sẽ chạy suy luận và trích xuất hình ảnh cho {num_to_run}/{len(df_clips)} clips test...")

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Bảng màu hiển thị (BGR):
    # Class 0: BG (Không tô)
    # Class 1: Fire (Đỏ tươi: BGR=[0, 0, 255])
    # Class 2: Smoke (Vàng cam: BGR=[0, 165, 255])
    COLOR_FIRE = (0, 0, 255)
    COLOR_SMOKE = (0, 165, 255)

    def overlay_mask(image_bgr, mask, alpha=0.5):
        overlay = image_bgr.copy()
        # Fire
        overlay[mask == 1] = (
            (1 - alpha) * overlay[mask == 1] + alpha * np.array(COLOR_FIRE)
        ).astype(np.uint8)
        # Smoke
        overlay[mask == 2] = (
            (1 - alpha) * overlay[mask == 2] + alpha * np.array(COLOR_SMOKE)
        ).astype(np.uint8)

        # Vẽ viền đốm cháy để nhìn rõ vị trí
        for cls_id, col in [(1, COLOR_FIRE), (2, COLOR_SMOKE)]:
            bin_m = (mask == cls_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, col, 2)
        return overlay

    def add_label(img, text, color=(255, 255, 255), bg_color=(30, 30, 30)):
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 32), bg_color, -1)
        cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        return img

    results_files = {}

    for clip_idx in range(num_to_run):
        row = df_clips.iloc[clip_idx]
        vid_name = row['video_name']
        img_rel_paths = row['image_paths'].split(";")
        mask_rel_paths = row['mask_paths'].split(";")

        raw_frames = []
        norm_frames = []
        gt_masks = []

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

            raw_frames.append(img)
            gt_masks.append(mask)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_norm = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
            norm_frames.append(img_norm.transpose(2, 0, 1))

        # --- A. INFERENCE STAGE 1 (SegFormer - Per-frame) ---
        s1_preds = []
        with torch.no_grad():
            for t in range(8):
                t_input = torch.from_numpy(norm_frames[t]).unsqueeze(0).float().to(device)
                out = model_s1(pixel_values=t_input)
                logits = out.logits
                logits_up = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
                pred_s1 = torch.argmax(logits_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                s1_preds.append(pred_s1)

        # --- B. INFERENCE STAGE 2 (TimeSformer - Spatio-Temporal Clip) ---
        with torch.no_grad():
            clip_tensor = torch.from_numpy(np.stack(norm_frames, axis=0)).unsqueeze(0).float().to(device)
            out_s2 = model_s2(clip_tensor)
            preds_s2 = torch.argmax(out_s2, dim=2).squeeze(0).cpu().numpy().astype(np.uint8) # (8, 512, 512)

        # --- C. TẠO ẢNH SO SÁNH 4-PANEL CHO KHUNG HÌNH ĐẠI DIỆN ---
        # Chọn khung hình có nhiều lửa/khói nhất hoặc frame giữa (t=4)
        best_t = 4
        max_target_pixels = -1
        for t in range(8):
            fg_px = np.sum(gt_masks[t] > 0)
            if fg_px > max_target_pixels:
                max_target_pixels = fg_px
                best_t = t

        for t_inspect in [best_t]:
            raw_rgb = raw_frames[t_inspect]
            gt_m = gt_masks[t_inspect]
            p1_m = s1_preds[t_inspect]
            p2_m = preds_s2[t_inspect]

            # Tính IoU cho từng model ở frame này
            def get_f_s_iou(pred, gt):
                def iou_c(c):
                    inter = np.sum((pred == c) & (gt == c))
                    union = np.sum((pred == c) | (gt == c))
                    return (inter / union * 100) if union > 0 else 100.0
                return iou_c(1), iou_c(2)

            f_iou_s1, s_iou_s1 = get_f_s_iou(p1_m, gt_m)
            f_iou_s2, s_iou_s2 = get_f_s_iou(p2_m, gt_m)

            # Panel 1: Raw
            p1 = add_label(raw_rgb.copy(), f"1. Raw Frame: {vid_name[:20]}_f{t_inspect}", color=(255, 255, 255))
            # Panel 2: GT
            p2 = overlay_mask(raw_rgb.copy(), gt_m)
            p2 = add_label(p2, "2. Ground Truth Mask (Red:Fire, Orange:Smoke)", color=(0, 255, 0))
            # Panel 3: Stage 1
            p3 = overlay_mask(raw_rgb.copy(), p1_m)
            p3 = add_label(p3, f"3. Stage 1 SegFormer (Fire:{f_iou_s1:.1f}%, Smoke:{s_iou_s1:.1f}%)", color=(255, 200, 50))
            # Panel 4: Stage 2
            p4 = overlay_mask(raw_rgb.copy(), p2_m)
            p4 = add_label(p4, f"4. Stage 2 TimeSformer (Fire:{f_iou_s2:.1f}%, Smoke:{s_iou_s2:.1f}%)", color=(0, 255, 255))

            # Ghép lưới 2x2 (1024 x 1024)
            top_row = np.hstack([p1, p2])
            bottom_row = np.hstack([p3, p4])
            grid_2x2 = np.vstack([top_row, bottom_row])

            # Header tổng quan
            header = np.zeros((60, 1024, 3), dtype=np.uint8)
            cv2.putText(
                header,
                f"TEST COMPARISON | Clip {clip_idx+1}: {vid_name} | Frame {t_inspect}/7",
                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                header,
                f"Stage 1 Fg-IoU: {(f_iou_s1+s_iou_s1)/2:.1f}%  -->  Stage 2 Fg-IoU: {(f_iou_s2+s_iou_s2)/2:.1f}%",
                (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1
            )
            final_img = np.vstack([header, grid_2x2])

            out_filename = f"comparison_clip{clip_idx+1}_{vid_name[:18]}_f{t_inspect}.jpg"
            out_filepath = OUT_DIR / out_filename
            cv2.imwrite(str(out_filepath), final_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  ✅ Đã xuất ảnh so sánh: {out_filename}")

            # Đọc lại bytes để trả về cho local
            with open(out_filepath, "rb") as f:
                results_files[out_filename] = f.read()

    volume.commit()
    print("\n" + "=" * 80)
    print(f"🎉 SUY LUẬN VÀ TRỰC QUAN HÓA HOÀN TẤT! Đã lưu {len(results_files)} ảnh so sánh lên Volume.")
    print("=" * 80)
    return results_files

@app.local_entrypoint()
def main(max_clips: int = 4):
    from pathlib import Path
    print("🚀 Bắt đầu gọi hàm suy luận trên Modal GPU...")
    image_dict = run_inference_comparison.remote(max_clips=max_clips)

    local_dir = Path("docs/inference_comparisons")
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Đang lưu {len(image_dict)} ảnh so sánh về thư mục cục bộ: {local_dir}")
    for fname, img_bytes in image_dict.items():
        local_p = local_dir / fname
        with open(local_p, "wb") as f:
            f.write(img_bytes)
        print(f"  💾 Đã tải: {local_p}")

    print("\n" + "=" * 80)
    print(f"✨ TẤT CẢ ẢNH SO SÁNH ĐÃ SẴN SÀNG TẠI: docs/inference_comparisons/")
    print("=" * 80)
