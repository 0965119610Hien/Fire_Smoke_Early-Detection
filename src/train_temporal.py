"""
train_temporal.py
=================
Stage 2: Temporal Segmentation Model — TimeSformer + SegFormer Decoder
=========================================================================
Kiến trúc:
  - Backbone      : TimeSformer (Divided Space-Time Attention) - xử lý chuỗi T frame
  - Decoder Head  : Lightweight MLP Segmentation Head (3-class: BG/Fire/Smoke)
  - Input         : Chuỗi T=8 frame PNG 512×512 (từ annotation_frames/)
  - Output        : Per-frame segmentation mask 512×512 (3 lớp)
  - Labels        : Mask PNG từ annotation_labels/ (được gán nhãn bởi SegFormer-B2)

Cách chạy:
  modal run src/train_temporal.py

Weights lưu tại Modal Volume:
  /data/timesformer_temporal_v1.pth
"""

import modal
import os

app    = modal.App("fire-smoke-temporal-training-v2")
vol    = modal.Volume.from_name("fire_smoke_dataset")

# ─── Modal Image ──────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "opencv-python-headless",
        "numpy<2.0.0",
        "Pillow",
        "einops",
        "scikit-learn",
        "tqdm",
        "pandas",
        "matplotlib",
    )
    .add_local_dir("annotation_frames",       remote_path="/root/annotation_frames")
    .add_local_dir("annotation_labels",       remote_path="/root/annotation_labels")
    .add_local_dir("data_splits",             remote_path="/root/data_splits")
)


# =============================================================================
# MODAL GPU FUNCTION (H100)
# =============================================================================
@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="H100",
    timeout=86400,   # 24h tối đa
    memory=65536,    # 64 GB RAM
)
def train_temporal():
    import copy
    import json
    import random
    import numpy as np
    import pandas as pd
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from einops import rearrange
    from pathlib import Path
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    print("=" * 70)
    print("  STAGE 2 — TEMPORAL SEGMENTATION: TimeSformer")
    print("  GPU: H100 | Input: T=8 frames | Output: Per-frame Mask")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # =========================================================================
    # 1. CONFIG
    # =========================================================================
    CFG = {
        "frames_dir"  : "/root/annotation_frames",
        "labels_dir"  : "/root/annotation_labels",
        "train_csv"   : "/root/data_splits/train_manifest.csv",
        "val_csv"     : "/root/data_splits/val_manifest.csv",
        "save_dir"    : "/data",

        # Model
        "num_classes" : 3,      # 0=BG, 1=Fire, 2=Smoke
        "img_size"    : 224,    # TimeSformer dùng 224×224
        "patch_size"  : 16,
        "num_frames"  : 8,      # Số frame per clip
        "dim"         : 384,    # Embedding dim
        "depth"       : 8,      # Số transformer block
        "heads"       : 8,
        "mlp_ratio"   : 4.0,
        "drop"        : 0.1,
        "attn_drop"   : 0.1,
        "drop_path"   : 0.1,

        # Training
        "batch_size"  : 4,      # H100 80GB: batch=4 với T=8, 224x224
        "epochs"      : 40,
        "lr"          : 2e-4,
        "weight_decay": 1e-4,
        "patience"    : 8,

        # Data
        "num_workers" : 4,
        "pin_memory"  : True,
    }

    # =========================================================================
    # 2. KIẾN TRÚC TIMESFORMER + SEGMENTATION DECODER
    # =========================================================================

    def drop_path_fn(x, drop_prob=0., training=False):
        if drop_prob == 0. or not training:
            return x
        keep = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x.div(keep) * torch.rand(shape, dtype=x.dtype, device=x.device).floor_()

    class DropPath(nn.Module):
        def __init__(self, p=0.): super().__init__(); self.p = p
        def forward(self, x): return drop_path_fn(x, self.p, self.training)

    class Mlp(nn.Module):
        def __init__(self, d, hidden=None, drop=0.):
            super().__init__()
            h = hidden or d
            self.net = nn.Sequential(
                nn.Linear(d, h), nn.GELU(), nn.Dropout(drop),
                nn.Linear(h, d), nn.Dropout(drop)
            )
        def forward(self, x): return self.net(x)

    class DividedSpaceTimeAttention(nn.Module):
        """Temporal → Spatial Divided Attention (TimeSformer core block)"""
        def __init__(self, dim, heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., dp=0.):
            super().__init__()
            self.heads = heads
            self.scale = (dim // heads) ** -0.5

            self.time_norm  = nn.LayerNorm(dim)
            self.time_qkv   = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.time_proj  = nn.Linear(dim, dim)
            self.time_drop  = nn.Dropout(attn_drop)
            # temporal_fc initialized to zeros for stable transfer
            self.temporal_fc = nn.Linear(dim, dim)
            nn.init.zeros_(self.temporal_fc.weight)
            nn.init.zeros_(self.temporal_fc.bias)

            self.space_norm = nn.LayerNorm(dim)
            self.space_qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.space_proj = nn.Linear(dim, dim)
            self.space_drop = nn.Dropout(attn_drop)

            self.mlp_norm   = nn.LayerNorm(dim)
            self.mlp        = Mlp(dim, int(dim * 4), drop=proj_drop)
            self.dp1        = DropPath(dp)
            self.dp2        = DropPath(dp)
            self.dp3        = DropPath(dp)

        def _attn(self, qkv, drop):
            h = self.heads
            q, k, v = qkv.chunk(3, dim=-1)
            q = rearrange(q, 'b n (h d) -> b h n d', h=h)
            k = rearrange(k, 'b n (h d) -> b h n d', h=h)
            v = rearrange(v, 'b n (h d) -> b h n d', h=h)
            attn = drop((q @ k.transpose(-2,-1) * self.scale).softmax(dim=-1))
            return rearrange(attn @ v, 'b h n d -> b n (h d)')

        def forward(self, x, T, N):
            # x: [B, T*N, D]
            B, TN, D = x.shape

            # Temporal attention
            xt = rearrange(x, 'b (t n) d -> (b n) t d', t=T, n=N)
            xt = xt + self.dp1(self.time_proj(self._attn(
                self.time_qkv(self.time_norm(xt)), self.time_drop)))
            xt = rearrange(xt, '(b n) t d -> b (t n) d', b=B, n=N)
            x = x + self.temporal_fc(xt - x)

            # Spatial attention
            xs = rearrange(x, 'b (t n) d -> (b t) n d', t=T, n=N)
            xs = xs + self.dp2(self.space_proj(self._attn(
                self.space_qkv(self.space_norm(xs)), self.space_drop)))
            xs = rearrange(xs, '(b t) n d -> b (t n) d', b=B, t=T)

            x = x + self.dp3(self.mlp(self.mlp_norm(xs)))
            return x

    class TimeSformerSegmentation(nn.Module):
        """
        TimeSformer Backbone + All-MLP Segmentation Decoder
        Input : [B, C, T, H, W] = [B, 3, 8, 224, 224]
        Output: [B, T, num_classes, H, W] (upsample về kích thước gốc)
        """
        def __init__(self, cfg):
            super().__init__()
            dim         = cfg["dim"]
            patch_size  = cfg["patch_size"]
            img_size    = cfg["img_size"]
            T           = cfg["num_frames"]
            num_classes = cfg["num_classes"]
            depth       = cfg["depth"]
            heads       = cfg["heads"]

            self.T = T
            self.patch_size = patch_size
            N = (img_size // patch_size) ** 2
            self.N = N

            # Patch embedding
            self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
            self.pos_embed   = nn.Parameter(torch.zeros(1, N, dim))
            self.time_embed  = nn.Parameter(torch.zeros(1, T, dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.time_embed, std=0.02)

            self.drop = nn.Dropout(cfg["drop"])

            dp_rates = [cfg["drop_path"] * i / depth for i in range(depth)]
            self.blocks = nn.ModuleList([
                DividedSpaceTimeAttention(
                    dim=dim, heads=heads,
                    attn_drop=cfg["attn_drop"],
                    proj_drop=cfg["drop"],
                    dp=dp_rates[i]
                )
                for i in range(depth)
            ])
            self.norm = nn.LayerNorm(dim)

            # Segmentation decoder: All-MLP (SegFormer-style)
            H_feat = img_size // patch_size   # 14
            self.decode = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, num_classes),
            )

        def forward(self, x):
            # x: [B, C, T, H, W]
            B, C, T, H, W = x.shape
            N = self.N

            # Patch embed per frame
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = self.patch_embed(x)             # [B*T, dim, H/P, W/P]
            x = rearrange(x, 'bt d h w -> bt (h w) d')  # [B*T, N, D]

            # Add spatial positional embedding
            x = x + self.pos_embed

            # Add temporal embedding
            x = rearrange(x, '(b t) n d -> b (t n) d', b=B, t=T)
            time_emb = self.time_embed.unsqueeze(2).expand(-1, -1, N, -1)
            time_emb = rearrange(time_emb, '1 t n d -> 1 (t n) d').expand(B, -1, -1)
            x = x + time_emb
            x = self.drop(x)

            # Transformer blocks
            for blk in self.blocks:
                x = blk(x, T, N)

            x = self.norm(x)  # [B, T*N, D]

            # Decode to class logits
            x = self.decode(x)  # [B, T*N, num_classes]
            x = rearrange(x, 'b (t n) c -> b t c n', t=T, n=N)

            # Reshape to spatial map
            H_feat = int(N ** 0.5)
            x = rearrange(x, 'b t c (h w) -> (b t) c h w', h=H_feat, w=H_feat)

            # Upsample về img_size
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
            x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
            return x

    # =========================================================================
    # 3. DATASET — đọc chuỗi frame PNG theo video clip
    # =========================================================================
    class FireSmokeClipDataset(Dataset):
        """
        Dataset đọc chuỗi T frame từ cùng 1 video clip (video_stem).
        Mỗi sample = (frames_tensor, masks_tensor)
          frames_tensor : [C, T, H, W] float32 normalized
          masks_tensor  : [T, H, W] long (0=BG, 1=Fire, 2=Smoke)
        """
        def __init__(self, manifest_csv, frames_dir, labels_dir,
                     num_frames=8, img_size=224, augment=False):
            df = pd.read_csv(manifest_csv)
            df["video_stem"] = df["frame_path"].str.split("/").str[0]
            self.groups = [g for _, g in df.groupby("video_stem")]
            self.frames_dir = Path(frames_dir)
            self.labels_dir = Path(labels_dir)
            self.T = num_frames
            self.S = img_size
            self.augment = augment
            print(f"  Dataset: {len(self.groups)} clips | {len(df)} frames total")

        def __len__(self):
            return len(self.groups)

        def _load_frame(self, frame_path, mask_path):
            img  = cv2.imread(str(self.frames_dir / frame_path))
            mask = cv2.imread(str(self.labels_dir / mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img  = np.zeros((self.S, self.S, 3), np.uint8)
                mask = np.zeros((self.S, self.S), np.uint8)
            img  = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (self.S, self.S))
            mask = cv2.resize(mask, (self.S, self.S), interpolation=cv2.INTER_NEAREST)
            mask = np.clip(mask, 0, 2)
            return img, mask

        def __getitem__(self, idx):
            group = self.groups[idx].reset_index(drop=True)
            rows  = group.sample(min(self.T, len(group)), random_state=idx).sort_index()
            # Pad nếu clip ngắn hơn T frames
            while len(rows) < self.T:
                rows = pd.concat([rows, rows.iloc[[-1]]], ignore_index=True)

            frames, masks = [], []
            # Augmentation params (nhất quán toàn clip)
            do_hflip = self.augment and random.random() > 0.5
            bright   = 1.0 + random.uniform(-0.2, 0.2) if self.augment else 1.0

            for _, row in rows.iterrows():
                img, mask = self._load_frame(row["frame_path"], row["mask_path"])

                if do_hflip:
                    img  = cv2.flip(img, 1)
                    mask = cv2.flip(mask, 1)

                img = np.clip(img.astype(np.float32) * bright, 0, 255)

                # Normalize ImageNet
                img = img / 255.0
                img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

                frames.append(img)
                masks.append(mask)

            frames = np.stack(frames, axis=0)              # [T, H, W, C]
            frames = np.transpose(frames, (3, 0, 1, 2))    # [C, T, H, W]
            masks  = np.stack(masks, axis=0)               # [T, H, W]

            return (
                torch.tensor(frames, dtype=torch.float32),
                torch.tensor(masks, dtype=torch.long),
            )

    # =========================================================================
    # 4. KHỞI TẠO MODEL, OPTIMIZER, LOSS
    # =========================================================================
    model = TimeSformerSegmentation(CFG).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: TimeSformerSegmentation")
    print(f"  Total params: {total_params / 1e6:.1f}M")

    # Differential LR: temporal attention lr×1, rest lr×0.1
    temporal_params, other_params = [], []
    for name, p in model.named_parameters():
        if "time" in name or "temporal" in name:
            temporal_params.append(p)
        else:
            other_params.append(p)

    optimizer = optim.AdamW([
        {"params": other_params,    "lr": CFG["lr"] * 0.1,  "name": "spatial"},
        {"params": temporal_params, "lr": CFG["lr"],        "name": "temporal"},
    ], weight_decay=CFG["weight_decay"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=1e-6
    )

    # Class-weighted cross-entropy để xử lý mất cân bằng BG >> Fire
    class_weights = torch.tensor([0.5, 2.0, 1.5], device=device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    def dice_loss(pred, target, num_classes=3, eps=1e-6):
        """Soft Dice loss — ổn định cho lớp nhỏ"""
        pred = F.softmax(pred, dim=1)
        total = 0.
        for c in range(1, num_classes):  # Skip background
            p = pred[:, c]
            t = (target == c).float()
            inter = (p * t).sum()
            total += 1 - (2 * inter + eps) / (p.sum() + t.sum() + eps)
        return total / (num_classes - 1)

    def combined_loss(pred_frames, masks):
        # pred_frames: [B, T, C, H, W]
        # masks      : [B, T, H, W]
        B, T, C, H, W = pred_frames.shape
        pred  = pred_frames.view(B * T, C, H, W)
        tgt   = masks.view(B * T, H, W)
        return 0.6 * ce_loss(pred, tgt) + 0.4 * dice_loss(pred, tgt)

    # =========================================================================
    # 5. DATASETS & DATALOADERS
    # =========================================================================
    ds_train = FireSmokeClipDataset(
        CFG["train_csv"], CFG["frames_dir"], CFG["labels_dir"],
        num_frames=CFG["num_frames"], img_size=CFG["img_size"], augment=True
    )
    ds_val = FireSmokeClipDataset(
        CFG["val_csv"], CFG["frames_dir"], CFG["labels_dir"],
        num_frames=CFG["num_frames"], img_size=CFG["img_size"], augment=False
    )

    dl_train = DataLoader(
        ds_train, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"]
    )
    dl_val = DataLoader(
        ds_val, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"]
    )

    # =========================================================================
    # 6. TRAINING LOOP
    # =========================================================================
    scaler = torch.cuda.amp.GradScaler()  # AMP Mixed Precision

    best_val_loss  = float("inf")
    no_improve     = 0
    best_weights   = None
    history        = []

    print(f"\n  Training: {len(ds_train)} clips | Validation: {len(ds_val)} clips")
    print(f"  Batch size: {CFG['batch_size']} | Frames per clip: {CFG['num_frames']}")
    print(f"  Epochs: {CFG['epochs']} | Patience: {CFG['patience']}")
    print("=" * 70)

    for epoch in range(CFG["epochs"]):
        # --- TRAIN ---
        model.train()
        train_loss = 0.
        n_train = 0

        bar = tqdm(dl_train, desc=f"[Epoch {epoch+1:02d}/{CFG['epochs']}] Train")
        for frames, masks in bar:
            frames = frames.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(frames)            # [B, T, C, H, W]
                loss = combined_loss(pred, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_train    += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = train_loss / max(n_train, 1)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.
        n_val = 0
        ious = {0: [], 1: [], 2: []}

        with torch.no_grad():
            for frames, masks in tqdm(dl_val, desc=f"[Epoch {epoch+1:02d}/{CFG['epochs']}] Val  "):
                frames = frames.to(device)
                masks  = masks.to(device)
                with torch.cuda.amp.autocast():
                    pred = model(frames)
                    loss = combined_loss(pred, masks)
                val_loss += loss.item()
                n_val    += 1

                # Compute IoU per class
                B, T, C, H, W = pred.shape
                p_flat = pred.view(B*T, C, H, W).argmax(dim=1).cpu().numpy().flatten()
                m_flat = masks.view(B*T, H, W).cpu().numpy().flatten()
                for c in range(3):
                    inter = ((p_flat == c) & (m_flat == c)).sum()
                    union = ((p_flat == c) | (m_flat == c)).sum()
                    if union > 0:
                        ious[c].append(inter / union)

        avg_val  = val_loss / max(n_val, 1)
        miou     = np.mean([np.mean(v) for v in ious.values() if v])
        iou_fire = np.mean(ious[1]) if ious[1] else 0.
        iou_smk  = np.mean(ious[2]) if ious[2] else 0.

        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train,
            "val_loss": avg_val,
            "mIoU": float(miou),
            "IoU_fire": float(iou_fire),
            "IoU_smoke": float(iou_smk),
        })

        print(f"\n  Epoch {epoch+1:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        print(f"  mIoU: {miou:.4f} | IoU_fire: {iou_fire:.4f} | IoU_smoke: {iou_smk:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve    = 0
            best_weights  = copy.deepcopy(model.state_dict())
            print(f"  [OK] New best Val Loss: {best_val_loss:.4f} — checkpoint saved!")
        else:
            no_improve += 1
            print(f"  No improve: {no_improve}/{CFG['patience']}")
            if no_improve >= CFG["patience"]:
                print(f"\n  [EARLY STOPPING] Epoch {epoch+1}")
                break

    # =========================================================================
    # 7. LƯU MODEL & HISTORY
    # =========================================================================
    import glob, re, json as json_mod

    model.load_state_dict(best_weights)
    existing = glob.glob("/data/timesformer_temporal_v*.pth")
    max_v = max(
        (int(re.search(r"_v(\d+)\.pth$", f).group(1))
         for f in existing if re.search(r"_v(\d+)\.pth$", f)),
        default=0,
    )
    save_path = f"/data/timesformer_temporal_v{max_v + 1}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": CFG,
        "best_val_loss": best_val_loss,
        "history": history,
    }, save_path)

    hist_path = f"/data/timesformer_training_history.json"
    with open(hist_path, "w") as f:
        json_mod.dump(history, f, indent=2)

    vol.commit()

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Model saved  : {save_path}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Best mIoU    : {max(h['mIoU'] for h in history):.4f}")
    print(f"  IoU_fire     : {max(h['IoU_fire'] for h in history):.4f}")
    print(f"  IoU_smoke    : {max(h['IoU_smoke'] for h in history):.4f}")
    print("=" * 70)


# =============================================================================
# LOCAL ENTRYPOINT
# =============================================================================
@app.local_entrypoint()
def main():
    print("[INFO] Launching TimeSformer Temporal Segmentation Training on Modal H100...")
    train_temporal.remote()
    print("[INFO] Job dispatched to Modal GPU. Monitor at https://modal.com/apps/usecolabgpu02")
