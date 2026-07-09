import modal
import os

app = modal.App("fire-smoke-temporal-evaluation")
volume = modal.Volume.from_name("fire_smoke_dataset")

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "opencv-python-headless", "einops", "numpy", "scikit-learn"
)

@app.function(image=image, volumes={"/data": volume}, gpu="T4", timeout=3600)
def evaluate_full_model_on_modal():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import rearrange, repeat
    import cv2
    import numpy as np
    import glob
    import re
    import os
    from sklearn.metrics import (
        classification_report, accuracy_score,
        roc_auc_score, average_precision_score,
        hamming_loss, multilabel_confusion_matrix
    )

    print("🚀 ĐÁNH GIÁ MÔ HÌNH FULL SPATIO-TEMPORAL...")

    # ==========================================
    # 1. KIẾN TRÚC (giống train_temporal.py)
    # ==========================================
    def drop_path(x, drop_prob=0., training=False):
        if drop_prob == 0. or not training: return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x.div(keep_prob) * torch.rand(shape, dtype=x.dtype, device=x.device).floor_()

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.):
            super().__init__(); self.drop_prob = drop_prob
        def forward(self, x): return drop_path(x, self.drop_prob, self.training)

    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
            super().__init__()
            hf = hidden_features or in_features; of = out_features or in_features
            self.fc1 = nn.Linear(in_features, hf); self.act = nn.GELU()
            self.fc2 = nn.Linear(hf, of); self.drop = nn.Dropout(drop)
        def forward(self, x): return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

    class Attention(nn.Module):
        def __init__(self, dim, heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
            super().__init__()
            self.heads = heads; head_dim = dim // heads; self.scale = head_dim ** -0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim); self.proj_drop = nn.Dropout(proj_drop)
        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = self.attn_drop((q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1))
            return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))

    class DividedSpaceTimeBlock(nn.Module):
        def __init__(self, dim, heads, mlp_ratio=4., qkv_bias=True,
                     drop=0., attn_drop=0., drop_path_rate=0.1):
            super().__init__()
            mlp_dim = int(dim * mlp_ratio)
            self.temporal_norm1 = nn.LayerNorm(dim)
            self.temporal_attn  = Attention(dim, heads=heads, qkv_bias=qkv_bias,
                                            attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc    = nn.Linear(dim, dim)
            self.norm1 = nn.LayerNorm(dim)
            self.attn  = Attention(dim, heads=heads, qkv_bias=qkv_bias,
                                   attn_drop=attn_drop, proj_drop=drop)
            self.norm2     = nn.LayerNorm(dim)
            self.mlp       = Mlp(in_features=dim, hidden_features=mlp_dim, drop=drop)
            self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        def forward(self, x, T):
            b, t, n, d = x.shape
            xt = rearrange(x, 'b t n d -> (b n) t d')
            res_temporal = self.drop_path(self.temporal_fc(self.temporal_attn(self.temporal_norm1(xt))))
            x = x + rearrange(res_temporal, '(b n) t d -> b t n d', b=b, n=n)
            xs = rearrange(x, 'b t n d -> (b t) n d')
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))
            x = x + rearrange(res_spatial, '(b t) n d -> b t n d', b=b, t=t)
            xm = rearrange(x, 'b t n d -> (b t n) d')
            x = x + rearrange(self.drop_path(self.mlp(self.norm2(xm))),
                               '(b t n) d -> b t n d', b=b, t=t, n=n)
            return x

    class TimeSformerFull(nn.Module):
        def __init__(self, image_size=224, patch_size=16, num_frames=8, num_classes=2,
                     dim=256, depth=6, heads=8, mlp_ratio=4., qkv_bias=True,
                     drop=0.1, attn_drop=0.1, drop_path_rate=0.1):
            super().__init__()
            self.num_frames = num_frames; self.patch_size = patch_size
            num_patches = (image_size // patch_size) ** 2
            self.to_patch_embedding = nn.Linear(3 * patch_size ** 2, dim)
            self.cls_token     = nn.Parameter(torch.zeros(1, 1, 1, dim))
            self.pos_embedding = nn.Parameter(torch.zeros(1, 1, num_patches + 1, dim))
            self.time_embedding = nn.Parameter(torch.zeros(1, num_frames, 1, dim))
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            self.layers = nn.ModuleList([
                DividedSpaceTimeBlock(dim=dim, heads=heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      drop=drop, attn_drop=attn_drop, drop_path_rate=dpr[i])
                for i in range(depth)
            ])
            self.norm     = nn.LayerNorm(dim)
            self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        def forward(self, video):
            x = self.to_patch_embedding(
                rearrange(video, 'b c t (h p1) (w p2) -> b t (h w) (c p1 p2)',
                          p1=self.patch_size, p2=self.patch_size))
            b, t, n, d = x.shape
            cls_tokens = repeat(self.cls_token, '() () () d -> b t () d', b=b, t=t)
            x = torch.cat((cls_tokens, x), dim=2)
            x = x + self.pos_embedding[:, :, :n + 1]
            x[:, :, 1:] = x[:, :, 1:] + self.time_embedding[:, :t, :]
            for layer in self.layers: x = layer(x, t)
            x = self.norm(x)
            return self.mlp_head(x[:, :, 0, :].mean(dim=1))

    # ==========================================
    # 2. VIDEO DATA LOADER (không augmentation)
    # ==========================================
    def load_video_clip(video_path, num_frames=8):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1: cap.release(); return None
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
        frames = []; last = np.zeros((224, 224, 3), dtype=np.float32)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None: frames.append(last.copy()); continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
            last = frame; frames.append(frame)
        cap.release()
        return np.transpose(np.stack(frames, axis=0), (3, 0, 1, 2)).astype(np.float32)

    def get_label(folder_name):
        n = folder_name.lower()
        return [1.0 if 'fire' in n else 0.0, 1.0 if 'smoke' in n else 0.0]

    def video_batch_generator(base_dir, num_frames=8, batch_size=8):
        video_paths, labels = [], []
        for folder in sorted(os.listdir(base_dir)):
            fp = os.path.join(base_dir, folder)
            if not os.path.isdir(fp): continue
            label = get_label(folder)
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4']:
                for vp in glob.glob(os.path.join(fp, ext)):
                    video_paths.append(vp); labels.append(label)
        print(f"  📹 {len(video_paths)} test videos")
        batch_clips, batch_labels = [], []
        for vp, lbl in zip(video_paths, labels):
            clip = load_video_clip(vp, num_frames=num_frames)
            if clip is None: continue
            batch_clips.append(clip); batch_labels.append(lbl)
            if len(batch_clips) == batch_size:
                yield torch.tensor(np.array(batch_clips)), torch.tensor(np.array(batch_labels))
                batch_clips, batch_labels = [], []
        if batch_clips:
            yield torch.tensor(np.array(batch_clips)), torch.tensor(np.array(batch_labels))

    # ==========================================
    # 3. LOAD MODEL (tự động version mới nhất)
    # ==========================================
    existing = glob.glob('/data/full_fire_smoke_weights_v*.pth')
    if not existing:
        print("❌ Không tìm thấy full model weights trong /data!")
        return

    latest_path = max(existing, key=lambda f: int(re.search(r'_v(\d+)\.pth$', f).group(1)))
    latest_v    = int(re.search(r'_v(\d+)\.pth$', latest_path).group(1))
    print(f"🔄 Load model: {latest_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSformerFull(
        image_size=224, patch_size=16, num_frames=8, num_classes=2,
        dim=256, depth=6, heads=8, mlp_ratio=4., qkv_bias=True,
        drop=0.1, attn_drop=0.1, drop_path_rate=0.1
    ).to(device)
    model.load_state_dict(torch.load(latest_path, map_location=device))
    model.eval()

    # ==========================================
    # 4. INFERENCE
    # ==========================================
    test_dir = '/data/video_dataset/test'
    test_gen = video_batch_generator(test_dir, num_frames=8, batch_size=8)

    all_preds, all_probs, all_targets = [], [], []
    print("⏳ Đang chạy inference trên test set...")
    with torch.no_grad():
        for clips, targets in test_gen:
            probs = torch.sigmoid(model(clips.to(device)))
            all_probs.append(probs.cpu().numpy())
            all_preds.append((probs > 0.5).float().cpu().numpy())
            all_targets.append(targets.numpy())

    all_probs   = np.vstack(all_probs)
    all_preds   = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    target_names = ['Fire (Lửa)', 'Smoke (Khói)']

    # ==========================================
    # 5. TÍNH METRICS
    # ==========================================
    report      = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0)
    exact_match = accuracy_score(all_targets, all_preds)
    h_loss      = hamming_loss(all_targets, all_preds)

    try:
        roc_auc_pc  = roc_auc_score(all_targets, all_probs, average=None)
        roc_auc_mac = roc_auc_score(all_targets, all_probs, average='macro')
    except ValueError:
        roc_auc_pc = [0.0, 0.0]; roc_auc_mac = 0.0

    try:
        ap_pc  = average_precision_score(all_targets, all_probs, average=None)
        ap_mac = average_precision_score(all_targets, all_probs, average='macro')
    except ValueError:
        ap_pc = [0.0, 0.0]; ap_mac = 0.0

    mcm = multilabel_confusion_matrix(all_targets, all_preds)

    # ==========================================
    # 6. BÁO CÁO
    # ==========================================
    full_report = f"""
==================================================
 📊 BÁO CÁO ĐÁNH GIÁ FULL SPATIO-TEMPORAL (v{latest_v})
     Model: TimeSformerFull | num_frames=8
==================================================

1. KẾT QUẢ CƠ BẢN (Threshold = 0.5)
--------------------------------------------------
{report}
- Exact Match Ratio  : {exact_match*100:.2f}%
- Hamming Loss       : {h_loss:.4f}

2. ĐÁNH GIÁ CHUYÊN SÂU (Threshold-Independent)
--------------------------------------------------
[ROC-AUC]
- Fire (Lửa)   : {roc_auc_pc[0]:.4f}
- Smoke (Khói) : {roc_auc_pc[1]:.4f}
- Macro Avg    : {roc_auc_mac:.4f}

[Average Precision / mAP]
- Fire (Lửa)   : {ap_pc[0]:.4f}
- Smoke (Khói) : {ap_pc[1]:.4f}
- Macro Avg    : {ap_mac:.4f}

3. CONFUSION MATRIX
--------------------------------------------------
[Fire]
  TN={mcm[0][0][0]:<5} FP={mcm[0][0][1]:<5}
  FN={mcm[0][1][0]:<5} TP={mcm[0][1][1]:<5}

[Smoke]
  TN={mcm[1][0][0]:<5} FP={mcm[1][0][1]:<5}
  FN={mcm[1][1][0]:<5} TP={mcm[1][1][1]:<5}
==================================================
"""
    print(full_report)

    save_path = f'/data/evaluation_temporal_v{latest_v}.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(full_report)
    print(f"📝 Kết quả lưu tại: {save_path}")


@app.local_entrypoint()
def main():
    evaluate_full_model_on_modal.remote()
