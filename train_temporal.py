import modal
import os

app = modal.App("fire-smoke-temporal-training")
volume = modal.Volume.from_name("fire_smoke_dataset")

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "opencv-python-headless", "einops", "numpy"
)

@app.function(image=image, volumes={"/data": volume}, gpu="H100", timeout=86400)
def train_full_model_on_modal():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from einops import rearrange, repeat
    import cv2
    import random
    import numpy as np
    import glob
    import re
    import copy
    import os

    print("🚀 GIAI ĐOẠN 2: FULL SPATIO-TEMPORAL TRAINING (GPU: H100)!")

    # ==========================================
    # 1. KIẾN TRÚC TIMESFORMER ĐẦY ĐỦ
    # Theo Facebook Research (vit.py):
    #   + DropPath (Stochastic Depth)
    #   + temporal_fc (init zeros để ổn định khi transfer)
    #   + qkv_bias=True
    # ==========================================

    def drop_path(x, drop_prob: float = 0., training: bool = False):
        """Stochastic Depth — regularization quan trọng cho temporal learning"""
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.):
            super().__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            return drop_path(x, self.drop_prob, self.training)

    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
            super().__init__()
            hidden_features = hidden_features or in_features
            out_features = out_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)
        def forward(self, x):
            return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

    class Attention(nn.Module):
        def __init__(self, dim, heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
            super().__init__()
            self.heads = heads
            head_dim = dim // heads
            self.scale = head_dim ** -0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = self.attn_drop((q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1))
            return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))

    class DividedSpaceTimeBlock(nn.Module):
        """
        Block đầy đủ theo Facebook Research:
        - temporal_fc: init zeros để khi load spatial weights,
          temporal contribution ban đầu = 0 → không phá vỡ spatial features
        - DropPath áp dụng cho cả temporal lẫn spatial residual
        """
        def __init__(self, dim, heads, mlp_ratio=4., qkv_bias=True,
                     drop=0., attn_drop=0., drop_path_rate=0.1):
            super().__init__()
            mlp_dim = int(dim * mlp_ratio)

            # Temporal Attention
            self.temporal_norm1 = nn.LayerNorm(dim)
            self.temporal_attn  = Attention(dim, heads=heads, qkv_bias=qkv_bias,
                                            attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc    = nn.Linear(dim, dim)  # init zeros ở _init_weights

            # Spatial Attention
            self.norm1 = nn.LayerNorm(dim)
            self.attn  = Attention(dim, heads=heads, qkv_bias=qkv_bias,
                                   attn_drop=attn_drop, proj_drop=drop)

            # MLP + DropPath
            self.norm2     = nn.LayerNorm(dim)
            self.mlp       = Mlp(in_features=dim, hidden_features=mlp_dim, drop=drop)
            self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        def forward(self, x, T):
            b, t, n, d = x.shape

            # --- TEMPORAL ATTENTION: attend across frames, per patch position ---
            xt = rearrange(x, 'b t n d -> (b n) t d')
            res_temporal = self.drop_path(self.temporal_fc(self.temporal_attn(self.temporal_norm1(xt))))
            x = x + rearrange(res_temporal, '(b n) t d -> b t n d', b=b, n=n)

            # --- SPATIAL ATTENTION: attend across patches, per frame ---
            xs = rearrange(x, 'b t n d -> (b t) n d')
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))
            x = x + rearrange(res_spatial, '(b t) n d -> b t n d', b=b, t=t)

            # --- MLP ---
            xm = rearrange(x, 'b t n d -> (b t n) d')
            x = x + rearrange(self.drop_path(self.mlp(self.norm2(xm))),
                               '(b t n) d -> b t n d', b=b, t=t, n=n)
            return x

    class TimeSformerFull(nn.Module):
        """
        TimeSformer hoàn chỉnh — Giai đoạn 2
        num_frames=8: học được biến đổi không-thời gian của lửa/khói
        """
        def __init__(self, image_size=224, patch_size=16, num_frames=8, num_classes=2,
                     dim=256, depth=6, heads=8, mlp_ratio=4., qkv_bias=True,
                     drop=0.1, attn_drop=0.1, drop_path_rate=0.1):
            super().__init__()
            self.num_frames = num_frames
            self.patch_size = patch_size
            num_patches = (image_size // patch_size) ** 2

            self.to_patch_embedding = nn.Linear(3 * patch_size ** 2, dim)
            self.cls_token     = nn.Parameter(torch.zeros(1, 1, 1, dim))
            self.pos_embedding = nn.Parameter(torch.zeros(1, 1, num_patches + 1, dim))

            # time_embedding: shape [1, num_frames, 1, dim]
            # Khi load từ spatial model (num_frames=1) sẽ được interpolate
            self.time_embedding = nn.Parameter(torch.zeros(1, num_frames, 1, dim))

            # Stochastic depth decay: tuyến tính từ 0 → drop_path_rate
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            self.layers = nn.ModuleList([
                DividedSpaceTimeBlock(
                    dim=dim, heads=heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop, attn_drop=attn_drop, drop_path_rate=dpr[i]
                ) for i in range(depth)
            ])
            self.norm     = nn.LayerNorm(dim)
            self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=.02)
                    if m.bias is not None: nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            # QUAN TRỌNG: temporal_fc init zeros
            # → Lúc đầu temporal contribution = 0, không phá vỡ spatial features đã học
            for layer in self.layers:
                nn.init.zeros_(layer.temporal_fc.weight)
                nn.init.zeros_(layer.temporal_fc.bias)

        def forward(self, video):
            # video: [B, C, T, H, W]
            x = self.to_patch_embedding(
                rearrange(video, 'b c t (h p1) (w p2) -> b t (h w) (c p1 p2)',
                          p1=self.patch_size, p2=self.patch_size)
            )
            b, t, n, d = x.shape

            cls_tokens = repeat(self.cls_token, '() () () d -> b t () d', b=b, t=t)
            x = torch.cat((cls_tokens, x), dim=2)  # [B, T, N+1, D]

            x = x + self.pos_embedding[:, :, :n + 1]
            x[:, :, 1:] = x[:, :, 1:] + self.time_embedding[:, :t, :]

            for layer in self.layers:
                x = layer(x, t)

            x = self.norm(x)
            return self.mlp_head(x[:, :, 0, :].mean(dim=1))  # CLS mean over frames

    # ==========================================
    # 2. TRANSFER LEARNING: Load Spatial Weights
    # ==========================================
    def load_spatial_weights(model, spatial_weights_path, device, target_num_frames=8):
        print(f"\n{'='*50}")
        print(f"🔄 TRANSFER LEARNING TỪ SPATIAL MODEL")
        print(f"{'='*50}")
        print(f"📂 Source: {spatial_weights_path}")

        spatial_sd = torch.load(spatial_weights_path, map_location=device)
        model_sd   = model.state_dict()

        loaded, skipped, resized = [], [], []

        for key, spatial_val in spatial_sd.items():
            if key not in model_sd:
                skipped.append(key)
                continue

            model_val = model_sd[key]

            # Xử lý time_embedding: interpolate T=1 → T=target_num_frames
            if key == 'time_embedding':
                # [1, 1, 1, D] → permute → [1, D, 1, 1] → interpolate → [1, D, T, 1] → permute lại
                t_perm = spatial_val.permute(0, 3, 2, 1)  # [1, D, 1, 1]
                t_resized = F.interpolate(t_perm, size=(1, target_num_frames),
                                          mode='bilinear', align_corners=False)
                model_sd[key] = t_resized.permute(0, 3, 2, 1)  # [1, T, 1, D]
                resized.append(f"  ↕ {key}: {list(spatial_val.shape)} → {list(model_sd[key].shape)}")
                continue

            if spatial_val.shape == model_val.shape:
                model_sd[key] = spatial_val
                loaded.append(key)
            else:
                skipped.append(f"  ✗ {key} shape mismatch {spatial_val.shape} vs {model_val.shape}")

        model.load_state_dict(model_sd, strict=False)

        print(f"\n  ✅ Loaded:  {len(loaded)} / {len(spatial_sd)} layers")
        print(f"  ↕  Resized: {len(resized)} layers")
        for r in resized: print(r)
        print(f"  ✗  Skipped: {len([s for s in skipped if '✗' not in str(s)])} layers")
        print(f"\n  🎯 Spatial features được kế thừa — chỉ train thêm temporal!\n")
        return model

    # ==========================================
    # 3. VIDEO DATA LOADER
    # Cấu trúc: /data/video_dataset/train/{fire,smoke,normal}/*.mp4
    # Multi-label: fire=[1,0], smoke=[0,1], normal=[0,0]
    # ==========================================
    def load_video_clip(video_path, num_frames=8, augment=False):
        """Sample num_frames frames từ video, trả về [C, T, H, W]"""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            cap.release()
            return None

        # Temporal sampling strategy
        if augment and total > num_frames:
            # Random clip start để augment temporal
            start = random.randint(0, total - num_frames)
            indices = list(range(start, start + num_frames))
        else:
            # Uniform sampling — đều đặn suốt video
            indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

        frames = []
        last_valid = np.zeros((224, 224, 3), dtype=np.float32)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                frames.append(last_valid.copy())
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
            last_valid = frame
            frames.append(frame)

        cap.release()

        frames = np.stack(frames, axis=0)  # [T, H, W, C]

        # Spatial augmentation — nhất quán trên toàn clip
        if augment:
            if random.random() > 0.5:
                frames = frames[:, :, ::-1, :].copy()        # Horizontal flip
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)             # Contrast
                beta  = random.randint(-15, 15) / 255.0      # Brightness
                frames = np.clip(frames * alpha + beta, 0.0, 1.0)

        # [T, H, W, C] → [C, T, H, W]
        return np.transpose(frames, (3, 0, 1, 2)).astype(np.float32)

    def get_label(folder_name):
        """Multi-label từ tên thư mục"""
        n = folder_name.lower()
        return [1.0 if 'fire' in n else 0.0,
                1.0 if 'smoke' in n else 0.0]

    def video_batch_generator(base_dir, num_frames=8, batch_size=8,
                               shuffle=True, augment=False):
        video_paths, labels = [], []
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            if not os.path.isdir(folder_path): continue
            label = get_label(folder)
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI']:
                for vp in glob.glob(os.path.join(folder_path, ext)):
                    video_paths.append(vp)
                    labels.append(label)

        label_counts = {'fire': sum(1 for l in labels if l[0] == 1.0),
                        'smoke': sum(1 for l in labels if l[1] == 1.0),
                        'normal': sum(1 for l in labels if l == [0.0, 0.0])}
        print(f"  📹 {len(video_paths)} videos | Fire: {label_counts['fire']} | Smoke: {label_counts['smoke']} | Normal: {label_counts['normal']}")

        if shuffle:
            pairs = list(zip(video_paths, labels))
            random.shuffle(pairs)
            video_paths, labels = zip(*pairs)

        batch_clips, batch_labels = [], []
        for vp, lbl in zip(video_paths, labels):
            clip = load_video_clip(vp, num_frames=num_frames, augment=augment)
            if clip is None: continue
            batch_clips.append(clip)
            batch_labels.append(lbl)
            if len(batch_clips) == batch_size:
                yield torch.tensor(np.array(batch_clips)), torch.tensor(np.array(batch_labels))
                batch_clips, batch_labels = [], []
        if batch_clips:
            yield torch.tensor(np.array(batch_clips)), torch.tensor(np.array(batch_labels))

    # ==========================================
    # 4. KHỞI TẠO & TRANSFER LEARNING
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    NUM_FRAMES = 8

    model = TimeSformerFull(
        image_size=224, patch_size=16, num_frames=NUM_FRAMES, num_classes=2,
        dim=256, depth=6, heads=8, mlp_ratio=4., qkv_bias=True,
        drop=0.1, attn_drop=0.1, drop_path_rate=0.1
    ).to(device)

    # === TRANSFER LEARNING ===
    spatial_weights_path = '/data/spatial_fire_smoke_weights_v2.pth'
    if os.path.exists(spatial_weights_path):
        model = load_spatial_weights(model, spatial_weights_path, device, target_num_frames=NUM_FRAMES)
    else:
        print(f"⚠️  Không tìm thấy {spatial_weights_path}. Train từ đầu...")

    # ==========================================
    # 5. DIFFERENTIAL LEARNING RATE
    # Spatial (pre-trained) → LR x0.1
    # Temporal (mới học)    → LR x1.0
    # ==========================================
    spatial_params, temporal_params = [], []
    for name, param in model.named_parameters():
        if any(k in name for k in ['temporal', 'time_embedding']):
            temporal_params.append(param)
        else:
            spatial_params.append(param)

    base_lr = 1e-4
    optimizer = optim.Adam([
        {'params': spatial_params,  'lr': base_lr * 0.1, 'name': 'spatial'},
        {'params': temporal_params, 'lr': base_lr,       'name': 'temporal'},
    ])

    print(f"\n⚙️  Differential Learning Rate:")
    print(f"   Spatial  params: lr = {base_lr * 0.1:.2e}  ({len(spatial_params)} tensors)")
    print(f"   Temporal params: lr = {base_lr:.2e}  ({len(temporal_params)} tensors)\n")

    # Class weights giải quyết mất cân bằng
    pos_weights = torch.tensor([1.18, 1.71]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # ==========================================
    # 6. TRAINING LOOP
    # ==========================================
    train_dir = '/data/video_dataset/train'
    val_dir   = '/data/video_dataset/val'

    BATCH_SIZE = 8    # Video tốn VRAM hơn nhiều so với ảnh
    EPOCHS     = 50   # Transfer learning hội tụ nhanh hơn
    PATIENCE   = 7

    best_val_loss      = float('inf')
    epochs_no_improve  = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    print(f"🏋️  BẮT ĐẦU TRAINING | Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Patience: {PATIENCE}\n")

    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        n_batches    = 0

        print(f"\n[EPOCH {epoch+1}/{EPOCHS}] --- TRAINING ---")
        train_gen = video_batch_generator(
            train_dir, num_frames=NUM_FRAMES,
            batch_size=BATCH_SIZE, shuffle=True, augment=True
        )
        for clips, targets in train_gen:
            clips, targets = clips.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(clips), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches    += 1
            if n_batches % 10 == 0:
                print(f"  Batch {n_batches:4d} | Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / max(n_batches, 1)

        # --- VALIDATION PHASE ---
        model.eval()
        running_val = 0.0
        n_val = 0
        print(f"[EPOCH {epoch+1}/{EPOCHS}] --- VALIDATION ---")
        with torch.no_grad():
            val_gen = video_batch_generator(
                val_dir, num_frames=NUM_FRAMES,
                batch_size=BATCH_SIZE, shuffle=False, augment=False
            )
            for clips, targets in val_gen:
                clips, targets = clips.to(device), targets.to(device)
                running_val += loss_fn(model(clips), targets).item()
                n_val       += 1

        avg_val_loss = running_val / max(n_val, 1)
        print(f"\n===> EPOCH {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"     🌟 Val Loss kỷ lục ({best_val_loss:.4f})! Lưu trọng số.")
        else:
            epochs_no_improve += 1
            print(f"     ⚠️  Không cải thiện. {epochs_no_improve}/{PATIENCE}")
            if epochs_no_improve >= PATIENCE:
                print(f"\n{'='*50}")
                print(f"🛑 EARLY STOPPING TẠI EPOCH {epoch+1}!")
                print(f"   Khôi phục về Val Loss tốt nhất: {best_val_loss:.4f}")
                print(f"{'='*50}")
                model.load_state_dict(best_model_weights)
                break

    # ==========================================
    # 7. AUTO-VERSIONING & LƯU MODEL
    # ==========================================
    existing = glob.glob('/data/full_fire_smoke_weights_v*.pth')
    max_v = max(
        (int(re.search(r'_v(\d+)\.pth$', f).group(1))
         for f in existing if re.search(r'_v(\d+)\.pth$', f)),
        default=0
    )
    save_path = f'/data/full_fire_smoke_weights_v{max_v + 1}.pth'
    torch.save(model.state_dict(), save_path)

    print(f"\n{'='*50}")
    print(f"🎉 ĐÃ HOÀN THÀNH GIAI ĐOẠN 2!")
    print(f"   Model lưu tại : {save_path}")
    print(f"   Kế thừa từ   : {spatial_weights_path}")
    print(f"   Best Val Loss : {best_val_loss:.4f}")
    print(f"   num_frames    : {NUM_FRAMES}")
    print(f"{'='*50}")


@app.local_entrypoint()
def main():
    train_full_model_on_modal.remote()
