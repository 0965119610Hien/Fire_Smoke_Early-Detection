import modal
import os

app = modal.App("fire-smoke-spatial-training")
volume = modal.Volume.from_name("fire_smoke_dataset")

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "opencv-python-headless", "einops", "numpy"
)

@app.function(image=image, volumes={"/data": volume}, gpu="A10G", timeout=86400)
def train_model_on_modal():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from einops import rearrange, repeat
    import cv2
    import random
    import numpy as np
    import glob
    import re
    import copy # Thêm thư viện để copy trọng số tốt nhất

    print("🚀 ĐÃ KHỞI ĐỘNG MÁY CHỦ MODAL THÀNH CÔNG (GPU: A10G)!")

    # --- KHỐI MẠNG LÕI ---
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
            return self.to_out(rearrange(torch.matmul(dots.softmax(dim=-1), v), 'b h n d -> b n (h d)'))

    class DividedSpaceTimeBlock(nn.Module):
        def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
            super().__init__()
            self.time_norm1 = nn.LayerNorm(dim)
            self.time_attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.space_norm1 = nn.LayerNorm(dim)
            self.space_attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_dim, drop=dropout)
        def forward(self, x, f):
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

    class TimeSformer(nn.Module):
        def __init__(self, image_size=224, patch_size=16, num_frames=8, num_classes=2, 
                     dim=256, depth=6, heads=8, dim_head=64, mlp_dim=512, dropout=0.1):
            super().__init__()
            num_patches = (image_size // patch_size) ** 2
            self.to_patch_embedding = nn.Linear(3 * patch_size ** 2, dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))
            self.pos_embedding = nn.Parameter(torch.randn(1, 1, num_patches + 1, dim))
            self.time_embedding = nn.Parameter(torch.randn(1, num_frames, 1, dim))
            self.layers = nn.ModuleList([DividedSpaceTimeBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])
            self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        def forward(self, video):
            x = self.to_patch_embedding(rearrange(video, 'b c f (h p1) (w p2) -> b f (h w) (c p1 p2)', p1=16, p2=16))
            b, t, n, d = x.shape
            x = torch.cat((repeat(self.cls_token, '() () () d -> b t () d', b=b, t=t), x), dim=2)
            x = x + self.pos_embedding[:, :, :n+1]
            x[:, :, 1:] = x[:, :, 1:] + self.time_embedding[:, :t, :]
            for layer in self.layers: x = layer(x, t)
            return self.mlp_head(x[:, :, 0, :].mean(dim=1))

    # --- HÀM LOAD DỮ LIỆU THỦ CÔNG ---
    def custom_yolo_to_classification_generator(image_dir, batch_size=32, shuffle=True):
        label_dir = image_dir.replace('images', 'labels')
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if shuffle: random.shuffle(image_files)

        batch_imgs, batch_targets = [], []
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            img = np.transpose(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224)).astype(np.float32) / 255.0, (2, 0, 1))
            target = [0.0, 0.0]
            txt_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) > 0:
                            if int(parts[0]) == 0: target[0] = 1.0 
                            elif int(parts[0]) == 1: target[1] = 1.0 
            
            batch_imgs.append(img)
            batch_targets.append(target)
            if len(batch_imgs) == batch_size:
                yield torch.tensor(np.array(batch_imgs)), torch.tensor(np.array(batch_targets))
                batch_imgs, batch_targets = [], []

        if len(batch_imgs) > 0:
            yield torch.tensor(np.array(batch_imgs)), torch.tensor(np.array(batch_targets))

    # --- KHỞI TẠO & HUẤN LUYỆN ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSformer(image_size=224, patch_size=16, num_frames=1, num_classes=2, dim=256, depth=6).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_dir = '/data/merged_dataset/train/images'
    val_dir = '/data/merged_dataset/val/images' # Thư mục Val để chạy Early Stopping
    
    epochs = 80
    batch_size = 64

    # --- CẤU HÌNH EARLY STOPPING ---
    patience = 5  # Số epoch tối đa chịu đựng nếu Loss không giảm
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # 1. GIAI ĐOẠN TRAINING
        model.train()
        running_train_loss = 0.0
        train_batches = 0
        train_gen = custom_yolo_to_classification_generator(train_dir, batch_size=batch_size, shuffle=True)
        
        for batch_imgs, batch_targets in train_gen:
            batch_imgs, batch_targets = batch_imgs.unsqueeze(2).to(device), batch_targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_imgs), batch_targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            train_batches += 1
            if train_batches % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch {train_batches} | Train Loss: {loss.item():.4f}")
                
        avg_train_loss = running_train_loss / train_batches

        # 2. GIAI ĐOẠN VALIDATION (Tính điểm để xem xét Early Stopping)
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        # Tắt gradient để tiết kiệm VRAM và tăng tốc tính toán
        with torch.no_grad():
            # Không cần shuffle tập Val
            val_gen = custom_yolo_to_classification_generator(val_dir, batch_size=batch_size, shuffle=False)
            for batch_imgs, batch_targets in val_gen:
                batch_imgs, batch_targets = batch_imgs.unsqueeze(2).to(device), batch_targets.to(device)
                outputs = model(batch_imgs)
                val_loss = loss_fn(outputs, batch_targets)
                running_val_loss += val_loss.item()
                val_batches += 1
                
        avg_val_loss = running_val_loss / (val_batches if val_batches > 0 else 1)
        
        print(f"---> HOÀN THÀNH EPOCH {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 3. KIỂM TRA EARLY STOPPING
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Lưu lại trạng thái của mô hình tốt nhất
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"     🌟 Val Loss giảm kỷ lục! Đã lưu trọng số tạm thời.")
        else:
            epochs_no_improve += 1
            print(f"     ⚠️ Val Loss không giảm. Cảnh báo: {epochs_no_improve}/{patience}")
            
            if epochs_no_improve >= patience:
                print("\n" + "="*50)
                print(f"🛑 KÍCH HOẠT EARLY STOPPING TẠI EPOCH {epoch+1}!")
                print(f"Khôi phục trọng số về phiên bản tốt nhất (Val Loss: {best_val_loss:.4f})")
                print("="*50 + "\n")
                # Nạp lại trọng số tốt nhất vào model trước khi thoát vòng lặp
                model.load_state_dict(best_model_weights)
                break

    # --- TỰ ĐỘNG ĐÁNH DẤU PHIÊN BẢN (AUTO-VERSIONING) ---
    # Lúc này mô hình đang giữ trọng số tốt nhất (từ epoch có Val Loss thấp nhất)
    existing_models = glob.glob('/data/spatial_fire_smoke_weights_v*.pth')
    max_version = 0
    for f in existing_models:
        match = re.search(r'_v(\d+)\.pth$', f)
        if match:
            max_version = max(max_version, int(match.group(1)))
    
    next_version = max_version + 1
    save_path = f'/data/spatial_fire_smoke_weights_v{next_version}.pth'
    
    torch.save(model.state_dict(), save_path)
    print(f"🎉 Đã lưu trọng số khối Spatial phiên bản tốt nhất (v{next_version}) tại: {save_path}")

@app.local_entrypoint()
def main():
    train_model_on_modal.remote()