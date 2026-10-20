import modal
import os

app = modal.App("fire-smoke-detection-training")
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
    import copy 

    print("🚀 ĐÃ KHỞI ĐỘNG MÁY CHỦ MODAL THÀNH CÔNG (GPU: A10G)!")

    # ==========================================
    # 1. KHỐI MẠNG LÕI TIMESFORMER (DETECTION)
    # ==========================================
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

    class TimeSformerDetection(nn.Module):
        def __init__(self, image_size=224, patch_size=16, num_frames=8, 
                     dim=256, depth=6, heads=8, dim_head=64, mlp_dim=512, dropout=0.1):
            super().__init__()
            num_patches = (image_size // patch_size) ** 2
            self.to_patch_embedding = nn.Linear(3 * patch_size ** 2, dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))
            self.pos_embedding = nn.Parameter(torch.randn(1, 1, num_patches + 1, dim))
            self.time_embedding = nn.Parameter(torch.randn(1, num_frames, 1, dim))
            self.layers = nn.ModuleList([DividedSpaceTimeBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])
            
            # Đổi sang Lớp Detection: Dự đoán 6 giá trị (P_fire, P_smoke, x, y, w, h) cho từng patch
            self.detection_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 6) 
            )
            
        def forward(self, video):
            x = self.to_patch_embedding(rearrange(video, 'b c f (h p1) (w p2) -> b f (h w) (c p1 p2)', p1=16, p2=16))
            b, t, n, d = x.shape
            x = torch.cat((repeat(self.cls_token, '() () () d -> b t () d', b=b, t=t), x), dim=2)
            x = x + self.pos_embedding[:, :, :n+1]
            x[:, :, 1:] = x[:, :, 1:] + self.time_embedding[:, :t, :]
            for layer in self.layers: x = layer(x, t)
            
            # Lấy 196 patches (bỏ cls_token ở index 0)
            patches = x[:, :, 1:, :] # shape: (b, 1, 196, dim)
            patches = patches.squeeze(1) # shape: (b, 196, dim)
            
            out = self.detection_head(patches) # shape: (b, 196, 6)
            # Dùng Sigmoid để ép toàn bộ xác suất và tọa độ về khoảng 0-1
            return torch.sigmoid(out)

    # ==========================================
    # 2. HÀM LOAD DỮ LIỆU DETECTION (CODE CHAY TỪ TXT)
    # ==========================================
    def custom_yolo_to_detection_generator(image_dir, batch_size=32, shuffle=True, augment=False):
        label_dir = image_dir.replace('images', 'labels')
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if shuffle: random.shuffle(image_files)

        batch_imgs, batch_targets = [], []
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            txt_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
            boxes = []
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            boxes.append([float(x) for x in parts])

            # --- DATA AUGMENTATION (XỬ LÝ CẢ BOX) ---
            if augment:
                is_flipped = random.random() > 0.5
                if is_flipped:
                    img = cv2.flip(img, 1)
                    # Lật tọa độ X của Bounding Box
                    for b in boxes:
                        b[1] = 1.0 - b[1] 
                
                if random.random() > 0.5:
                    alpha = random.uniform(0.8, 1.2) 
                    beta = random.randint(-15, 15)   
                    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            
            # --- TẠO TARGET TENSOR (LƯỚI 196 MAPS) ---
            # Shape: (196 patches, 6 values)
            target = np.zeros((196, 6), dtype=np.float32)
            
            for b in boxes:
                cls_id, x, y, w, h = b
                cls_id = int(cls_id)
                
                # Tìm Patch tương ứng với tâm của Bounding Box (Lưới 14x14)
                grid_x = int(x * 14)
                grid_y = int(y * 14)
                # Chặn viền
                if grid_x >= 14: grid_x = 13
                if grid_y >= 14: grid_y = 13
                
                patch_idx = grid_y * 14 + grid_x
                
                # Gán nhãn Class
                if cls_id == 0: target[patch_idx, 0] = 1.0 # Fire
                elif cls_id == 1: target[patch_idx, 1] = 1.0 # Smoke
                
                # Gán Box
                target[patch_idx, 2:6] = [x, y, w, h]
            
            batch_imgs.append(img)
            batch_targets.append(target)
            
            if len(batch_imgs) == batch_size:
                yield torch.tensor(np.array(batch_imgs)), torch.tensor(np.array(batch_targets))
                batch_imgs, batch_targets = [], []

        if len(batch_imgs) > 0:
            yield torch.tensor(np.array(batch_imgs)), torch.tensor(np.array(batch_targets))

    # ==========================================
    # 3. HÀM LOSS TỰ CODE CHO DETECTION
    # ==========================================
    def detection_loss(preds, targets):
        """
        preds, targets shape: (batch_size, 196, 6)
        """
        # Tách phần Classification (0, 1) và Box Regression (2, 3, 4, 5)
        pred_cls = preds[..., 0:2]
        target_cls = targets[..., 0:2]
        
        pred_box = preds[..., 2:6]
        target_box = targets[..., 2:6]
        
        # Mask kiểm tra xem Patch đó có vật thể không (Nếu có Lửa hoặc Khói thì = 1)
        obj_mask = (target_cls.sum(dim=-1) > 0).unsqueeze(-1) # shape: (batch, 196, 1)
        
        # 1. Loss Phân loại (Tính trên toàn bộ 196 patches để phạt các dự đoán sai)
        cls_loss = nn.BCELoss()(pred_cls, target_cls)
        
        # 2. Loss Bounding Box (CHỈ TÍNH TẠI CÁC PATCH CÓ VẬT THỂ)
        # Tính Sai số bình phương trung bình (MSE)
        box_mse = nn.MSELoss(reduction='none')(pred_box, target_box)
        # Dùng mask lọc ra và tính trung bình
        box_loss = (box_mse * obj_mask).sum() / (obj_mask.sum() + 1e-6)
        
        # Cộng gộp loss, đặt trọng số x5 cho Box để ép mô hình học khung chính xác
        return cls_loss + 5.0 * box_loss

    # ==========================================
    # 4. KHỞI TẠO & HUẤN LUYỆN
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Khởi tạo mạng Detection mới
    model = TimeSformerDetection(image_size=224, patch_size=16, num_frames=1, dim=256, depth=6).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_dir = '/data/Image_dataset/train/images'
    val_dir   = '/data/Image_dataset/val/images' 
    
    epochs = 100
    batch_size = 64

    # CẤU HÌNH EARLY STOPPING
    patience = 5  
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # --- GIAI ĐOẠN TRAINING ---
        model.train()
        running_train_loss = 0.0
        train_batches = 0
        
        train_gen = custom_yolo_to_detection_generator(
            train_dir, batch_size=batch_size, shuffle=True, augment=True
        )
        
        for batch_imgs, batch_targets in train_gen:
            batch_imgs, batch_targets = batch_imgs.unsqueeze(2).to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_imgs)
            loss = detection_loss(preds, batch_targets)
            
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            train_batches += 1
            if train_batches % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch {train_batches} | Train Loss: {loss.item():.4f}")
                
        avg_train_loss = running_train_loss / train_batches

        # --- GIAI ĐOẠN VALIDATION ---
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            val_gen = custom_yolo_to_detection_generator(
                val_dir, batch_size=batch_size, shuffle=False, augment=False
            )
            for batch_imgs, batch_targets in val_gen:
                batch_imgs, batch_targets = batch_imgs.unsqueeze(2).to(device), batch_targets.to(device)
                preds = model(batch_imgs)
                val_loss = detection_loss(preds, batch_targets)
                running_val_loss += val_loss.item()
                val_batches += 1
                
        avg_val_loss = running_val_loss / (val_batches if val_batches > 0 else 1)
        
        print(f"---> HOÀN THÀNH EPOCH {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- KIỂM TRA EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
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
                model.load_state_dict(best_model_weights)
                break

    # ==========================================
    # 4. LƯU MÔ HÌNH DETECTION
    # ==========================================
    existing_models = glob.glob('/data/spatial_fire_smoke_det_weights_v*.pth')
    max_version = 0
    for f in existing_models:
        match = re.search(r'_v(\d+)\.pth$', f)
        if match:
            max_version = max(max_version, int(match.group(1)))
    
    next_version = max_version + 1
    save_path = f'/data/spatial_fire_smoke_det_weights_v{next_version}.pth'
    
    torch.save(model.state_dict(), save_path)
    print(f"🎉 Đã lưu trọng số khối Spatial Detection tốt nhất (v{next_version}) tại: {save_path}")

@app.local_entrypoint()
def main():
    train_model_on_modal.remote()