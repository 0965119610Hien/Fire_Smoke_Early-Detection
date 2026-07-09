import modal
import os

app = modal.App("fire-smoke-evaluation")
volume = modal.Volume.from_name("fire_smoke_dataset")

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "opencv-python-headless", "einops", "numpy", "scikit-learn" 
)

@app.function(image=image, volumes={"/data": volume}, gpu="T4", timeout=3600)
def evaluate_model_on_modal():
    import torch
    import torch.nn as nn
    from einops import rearrange, repeat
    import cv2
    import numpy as np
    import glob
    import re
    from sklearn.metrics import (
        classification_report, 
        accuracy_score,
        roc_auc_score,
        average_precision_score,
        hamming_loss,
        multilabel_confusion_matrix
    )

    print("🚀 ĐANG KHỞI ĐỘNG TRÌNH ĐÁNH GIÁ ĐA CHIỀU TRÊN MODAL...")

    # --- KHỐI MẠNG LÕI TIMESFORMER ---
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

    # --- HÀM LOAD DỮ LIỆU TEST ---
    def custom_yolo_to_classification_generator(image_dir, batch_size=32):
        label_dir = image_dir.replace('images', 'labels')
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
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

    # --- TÌM VERSION MỚI NHẤT ĐỂ ĐÁNH GIÁ ---
    existing_models = glob.glob('/data/spatial_fire_smoke_weights_v*.pth')
    if not existing_models:
        print("❌ LỖI: Không tìm thấy file model nào trong Volume /data!")
        return

    latest_version = 0
    latest_model_path = existing_models[0]
    
    for f in existing_models:
        match = re.search(r'_v(\d+)\.pth$', f)
        if match:
            v = int(match.group(1))
            if v > latest_version:
                latest_version = v
                latest_model_path = f

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSformer(image_size=224, patch_size=16, num_frames=1, num_classes=2, dim=256, depth=6).to(device)
    
    print(f"🔄 Tự động load Model phiên bản mới nhất: {latest_model_path}")
    model.load_state_dict(torch.load(latest_model_path, map_location=device))
    model.eval()

    test_dir = '/data/merged_dataset/test/images'
    test_gen = custom_yolo_to_classification_generator(test_dir, batch_size=64)
    
    all_preds, all_probs, all_targets = [], [], []

    print(f"⏳ Đang tiến hành chạy inference trên tập Test...")
    with torch.no_grad():
        for batch_imgs, batch_targets in test_gen:
            # Lấy xác suất nguyên thủy (0.0 -> 1.0) để tính toán ROC-AUC và mAP
            probs = torch.sigmoid(model(batch_imgs.unsqueeze(2).to(device)))
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append((probs > 0.5).float().cpu().numpy())
            all_targets.append(batch_targets.numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    target_names = ['Fire (Lửa)', 'Smoke (Khói)']

    # ========================================================
    # TÍNH TOÁN CÁC BENCHMARK CHUYÊN SÂU
    # ========================================================
    
    # 1. Cơ bản: Report & Exact Match
    report = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0)
    exact_match = accuracy_score(all_targets, all_preds)
    
    # 2. Hamming Loss: Tỉ lệ dự đoán sai trên tổng số nhãn (Càng thấp càng tốt)
    h_loss = hamming_loss(all_targets, all_preds)
    
    # 3. ROC-AUC: Đo lường độ nhạy của mô hình với ngưỡng
    try:
        roc_auc_per_class = roc_auc_score(all_targets, all_probs, average=None)
        roc_auc_macro = roc_auc_score(all_targets, all_probs, average='macro')
    except ValueError:
        roc_auc_per_class = [0.0, 0.0]
        roc_auc_macro = 0.0
        
    # 4. Average Precision (mAP): Cực kỳ quan trọng với tập dữ liệu mất cân bằng
    try:
        ap_per_class = average_precision_score(all_targets, all_probs, average=None)
        ap_macro = average_precision_score(all_targets, all_probs, average='macro')
    except ValueError:
        ap_per_class = [0.0, 0.0]
        ap_macro = 0.0
        
    # 5. Multilabel Confusion Matrix: TN, FP, FN, TP cho từng class riêng biệt
    mcm = multilabel_confusion_matrix(all_targets, all_preds)

    # --- TẠO BÁO CÁO TOÀN DIỆN ---
    full_report_text = f"""
==================================================
 📊 BÁO CÁO ĐÁNH GIÁ TOÀN DIỆN (VERSION {latest_version})
==================================================

1. KẾT QUẢ PHÂN LOẠI CƠ BẢN (Threshold = 0.5)
--------------------------------------------------
{report}

- Độ chính xác tuyệt đối (Exact Match Ratio) : {exact_match*100:.2f}%
- Hamming Loss (Tỷ lệ dự đoán nhãn sai)      : {h_loss:.4f}

2. ĐÁNH GIÁ CHUYÊN SÂU (Threshold-Independent)
--------------------------------------------------
[ROC-AUC] (Khả năng phân biệt Thật/Giả)
- Fire (Lửa)   : {roc_auc_per_class[0]:.4f}
- Smoke (Khói) : {roc_auc_per_class[1]:.4f}
- Trung bình   : {roc_auc_macro:.4f}

[Average Precision / mAP] (Hiệu năng với dữ liệu lệch)
- Fire (Lửa)   : {ap_per_class[0]:.4f}
- Smoke (Khói) : {ap_per_class[1]:.4f}
- Trung bình   : {ap_macro:.4f}

3. MA TRẬN NHẦM LẪN CHI TIẾT TỪNG LỚP
--------------------------------------------------
[Lớp: Fire (Lửa)]
[[True Negative (TN): {mcm[0][0][0]:<5}  | False Positive (FP): {mcm[0][0][1]:<5} ]
 [False Negative (FN): {mcm[0][1][0]:<5} | True Positive (TP): {mcm[0][1][1]:<5} ]]

[Lớp: Smoke (Khói)]
[[True Negative (TN): {mcm[1][0][0]:<5}  | False Positive (FP): {mcm[1][0][1]:<5} ]
 [False Negative (FN): {mcm[1][1][0]:<5} | True Positive (TP): {mcm[1][1][1]:<5} ]]
==================================================
"""

    print(full_report_text)

    # Lưu kết quả xuống file để tiện theo dõi sau này
    eval_save_path = f'/data/evaluation_v{latest_version}.txt'
    with open(eval_save_path, 'w', encoding='utf-8') as f:
        f.write(full_report_text)
    
    print(f"📝 Đã lưu lại lịch sử kết quả đánh giá tại: {eval_save_path}")

@app.local_entrypoint()
def main():
    evaluate_model_on_modal.remote()