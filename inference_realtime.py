"""
inference_realtime.py
---------------------
Real-time fire & smoke detection từ camera stream / video file
sử dụng Full TimeSformer Spatio-Temporal model.

Cách dùng:
    # Từ camera (webcam index 0):
    python inference_realtime.py --source 0 --weights /data/full_fire_smoke_weights_v1.pth

    # Từ RTSP stream:
    python inference_realtime.py --source rtsp://ip:port/stream --weights /path/to/weights.pth

    # Từ file video:
    python inference_realtime.py --source /path/to/video.mp4 --weights /path/to/weights.pth
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import cv2
import numpy as np
from collections import deque
import time
import os


# ==========================================
# KIẾN TRÚC TIMESFORMER FULL
# (copy từ train_temporal.py để standalone)
# ==========================================

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return x.div(keep_prob) * torch.rand(shape, dtype=x.dtype, device=x.device).floor_()

class DropPath(nn.Module):
    def __init__(self, dp=0.): super().__init__(); self.dp = dp
    def forward(self, x): return drop_path(x, self.dp, self.training)

class Mlp(nn.Module):
    def __init__(self, in_f, hid_f=None, out_f=None, drop=0.):
        super().__init__()
        hid_f = hid_f or in_f; out_f = out_f or in_f
        self.net = nn.Sequential(nn.Linear(in_f, hid_f), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(hid_f, out_f), nn.Dropout(drop))
    def forward(self, x): return self.net(x)

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
                 drop=0., attn_drop=0., drop_path_rate=0.):
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
        self.mlp       = Mlp(dim, mlp_dim, drop=drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, T):
        b, t, n, d = x.shape
        xt = rearrange(x, 'b t n d -> (b n) t d')
        res_t = self.drop_path(self.temporal_fc(self.temporal_attn(self.temporal_norm1(xt))))
        x = x + rearrange(res_t, '(b n) t d -> b t n d', b=b, n=n)
        xs = rearrange(x, 'b t n d -> (b t) n d')
        res_s = self.drop_path(self.attn(self.norm1(xs)))
        x = x + rearrange(res_s, '(b t) n d -> b t n d', b=b, t=t)
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
# INFERENCE ENGINE
# ==========================================

class FireSmokeDetector:
    """
    Real-time detector sử dụng Sliding Window.
    Buffer num_frames frames gần nhất, chạy inference liên tục.
    """
    COLORS = {
        'fire':        (0, 80, 255),   # Đỏ-cam
        'smoke':       (180, 180, 180), # Xám
        'fire+smoke':  (0, 50, 200),   # Đỏ đậm
        'normal':      (0, 200, 80),   # Xanh lá
    }

    def __init__(self, weights_path, num_frames=8, threshold=0.5,
                 device=None, inference_interval=5):
        self.num_frames        = num_frames
        self.threshold         = threshold
        self.inference_interval = inference_interval  # Chạy inference mỗi N frames

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"💻 Device: {self.device}")

        # Load model
        print(f"📂 Loading model từ: {weights_path}")
        self.model = TimeSformerFull(
            image_size=224, patch_size=16, num_frames=num_frames, num_classes=2,
            dim=256, depth=6, heads=8, mlp_ratio=4., qkv_bias=True,
            drop=0.1, attn_drop=0.1, drop_path_rate=0.1
        ).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        print("✅ Model loaded!")

        # Frame buffer (sliding window)
        self.frame_buffer = deque(maxlen=num_frames)
        self.frame_count  = 0

        # Kết quả mới nhất
        self.last_fire_prob  = 0.0
        self.last_smoke_prob = 0.0
        self.last_label      = 'normal'

    def preprocess_frame(self, frame):
        """Chuẩn hóa 1 frame: BGR→RGB→resize→normalize→CHW"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
        return np.transpose(frame, (2, 0, 1))  # [C, H, W]

    @torch.no_grad()
    def infer(self):
        """Chạy inference trên buffer hiện tại"""
        if len(self.frame_buffer) < self.num_frames:
            return  # Chưa đủ frames

        # Stack frames: [T, C, H, W] → [C, T, H, W] → [1, C, T, H, W]
        clip = np.stack(list(self.frame_buffer), axis=0)    # [T, C, H, W]
        clip = np.transpose(clip, (1, 0, 2, 3))             # [C, T, H, W]
        tensor = torch.tensor(clip[np.newaxis]).to(self.device)  # [1, C, T, H, W]

        logits = self.model(tensor)
        probs  = torch.sigmoid(logits)[0].cpu().numpy()

        self.last_fire_prob  = float(probs[0])
        self.last_smoke_prob = float(probs[1])

        fire  = self.last_fire_prob  > self.threshold
        smoke = self.last_smoke_prob > self.threshold

        if fire and smoke: self.last_label = 'fire+smoke'
        elif fire:         self.last_label = 'fire'
        elif smoke:        self.last_label = 'smoke'
        else:              self.last_label = 'normal'

    def process_frame(self, frame):
        """Xử lý 1 frame: thêm vào buffer, infer nếu đến lượt"""
        self.frame_buffer.append(self.preprocess_frame(frame))
        self.frame_count += 1

        # Chỉ infer mỗi inference_interval frames để tiết kiệm tài nguyên
        if self.frame_count % self.inference_interval == 0:
            self.infer()

        return self.draw_overlay(frame)

    def draw_overlay(self, frame):
        """Vẽ overlay kết quả lên frame"""
        h, w = frame.shape[:2]
        color = self.COLORS.get(self.last_label, (0, 200, 80))

        # Viền màu theo trạng thái
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4)

        # Overlay bán trong suốt ở góc trên trái
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 110), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Text thông tin
        label_text = {
            'normal':   '✓ NORMAL',
            'fire':     '🔥 FIRE DETECTED!',
            'smoke':    '💨 SMOKE DETECTED!',
            'fire+smoke': '🔥💨 FIRE + SMOKE!',
        }.get(self.last_label, 'NORMAL')

        cv2.putText(frame, label_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Fire:  {self.last_fire_prob*100:5.1f}%", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)
        cv2.putText(frame, f"Smoke: {self.last_smoke_prob*100:5.1f}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # Alert nhấp nháy nếu phát hiện
        if self.last_label != 'normal' and (self.frame_count // 15) % 2 == 0:
            cv2.putText(frame, "⚠ ALERT!", (w - 160, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        return frame


def run(args):
    detector = FireSmokeDetector(
        weights_path=args.weights,
        num_frames=args.num_frames,
        threshold=args.threshold,
        inference_interval=args.interval,
    )

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        print(f"❌ Không thể mở source: {args.source}")
        return

    print(f"\n▶ Bắt đầu inference | Source: {args.source}")
    print(f"  num_frames: {args.num_frames} | threshold: {args.threshold}")
    print(f"  Nhấn 'q' để thoát\n")

    fps_time = time.time()
    fps_count = 0
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break

        result = detector.process_frame(frame)

        # FPS counter
        fps_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count = 0; fps_time = time.time()

        cv2.putText(result, f"FPS: {fps_display:.1f}", (10, result.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Fire & Smoke Detection — TimeSformer", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Đã dừng.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time Fire & Smoke Detection')
    parser.add_argument('--source',     type=str,   default='0',
                        help='Video source: 0=webcam, rtsp://..., /path/to/video.mp4')
    parser.add_argument('--weights',    type=str,   required=True,
                        help='Path to full_fire_smoke_weights_vN.pth')
    parser.add_argument('--num_frames', type=int,   default=8,
                        help='Số frames trong sliding window (default: 8)')
    parser.add_argument('--threshold',  type=float, default=0.5,
                        help='Ngưỡng phân loại (default: 0.5)')
    parser.add_argument('--interval',   type=int,   default=5,
                        help='Chạy inference mỗi N frames (default: 5)')
    args = parser.parse_args()
    run(args)
