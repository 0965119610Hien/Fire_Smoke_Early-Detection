# -*- coding: utf-8 -*-
"""
PIPELINE CẮT VIDEO ĐA TẦNG CHO DATASET FIRE/SMOKE DETECTION
=============================================================
Tầng 1: Scene-cut detection (PySceneDetect)      -> không cắt vắt ngang giữa 2 cảnh
Tầng 2: Optical Flow + Adaptive Threshold        -> tín hiệu chuyển động
Tầng 3: Color HSV masking (fire/smoke)           -> tín hiệu màu sắc đặc trưng
Tầng 4: TimeSformer spatial model (đã train)     -> tín hiệu semantic

TĂNG TỐC:
- Tầng 2+3: Gộp vào 1 lần đọc video duy nhất (thay vì đọc 4 lần)
- Tầng 4: GPU batch inference (stride=5)
- Song song: mỗi video xử lý trên 1 Modal container riêng (modal.map)

Chạy trên Modal GPU:
    modal run prepare_dataset.py

Cài đặt cần thiết:
    pip install modal
"""

import modal
import os

# =====================================================================
# MODAL CONFIGURATION
# =====================================================================
app = modal.App("fire-smoke-prepare-dataset")
volume = modal.Volume.from_name("fire_smoke_dataset")

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libglib2.0-0", "libgl1-mesa-glx")
    .pip_install(
        "torch",
        "torchvision",
        "opencv-python-headless",
        "einops",
        "numpy",
        "moviepy",
        "scenedetect[opencv]",
    )
)


# =====================================================================
# HÀM XỬ LÝ 1 VIDEO — chạy song song trên nhiều container
# =====================================================================
@app.function(image=image, volumes={"/data": volume}, gpu="A10G", timeout=3600)
def process_one_video(args):
    """
    Nhận (in_path, out_path, class_name).
    Chạy toàn bộ pipeline cho 1 video, trả về (filename, success, message).
    """
    in_path, out_path, class_name = args

    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    from einops import rearrange, repeat

    try:
        from moviepy import VideoFileClip
        _MOVIEPY_V2 = True
    except ImportError:
        from moviepy.editor import VideoFileClip
        _MOVIEPY_V2 = False

    def _make_subclip(clip, s, e):
        return clip.subclipped(s, e) if _MOVIEPY_V2 else clip.subclip(s, e)

    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        HAS_SCENEDETECT = True
    except ImportError:
        HAS_SCENEDETECT = False

    filename = os.path.basename(in_path)

    # ------------------------------------------------------------------
    # CẤU HÌNH
    # ------------------------------------------------------------------
    MODEL_WEIGHTS_PATH   = "/data/spatial_fire_smoke_weights_v2.pth"
    MODEL_IMAGE_SIZE     = 224
    MODEL_SAMPLE_STRIDE  = 5
    MODEL_CONF_THRESHOLD = 0.5
    MIN_SCENE_LEN_SEC    = 1.0
    PADDING_SEC          = 1.5
    MIN_DURATION_SEC     = 2.0
    ADAPTIVE_PERCENTILE  = 75
    WEIGHTS_CLASS_1 = {"motion": 0.30, "color": 0.25, "model": 0.45}
    WEIGHTS_CLASS_0 = {"motion": 0.55, "color": 0.45, "model": 0.00}

    # HSV ranges
    FIRE_HSV_LOWER = np.array([0,   100, 150])
    FIRE_HSV_UPPER = np.array([35,  255, 255])
    SMOKE_HSV_LOWER = np.array([0,    0,  60])
    SMOKE_HSV_UPPER = np.array([180,  60, 220])

    # ------------------------------------------------------------------
    # TẦNG 1: SCENE DETECTION
    # ------------------------------------------------------------------
    def detect_scenes(video_path, fps, total_frames):
        if not HAS_SCENEDETECT:
            return [(0, total_frames - 1)]
        try:
            video = open_video(video_path)
            sm = SceneManager()
            sm.add_detector(ContentDetector(threshold=27.0))
            sm.detect_scenes(video)
            scene_list = sm.get_scene_list()
            if not scene_list:
                return [(0, total_frames - 1)]
            scenes = [(s.get_frames(), max(s.get_frames(), e.get_frames() - 1))
                      for s, e in scene_list]
            min_len = MIN_SCENE_LEN_SEC * fps
            merged = []
            for s, e in scenes:
                if merged and (e - s) < min_len:
                    merged[-1] = (merged[-1][0], e)
                else:
                    merged.append((s, e))
            return merged or [(0, total_frames - 1)]
        except Exception:
            return [(0, total_frames - 1)]

    def frame_to_scene_bounds(frame_idx, scenes):
        for s, e in scenes:
            if s <= frame_idx <= e:
                return s, e
        return scenes[0] if scenes else (frame_idx, frame_idx)

    # ------------------------------------------------------------------
    # TẦNG 2+3: ĐỌC VIDEO 1 LẦN — tính motion + color đồng thời
    # ------------------------------------------------------------------
    def compute_motion_and_color_scores(video_path):
        """
        1 lần đọc video -> trả về (motion_scores, color_scores).
        motion: Farneback optical flow magnitude trung bình mỗi frame.
        color : tỉ lệ pixel fire/smoke mỗi frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.array([]), np.array([])

        ret, frame1 = cap.read()
        if not ret:
            cap.release()
            return np.array([]), np.array([])

        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        # Frame 0: motion = 0, color = tính luôn
        hsv0 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        total_px = hsv0.shape[0] * hsv0.shape[1]
        c0 = (cv2.countNonZero(cv2.inRange(hsv0, FIRE_HSV_LOWER, FIRE_HSV_UPPER)) +
              cv2.countNonZero(cv2.inRange(hsv0, SMOKE_HSV_LOWER, SMOKE_HSV_UPPER))) / total_px

        motion_scores = [0.0]
        color_scores  = [c0]

        while True:
            ret, frame2 = cap.read()
            if not ret:
                break
            nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prvs, nxt, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_scores.append(float(np.mean(mag)))

            # Color
            hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
            ratio = (cv2.countNonZero(cv2.inRange(hsv, FIRE_HSV_LOWER, FIRE_HSV_UPPER)) +
                     cv2.countNonZero(cv2.inRange(hsv, SMOKE_HSV_LOWER, SMOKE_HSV_UPPER))) / total_px
            color_scores.append(ratio)

            prvs = nxt

        cap.release()

        motion = np.array(motion_scores)
        color  = np.array(color_scores)
        if motion.max() > 0: motion = motion / motion.max()
        if color.max()  > 0: color  = color  / color.max()
        return motion, color

    # ------------------------------------------------------------------
    # TẦNG 4: MODEL TIMESFORMER
    # ------------------------------------------------------------------
    class Mlp(nn.Module):
        def __init__(self, in_f, hidden=None, out_f=None, drop=0.):
            super().__init__()
            self.fc1  = nn.Linear(in_f, hidden or in_f)
            self.act  = nn.GELU()
            self.fc2  = nn.Linear(hidden or in_f, out_f or in_f)
            self.drop = nn.Dropout(drop)
        def forward(self, x):
            return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

    class Attention(nn.Module):
        def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
            super().__init__()
            inner = dim_head * heads
            self.heads  = heads
            self.scale  = dim_head ** -0.5
            self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
            self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        def forward(self, x):
            h = self.heads
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                          self.to_qkv(x).chunk(3, dim=-1))
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            return self.to_out(rearrange(torch.matmul(dots.softmax(dim=-1), v),
                                         'b h n d -> b n (h d)'))

    class DST(nn.Module):
        def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
            super().__init__()
            self.tn1 = nn.LayerNorm(dim)
            self.ta  = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.sn1 = nn.LayerNorm(dim)
            self.sa  = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.n2  = nn.LayerNorm(dim)
            self.mlp = Mlp(dim, mlp_dim, drop=dropout)
        def forward(self, x, f):
            b, t, n, d = x.shape
            xt = rearrange(x, 'b t n d -> (b n) t d')
            xt = xt + self.ta(self.tn1(xt))
            x  = rearrange(xt, '(b n) t d -> b t n d', b=b, n=n)
            xs = rearrange(x, 'b t n d -> (b t) n d')
            xs = xs + self.sa(self.sn1(xs))
            x  = rearrange(xs, '(b t) n d -> b t n d', b=b, t=t)
            xm = rearrange(x, 'b t n d -> (b t n) d')
            xm = xm + self.mlp(self.n2(xm))
            return rearrange(xm, '(b t n) d -> b t n d', b=b, t=t, n=n)

    class TimeSformer(nn.Module):
        def __init__(self, image_size=224, patch_size=16, num_frames=1, num_classes=2,
                     dim=256, depth=6, heads=8, dim_head=64, mlp_dim=512, dropout=0.1):
            super().__init__()
            np_ = (image_size // patch_size) ** 2
            self.to_patch  = nn.Linear(3 * patch_size**2, dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))
            self.pos_emb   = nn.Parameter(torch.randn(1, 1, np_ + 1, dim))
            self.time_emb  = nn.Parameter(torch.randn(1, num_frames, 1, dim))
            self.layers    = nn.ModuleList([DST(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])
            self.head      = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        def forward(self, video):
            x = self.to_patch(rearrange(video, 'b c f (h p1) (w p2) -> b f (h w) (c p1 p2)', p1=16, p2=16))
            b, t, n, d = x.shape
            x = torch.cat((repeat(self.cls_token, '() () () d -> b t () d', b=b, t=t), x), dim=2)
            x = x + self.pos_emb[:, :, :n+1]
            x[:, :, 1:] = x[:, :, 1:] + self.time_emb[:, :t, :]
            for layer in self.layers:
                x = layer(x, t)
            return self.head(x[:, :, 0, :].mean(dim=1))

    def load_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = TimeSformer(image_size=224, patch_size=16, num_frames=1, num_classes=2, dim=256, depth=6).to(device)
        m.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        m.eval()
        return m, device

    def compute_model_scores(video_path, model, device):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return np.zeros(1), np.zeros(1)

        batch_imgs, batch_idx = [], []
        frame_i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_i % MODEL_SAMPLE_STRIDE == 0:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)).astype(np.float32) / 255.0
                batch_imgs.append(np.transpose(img, (2, 0, 1)))
                batch_idx.append(frame_i)
            frame_i += 1
        cap.release()

        if not batch_imgs:
            return np.zeros(total_frames), np.zeros(total_frames)

        with torch.no_grad():
            tensor = torch.tensor(np.array(batch_imgs)).unsqueeze(2).to(device)
            probs_list = []
            for i in range(0, tensor.shape[0], 32):
                logits = model(tensor[i:i+32])
                probs_list.append(torch.sigmoid(logits).cpu().numpy())
            probs = np.vstack(probs_list)

        full_idx = np.arange(total_frames)
        return (np.interp(full_idx, np.array(batch_idx), probs[:, 0]),
                np.interp(full_idx, np.array(batch_idx), probs[:, 1]))

    # ------------------------------------------------------------------
    # FUSION
    # ------------------------------------------------------------------
    def _smooth(scores, win=5):
        if len(scores) < win:
            return scores
        return np.convolve(scores, np.ones(win)/win, mode="same")

    def fuse_and_select(motion, color, fire, smoke, fps, total_frames, scenes, weights):
        # Nếu model_s rỗng, fallback về zeros
        if len(fire) > 0 and len(smoke) > 0:
            model_s = np.maximum(fire, smoke)
        else:
            model_s = np.zeros(total_frames)

        n = min(len(motion), len(color), len(model_s))
        # Guard: nếu bất kỳ mảng nào rỗng -> không xử lý được
        if n == 0:
            return None

        motion, color, model_s = motion[:n], color[:n], model_s[:n]

        fused = (weights["motion"] * motion +
                 weights["color"]  * color +
                 weights["model"]  * model_s)
        fused = _smooth(fused, win=max(3, int(fps // 5)))

        # Guard: fused rỗng sau smooth
        if len(fused) == 0:
            return None

        active = np.where(fused > 0.15)[0]
        if len(active) == 0 and len(fused) > 0:
            # Fallback sang adaptive percentile — chỉ gọi khi mảng không rỗng
            threshold = np.percentile(fused, ADAPTIVE_PERCENTILE)
            active = np.where(fused > threshold)[0]
        if len(active) == 0:
            return None

        sf, ef = int(active[0]), int(active[-1])
        ss, _ = frame_to_scene_bounds(sf, scenes)
        _, se = frame_to_scene_bounds(ef, scenes)
        sf = max(sf, ss)
        ef = min(ef, se)

        st = max(0.0, sf / fps - PADDING_SEC)
        et = min(total_frames / fps, ef / fps + PADDING_SEC)
        if (et - st) < MIN_DURATION_SEC:
            return None
        return st, et

    # ------------------------------------------------------------------
    # PIPELINE CHO 1 VIDEO
    # ------------------------------------------------------------------
    # Kiểm tra output đã tồn tại chưa
    if os.path.exists(out_path):
        return filename, True, "skipped (already exists)"

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        return filename, False, "cannot open video"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames < 2:
        return filename, False, "too short"

    # Tier 1: Scene detection
    scenes = detect_scenes(in_path, fps, total_frames)

    # Tier 2+3: 1 lần đọc -> motion + color
    motion_scores, color_scores = compute_motion_and_color_scores(in_path)

    weights = WEIGHTS_CLASS_1 if class_name == "class_1" else WEIGHTS_CLASS_0

    # Tier 4: Model inference (GPU)
    try:
        model, device = load_model()
        fire_scores, smoke_scores = compute_model_scores(in_path, model, device)
        if class_name == "class_0":
            # class_0: chỉ log cảnh báo, không dùng model để chọn vùng
            if fire_scores.max() > MODEL_CONF_THRESHOLD or smoke_scores.max() > MODEL_CONF_THRESHOLD:
                print(f"[WARN] class_0 '{filename}': fire={fire_scores.max():.2f} smoke={smoke_scores.max():.2f}")
            fire_scores = np.zeros(total_frames)
            smoke_scores = np.zeros(total_frames)
    except Exception as e:
        print(f"[WARN] load model failed ({e}), skipping Tier 4")
        fire_scores = np.zeros(total_frames)
        smoke_scores = np.zeros(total_frames)

    result = fuse_and_select(motion_scores, color_scores, fire_scores, smoke_scores,
                              fps, total_frames, scenes, weights)
    if result is None:
        return filename, False, "no active region found"

    start_time, end_time = result
    print(f"[CUT] [{class_name}] {filename}: {start_time:.1f}s -> {end_time:.1f}s")

    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        full_clip = VideoFileClip(in_path)
        clip = _make_subclip(full_clip, start_time, end_time)
        clip.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
        clip.close()
        full_clip.close()
        return filename, True, f"{start_time:.1f}s -> {end_time:.1f}s"
    except Exception as e:
        return filename, False, f"write error: {e}"


# =====================================================================
# HÀM ĐIỀU PHỐI — thu thập danh sách video và map song song
# =====================================================================
@app.function(image=image, volumes={"/data": volume}, timeout=86400)
def prepare_dataset_on_modal():
    import os

    input_root_dir = "/data/unzipped_data/dataset_merged/dataset_merged"
    output_root_dir = "/data/cleaned_dataset_merged"

    # Thu thập tất cả video cần xử lý
    tasks = []
    skipped_count = 0
    for class_name in ["class_0", "class_1"]:
        class_path = os.path.join(input_root_dir, class_name)
        if not os.path.exists(class_path):
            print(f"[SKIP] Not found: {class_path}")
            continue
        for sub_folder in os.listdir(class_path):
            sub_folder_path = os.path.join(class_path, sub_folder)
            if not os.path.isdir(sub_folder_path):
                continue
            out_sub = os.path.join(output_root_dir, class_name, sub_folder)
            for file in os.listdir(sub_folder_path):
                if not file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    continue
                in_path  = os.path.join(sub_folder_path, file)
                out_path = os.path.join(out_sub, "cleaned_" + file)
                # Lọc ngay tại coordinator — không spawn container cho video đã xong
                if os.path.exists(out_path):
                    skipped_count += 1
                    continue
                tasks.append((in_path, out_path, class_name))

    print(f"[INFO] Already done (skipped): {skipped_count}")
    print(f"[INFO] Videos to process now : {len(tasks)}")

    # Chạy song song — mỗi video 1 container riêng
    ok, fail, skip = 0, 0, 0
    for filename, success, msg in process_one_video.map(tasks):
        if "skipped" in msg:
            skip += 1
        elif success:
            ok += 1
            print(f"[OK]   {filename}: {msg}")
        else:
            fail += 1
            print(f"[FAIL] {filename}: {msg}")

    volume.commit()
    print(f"\n[DONE] OK={ok}  FAIL={fail}  SKIP={skip}")
    print(f"Output: {output_root_dir}")


# =====================================================================
# MODAL ENTRY POINT — chay: modal run prepare_dataset.py
# =====================================================================
@app.local_entrypoint()
def main():
    prepare_dataset_on_modal.remote()


# =====================================================================
# LOCAL FALLBACK — chay: python prepare_dataset.py
# =====================================================================
if __name__ == "__main__":
    print("Chay tren Modal GPU: modal run prepare_dataset.py")