"""
src/fire_core.py
================
Shared core for the two-stage Fire/Smoke segmentation pipeline.

Everything that used to be defined *inside* the Modal training functions lives
here so that both training scripts, the evaluation script and the unit tests
use one single implementation:

- Preprocessing: letterbox, multi-scale CLAHE / neutrality / DCP enhancement
  (`enhance_base`) and the optional temporal motion term (`apply_motion`).
- Datasets: `StaticSegDataset` (Stage 1) and `VideoClipDataset` (Stage 2).
- Models: `TemporalRefinementModule`, `SpatioTemporalSegFormer`.
- Losses: `seg_ce_dice_loss`, `SpatioTemporalLoss`.
- Metrics: `confusion_matrix_3c`, `iou_dice_from_cm`.
- Run bookkeeping: config check on resume, checkpoint archiving, RNG state,
  best-record tracking with the correct "best first, latest last" ordering.

Heavy optional dependencies (transformers) are imported lazily.
"""

import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# =============================================================================
# CONSTANTS
# =============================================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_CLASSES = 3            # 0 = background, 1 = fire, 2 = smoke
IGNORE_INDEX = 255         # letterbox padding in masks; ignored by loss & metrics
PAD_COLOR = (114, 114, 114)
PREPROCESS_VERSION = "v3.1"

# Config keys that change the model / input definition. Resuming a run whose
# saved config differs on any of these keys is refused.
ARCH_CONFIG_KEYS = ("causal", "temporal_hidden", "num_heads", "max_T", "letterbox", "preprocess_version")

# MiT-B2 encoder + SegFormer all-MLP decoder (same as "nvidia/mit-b2").
MIT_B2_CONFIG = dict(
    num_channels=3,
    num_encoder_blocks=4,
    depths=[3, 4, 6, 3],
    sr_ratios=[8, 4, 2, 1],
    hidden_sizes=[64, 128, 320, 512],
    patch_sizes=[7, 3, 3, 3],
    strides=[4, 2, 2, 2],
    num_attention_heads=[1, 2, 5, 8],
    mlp_ratios=[4, 4, 4, 4],
    decoder_hidden_size=768,
)


# =============================================================================
# 1. PREPROCESSING
# =============================================================================
def letterbox_image_and_mask(img, mask, target_size=(512, 512), pad_color=PAD_COLOR,
                             mask_pad_value=IGNORE_INDEX):
    """Aspect-preserving resize + constant padding to `target_size` (h, w).

    The image is padded with `pad_color`; the mask with `mask_pad_value`
    (IGNORE_INDEX by default so padding is neither trained on nor counted as
    background by the metrics). `mask` may be None.
    """
    th, tw = target_size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    pad_w, pad_h = tw - nw, th - nh
    pad_left, pad_top = pad_w // 2, pad_h // 2
    pad_right, pad_bottom = pad_w - pad_left, pad_h - pad_top

    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=pad_color)
    if mask is None:
        return img_padded, None
    mask_resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    mask_padded = cv2.copyMakeBorder(mask_resized, pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=int(mask_pad_value))
    return img_padded, mask_padded


def enhance_base(img_bgr):
    """Per-frame enhancement (no temporal information).

    1. Multi-scale adaptive CLAHE on L: micro (4x4, clip 3.5) + macro (16x16, clip 2.0).
    2. Achromatic colour-neutrality map W_neutral (smoke is grey-ish).
    3. Local Dark Channel Prior (aerosol optical density proxy).
    4. Weighted smoke boost: smoke_boost = (w_neutral * dark_norm) * 28.0.
    """
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
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def apply_motion(cur_base, prev_base, strength=0.15):
    """Brighten moving regions. Both inputs are *base* (enhance_base) frames, so
    the result for a given physical frame only depends on itself and its
    predecessor, never on its position inside a sliding window."""
    if prev_base is None:
        return cur_base
    diff = cv2.absdiff(cur_base, prev_base)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    w = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32) / 255.0 * strength
    return np.clip(cur_base.astype(np.float32) * (1.0 + w[:, :, None]), 0, 255).astype(np.uint8)


def enhance_smoke_and_fire_v2(img_bgr, prev_frame_bgr=None):
    """Backward-compatible wrapper. `prev_frame_bgr` must be the previous frame's
    *base* enhancement (see `enhance_base`)."""
    return apply_motion(enhance_base(img_bgr), prev_frame_bgr)


def normalize_to_chw(img_bgr):
    """BGR uint8 -> ImageNet-normalised float32 (3, H, W)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((img_rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


# =============================================================================
# 2. FILE READING / MASK VALIDATION
# =============================================================================
def validate_mask(mask, path="<mask>"):
    """Masks must only contain {0, 1, 2}. No silent clipping (255 would become smoke)."""
    vals = np.unique(mask)
    if not set(vals.tolist()).issubset({0, 1, 2}):
        raise ValueError(f"Mask {path} có giá trị ngoài {{0,1,2}}: {vals[:10]}")
    return mask.astype(np.uint8)


def read_image(path, strict=True, fallback_size=(512, 512)):
    img = cv2.imread(str(path))
    if img is None:
        if strict:
            raise FileNotFoundError(f"Không đọc được ảnh: {path}")
        return np.zeros((fallback_size[0], fallback_size[1], 3), dtype=np.uint8)
    return img


def read_mask(path, strict=True, fallback_shape=(512, 512)):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        if strict:
            raise FileNotFoundError(f"Không đọc được mask: {path}")
        return np.zeros(fallback_shape, dtype=np.uint8)
    return validate_mask(mask, path)


# =============================================================================
# 3. DATASETS
# =============================================================================
class StaticSegDataset(Dataset):
    """Stage 1 static images. Pipeline: enhance_base (original resolution) ->
    letterbox (image pad 114, mask pad IGNORE_INDEX) -> physics-aware augmentation."""

    def __init__(self, manifest_csv, data_root, target_size=(512, 512), is_training=False, strict=True):
        self.df = pd.read_csv(manifest_csv)
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.is_training = is_training
        self.strict = strict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_root / row["filename"]
        mask_path = self.data_root / row["mask_filename"]

        img = enhance_base(read_image(img_path, self.strict, self.target_size))
        mask = read_mask(mask_path, self.strict, img.shape[:2])
        img_pad, mask_pad = letterbox_image_and_mask(img, mask, self.target_size)

        # Physics-aware augmentation (train only, NO vertical flip)
        if self.is_training:
            if random.random() < 0.5:
                img_pad = cv2.flip(img_pad, 1)
                mask_pad = cv2.flip(mask_pad, 1)
            if random.random() < 0.5:
                alpha = 1.0 + random.uniform(-0.2, 0.2)
                beta = random.uniform(-20, 20)
                img_pad = np.clip(alpha * img_pad.astype(np.float32) + beta, 0, 255).astype(np.uint8)
            if random.random() < 0.2:
                ksize = random.choice([3, 5])
                img_pad = cv2.GaussianBlur(img_pad, (ksize, ksize), 0)

        img_tensor = torch.from_numpy(normalize_to_chw(img_pad)).float()
        mask_tensor = torch.from_numpy(mask_pad).long()
        return img_tensor, mask_tensor


def clip_frame_keys(row):
    """[(video_name, frame_name), ...] for every position of a clip row."""
    return [(row["video_name"], name) for name in str(row["frame_names"]).split(";")]


def clip_labeled_positions(row):
    """Positions of a clip that carry a label (an empty mask path = context-only frame)."""
    return [i for i, p in enumerate(str(row["mask_paths"]).split(";")) if p]


class VideoClipDataset(Dataset):
    """Stage 2 T-frame clips. Per frame: enhance_base on the original frame ->
    optional motion term against the previous *base* frame -> letterbox (same
    as Stage 1) -> clip-consistent augmentation.

    An empty entry in `mask_paths` marks an unlabeled context frame: its mask is
    all IGNORE_INDEX, so it only provides temporal context (dense clips)."""

    def __init__(self, clips_csv, data_root, target_size=(512, 512), is_training=False,
                 use_motion=True, letterbox=True, strict=True):
        self.df = pd.read_csv(clips_csv)
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.is_training = is_training
        self.use_motion = use_motion
        self.letterbox = letterbox
        self.strict = strict

    def __len__(self):
        return len(self.df)

    def frame_keys(self, idx):
        return clip_frame_keys(self.df.iloc[idx])

    def _fit(self, img, mask):
        if self.letterbox:
            return letterbox_image_and_mask(img, mask, self.target_size)
        th, tw = self.target_size
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_paths = row["image_paths"].split(";")
        mask_rel_paths = row["mask_paths"].split(";")

        # Clip-synchronous augmentation (NO vertical flip)
        do_flip = self.is_training and (random.random() < 0.5)
        brightness_factor = (1.0 + random.uniform(-0.15, 0.15)) if self.is_training else 1.0

        clip_frames, clip_masks = [], []
        prev_base = None
        for img_rel, mask_rel in zip(img_rel_paths, mask_rel_paths):
            img = read_image(self.data_root / img_rel, self.strict, self.target_size)
            if mask_rel:
                mask = read_mask(self.data_root / mask_rel, self.strict, img.shape[:2])
            else:
                mask = np.full(img.shape[:2], IGNORE_INDEX, dtype=np.uint8)
            if mask.shape[:2] != img.shape[:2]:
                raise ValueError(f"Kích thước mask {mask.shape[:2]} != ảnh {img.shape[:2]}: {mask_rel}")

            base = enhance_base(img)
            img = apply_motion(base, prev_base) if self.use_motion else base
            prev_base = base

            img, mask = self._fit(img, mask)

            if do_flip:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            if brightness_factor != 1.0:
                img = np.clip(img.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

            clip_frames.append(normalize_to_chw(img))
            clip_masks.append(mask)

        clip_frames = torch.from_numpy(np.stack(clip_frames, axis=0)).float()  # (T, 3, H, W)
        clip_masks = torch.from_numpy(np.stack(clip_masks, axis=0)).long()     # (T, H, W)
        return clip_frames, clip_masks


# =============================================================================
# 4. MODELS
# =============================================================================
def build_segformer_b2(num_labels=NUM_CLASSES, pretrained=True):
    """SegFormer-B2. `pretrained=True` downloads ImageNet MiT-B2 weights;
    `pretrained=False` builds the same architecture offline (for loading a checkpoint)."""
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    if pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2", num_labels=num_labels, ignore_mismatched_sizes=True)
    return SegformerForSemanticSegmentation(SegformerConfig(num_labels=num_labels, **MIT_B2_CONFIG))


class TemporalRefinementModule(nn.Module):
    """Temporal reasoning over T frames on the fused decoder features.

    - 1x1 projection + 2x average pooling (128x128 -> 64x64) to save memory.
    - 1D temporal conv (k=3), left-padded when causal.
    - Learned temporal positional embedding (max_T positions).
    - Temporal multi-head attention, with a causal mask when causal=True:
      the output at frame t only depends on frames <= t (streaming-safe).
    - Zero-initialised residual gate (gamma=0) -> identical to Stage 1 at init.
    """

    def __init__(self, in_channels=768, hidden_channels=128, num_heads=4,
                 dropout=0.1, max_T=16, causal=True):
        super().__init__()
        self.causal = causal
        self.max_T = max_T
        self.in_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.temp_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_T, hidden_channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.temp_attn = nn.MultiheadAttention(
            embed_dim=hidden_channels, num_heads=num_heads, dropout=dropout, batch_first=True)
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
        # z: (B*T, C, H, W)
        if T > self.max_T:
            raise ValueError(f"T={T} > max_T={self.max_T}")
        BT, C, H, W = z.shape
        feat = self.pool(self.in_proj(z))
        _, hid, h, w = feat.shape

        x = feat.view(B, T, hid, h, w).permute(0, 3, 4, 2, 1).reshape(B * h * w, hid, T)
        pad = (2, 0) if self.causal else (1, 1)
        x = x + self.temp_conv(F.pad(x, pad))

        seq = x.permute(0, 2, 1) + self.pos_embed[:, :T]
        mask = None
        if self.causal:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=z.device), diagonal=1)
        n = self.norm1(seq)
        a, _ = self.temp_attn(n, n, n, attn_mask=mask, need_weights=False)
        seq = seq + a
        seq = seq + self.mlp(self.norm2(seq))

        out = seq.reshape(B, h, w, T, hid).permute(0, 3, 4, 1, 2).reshape(BT, hid, h, w)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return z + self.gamma * self.out_proj(out)


class SpatioTemporalSegFormer(nn.Module):
    """Stage 1 SegFormer (encoder + all-MLP decoder) with a temporal refinement
    block inserted between the fused decoder features and the classifier.

    - Frozen encoder (no graph through T frames) -> large VRAM savings.
    - Decode head + temporal module are trained.
    - Decode-head BatchNorm running stats are frozen by default: a batch is
      T near-identical frames of one clip, which would drift the Stage 1 stats.
    """

    def __init__(self, pretrained_segformer, freeze_backbone=True, freeze_decode_bn=True,
                 temporal_hidden=128, num_heads=4, dropout=0.1, max_T=16, causal=True, verbose=True):
        super().__init__()
        self.segformer = pretrained_segformer.segformer
        self.decode_head = pretrained_segformer.decode_head
        self.temporal_refinement = TemporalRefinementModule(
            in_channels=self.decode_head.linear_fuse.out_channels,
            hidden_channels=temporal_hidden,
            num_heads=num_heads,
            dropout=dropout,
            max_T=max_T,
            causal=causal,
        )
        self.freeze_backbone = freeze_backbone
        self.freeze_decode_bn = freeze_decode_bn
        if freeze_backbone:
            for p in self.segformer.parameters():
                p.requires_grad = False
            self.segformer.eval()
            if verbose:
                print("🔒 Đã ĐÓNG BĂNG SegFormer Spatial Backbone (Bảo toàn 100% đặc trưng Stage 1 & giải phóng VRAM).")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.segformer.eval()
        if self.freeze_decode_bn:
            self.decode_head.batch_norm.eval()  # affine weight/bias still trainable
        return self

    def encode_frames(self, x_flat):
        """(N, 3, H, W) -> fused decoder features (N, C, H/4, W/4). Per-frame only,
        so results can be cached frame by frame in streaming inference."""
        N = x_flat.shape[0]
        if self.freeze_backbone:
            with torch.no_grad():
                hidden_states = [h.detach() for h in
                                 self.segformer(x_flat, output_hidden_states=True).hidden_states]
        else:
            hidden_states = self.segformer(x_flat, output_hidden_states=True).hidden_states

        all_hidden_states = ()
        for h, mlp in zip(hidden_states, self.decode_head.linear_c):
            h_proj = mlp(h).permute(0, 2, 1).reshape(N, -1, h.shape[2], h.shape[3])
            h_up = F.interpolate(h_proj, size=hidden_states[0].shape[2:], mode="bilinear", align_corners=False)
            all_hidden_states += (h_up,)

        fused = self.decode_head.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        fused = self.decode_head.batch_norm(fused)
        return self.decode_head.activation(fused)

    def decode_fused(self, fused, B, T, out_size):
        """Temporal refinement + classifier. fused: (B*T, C, h, w) -> (B, T, K, H, W)."""
        fused_temporal = self.temporal_refinement(fused, B, T)
        logits = self.decode_head.classifier(self.decode_head.dropout(fused_temporal))
        logits_up = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits_up.view(B, T, logits_up.shape[1], out_size[0], out_size[1])

    def forward(self, pixel_values):
        # pixel_values: (B, T, 3, H, W)
        B, T, C, H, W = pixel_values.shape
        fused = self.encode_frames(pixel_values.reshape(B * T, C, H, W))
        return self.decode_fused(fused, B, T, (H, W))


def load_stage2_state_dict(model, state_dict, legacy_ok=False):
    """Strict load. With `legacy_ok`, a pre-T7 checkpoint (no pos_embed) is
    accepted: pos_embed is zeroed, which reproduces the legacy module exactly
    when the model was built with causal=False."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    legacy_missing = {"temporal_refinement.pos_embed"}
    if unexpected or (set(missing) - (legacy_missing if legacy_ok else set())):
        raise RuntimeError(f"Checkpoint không khớp kiến trúc. missing={missing}, unexpected={unexpected}")
    if missing:
        with torch.no_grad():
            model.temporal_refinement.pos_embed.zero_()
    return bool(missing)


# =============================================================================
# 5. LOSSES
# =============================================================================
def dice_loss_fg(logits, targets, classes=(1, 2), smooth=1.0):
    """Batch-level soft Dice over foreground classes only, ignoring padding."""
    probs = F.softmax(logits.float(), dim=1)
    valid = targets != IGNORE_INDEX
    losses = []
    for c in classes:
        p = probs[:, c] * valid
        t = ((targets == c) & valid).float()
        inter = (p * t).sum()
        card = p.sum() + t.sum()
        losses.append(1.0 - (2 * inter + smooth) / (card + smooth))
    return torch.stack(losses).mean()


def dice_loss_per_image_all(logits, targets, smooth=1.0):
    """Legacy Dice (per image, all classes incl. background) kept for ablation."""
    probs = F.softmax(logits.float(), dim=1)
    valid = (targets != IGNORE_INDEX).float()
    num_classes = logits.shape[1]
    total = 0.0
    for c in range(num_classes):
        p = probs[:, c] * valid
        t = (targets == c).float()
        inter = (p * t).sum(dim=(1, 2))
        card = (p + t).sum(dim=(1, 2))
        total = total + (1.0 - ((2.0 * inter + smooth) / (card + smooth)).mean())
    return total / num_classes


DICE_MODES = {"batch_fg": dice_loss_fg, "per_image_all": dice_loss_per_image_all}


def seg_ce_dice_loss(logits, targets, class_weights, dice_weight=1.0, dice_mode="batch_fg"):
    """logits (N, K, H, W), targets (N, H, W) -> (total, ce, dice)."""
    ce = F.cross_entropy(logits.float(), targets, weight=class_weights, ignore_index=IGNORE_INDEX)
    dl = DICE_MODES[dice_mode](logits, targets)
    return ce + dice_weight * dl, ce, dl


class SpatioTemporalLoss(nn.Module):
    """CE + Dice + temporal smoothness. Smoothness only penalises probability
    changes where the ground truth itself is stable between consecutive frames,
    so fire/smoke onset is never discouraged."""

    def __init__(self, class_weights=None, dice_weight=1.0, temp_smooth_weight=0.05, dice_mode="batch_fg"):
        super().__init__()
        self.class_weights = class_weights
        self.dice_weight = dice_weight
        self.temp_weight = temp_smooth_weight
        self.dice_mode = dice_mode

    def temporal_smoothness(self, logits, targets):
        b, t, c, h, w = logits.shape
        probs_seq = F.softmax(logits.float(), dim=2)
        diff = (probs_seq[:, 1:] - probs_seq[:, :-1]).abs().sum(dim=2)  # (b, t-1, h, w)
        tg = targets.view(b, t, h, w)
        stable = ((tg[:, 1:] == tg[:, :-1]) & (tg[:, 1:] != IGNORE_INDEX)).float()
        return (diff * stable).sum() / stable.sum().clamp_min(1.0)

    def forward(self, logits, targets):
        b, t, c, h, w = logits.shape
        _, loss_ce, loss_dice = seg_ce_dice_loss(
            logits.reshape(b * t, c, h, w), targets.reshape(b * t, h, w),
            self.class_weights, self.dice_weight, self.dice_mode)
        loss_temp = self.temporal_smoothness(logits, targets)
        total = loss_ce + self.dice_weight * loss_dice + self.temp_weight * loss_temp
        return total, loss_ce, loss_dice, loss_temp


# =============================================================================
# 6. METRICS
# =============================================================================
def confusion_matrix_3c(preds, targets, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    """Confusion matrix cm[gt, pred] (int64 numpy), ignoring `ignore_index` pixels."""
    preds = torch.as_tensor(preds).reshape(-1).long()
    targets = torch.as_tensor(targets).reshape(-1).long()
    valid = targets != ignore_index
    idx = targets[valid] * num_classes + preds[valid].to(targets.device)
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.view(num_classes, num_classes).cpu().numpy().astype(np.int64)


def iou_dice_from_cm(cm):
    """Per-class IoU / Dice from a confusion matrix. Undefined classes (empty
    union) give NaN and are skipped by the means."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    union = tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        ious = np.where(union > 0, tp / union, np.nan)
        dices = np.where(union > 0, 2 * tp / (2 * tp + fp + fn), np.nan)

    def _nanmean(x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.nanmean(x)) if np.any(~np.isnan(x)) else float("nan")

    return {
        "iou": [float(v) for v in ious],
        "dice": [float(v) for v in dices],
        "iou_bg": float(ious[0]), "iou_fire": float(ious[1]), "iou_smoke": float(ious[2]),
        "dice_fire": float(dices[1]), "dice_smoke": float(dices[2]),
        "fg_miou": _nanmean(ious[1:]),
        "all_miou": _nanmean(ious),
    }


# =============================================================================
# 7. RUN BOOKKEEPING (run_id, config, archive, resume, RNG)
# =============================================================================
def make_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def check_resume_config(saved_config, current_config, keys=ARCH_CONFIG_KEYS):
    """Raise ValueError when a checkpoint's config differs on architecture keys."""
    if not saved_config:
        raise ValueError("Checkpoint không có `config` (checkpoint cũ trước khi sửa) -> không thể resume an toàn. "
                         "Hãy chạy lại không có --resume.")
    diffs = {k: (saved_config.get(k), current_config.get(k)) for k in keys
             if saved_config.get(k) != current_config.get(k)}
    if diffs:
        raise ValueError(f"Config checkpoint khác cấu hình hiện tại ở khóa kiến trúc: {diffs}. Từ chối resume.")


def read_checkpoint_run_id(path):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return ck.get("run_id") if isinstance(ck, dict) else None
    except Exception:
        return None


def archive_existing_checkpoints(paths, archive_root, timestamp=None):
    """Move every existing file in `paths` into archive_root/<old_run_id|legacy>_<timestamp>/.
    Returns the archive directory or None if nothing existed."""
    existing = [Path(p) for p in paths if Path(p).exists()]
    if not existing:
        return None
    old_run_id = None
    for p in existing:
        old_run_id = read_checkpoint_run_id(p)
        if old_run_id:
            break
    timestamp = timestamp or make_run_id()
    dest = Path(archive_root) / f"{old_run_id or 'legacy'}_{timestamp}"
    dest.mkdir(parents=True, exist_ok=True)
    for p in existing:
        shutil.move(str(p), str(dest / p.name))
    return dest


def capture_rng_state():
    return {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.as_tensor(s, dtype=torch.uint8).cpu() for s in state["cuda"]])


class BestTracker:
    """Records of the best val metrics (Fg-mIoU, Fire IoU, Smoke IoU)."""

    def __init__(self, best_val_miou=0.0, best_fire_iou=0.0, best_smoke_iou=0.0, best_epoch=0):
        self.best_val_miou = best_val_miou
        self.best_fire_iou = best_fire_iou
        self.best_smoke_iou = best_smoke_iou
        self.best_epoch = best_epoch

    def update(self, epoch, fg_miou, iou_fire, iou_smoke):
        """Returns the list of improved records among ('best', 'fire', 'smoke')."""
        improved = []
        if fg_miou > self.best_val_miou:
            self.best_val_miou, self.best_epoch = fg_miou, epoch
            improved.append("best")
        if iou_fire > self.best_fire_iou:
            self.best_fire_iou = iou_fire
            improved.append("fire")
        if iou_smoke > self.best_smoke_iou:
            self.best_smoke_iou = iou_smoke
            improved.append("smoke")
        return improved

    def state(self):
        return {"best_val_miou": self.best_val_miou, "best_fire_iou": self.best_fire_iou,
                "best_smoke_iou": self.best_smoke_iou, "best_epoch": self.best_epoch}


def build_latest_payload(epoch, run_id, config, tracker, metrics, gamma):
    """Metadata of the LATEST checkpoint. Must be called AFTER tracker.update()
    of the same epoch so the saved records include this epoch's results."""
    return {
        "epoch": epoch,
        "run_id": run_id,
        "config": config,
        **tracker.state(),
        "fg_miou": metrics["fg_miou"], "iou_fire": metrics["iou_fire"], "iou_smoke": metrics["iou_smoke"],
        "gamma": gamma,
    }


def epoch_checkpoint_step(epoch, metrics, tracker, save_best, save_latest, run_id, config, gamma):
    """Per-epoch checkpoint policy: update records -> save improved best
    checkpoints -> save LATEST last (so a resume never restores stale records).
    `save_best(kind)` and `save_latest(payload)` perform the I/O."""
    improved = tracker.update(epoch, metrics["fg_miou"], metrics["iou_fire"], metrics["iou_smoke"])
    for kind in improved:
        save_best(kind)
    save_latest(build_latest_payload(epoch, run_id, config, tracker, metrics, gamma))
    return improved


def append_history(history_path, record):
    """Append one epoch record to a JSON list file."""
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    history.append(record)
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
