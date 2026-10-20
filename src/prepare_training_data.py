"""
prepare_training_data.py
========================
Script tự động chuẩn bị dữ liệu huấn luyện sau khi gán nhãn SegFormer-B2:

  BƯỚC 1: Chia tập Train / Validation theo Video Clip (80/20)
           - Group split: 1 video chỉ thuộc Train HOẶC Validation (tránh data leakage)
           - Stratified: đảm bảo tỷ lệ nhãn (fire/smoke/both/empty) cân bằng
  
  BƯỚC 2: Phân tích mất cân bằng lớp và Over-sampling trên tập Train
           - Phân tích phân bố fire_only / smoke_only / both / empty
           - Over-sample (hard-copy hoặc symlink) các frame thiếu lên mức cân bằng
  
  BƯỚC 3: Augmentation đồng bộ (ảnh + mask) chỉ trên tập Train
           - Horizontal Flip, Vertical Flip, 90/180/270 rotate
           - Color Jitter (brightness, contrast, saturation)
           - Gaussian Blur, Gaussian Noise
  
  BƯỚC 4: Xuất manifest CSV chi tiết cho cả 3 tập (train_aug, val, test)

Cách chạy:
  python src/prepare_training_data.py

Outputs:
  - data_splits/train_manifest.csv   (sau augmentation)
  - data_splits/val_manifest.csv
  - data_splits/test_manifest.csv
  - data_splits/split_statistics.json
"""

import os
import json
import shutil
import random
import logging
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
FRAMES_DIR     = ROOT / "annotation_frames"     # Ảnh gốc tập train
LABELS_DIR     = ROOT / "annotation_labels"     # Masks gốc tập train
TEST_FRAMES    = ROOT / "test_annotation_frames"
TEST_LABELS    = ROOT / "test_annotation_labels"
OUT_DIR        = ROOT / "data_splits"

MANIFEST_CSV   = LABELS_DIR / "labeling_manifest.csv"
TEST_MANIFEST  = TEST_LABELS / "labeling_manifest.csv"

VAL_RATIO      = 0.20   # 20% video clips -> Validation
SEED           = 42
AUG_SUFFIX     = "_aug"

# Augmentation: số lần nhân mỗi frame thiếu tối đa
OVERSAMPLE_TARGET_PER_CAT = 2500  # Target: min 2500 frame cho mỗi category

# Augmentation transforms (áp dụng lên ảnh VÀ mask đồng bộ)
AUG_TRANSFORMS = [
    "hflip",          # Horizontal flip
    "vflip",          # Vertical flip
    "rot90",          # Xoay 90 độ
    "rot270",         # Xoay 270 độ
    "bright_up",      # Tăng sáng
    "bright_down",    # Giảm sáng
    "gauss_blur",     # Gaussian blur
    "gauss_noise",    # Gaussian noise
    "hflip_rot90",    # Flip + Rotate 90
]

random.seed(SEED)
np.random.seed(SEED)


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def get_category(row):
    if row["has_fire"] and row["has_smoke"]:
        return "both"
    elif row["has_fire"]:
        return "fire_only"
    elif row["has_smoke"]:
        return "smoke_only"
    return "empty"


def apply_aug_to_image(img: np.ndarray, transform: str) -> np.ndarray:
    """Áp dụng augmentation cho ảnh (BGR / Grayscale)."""
    if transform == "hflip":
        return cv2.flip(img, 1)
    elif transform == "vflip":
        return cv2.flip(img, 0)
    elif transform == "rot90":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif transform == "rot270":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif transform == "hflip_rot90":
        return cv2.rotate(cv2.flip(img, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif transform == "bright_up":
        if len(img.shape) == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return np.clip(img.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
    elif transform == "bright_down":
        if len(img.shape) == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.7, 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
    elif transform == "gauss_blur":
        if len(img.shape) == 3:
            return cv2.GaussianBlur(img, (5, 5), 1.2)
        return img  # Mask không blur
    elif transform == "gauss_noise":
        if len(img.shape) == 3:
            noise = np.random.normal(0, 15, img.shape).astype(np.float32)
            return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img  # Mask không thêm noise
    return img


def augment_pair(img_path: Path, mask_path: Path, transform: str, out_img_path: Path, out_mask_path: Path):
    """Augment đồng bộ cặp (ảnh gốc, mask) và lưu vào đường dẫn mới."""
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return False

    img_aug  = apply_aug_to_image(img, transform)
    mask_aug = apply_aug_to_image(mask, transform)  # Mask chỉ áp dụng transform hình học

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img_path), img_aug)
    cv2.imwrite(str(out_mask_path), mask_aug)
    return True


# ─── BƯỚC 1: VIDEO-LEVEL GROUP SPLIT ─────────────────────────────────────────
def split_train_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chia theo video clip:
    - 1 clip chỉ nằm hoàn toàn trong Train hoặc Val (tránh data leakage)
    - Stratified: đảm bảo tỷ lệ 'category' tương đồng giữa 2 tập
    """
    logger.info("=" * 65)
    logger.info("  BUOC 1: VIDEO-LEVEL GROUP SPLIT (80/20)")
    logger.info("=" * 65)

    df = df.copy()
    df["category"] = df.apply(get_category, axis=1)
    df["video_stem"] = df["frame_path"].str.split("/").str[0]

    # Xác định dominant category của từng video
    video_cats = (
        df.groupby("video_stem")["category"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"category": "dominant_cat"})
    )

    val_videos = []
    train_videos = []

    for cat_name, group in video_cats.groupby("dominant_cat"):
        stems = group["video_stem"].tolist()
        random.shuffle(stems)
        n_val = max(1, int(len(stems) * VAL_RATIO))
        val_videos.extend(stems[:n_val])
        train_videos.extend(stems[n_val:])

    df_train = df[df["video_stem"].isin(train_videos)].copy()
    df_val   = df[df["video_stem"].isin(val_videos)].copy()

    logger.info(f"  Train videos : {len(train_videos)} clips -> {len(df_train)} frames")
    logger.info(f"  Val videos   : {len(val_videos)} clips -> {len(df_val)} frames")
    logger.info(f"  Train cat distribution: {df_train['category'].value_counts().to_dict()}")
    logger.info(f"  Val cat distribution  : {df_val['category'].value_counts().to_dict()}")
    return df_train, df_val


# ─── BƯỚC 2 & 3: OVER-SAMPLING + AUGMENTATION CHỈ TRÊN TRAIN ─────────────────
def augment_train(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý mất cân bằng lớp bằng augmentation đồng bộ.
    Mục tiêu: đạt tối thiểu OVERSAMPLE_TARGET_PER_CAT frames mỗi category.
    """
    logger.info("=" * 65)
    logger.info("  BUOC 2+3: CLASS BALANCE + AUGMENTATION (TRAIN ONLY)")
    logger.info("=" * 65)

    aug_records = []

    for cat_name, group in df_train.groupby("category"):
        current_n = len(group)
        target_n  = OVERSAMPLE_TARGET_PER_CAT
        deficit   = max(0, target_n - current_n)

        logger.info(f"  Category [{cat_name}]: {current_n} frames -> target {target_n} (deficit: {deficit})")

        if deficit == 0:
            aug_records.append(group)
            continue

        # Lặp qua các transform để đạt đủ số lượng
        transforms_cycle = AUG_TRANSFORMS * ((deficit // len(AUG_TRANSFORMS)) + 1)
        transforms_cycle = transforms_cycle[:deficit]

        samples = group.sample(n=deficit, replace=True, random_state=SEED).reset_index(drop=True)

        new_rows = []
        for i, (_, row) in enumerate(samples.iterrows()):
            transform = transforms_cycle[i % len(AUG_TRANSFORMS)]
            stem      = Path(row["frame_path"]).stem
            parent    = Path(row["frame_path"]).parent

            aug_frame_name = f"{stem}{AUG_SUFFIX}_{transform}.png"
            aug_mask_name  = f"{stem}{AUG_SUFFIX}_{transform}_mask.png"

            aug_frame_path = parent / aug_frame_name
            aug_mask_path  = parent / aug_mask_name

            # Tạo ảnh augment
            orig_img_path  = FRAMES_DIR / row["frame_path"]
            orig_mask_path = LABELS_DIR / row["mask_path"]

            out_img_path  = FRAMES_DIR / aug_frame_path
            out_mask_path = LABELS_DIR / aug_mask_path

            success = augment_pair(orig_img_path, orig_mask_path, transform, out_img_path, out_mask_path)
            if not success:
                continue

            new_row = row.copy()
            new_row["frame_path"] = str(aug_frame_path).replace("\\", "/")
            new_row["mask_path"]  = str(aug_mask_path).replace("\\", "/")
            new_row["augmented"]  = transform
            new_rows.append(new_row)

        if new_rows:
            aug_df = pd.DataFrame(new_rows)
            group["augmented"] = "original"
            aug_records.append(group)
            aug_records.append(aug_df)
            logger.info(f"    -> {len(new_rows)} frames augmented and saved.")
        else:
            group["augmented"] = "original"
            aug_records.append(group)

    df_train_aug = pd.concat(aug_records, ignore_index=True)

    # Đánh dấu bản gốc
    if "augmented" not in df_train.columns:
        df_train["augmented"] = "original"

    logger.info(f"  Total Train frames after augmentation: {len(df_train_aug)}")
    logger.info(f"  Final category distribution:")
    for c, n in df_train_aug["category"].value_counts().items():
        logger.info(f"    {c}: {n}")

    return df_train_aug


# ─── BƯỚC 4: LƯU MANIFESTS ────────────────────────────────────────────────────
def save_manifests(df_train_aug, df_val, df_test):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_train_aug.to_csv(OUT_DIR / "train_manifest.csv", index=False, encoding="utf-8")
    df_val.to_csv(OUT_DIR / "val_manifest.csv", index=False, encoding="utf-8")
    df_test.to_csv(OUT_DIR / "test_manifest.csv", index=False, encoding="utf-8")

    # JSON statistics
    stats = {
        "train": {
            "total_frames": len(df_train_aug),
            "original_frames": int((df_train_aug.get("augmented", "original") == "original").sum()),
            "augmented_frames": int((df_train_aug.get("augmented", "original") != "original").sum()),
            "category_distribution": df_train_aug["category"].value_counts().to_dict(),
            "avg_fire_pixels": float(df_train_aug["fire_pixels"].mean()),
            "avg_smoke_pixels": float(df_train_aug["smoke_pixels"].mean()),
        },
        "val": {
            "total_frames": len(df_val),
            "category_distribution": df_val["category"].value_counts().to_dict(),
            "avg_fire_pixels": float(df_val["fire_pixels"].mean()),
            "avg_smoke_pixels": float(df_val["smoke_pixels"].mean()),
        },
        "test": {
            "total_frames": len(df_test),
            "category_distribution": df_test["category"].value_counts().to_dict(),
            "avg_fire_pixels": float(df_test["fire_pixels"].mean()),
            "avg_smoke_pixels": float(df_test["smoke_pixels"].mean()),
        },
    }

    with open(OUT_DIR / "split_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("  Manifests saved to data_splits/")
    return stats


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 65)
    logger.info("  FIRE/SMOKE DATASET PREPARATION PIPELINE")
    logger.info("  Train/Val Split + Class Balance + Augmentation")
    logger.info("=" * 65)

    # Load manifests
    df = pd.read_csv(MANIFEST_CSV)
    df["category"] = df.apply(get_category, axis=1)
    df["augmented"] = "original"

    df_test = pd.read_csv(TEST_MANIFEST)
    df_test["category"] = df_test.apply(get_category, axis=1)
    df_test["augmented"] = "original"

    logger.info(f"  Train+Val source: {len(df)} frames from {MANIFEST_CSV}")
    logger.info(f"  Test source     : {len(df_test)} frames from {TEST_MANIFEST}")

    # Step 1: Group split
    df_train_raw, df_val = split_train_val(df)

    # Step 2+3: Augmentation on train only
    df_train_aug = augment_train(df_train_raw)

    # Step 4: Save manifests
    stats = save_manifests(df_train_aug, df_val, df_test)

    # Final summary
    logger.info("=" * 65)
    logger.info("  PIPELINE COMPLETE — DATA SPLITS READY")
    logger.info("=" * 65)
    logger.info(f"  Train (augmented) : {stats['train']['total_frames']:,} frames")
    logger.info(f"  Validation        : {stats['val']['total_frames']:,} frames (NO augmentation)")
    logger.info(f"  Test              : {stats['test']['total_frames']:,} frames (NO augmentation, isolated)")
    logger.info(f"  Manifests         : data_splits/{{train,val,test}}_manifest.csv")
    logger.info(f"  Statistics JSON   : data_splits/split_statistics.json")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
