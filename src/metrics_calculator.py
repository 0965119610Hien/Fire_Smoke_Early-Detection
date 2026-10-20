"""
src/metrics_calculator.py
==========================
Bộ thư viện tính toán toàn bộ các độ đo đánh giá khoa học (Evaluation Metrics)
cho bài toán Phát hiện sớm Lửa & Khói:

1. Binary Classification Metrics (Cấp độ khung hình / Cảnh báo):
   - ACC: Overall Accuracy = (TP + TN) / (TP + TN + FP + FN)
   - TPR / Recall / Sensitivity: TP / (TP + FN)
   - FPR / FAR (False Alarm Rate): FP / (FP + TN)
   - TNR / Specificity: TN / (TN + FP)
   - FNR / Missed Alarm Rate: FN / (TP + FN) = 1 - TPR
   - PPV / Precision: TP / (TP + FP)
   - FDR (False Discovery Rate): FP / (TP + FP) = 1 - PPV
   - F1-Score: 2 * (PPV * TPR) / (PPV + TPR)
   - MCC (Matthews Correlation Coefficient):
     (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

2. Localization Metrics (Cấp độ vùng / Đốm phân đoạn liên thông):
   - LTPR (Localization True Positive Rate): Tỷ lệ các vùng lửa/khói thực tế được phát hiện
   - LPPV (Localization Positive Predictive Value): Tỷ lệ các vùng cảnh báo là chính xác
   - Mean Similarity S: Độ tương đồng trung bình Dice / IoU giữa các vùng dự đoán và thực tế

3. Pixel-level Segmentation Metrics:
   - mIoU, IoU Fire, IoU Smoke, Dice Fire, Dice Smoke

4. Temporal Video Early Detection Metrics (Dành riêng cho Stage 2 Video):
   - Detection Latency / Time-to-Detect (TTD): Số frame từ khi lửa/khói xuất hiện đến khi báo động
   - Temporal Stability / Persistence: Độ ổn định cảnh báo qua các frame liên tiếp

5. Speed Metrics:
   - Latency (ms/frame) và FPS (Frames Per Second)
"""

import math
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F

def compute_binary_classification_metrics(y_true, y_pred):
    """
    Tính toán toàn diện 9 chỉ số phân loại nhị phân từ nhãn thực tế và nhãn dự đoán (0 hoặc 1).
    Hỗ trợ xử lý an toàn các trường hợp mẫu bằng 0 (tránh division by zero).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn
    eps = 1e-7

    acc = (tp + tn) / max(total, 1)
    tpr = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    fpr = fp / max(fp + tn, 1) if (fp + tn) > 0 else 0.0
    tnr = tn / max(tn + fp, 1) if (tn + fp) > 0 else 0.0
    fnr = fn / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    ppv = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    fdr = fp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0

    if ppv + tpr > 0:
        f1 = 2 * (ppv * tpr) / (ppv + tpr)
    else:
        f1 = 0.0

    # Matthews Correlation Coefficient (MCC)
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = (mcc_num / mcc_den) if mcc_den > 0 else 0.0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Total": total,
        "ACC": float(acc),
        "TPR_Recall": float(tpr),
        "FPR_FAR": float(fpr),
        "TNR_Specificity": float(tnr),
        "FNR_MissRate": float(fnr),
        "PPV_Precision": float(ppv),
        "FDR": float(fdr),
        "F1_Score": float(f1),
        "MCC": float(mcc)
    }

def extract_connected_components(binary_mask, min_area=16):
    """
    Trích xuất các đốm liên thông (blobs) từ mask nhị phân:
    Trả về danh sách các blob mask riêng biệt có diện tích >= min_area.
    """
    binary_mask = (binary_mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    blobs = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            blob = (labels == i).astype(np.uint8)
            blobs.append(blob)
    return blobs

def compute_localization_metrics(pred_masks, gt_masks, target_class=1, iou_thresh=0.2, min_area=16):
    """
    Tính toán các chỉ số định vị vùng (Localization Metrics):
    - LTPR (Localization True Positive Rate): Tỷ lệ vùng Ground Truth tìm thấy
    - LPPV (Localization Positive Predictive Value): Tỷ lệ vùng Dự đoán đúng
    - Mean Similarity S: Độ tương đồng trung bình Dice (Sørensen-Dice) của các cặp matched
    
    Args:
        pred_masks: List hoặc array các mask dự đoán (H, W)
        gt_masks: List hoặc array các mask thực tế (H, W)
        target_class: 1 cho Fire, 2 cho Smoke
        iou_thresh: Ngưỡng IoU tối thiểu để coi là phát hiện đúng vùng
    """
    total_gt_regions = 0
    detected_gt_regions = 0
    total_pred_regions = 0
    correct_pred_regions = 0
    similarities = []

    for pred_m, gt_m in zip(pred_masks, gt_masks):
        p_bin = (pred_m == target_class).astype(np.uint8)
        g_bin = (gt_m == target_class).astype(np.uint8)

        p_blobs = extract_connected_components(p_bin, min_area=min_area)
        g_blobs = extract_connected_components(g_bin, min_area=min_area)

        total_pred_regions += len(p_blobs)
        total_gt_regions += len(g_blobs)

        if len(g_blobs) == 0 and len(p_blobs) == 0:
            continue

        matched_p = set()
        matched_g = set()

        for g_idx, g_blob in enumerate(g_blobs):
            best_iou = 0.0
            best_sim = 0.0
            best_p_idx = -1

            for p_idx, p_blob in enumerate(p_blobs):
                inter = np.sum((g_blob == 1) & (p_blob == 1))
                union = np.sum((g_blob == 1) | (p_blob == 1))
                if union == 0:
                    continue
                iou = inter / union
                # Dice similarity
                sim = (2.0 * inter) / (np.sum(g_blob) + np.sum(p_blob))

                if iou > best_iou:
                    best_iou = iou
                    best_sim = sim
                    best_p_idx = p_idx

            if best_iou >= iou_thresh and best_p_idx != -1:
                detected_gt_regions += 1
                matched_g.add(g_idx)
                matched_p.add(best_p_idx)
                similarities.append(best_sim)

        correct_pred_regions += len(matched_p)

    ltpr = (detected_gt_regions / total_gt_regions) if total_gt_regions > 0 else 1.0
    lppv = (correct_pred_regions / total_pred_regions) if total_pred_regions > 0 else (1.0 if total_gt_regions == 0 else 0.0)
    mean_s = float(np.mean(similarities)) if len(similarities) > 0 else 0.0

    return {
        "Total_GT_Regions": total_gt_regions,
        "Detected_GT_Regions": detected_gt_regions,
        "Total_Pred_Regions": total_pred_regions,
        "Correct_Pred_Regions": correct_pred_regions,
        "LTPR": float(ltpr),
        "LPPV": float(lppv),
        "Mean_Similarity_S": float(mean_s),
        "Num_Matches": len(similarities)
    }

def compute_pixel_segmentation_metrics(pred_masks, gt_masks, num_classes=3):
    """
    Tính toán các độ đo phân đoạn ngữ nghĩa cấp độ pixel:
    - IoU từng lớp (Background, Fire, Smoke)
    - Dice từng lớp
    - mIoU (Mean IoU của toàn bộ lớp và Foreground mIoU)
    """
    ious = []
    dices = []

    for c in range(num_classes):
        inter_total = 0
        union_total = 0
        card_total = 0

        for p_m, g_m in zip(pred_masks, gt_masks):
            p = (p_m == c)
            g = (g_m == c)
            inter = np.sum(p & g)
            union = np.sum(p | g)
            card = np.sum(p) + np.sum(g)

            inter_total += inter
            union_total += union
            card_total += card

        iou = (inter_total + 1e-6) / (union_total + 1e-6)
        dice = (2.0 * inter_total + 1e-6) / (card_total + 1e-6)
        ious.append(float(iou))
        dices.append(float(dice))

    fg_miou = (ious[1] + ious[2]) / 2.0
    all_miou = sum(ious) / num_classes

    return {
        "IoU_Background": ious[0],
        "IoU_Fire": ious[1],
        "IoU_Smoke": ious[2],
        "Dice_Background": dices[0],
        "Dice_Fire": dices[1],
        "Dice_Smoke": dices[2],
        "Fg_mIoU": fg_miou,
        "All_mIoU": all_miou
    }

def measure_fps(inference_fn, sample_input, num_warmup=10, num_runs=50):
    """
    Đo tốc độ xử lý thực tế: Latency (ms) và FPS (Frames Per Second) trên GPU/CPU.
    """
    # Khởi động (warmup)
    for _ in range(num_warmup):
        _ = inference_fn(sample_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_runs):
        _ = inference_fn(sample_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    avg_latency_ms = (total_time / num_runs) * 1000.0
    fps = num_runs / total_time

    return {
        "Avg_Latency_ms": float(avg_latency_ms),
        "FPS": float(fps),
        "Num_Runs": num_runs
    }
