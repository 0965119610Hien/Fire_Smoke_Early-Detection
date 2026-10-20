import os
import shutil
import random
from pathlib import Path
from collections import Counter

# Try importing sklearn and cv2, install if missing
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Installing scikit-learn...")
    os.system("pip install scikit-learn")
    from sklearn.model_selection import train_test_split

try:
    import cv2
except ImportError:
    print("Installing opencv-python...")
    os.system("pip install opencv-python")
    import cv2

import numpy as np

def get_image_category_and_counts(label_path):
    if not os.path.exists(label_path):
        return -1, 0, 0 # Background/empty, 0 class0, 0 class1
    
    c0 = 0
    c1 = 0
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                try:
                    cls_id = int(parts[0])
                    if cls_id == 0:
                        c0 += 1
                    elif cls_id == 1:
                        c1 += 1
                except ValueError:
                    pass
                    
    if c0 > 0 and c1 > 0:
        return 2, c0, c1
    elif c0 > 0:
        return 0, c0, c1
    elif c1 > 0:
        return 1, c0, c1
    else:
        return -1, c0, c1

def augment_image_and_label(img_path, label_path, dest_img_path, dest_label_path, aug_type):
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        return False
        
    # Read labels
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([float(p) for p in parts[:5]])
                    
    new_labels = []
    
    if aug_type == 'hflip':
        img = cv2.flip(img, 1) # horizontal flip
        for lbl in labels:
            cls_id, x, y, w, h = lbl
            new_x = 1.0 - x # flip x center
            new_labels.append([cls_id, new_x, y, w, h])
            
    elif aug_type == 'brightness':
        # Randomly adjust brightness by a factor between 0.7 and 1.3
        factor = random.uniform(0.7, 1.3)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        new_labels = labels # Bboxes remain unchanged
        
    # Write new image
    cv2.imwrite(str(dest_img_path), img)
    
    # Write new labels
    with open(dest_label_path, 'w') as f:
        for lbl in new_labels:
            f.write(f"{int(lbl[0])} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
            
    return True

def main():
    base_dir = r"e:\NCKH\Code\Fire_Smoke_Early-Detection\Final_Merged_Dataset"
    images_dir = Path(base_dir) / "images"
    labels_dir = Path(base_dir) / "labels"
    
    output_dir = Path(r"e:\NCKH\Code\Fire_Smoke_Early-Detection\Dataset_Balanced_Split")
    
    if not images_dir.exists():
        print("Final_Merged_Dataset not found.")
        return

    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    categories = []
    
    print("Categorizing images for stratified split...")
    for img_name in images:
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = labels_dir / label_name
        cat, _, _ = get_image_category_and_counts(label_path)
        categories.append(cat)
            
    print(f"Total images: {len(images)}")
    cat_counts = Counter(categories)
    print(f"Image Categories -> Empty: {cat_counts[-1]}, Only 0: {cat_counts[0]}, Only 1: {cat_counts[1]}, Both: {cat_counts[2]}")
    
    # 80/10/10 Split
    X_train, X_temp, y_train, y_temp = train_test_split(images, categories, test_size=0.2, stratify=categories, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    print(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Copying files
    splits = {
        'train': X_train,
        'val': X_val,
        'test': X_test
    }
    
    train_c0_bboxes = 0
    train_c1_bboxes = 0
    train_cat_1_images = []
    
    for split_name, img_list in splits.items():
        split_img_dir = output_dir / split_name / "images"
        split_lbl_dir = output_dir / split_name / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Copying {split_name} set ({len(img_list)} images)...")
        for img_name in img_list:
            label_name = os.path.splitext(img_name)[0] + '.txt'
            src_img = images_dir / img_name
            src_lbl = labels_dir / label_name
            
            dest_img = split_img_dir / img_name
            dest_lbl = split_lbl_dir / label_name
            
            if src_img.exists():
                shutil.copy2(src_img, dest_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dest_lbl)
                
            # Count bboxes for train set to calculate imbalance later
            if split_name == 'train':
                cat, c0, c1 = get_image_category_and_counts(dest_lbl)
                train_c0_bboxes += c0
                train_c1_bboxes += c1
                if cat == 1: # Only class 1
                    train_cat_1_images.append(img_name)
                    
    # Balance Train Set
    print("\n--- Balancing Train Set ---")
    print(f"Train BBoxes -> Class 0: {train_c0_bboxes}, Class 1: {train_c1_bboxes}")
    
    deficit = train_c0_bboxes - train_c1_bboxes
    
    if deficit > 0:
        print(f"Class 1 (Smoke) is minority. Need approximately {deficit} more bboxes.")
        # Augment class 1 images
        if len(train_cat_1_images) == 0:
            print("No purely class 1 images found to augment.")
        else:
            aug_types = ['hflip', 'brightness']
            num_augs_needed = int(deficit / (train_c1_bboxes / len(X_train)) ) if train_c1_bboxes > 0 else deficit
            # A rough heuristic: just augment `deficit` images. Each image has >=1 bbox.
            # We'll just randomly sample `deficit` times from train_cat_1_images.
            # If deficit is large, we might sample the same image multiple times with different augs.
            
            images_to_augment = random.choices(train_cat_1_images, k=deficit)
            
            train_img_dir = output_dir / 'train' / 'images'
            train_lbl_dir = output_dir / 'train' / 'labels'
            
            print(f"Applying augmentation to {len(images_to_augment)} images...")
            for i, img_name in enumerate(images_to_augment):
                label_name = os.path.splitext(img_name)[0] + '.txt'
                src_img = train_img_dir / img_name
                src_lbl = train_lbl_dir / label_name
                
                aug_type = random.choice(aug_types)
                new_img_name = f"aug_{i}_{aug_type}_{img_name}"
                new_lbl_name = os.path.splitext(new_img_name)[0] + '.txt'
                
                dest_img = train_img_dir / new_img_name
                dest_lbl = train_lbl_dir / new_lbl_name
                
                success = augment_image_and_label(src_img, src_lbl, dest_img, dest_lbl, aug_type)
                if success:
                    train_c1_bboxes += 1 # Rough estimate
                    
            print(f"Finished augmentation. Class 1 has been oversampled.")
            
    elif deficit < 0:
        print(f"Class 0 (Fire) is minority. Need approximately {-deficit} more bboxes. (Script currently optimizes for Class 1 deficit, but can be adapted easily)")
    else:
        print("Classes are perfectly balanced!")

    print("Dataset splitting and balancing complete!")
    print(f"Output saved to: {output_dir}")

if __name__ == '__main__':
    main()
