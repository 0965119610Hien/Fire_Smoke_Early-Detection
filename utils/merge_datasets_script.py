import os
import shutil
from pathlib import Path

def is_label_empty(label_path):
    if not os.path.exists(label_path):
        return True
    with open(label_path, 'r') as f:
        content = f.read().strip()
        if not content:
            return True
    return False

def copy_with_collision_handling(img_path, label_path, dest_image_dir, dest_label_dir, img_name, label_name):
    dest_img_path = Path(dest_image_dir) / img_name
    dest_lbl_path = Path(dest_label_dir) / label_name
    
    counter = 1
    while dest_img_path.exists():
        name, ext = os.path.splitext(img_name)
        new_img_name = f"{name}_{counter}{ext}"
        new_lbl_name = f"{name}_{counter}.txt"
        dest_img_path = Path(dest_image_dir) / new_img_name
        dest_lbl_path = Path(dest_label_dir) / new_lbl_name
        counter += 1
    
    shutil.copy2(img_path, dest_img_path)
    if os.path.exists(label_path):
        shutil.copy2(label_path, dest_lbl_path)
    else:
        # Create empty label file if it doesn't exist
        with open(dest_lbl_path, 'w') as f:
            pass

def merge_datasets(image_dataset_srcs, my_project_srcs, dest_image_dir, dest_label_dir):
    os.makedirs(dest_image_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)
    
    # Process image_dataset (filter empty labels)
    for src_dir in image_dataset_srcs:
        print(f"Processing {src_dir} (filtering empty labels)...")
        src_path = Path(src_dir)
        if not src_path.exists():
            print(f"Warning: {src_dir} does not exist.")
            continue
            
        images_dir = src_path / 'images'
        labels_dir = src_path / 'labels'
        if not images_dir.exists():
            print(f"Warning: images dir missing in {src_dir}")
            continue
            
        for img_name in os.listdir(images_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue
            
            img_path = images_dir / img_name
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = labels_dir / label_name
            
            if not is_label_empty(label_path):
                copy_with_collision_handling(img_path, label_path, dest_image_dir, dest_label_dir, img_name, label_name)

    # Process My First Project (merge directly)
    for src_dir in my_project_srcs:
        print(f"Processing {src_dir} (merging all)...")
        src_path = Path(src_dir)
        if not src_path.exists():
            print(f"Warning: {src_dir} does not exist.")
            continue
            
        images_dir = src_path / 'images'
        labels_dir = src_path / 'labels'
        if not images_dir.exists():
            print(f"Warning: images dir missing in {src_dir}")
            continue
            
        for img_name in os.listdir(images_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue
                
            img_path = images_dir / img_name
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = labels_dir / label_name
            
            copy_with_collision_handling(img_path, label_path, dest_image_dir, dest_label_dir, img_name, label_name)

if __name__ == '__main__':
    base_dir = r"e:\NCKH\Code\Fire_Smoke_Early-Detection"
    
    # Sources from image_dataset
    image_dataset_dir = os.path.join(base_dir, "image_dataset", "merged_dataset")
    image_dataset_srcs = [
        os.path.join(image_dataset_dir, split) for split in ['train', 'val', 'test']
    ]
    
    # Source from My First Project
    my_project_dir = os.path.join(base_dir, "My First Project.yolov8")
    my_project_srcs = [
        os.path.join(my_project_dir, split) for split in ['train', 'valid', 'test']
    ]
            
    final_merged_dir = os.path.join(base_dir, "Final_Merged_Dataset")
    dest_image_dir = os.path.join(final_merged_dir, "images")
    dest_label_dir = os.path.join(final_merged_dir, "labels")
    
    print("Starting dataset merge...")
    merge_datasets(image_dataset_srcs, my_project_srcs, dest_image_dir, dest_label_dir)
    print("Dataset merge completed successfully!")
    print(f"Total images in final dataset: {len(os.listdir(dest_image_dir))}")
    print(f"Total labels in final dataset: {len(os.listdir(dest_label_dir))}")
