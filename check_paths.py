"""
Script kiem tra duong dan tren Modal Volume.
Chay: modal run check_paths.py
"""
import modal

app = modal.App("check-paths")
volume = modal.Volume.from_name("fire_smoke_dataset")

image = modal.Image.debian_slim()


@app.function(image=image, volumes={"/data": volume})
def check_paths():
    import os

    # In cau truc cap 1-2 de tim dung ten thu muc
    print("=" * 60)
    print("VOLUME /data - TOP LEVEL (2 levels)")
    print("=" * 60)

    def list_top(path, depth=0, max_depth=2):
        if not os.path.exists(path):
            print("  " * depth + "[NOT FOUND] " + path)
            return
        if depth > max_depth:
            return
        try:
            entries = sorted(os.listdir(path))
        except PermissionError:
            return
        for entry in entries:
            full = os.path.join(path, entry)
            indent = "  " * depth
            if os.path.isdir(full):
                n = len(os.listdir(full))
                print(indent + "[DIR]  " + entry + "/  (" + str(n) + " items)")
                list_top(full, depth + 1, max_depth)
            else:
                size = os.path.getsize(full)
                print(indent + "[FILE] " + entry + "  (" + str(size) + " bytes)")

    list_top("/data", max_depth=2)

    # Kiem tra truc tiep cac duong dan quan trong
    print("\n" + "=" * 60)
    print("CHECKING IMPORTANT PATHS")
    print("=" * 60)

    important_paths = [
        "/data",
        "/data/spatial_fire_smoke_weights_v2.pth",
        "/data/fire_smoke_dataset",
        "/data/fire_smoke_dataset/spatial_fire_smoke_weights_v2.pth",
        "/data/fire_smoke_dataset/unzipped_data",
        "/data/fire_smoke_dataset/unzipped_data/dataset_merged",
        "/data/fire_smoke_dataset/unzipped_data/dataset_merged/dataset_merged",
        "/data/fire_smoke_dataset/unzipped_data/dataset_merged/dataset_merged/class_0",
        "/data/fire_smoke_dataset/unzipped_data/dataset_merged/dataset_merged/class_1",
        "/data/merged_dataset",
        "/data/merged_dataset/train",
        "/data/merged_dataset/val",
        "/data/cleaned_dataset_merged",
    ]

    for p in important_paths:
        if os.path.exists(p):
            if os.path.isdir(p):
                n = len(os.listdir(p))
                print("  [OK]  DIR   " + p + "  (" + str(n) + " items)")
            else:
                size = os.path.getsize(p)
                print("  [OK]  FILE  " + p + "  (" + str(size) + " bytes)")
        else:
            print("  [--]  MISS  " + p)

    print("=" * 60)
    print("DONE")
    print("=" * 60)


@app.local_entrypoint()
def main():
    check_paths.remote()
