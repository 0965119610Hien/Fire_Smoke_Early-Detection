import cv2, os, glob, csv, numpy as np

BASE      = r"e:\NCKH\Code\Fire_Smoke_Early-Detection"
DATA_DIR  = os.path.join(BASE, "data") if os.path.exists(os.path.join(BASE, "data")) else BASE
VID_DIR   = os.path.join(DATA_DIR, "all_videos_raw")
OUT_DIR   = os.path.join(DATA_DIR, "frames_for_annotation")
MANIFEST  = os.path.join(OUT_DIR, "manifest.csv")
FPS_OUT   = 1
SIZE      = 640
JPEG_Q    = 92

def clahe_lab(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)

os.makedirs(OUT_DIR, exist_ok=True)
videos = sorted(glob.glob(os.path.join(VID_DIR, "*.mp4")) +
                glob.glob(os.path.join(VID_DIR, "*.avi")) +
                glob.glob(os.path.join(VID_DIR, "*.mov")) +
                glob.glob(os.path.join(VID_DIR, "*.mkv")))

stats = {"videos": 0, "frames": 0, "skipped": 0}
fields = ["video_stem","frame_idx","timestamp_sec","frame_orig","width","height","rel_path"]

with open(MANIFEST, "w", newline="", encoding="utf-8") as mf:
    w = csv.DictWriter(mf, fieldnames=fields)
    w.writeheader()
    for i, vpath in enumerate(videos, 1):
        stem = os.path.splitext(os.path.basename(vpath))[0]
        stem_safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in stem)
        out_vdir = os.path.join(OUT_DIR, stem_safe)
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print("[%3d/%d] SKIP: %s" % (i, len(videos), stem[:50]), flush=True)
            stats["skipped"] += 1
            continue
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if orig_fps <= 0 or total_f <= 0:
            cap.release(); stats["skipped"] += 1; continue
        duration = total_f / orig_fps
        step     = max(1, round(orig_fps / FPS_OUT))
        os.makedirs(out_vdir, exist_ok=True)
        frame_idx = 0
        fno = 0
        while fno < total_f:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ret, frame = cap.read()
            if ret and frame is not None:
                fr = cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                fe = clahe_lab(fr)
                out_path = os.path.join(out_vdir, "frame_%05d.jpg" % frame_idx)
                cv2.imwrite(out_path, fe, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                w.writerow({"video_stem": stem, "frame_idx": frame_idx,
                            "timestamp_sec": round(fno/orig_fps, 3), "frame_orig": fno,
                            "width": orig_w, "height": orig_h,
                            "rel_path": os.path.relpath(out_path, BASE)})
                frame_idx += 1
                stats["frames"] += 1
            fno += step
        cap.release()
        stats["videos"] += 1
        print("[%3d/%d] OK  %-55s | %6.1fs -> %d frames" % (i, len(videos), stem_safe[:55], duration, frame_idx), flush=True)

print("HOAN THANH: %d videos | %d frames | %d skipped" % (stats["videos"], stats["frames"], stats["skipped"]), flush=True)