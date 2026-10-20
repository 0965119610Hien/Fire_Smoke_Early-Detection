"""
scripts/compare_evals.py
========================
Merges several docs/reports/eval_*.json files (same test frames) into one
comparison table.

  python scripts/compare_evals.py docs/reports/eval_A.json docs/reports/eval_B.json --out docs/reports/compare.md
"""

import argparse
import json
import math
from pathlib import Path


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.2f}"


def ci(c):
    return f"[{pct(c[0])}, {pct(c[1])}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for p in args.evals:
        r = json.loads(Path(p).read_text(encoding="utf-8"))
        mode = r.get("clip_mode", "sparse")
        for cfg, res in r["results"].items():
            # Stage 1 does not depend on the Stage 2 clip layout: label it by its input frames
            label = f"{cfg} [{mode}]"
            t = res["test"]
            rows.append((label, r["run_id"], t, r["checkpoints"]["stage2"]["epoch"] if cfg != "stage1_framewise" else
                         r["checkpoints"]["stage1"]["epoch"]))

    L = ["# So sánh đánh giá trên tập test", "",
         "Cùng 208 frame test có nhãn (mỗi frame tính 1 lần); ngưỡng chọn trên val cho từng cấu hình.",
         "`[sparse]` = clip 1–2 FPS gồm các frame đã có nhãn; `[dense10fps]` = 7 frame ngữ cảnh 10 FPS trước mỗi frame có nhãn.",
         "", "## Pixel level (%)", "",
         "| Cấu hình | Run | Epoch | Frames | IoU Fire [95% CI] | IoU Smoke [95% CI] | Fg-mIoU [95% CI] | All-mIoU |",
         "|---|---|---|---|---|---|---|---|"]
    for label, run, t, ep in rows:
        p, c = t["pixel"], t["bootstrap_ci95"]
        L.append(f"| {label} | {run} | {ep} | {t['num_frames']} | {pct(p['iou_fire'])} {ci(c['iou_fire'])} | "
                 f"{pct(p['iou_smoke'])} {ci(c['iou_smoke'])} | {pct(p['fg_miou'])} {ci(c['fg_miou'])} | {pct(p['all_miou'])} |")
    L += ["", "## Frame level — báo động (lửa hoặc khói)", "",
          "| Cấu hình | Recall | FAR | Precision | F1 | MCC | Fire Recall / FAR | Smoke Recall / FAR |",
          "|---|---|---|---|---|---|---|---|"]
    for label, run, t, ep in rows:
        a, f, s = t["frame"]["any"], t["frame"]["fire"], t["frame"]["smoke"]
        L.append(f"| {label} | {pct(a['Recall'])} | {pct(a['FAR'])} | {pct(a['Precision'])} | {pct(a['F1'])} | {a['MCC']:.3f} | "
                 f"{pct(f['Recall'])} / {pct(f['FAR'])} | {pct(s['Recall'])} / {pct(s['FAR'])} |")
    L += ["", "## Object level", "", "| Cấu hình | Fire LTPR / LPPV | Smoke LTPR / LPPV | Pred regions fire / smoke |",
          "|---|---|---|---|"]
    for label, run, t, ep in rows:
        o = t["object"]
        L.append(f"| {label} | {pct(o['fire']['LTPR'])} / {pct(o['fire']['LPPV'])} | {pct(o['smoke']['LTPR'])} / "
                 f"{pct(o['smoke']['LPPV'])} | {o['fire']['pred_regions']} / {o['smoke']['pred_regions']} |")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
