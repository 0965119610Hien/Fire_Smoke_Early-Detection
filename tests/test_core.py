"""CPU-only unit tests for src/fire_core.py (no internet needed)."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import fire_core as fc  # noqa: E402

transformers = pytest.importorskip("transformers")
from transformers import SegformerConfig, SegformerForSemanticSegmentation  # noqa: E402

H = W = 128
T = 4


def make_hf_model(seed=0):
    torch.manual_seed(seed)
    return SegformerForSemanticSegmentation(SegformerConfig(num_labels=3))


def make_st_model(causal=True, gamma=0.0, seed=0, freeze_decode_bn=True):
    hf = make_hf_model(seed)
    model = fc.SpatioTemporalSegFormer(hf, causal=causal, freeze_decode_bn=freeze_decode_bn, verbose=False)
    with torch.no_grad():
        model.temporal_refinement.gamma.fill_(gamma)
    return hf, model


# 1. Shape -------------------------------------------------------------------
def test_output_shape():
    _, model = make_st_model()
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, T, 3, H, W))
    assert out.shape == (2, T, 3, H, W)


# 2. gamma = 0 equals Stage 1 -------------------------------------------------
def test_gamma_zero_equals_stage1():
    hf, model = make_st_model(gamma=0.0)
    hf.eval()
    model.eval()
    x = torch.randn(1, T, 3, H, W)
    with torch.no_grad():
        out = model(x)
        ref = F.interpolate(hf(pixel_values=x[0]).logits, size=(H, W), mode="bilinear", align_corners=False)
    assert torch.allclose(out[0], ref, atol=1e-5)


# 3. Causality ----------------------------------------------------------------
@pytest.mark.parametrize("causal", [True, False])
def test_causal_temporal_module(causal):
    _, model = make_st_model(causal=causal, gamma=1.0)
    model.eval()
    x = torch.randn(1, T, 3, H, W)
    x2 = x.clone()
    x2[:, -1] = torch.randn(3, H, W)
    with torch.no_grad():
        a, b = model(x), model(x2)
    unchanged = torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)
    assert unchanged == causal


# 4. Frozen decode-head BN ----------------------------------------------------
def test_decode_bn_frozen():
    _, model = make_st_model()
    model.train()
    bn = model.decode_head.batch_norm
    assert bn.training is False
    assert model.segformer.training is False
    assert model.temporal_refinement.training is True
    before = bn.running_mean.clone()
    model(torch.randn(1, T, 3, H, W))
    assert torch.equal(before, bn.running_mean)


# 5. Dice ---------------------------------------------------------------------
def test_dice_ignores_pad():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 16, 16)
    targets = torch.randint(0, 3, (2, 16, 16))
    t2 = targets.clone()
    t2[:, :4] = fc.IGNORE_INDEX
    l2 = logits.clone()
    l2[:, :, :4] = torch.randn(2, 3, 4, 16) * 10  # arbitrary predictions inside the pad
    base = fc.dice_loss_fg(logits, t2)
    assert torch.allclose(base, fc.dice_loss_fg(l2, t2))
    ce = fc.seg_ce_dice_loss(logits, t2, None)[1]
    ce2 = fc.seg_ce_dice_loss(l2, t2, None)[1]
    assert torch.allclose(ce, ce2)


def test_dice_all_background_is_zero():
    logits = torch.zeros(4, 3, 32, 32)
    logits[:, 0] = 20.0
    targets = torch.zeros(4, 32, 32, dtype=torch.long)
    assert fc.dice_loss_fg(logits, targets).item() < 1e-3


# 6. Temporal smoothness ------------------------------------------------------
def test_smoothness_ignores_gt_transitions():
    crit = fc.SpatioTemporalLoss(dice_weight=1.0, temp_smooth_weight=0.05)
    torch.manual_seed(0)
    logits = torch.randn(1, 2, 3, 8, 8)
    targets = torch.zeros(1, 2, 8, 8, dtype=torch.long)
    targets[0, 1, 3, 3] = 1  # onset of fire at one pixel
    base = crit.temporal_smoothness(logits, targets)
    l2 = logits.clone()
    l2[0, 1, :, 3, 3] = torch.tensor([-10.0, 10.0, -10.0])  # big change exactly where GT changes
    assert torch.allclose(crit.temporal_smoothness(l2, targets), base)
    l3 = logits.clone()
    l3[0, 1, :, 0, 0] = torch.tensor([-10.0, 10.0, -10.0])  # change where GT is stable
    assert crit.temporal_smoothness(l3, targets) > base - 1e-6


# 7. Motion -------------------------------------------------------------------
def test_apply_motion_none_is_identity():
    x = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    assert fc.apply_motion(x, None) is x


def test_base_independent_of_position():
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (48, 64, 3), dtype=np.uint8) for _ in range(4)]

    def processed(seq):
        out, prev = [], None
        for f in seq:
            base = fc.enhance_base(f)
            out.append(fc.apply_motion(base, prev))
            prev = base
        return out

    a = processed(frames[0:3])  # frame 2 at position 2
    b = processed(frames[1:4])  # frame 2 at position 1
    assert np.array_equal(a[2], b[1])
    assert np.array_equal(fc.enhance_base(frames[2]), fc.enhance_base(frames[2].copy()))


# 8. Mask validation / file reading ------------------------------------------
def test_validate_mask_rejects_255():
    m = np.zeros((4, 4), dtype=np.uint8)
    m[0, 0] = 255
    with pytest.raises(ValueError):
        fc.validate_mask(m, "x.png")
    ok = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    assert fc.validate_mask(ok).dtype == np.uint8


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fc.read_image(tmp_path / "nope.png")
    with pytest.raises(FileNotFoundError):
        fc.read_mask(tmp_path / "nope_mask.png")


def test_letterbox_pads_mask_with_ignore():
    img = np.full((100, 200, 3), 50, dtype=np.uint8)
    mask = np.ones((100, 200), dtype=np.uint8)
    li, lm = fc.letterbox_image_and_mask(img, mask, (128, 128))
    assert li.shape == (128, 128, 3) and lm.shape == (128, 128)
    assert lm[0, 0] == fc.IGNORE_INDEX and lm[64, 64] == 1
    assert tuple(li[0, 0]) == fc.PAD_COLOR


def test_video_dataset_letterbox(tmp_path):
    vid = tmp_path / "v"
    vid.mkdir()
    names = []
    for i in range(3):
        cv2.imwrite(str(vid / f"frame_{i:06d}.png"), np.random.randint(0, 255, (90, 160, 3), dtype=np.uint8))
        m = np.zeros((90, 160), dtype=np.uint8)
        m[10:20, 10:20] = 2
        cv2.imwrite(str(vid / f"frame_{i:06d}_mask.png"), m)
        names.append(f"frame_{i:06d}.png")
    import pandas as pd
    csv = tmp_path / "clips.csv"
    pd.DataFrame([{
        "video_name": "v", "frame_names": ";".join(names),
        "image_paths": ";".join(f"v/{n}" for n in names),
        "mask_paths": ";".join(f"v/{n[:-4]}_mask.png" for n in names),
    }]).to_csv(csv, index=False)
    ds = fc.VideoClipDataset(csv, tmp_path, target_size=(64, 64))
    frames, masks = ds[0]
    assert frames.shape == (3, 3, 64, 64) and masks.shape == (3, 64, 64)
    assert (masks == fc.IGNORE_INDEX).any() and (masks == 2).any()
    assert ds.frame_keys(0) == [("v", n) for n in names]


def test_dense_clip_context_frames_are_ignored(tmp_path):
    vid = tmp_path / "v"
    vid.mkdir()
    names = []
    for i in range(4):
        cv2.imwrite(str(vid / f"raw_{i:06d}.jpg"), np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8))
        names.append(f"raw_{i:06d}.jpg")
    m = np.zeros((60, 80), dtype=np.uint8)
    m[5:15, 5:15] = 1
    cv2.imwrite(str(vid / "raw_000003_mask.png"), m)
    import pandas as pd
    csv = tmp_path / "dense.csv"
    pd.DataFrame([{
        "video_name": "v", "frame_names": "ctx_0;ctx_1;ctx_2;frame_000003.png",
        "image_paths": ";".join(f"v/{n}" for n in names),
        "mask_paths": ";;;v/raw_000003_mask.png",
    }]).to_csv(csv, index=False)
    ds = fc.VideoClipDataset(csv, tmp_path, target_size=(64, 64))
    _, masks = ds[0]
    assert (masks[:3] == fc.IGNORE_INDEX).all()
    assert (masks[3] == 1).any() and (masks[3] == 0).any()
    assert fc.clip_labeled_positions(ds.df.iloc[0]) == [3]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from evaluate_test import assign_latest_positions
    assert assign_latest_positions(ds.df) == {0: [3]}


# 9. Checkpoint ordering / resume / archive ----------------------------------
def test_latest_saved_after_best_update():
    tracker = fc.BestTracker(best_val_miou=0.5, best_fire_iou=0.6, best_smoke_iou=0.4, best_epoch=3)
    calls = []
    saved = {}

    def save_best(kind):
        calls.append(("best", kind))

    def save_latest(payload):
        calls.append(("latest", None))
        saved.update(payload)

    metrics = {"fg_miou": 0.7, "iou_fire": 0.65, "iou_smoke": 0.3}
    improved = fc.epoch_checkpoint_step(5, metrics, tracker, save_best, save_latest, "rid", {"a": 1}, 0.01)
    assert improved == ["best", "fire"]
    assert calls[-1] == ("latest", None) and len(calls) == 3
    assert saved["best_val_miou"] == 0.7 and saved["best_fire_iou"] == 0.65
    assert saved["best_smoke_iou"] == 0.4 and saved["best_epoch"] == 5
    assert saved["run_id"] == "rid" and saved["config"] == {"a": 1}


def test_resume_config_check():
    cfg = {"causal": True, "temporal_hidden": 128, "num_heads": 4, "max_T": 16,
           "letterbox": True, "preprocess_version": "v3.1"}
    fc.check_resume_config(dict(cfg), cfg)
    with pytest.raises(ValueError):
        fc.check_resume_config({**cfg, "causal": False}, cfg)
    with pytest.raises(ValueError):
        fc.check_resume_config(None, cfg)


def test_archive_existing_checkpoints(tmp_path):
    a, b = tmp_path / "best.pth", tmp_path / "latest.pth"
    torch.save({"run_id": "20250101-000000"}, a)
    torch.save({"x": 1}, b)
    dest = fc.archive_existing_checkpoints([a, b, tmp_path / "missing.pth"], tmp_path / "archive", "ts")
    assert dest.name == "20250101-000000_ts"
    assert not a.exists() and not b.exists()
    assert (dest / "best.pth").exists() and (dest / "latest.pth").exists()
    assert fc.archive_existing_checkpoints([a], tmp_path / "archive") is None


def test_rng_roundtrip():
    state = fc.capture_rng_state()
    x = (np.random.rand(), torch.rand(1).item())
    fc.restore_rng_state(state)
    assert (np.random.rand(), torch.rand(1).item()) == x


# Metrics ---------------------------------------------------------------------
def test_confusion_matrix_ignores_pad():
    t = torch.tensor([[0, 1, 2, fc.IGNORE_INDEX]])
    p = torch.tensor([[0, 1, 1, 2]])
    cm = fc.confusion_matrix_3c(p, t)
    assert cm.sum() == 3 and cm[2, 1] == 1
    m = fc.iou_dice_from_cm(cm)
    assert m["iou_bg"] == 1.0 and m["iou_fire"] == 0.5 and m["iou_smoke"] == 0.0


def test_legacy_stage2_state_dict_loads_exactly():
    """A pre-T7 checkpoint (no pos_embed) reproduces the legacy module with causal=False."""
    _, model = make_st_model(causal=False, gamma=0.5)
    sd = {k: v for k, v in model.state_dict().items() if k != "temporal_refinement.pos_embed"}
    _, other = make_st_model(causal=False, seed=1)
    assert fc.load_stage2_state_dict(other, sd, legacy_ok=True) is True
    assert torch.count_nonzero(other.temporal_refinement.pos_embed) == 0
    with pytest.raises(RuntimeError):
        fc.load_stage2_state_dict(other, sd, legacy_ok=False)
