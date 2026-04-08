# Orange Model Improvements — Design Spec
**Date:** 2026-04-08  
**Goal:** Make the model reliably learn identity mapping on orange images as a foundation for understanding CNN training pipelines.

---

## Context

The project is a fully convolutional network (SimpleNet) that maps colored orange images to identical targets (identity mapping). The model initially produced blurry-but-recognizable output, then degraded to pure white after Sigmoid was disabled. Root cause: unconstrained outputs grew beyond [0,1] and were saturated to 1.0 by `np.clip`, rendering all-white images.

---

## Section 1: Architecture (model.py)

**Problem:** 16-channel bottleneck limits representational capacity. No BatchNorm causes unstable training. Disabled Sigmoid causes output saturation.

**Changes:**
- Increase conv channels from 16 to 64 throughout
- Add `BatchNorm2d` after each conv layer, before activation
- Re-enable `Sigmoid` on the final layer output

**New layer stack:**
```
Conv2d(3 → 64, k=3, pad=1) → BatchNorm2d(64) → ReLU
Conv2d(64 → 64, k=3, pad=1) → BatchNorm2d(64) → ReLU
Conv2d(64 → 3, k=3, pad=1) → BatchNorm2d(3) → Sigmoid
```

**Why Sigmoid:** Targets are normalized to [0,1]. Sigmoid constrains outputs to the same range, eliminating saturation at the source. `np.clip` in infer.py becomes a safety net rather than the primary constraint.

---

## Section 2: Training (train.py + config.yaml)

**Problem:** LR=0.001 is too aggressive — model diverges after initial learning. No scheduler means LR stays constant even as loss plateaus.

**Changes:**
- Lower base LR: `0.001 → 0.0001`
- Add `CosineAnnealingLR` scheduler: decays LR smoothly from 0.0001 to 0.000001 over training
- Increase epochs: `500 → 1000` (lower LR needs more steps to converge)
- Add config entries: `scheduler: cosine`, `min_lr: 0.000001`

**Optimizer and loss unchanged:** Adam, MSELoss, batch size 4, early stopping.

---

## Section 3: Data Augmentation (train.py)

**Problem:** No augmentation — 256 image pairs is small, model can memorize positions.

**Changes:**
- `RandomHorizontalFlip(p=0.5)` applied to both input and target in sync
- `RandomVerticalFlip(p=0.5)` applied to both input and target in sync
- Use `torchvision.transforms.functional` for deterministic per-sample transforms (same random state applied to both tensors)

**What's excluded:** No color jitter, no rotation — these would create input/target mismatches for the identity task.

---

## Files to Change

| File | Changes |
|------|---------|
| `model.py` | Channels 16→64, add BatchNorm2d, re-enable Sigmoid |
| `train.py` | Lower LR, add CosineAnnealingLR, add augmentation in `__getitem__` |
| `config.yaml` | Add `min_lr`, `scheduler`, update `epochs` and `learning_rate` |

---

## Success Criteria

- Model produces a recognizable orange image in `result.png` after training
- Training loss decreases monotonically without divergence
- No pure white or pure black output
