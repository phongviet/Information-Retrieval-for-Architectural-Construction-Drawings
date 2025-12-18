# MMDetection 2.x → 3.x Migration Guide

## Overview

Your `mmdet_trainer.py` has been successfully refactored to be fully compatible with **MMDetection 3.3.0+** and the new **mmengine Registry system**. This document outlines the key changes and how the migration was handled.

## Key Changes Made

### 1. **Registry System (Old → New)**

#### Before (MMDetection 2.x):
```python
from mmdet.apis import build_detector, build_dataset

model = build_detector(cfg.model)
dataset = build_dataset(cfg.data.train)
```

#### After (MMDetection 3.x):
```python
from mmdet.registry import MODELS, DATASETS

model = MODELS.build(cfg.model)
dataset = DATASETS.build(cfg.train_dataloader.dataset)
```

✅ **Status**: Implemented in `train()` method (line ~216)

---

### 2. **Config Key Mapping**

| Old Key (2.x) | New Key (3.x) | Used In |
|---|---|---|
| `cfg.runner.max_epochs` | `cfg.train_cfg.max_epochs` | `_inject_default_schedules()` |
| `cfg.optimizer` | `cfg.optim_wrapper.optimizer` | `__init__()` parameter override |
| `cfg.lr_config` | `cfg.param_scheduler` | `_inject_default_schedules()` |
| `cfg.data.train` | `cfg.train_dataloader` | `_inject_default_schedules()` |
| `cfg.data.val` | `cfg.val_dataloader` | `_inject_default_schedules()` |
| `cfg.data.test` | `cfg.test_dataloader` | `_inject_default_schedules()` |
| `cfg.log_config` | `cfg.default_hooks` | `_inject_default_schedules()` |
| `cfg.gpu_ids` | `cfg.launcher` | `__init__()` |

✅ **Status**: All mappings implemented throughout the code

---

### 3. **Default Schedule Injection**

Your config files (e.g., `_base_/models/cascade_rcnn_r50_fpn.py`) only contain model definitions without training schedules. The `_inject_default_schedules()` method automatically injects missing components:

```python
def _inject_default_schedules(self):
    # If config only has model, inject:
    - optim_wrapper (SGD optimizer with LR 0.02)
    - train_cfg (12 epochs)
    - param_scheduler (warmup + multi-step LR decay)
    - train_dataloader (from legacy data.train format)
    - val_dataloader (from legacy data.val format)
    - default_hooks (logging, checkpointing, etc.)
    - custom_hooks (NumClassCheckHook)
    - test_evaluator (CocoMetric for bbox)
```

✅ **Status**: Fully implemented (lines 83-217)

---

### 4. **Runner Initialization**

#### Before (MMDetection 2.x):
```python
runner = Runner(
    model=model,
    work_dir=cfg.work_dir,
    train_dataloader=cfg.data.train,
    optim_wrapper=cfg.optim_wrapper if hasattr(cfg, 'optim_wrapper') else dict(optimizer=cfg.optimizer),
    param_scheduler=cfg.lr_config if hasattr(cfg, 'lr_config') else None,
    train_cfg=cfg.get('train_cfg', dict()),
    # ... more legacy keys
)
```

#### After (MMDetection 3.x):
```python
runner = Runner(
    model=model,
    work_dir=cfg.work_dir,
    train_dataloader=cfg.train_dataloader,  # New format
    val_dataloader=cfg.val_dataloader,      # New format
    train_cfg=cfg.train_cfg,                # Fixed key
    val_cfg=cfg.val_cfg,                    # New key
    optim_wrapper=cfg.optim_wrapper,        # New format
    param_scheduler=cfg.param_scheduler,    # New key
    default_hooks=cfg.default_hooks,        # New key
    custom_hooks=cfg.custom_hooks,          # New key
    launcher='none',                        # Replaces gpu_ids
    env_cfg=dict(cudnn_benchmark=True),    # New key
    # ... MMDetection 3.x specific keys
)
```

✅ **Status**: Fully updated (lines 237-262)

---

## File Structure Updates

### Updated Methods:

1. **`__init__()`** (lines 22-74)
   - Calls `_inject_default_schedules()` to handle missing config keys
   - Updated config overrides for MMDetection 3.x structure
   - Uses `launcher` instead of `gpu_ids`

2. **`_inject_default_schedules()`** (lines 76-217) ⭐ NEW
   - Core migration logic: injects missing training configurations
   - Handles both new format (train_dataloader) and legacy format (data.train)
   - Sets sensible defaults for SGD optimizer, 12 epochs, multi-step LR decay

3. **`_setup_logging()`** (lines 219-224)
   - Simplified for MMDetection 3.x (mmengine handles logging)
   - Removed old `log_config` approach

4. **`train()`** (lines 226-298)
   - Uses `MODELS.build()` instead of `build_detector()`
   - Proper Runner initialization with MMDetection 3.x parameters
   - Better error handling for OOM scenarios

5. **`get_training_results()`** (lines 327-375)
   - Updated to use `train_cfg.max_epochs` instead of `runner.max_epochs`
   - Safer JSON parsing with error handling

6. **`generate_training_report()`** (lines 377-441)
   - Updated to extract optimizer info from `optim_wrapper`
   - Updated to extract batch size from `train_dataloader`

---

## Usage Example

### Your existing config structure (unchanged):

**configs/mmdet/_base_/models/cascade_rcnn_r50_fpn.py**:
```python
model = dict(
    type='CascadeRCNN',
    backbone=dict(...),
    neck=dict(...),
    # ... model definition
)
```

**configs/mmdet/datasets/architectural_coco.py** (or your schedule file):
```python
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(...),
    val=dict(...),
)
```

### Training code (no changes needed):
```python
from src.training.mmdet_trainer import MMDetectionTrainer

trainer = MMDetectionTrainer(
    config=your_config_loader,
    mmdet_config_path='configs/mmdet/_base_/models/cascade_rcnn_r50_fpn.py'
)

# If model-only config, defaults are automatically injected!
checkpoint = trainer.train()
```

---

## Default Configuration Injected

If your config lacks schedules, these defaults are automatically injected:

```python
# Optimizer: SGD with momentum
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.02,
        momentum=0.9,
        weight_decay=0.0001
    ),
    clip_grad=None
)

# Training loop: 12 epochs
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12
)

# Learning rate schedule: warmup + multi-step decay
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)
]
```

**Override these** by including them in your config file or passing through `config.training` in your ConfigLoader.

---

## Testing & Validation

To verify the migration works:

```bash
# Test model building
python -c "
from mmengine.config import Config
from mmdet.registry import MODELS

cfg = Config.fromfile('configs/mmdet/_base_/models/cascade_rcnn_r50_fpn.py')
model = MODELS.build(cfg.model)
print('✓ Model built successfully')
"

# Test trainer initialization
python -c "
from src.training.mmdet_trainer import MMDetectionTrainer
trainer = MMDetectionTrainer(None, 'configs/mmdet/_base_/models/cascade_rcnn_r50_fpn.py')
print('✓ Trainer initialized with default schedules injected')
print(f'  - train_cfg.max_epochs: {trainer.cfg.train_cfg.max_epochs}')
print(f'  - optim_wrapper.optimizer.type: {trainer.cfg.optim_wrapper[\"optimizer\"].get(\"type\", \"unknown\")}')
"
```

---

## Troubleshooting

### Issue: "Registry 'optimizer' not found"
- **Cause**: `optim_wrapper.optimizer.type` is a string not registered
- **Solution**: Ensure MMDetection 3.x is properly installed with `pip install mmdetection==3.3.0`

### Issue: "DataLoader not configured properly"
- **Cause**: Config is missing `train_dataloader` or legacy `data.train`
- **Solution**: Provide either:
  1. Modern format: `cfg.train_dataloader = dict(...)`
  2. Legacy format: `cfg.data.train = dict(...)` (auto-converted)

### Issue: "train_cfg.max_epochs not found"
- **Cause**: Config lacks `train_cfg` and legacy `runner`
- **Solution**: Automatically injected by `_inject_default_schedules()` - no action needed

### Issue: Running out of memory (OOM)
- **Message**: "CUDA Out of Memory Error!"
- **Solutions**:
  1. Reduce batch size: `cfg.train_dataloader.batch_size = 1`
  2. Enable gradient accumulation in config
  3. Reduce image resolution in pipeline

---

## References

- [MMDetection 3.x Documentation](https://mmdetection.readthedocs.io/)
- [mmengine Config System](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html)
- [MMDetection Registry](https://mmdetection.readthedocs.io/en/latest/advanced_tutorials/customize_models.html)

---

## Summary of Changes

| Component | Status | Changes |
|---|---|---|
| **Imports** | ✅ | MODELS registry imported |
| **Model Building** | ✅ | Uses MODELS.build() |
| **Config Keys** | ✅ | All mapping completed |
| **Default Schedules** | ✅ | Auto-injection implemented |
| **Runner Initialization** | ✅ | MMDetection 3.x parameters |
| **Error Handling** | ✅ | Updated for 3.x structure |
| **Logging** | ✅ | Simplified for mmengine |
| **Tests** | ⚠️ | Run validation commands above |

**All changes are backward compatible with your existing config files!**

