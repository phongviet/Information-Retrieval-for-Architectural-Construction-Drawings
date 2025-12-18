# MMDetection 3.x Quick Start Guide

## ✅ Your Code is Ready!

The `mmdet_trainer.py` has been successfully migrated to MMDetection 3.3.0+. However, you need to provide proper data configuration.

---

## 🚀 Quick Start (2 Options)

### Option A: Use the Ready-Made Composite Config ⭐ EASIEST

```bash
python main.py train \
  --framework mmdetection \
  --mmdet-config configs/mmdet/cascade_rcnn_architectural_full.py
```

This combines:
- ✅ Model: Cascade R-CNN ResNet-50 + FPN
- ✅ Data: COCO format dataset (4 classes: door, window, wall, object)
- ✅ Schedule: SGD optimizer, 12 epochs, multi-step LR decay

---

### Option B: Create Your Own Composite Config

Create `configs/mmdet/my_training_config.py`:

```python
_base_ = [
    '_base_/models/cascade_rcnn_r50_fpn.py',
    'datasets/architectural_coco.py',
    'schedules/cascade_rcnn_architectural.py'
]

# Optional customizations:
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20)  # More epochs
optim_wrapper['optimizer']['lr'] = 0.01  # Higher learning rate
```

Then run:
```bash
python main.py train \
  --framework mmdetection \
  --mmdet-config configs/mmdet/my_training_config.py
```

---

## 📋 What Was Fixed

Your code now uses **MMDetection 3.x Registry system**:

| Old (2.x) | New (3.x) | Status |
|-----------|-----------|--------|
| `build_detector()` | `MODELS.build()` | ✅ Implemented |
| `cfg.runner.max_epochs` | `cfg.train_cfg.max_epochs` | ✅ Implemented |
| `cfg.optimizer` | `cfg.optim_wrapper.optimizer` | ✅ Implemented |
| `cfg.data.train` | `cfg.train_dataloader` | ✅ Implemented |
| `cfg.lr_config` | `cfg.param_scheduler` | ✅ Implemented |
| Missing schedules | Auto-injected defaults | ✅ Implemented |

---

## 🎯 Your Data Requirements

Your data must be in **COCO format**:

```
data/cubicasa5k_coco/
├── annotations/
│   ├── train.json         ← Training annotations
│   ├── val.json          ← Validation annotations
│   └── test.json         ← (optional) Test annotations
├── train2017/            ← Training images
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
├── val2017/              ← Validation images
│   └── ...
└── test2017/             ← (optional) Test images
    └── ...
```

**Verify your data:**
```bash
python -c "
import json
with open('data/cubicasa5k_coco/annotations/train.json') as f:
    data = json.load(f)
    print(f'✓ Train images: {len(data[\"images\"])}')
    print(f'✓ Train annotations: {len(data[\"annotations\"])}')
    print(f'✓ Classes: {[c[\"name\"] for c in data[\"categories\"]]}')
"
```

---

## 🔧 Customization Examples

### Example 1: Adjust Training Duration
```bash
# In your config or via CLI (if supported)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30)
```

### Example 2: Adjust Learning Rate
```bash
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.05,  # Increase from 0.02
        momentum=0.9,
        weight_decay=0.0001
    )
)
```

### Example 3: Adjust Batch Size
```bash
train_dataloader = dict(batch_size=4)  # Increase from 2
```

### Example 4: Use Different Optimizer
```bash
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam',  # Switch to Adam
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.0001
    )
)
```

---

## 📖 Documentation Files Created

1. **MMDETECTION_3X_MIGRATION.md** - Complete migration reference
2. **MMDETECTION_3X_CONFIG_QUICK_REF.md** - Config key mappings (before/after)
3. **MMDETECTION_DATA_CONFIGURATION.md** - Data setup and troubleshooting
4. **cascade_rcnn_architectural_full.py** - Ready-to-use composite config

---

## ⚡ Common Commands

```bash
# Train with composite config
python main.py train --framework mmdetection --mmdet-config configs/mmdet/cascade_rcnn_architectural_full.py

# Train with custom config
python main.py train --framework mmdetection --mmdet-config configs/mmdet/my_training_config.py

# Train and resume from checkpoint
python main.py train --framework mmdetection --mmdet-config configs/mmdet/cascade_rcnn_architectural_full.py --pretrained models/checkpoint.pth

# List available configs
ls -la configs/mmdet/
```

---

## 🐛 Troubleshooting

### "No train_dataloader or data.train config found"
**Solution**: Use `cascade_rcnn_architectural_full.py` instead of just the model config

### "Annotation file not found at data/cubicasa5k_coco/..."
**Solution**: Make sure your COCO dataset is properly prepared at the expected location

### "CocoDataset type not registered"
**Solution**: Ensure MMDetection 3.3.0+ is installed: `pip install mmdetection==3.3.0`

### "MODELS registry not found"
**Solution**: You're using old MMDetection. Update: `pip install -U mmdetection>=3.3.0`

---

## 📝 What the Trainer Does

When you call `trainer.train()`:

1. **Loads config** from your Python file
2. **Injects defaults** (optimizer, schedules, hooks) if missing
3. **Validates data** configuration is complete
4. **Builds model** using `MODELS.build()`
5. **Initializes Runner** with all MMDetection 3.x parameters
6. **Runs training loop** with validation
7. **Saves checkpoints** to `runs/mmdet_train/`
8. **Returns best checkpoint** path

---

## ✅ Validation Checklist

Before running training:

- [ ] MMDetection 3.3.0+ installed: `pip show mmdetection`
- [ ] Data in COCO format at correct paths
- [ ] Config file exists: `configs/mmdet/cascade_rcnn_architectural_full.py`
- [ ] Annotation JSON files exist: `data/cubicasa5k_coco/annotations/*.json`
- [ ] Image directories exist: `data/cubicasa5k_coco/train2017/`, `val2017/`, etc.

If all checks pass, you're ready to train! 🚀

---

## 🎓 Next Steps

1. Verify your COCO dataset is prepared
2. Run: `python main.py train --framework mmdetection --mmdet-config configs/mmdet/cascade_rcnn_architectural_full.py`
3. Monitor training in `runs/mmdet_train/` directory
4. Results saved to `models/mmdet_cascade_rcnn.pth`

Need help? Check the documentation files for detailed info! 📚

