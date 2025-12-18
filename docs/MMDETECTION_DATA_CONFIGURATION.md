# MMDetection 3.x Training Configuration Guide

## Problem: "No train_dataloader or data.train config found"

Your config file is missing the data (dataset) configuration. You're using a **model-only config** (like `_base_/models/cascade_rcnn_r50_fpn.py`) which defines the neural network architecture but not the training data.

---

## Solution 1: Use an existing composite config (RECOMMENDED)

The best approach is to use a config file that combines both model AND dataset definitions:

```bash
python main.py train \
  --framework mmdetection \
  --mmdet-config configs/mmdet/_base_/models/cascade_rcnn_r50_fpn.py \
  --data data/cubicasa5k_coco/data.yaml
```

Or create a composite config file:

**configs/mmdet/cascade_rcnn_architectural_full.py**:
```python
# Inherit from both model and dataset configs
_base_ = [
    '_base_/models/cascade_rcnn_r50_fpn.py',
    'datasets/architectural_coco.py',
    'schedules/cascade_rcnn_architectural.py'
]

# Optional: override any values
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    )
)
```

Then run:
```bash
python main.py train \
  --framework mmdetection \
  --mmdet-config configs/mmdet/cascade_rcnn_architectural_full.py
```

---

## Solution 2: Create your own composite config

Create a new file that merges model + data + schedules:

**configs/mmdet/your_training_config.py**:
```python
# Import base configs
from mmengine.config import read_base

with read_base():
    from ._base_.models.cascade_rcnn_r50_fpn import *
    from .datasets.architectural_coco import *
    from .schedules.cascade_rcnn_architectural import *

# Optional: customize
model['num_classes'] = 4
train_cfg['max_epochs'] = 20
optim_wrapper['optimizer']['lr'] = 0.01
```

---

## Solution 3: Python API - Manual Configuration

If you're using the Python API programmatically:

```python
from mmengine.config import Config
from src.training.mmdet_trainer import MMDetectionTrainer
from src.utils.config_loader import ConfigLoader

# Load your phase 1 config
config = ConfigLoader('config/config.yaml')

# Load MMDetection base config
trainer = MMDetectionTrainer(config, 'configs/mmdet/_base_/models/cascade_rcnn_r50_fpn.py')

# Manually configure data BEFORE calling train()
trainer.cfg.train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        ann_file='data/cubicasa5k_coco/annotations/train.json',
        img_prefix='data/cubicasa5k_coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Normalize', 
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]
    )
)

trainer.cfg.val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        ann_file='data/cubicasa5k_coco/annotations/val.json',
        img_prefix='data/cubicasa5k_coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='Normalize', 
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ]
    )
)

# Now safe to train
checkpoint = trainer.train()
```

---

## Check Your Data Structure

Make sure your COCO dataset is properly organized:

```
data/cubicasa5k_coco/
├── annotations/
│   ├── train.json         ← Required
│   ├── val.json          ← Required
│   └── test.json         ← Optional
├── train2017/            ← Required (training images)
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── val2017/              ← Required (validation images)
│   └── ...
└── test2017/             ← Optional
    └── ...
```

Verify with:
```bash
python -c "
import json
with open('data/cubicasa5k_coco/annotations/train.json') as f:
    data = json.load(f)
    print(f'Train images: {len(data[\"images\"])}')
    print(f'Train annotations: {len(data[\"annotations\"])}')
    print(f'Categories: {[c[\"name\"] for c in data[\"categories\"]]}')
"
```

---

## Default Config Structure (When Data is Missing)

When you use a model-only config without data, the trainer automatically injects:

```python
# Auto-injected defaults
train_cfg = {
    'type': 'EpochBasedTrainLoop',
    'max_epochs': 12
}

optim_wrapper = {
    'type': 'OptimWrapper',
    'optimizer': {
        'type': 'SGD',
        'lr': 0.02,
        'momentum': 0.9,
        'weight_decay': 0.0001
    }
}

# BUT this placeholder dataset is created if data is missing:
train_dataloader = {
    'batch_size': 2,
    'num_workers': 2,
    'dataset': {
        'type': 'CocoDataset',
        'ann_file': 'data/cubicasa5k_coco/annotations/train.json',  # ← Placeholder
        'img_prefix': 'data/cubicasa5k_coco/train2017/',             # ← Placeholder
        'pipeline': []                                                 # ← Placeholder
    }
}
```

The trainer will **check at train() time** if these placeholder values are valid and fail with a helpful error if:
- The annotation file doesn't exist
- The image directory doesn't exist
- The pipeline is empty (which would cause runtime errors)

---

## Quick Reference: Which Config to Use

| Scenario | Config File | Command |
|----------|-----------|---------|
| Just testing model build | `_base_/models/cascade_rcnn_r50_fpn.py` | Will fail at `train()` - data needed |
| Full training setup | `cascade_rcnn_architectural_full.py` (if exists) | Works out-of-the-box |
| Using Python API | Any config + manual `cfg.train_dataloader` | Works after manual setup |
| Custom training | Your composite config | Works if properly configured |

---

## Config Inheritance Examples

### Example 1: Simple inheritance
```python
# configs/mmdet/my_config.py
_base_ = [
    '_base_/models/cascade_rcnn_r50_fpn.py',
    'datasets/architectural_coco.py',
    'schedules/cascade_rcnn_architectural.py'
]
```

### Example 2: With overrides
```python
# configs/mmdet/my_config_v2.py
_base_ = [
    '_base_/models/cascade_rcnn_r50_fpn.py',
    'datasets/architectural_coco.py',
    'schedules/cascade_rcnn_architectural.py'
]

# Customize
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)
```

### Example 3: Data from different source
```python
# configs/mmdet/my_custom_data_config.py
_base_ = [
    '_base_/models/cascade_rcnn_r50_fpn.py',
    'schedules/cascade_rcnn_architectural.py'
]

# Custom data from YOLO format or other source
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='YOLODataset',  # Different dataset type
        data_root='data/yolo_format/',
        ann_file='train.txt',
        pipeline=[...]
    )
)
```

---

## Troubleshooting

### Error: "annotation file not found at data/cubicasa5k_coco/annotations/train.json"
- **Cause**: You used a model-only config, trainer created a placeholder config pointing to default COCO dataset
- **Fix**: Provide proper data configuration via one of the solutions above

### Error: "pipeline is empty or missing"
- **Cause**: Dataset config missing the image processing pipeline
- **Fix**: Include `datasets/architectural_coco.py` or define pipeline manually

### Error: "CocoDataset type not registered"
- **Cause**: MMDetection not properly installed
- **Fix**: `pip install mmdetection==3.3.0`

### Error: "MODELS registry not found"
- **Cause**: Using old MMDetection 2.x API with 3.x code
- **Fix**: Update MMDetection: `pip install -U mmdetection>=3.3.0`

---

## Summary

✅ **Best Practice**: Create a composite config that inherits from:
1. `_base_/models/cascade_rcnn_r50_fpn.py` (model architecture)
2. `datasets/architectural_coco.py` (data configuration)
3. `schedules/cascade_rcnn_architectural.py` (training parameters)

Then run:
```bash
python main.py train --framework mmdetection --mmdet-config configs/mmdet/cascade_rcnn_architectural_full.py
```

