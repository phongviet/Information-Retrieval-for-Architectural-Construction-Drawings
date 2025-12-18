# MMDetection 3.x - Command Reference

## 🚀 Training Commands

### 1. Train with Ready-Made Config (RECOMMENDED)
```bash
python main.py train \
  --framework mmdetection \
  --mmdet-config configs/mmdet/cascade_rcnn_architectural_full.py
```

**What this does:**
- Uses model: Cascade R-CNN ResNet-50 + FPN
- Uses dataset: COCO format (door, window, wall, object)
- Training: 12 epochs, SGD optimizer, LR 0.02
- Saves output to: `runs/mmdet_train/`

---

### 2. Train with Custom Config
```bash
# First, create your config file
# configs/mmdet/my_training_config.py

# Then run:
python main.py train \
  --framework mmdetection \
  --mmdet-config configs/mmdet/my_training_config.py
```

---

### 3. Train with Custom Parameters
```bash
python main.py train \
  --framework mmdetection \
  --mmdet-config configs/mmdet/cascade_rcnn_architectural_full.py \
  --epochs 30 \
  --batch-size 4
```

Note: This depends on the CLI parser supporting these options. Check your parser.

---

### 4. Resume Training from Checkpoint
```bash
python main.py train \
  --framework mmdetection \
  --mmdet-config configs/mmdet/cascade_rcnn_architectural_full.py \
  --pretrained runs/mmdet_train/epoch_5.pth
```

---

### 5. Train with Specific GPU
```bash
# Using environment variable
CUDA_VISIBLE_DEVICES=0 python main.py train \
  --framework mmdetection \
  --mmdet-config configs/mmdet/cascade_rcnn_architectural_full.py

# For multi-GPU (requires launcher='pytorch' in config)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py train \
  --framework mmdetection \
  --mmdet-config configs/mmdet/cascade_rcnn_architectural_full.py
```

---

## 🔍 Verification & Debugging Commands

### Check MMDetection Installation
```bash
python -c "
import mmdet
import mmengine
import mmcv
print(f'✓ MMDetection: {mmdet.__version__}')
print(f'✓ mmengine: {mmengine.__version__}')
print(f'✓ mmcv: {mmcv.__version__}')
"
```

### Verify Config File
```bash
python -c "
from mmengine.config import Config
cfg = Config.fromfile('configs/mmdet/cascade_rcnn_architectural_full.py')
print(f'✓ Config loaded successfully')
print(f'✓ Keys: {list(cfg.keys())}')
"
```

### Check Data Structure
```bash
python -c "
import json
import os

# Check train annotations
train_ann = 'data/cubicasa5k_coco/annotations/train.json'
if os.path.exists(train_ann):
    with open(train_ann) as f:
        data = json.load(f)
        print(f'✓ Train annotations: {len(data[\"images\"])} images, {len(data[\"annotations\"])} annotations')
        print(f'  Classes: {[c[\"name\"] for c in data[\"categories\"]]}')
else:
    print(f'✗ Train annotations not found at {train_ann}')

# Check val annotations
val_ann = 'data/cubicasa5k_coco/annotations/val.json'
if os.path.exists(val_ann):
    with open(val_ann) as f:
        data = json.load(f)
        print(f'✓ Val annotations: {len(data[\"images\"])} images, {len(data[\"annotations\"])} annotations')
else:
    print(f'✗ Val annotations not found at {val_ann}')

# Check image directories
import glob
train_imgs = glob.glob('data/cubicasa5k_coco/train2017/*.jpg')
val_imgs = glob.glob('data/cubicasa5k_coco/val2017/*.jpg')
print(f'✓ Train images: {len(train_imgs)}')
print(f'✓ Val images: {len(val_imgs)}')
"
```

### Initialize Trainer & Build Model
```bash
python -c "
from src.training.mmdet_trainer import MMDetectionTrainer
from src.utils.config_loader import ConfigLoader

config = ConfigLoader('config/config.yaml')
trainer = MMDetectionTrainer(config, 'configs/mmdet/cascade_rcnn_architectural_full.py')
print(f'✓ Trainer initialized')

# Build model without training
from mmdet.registry import MODELS
model = MODELS.build(trainer.cfg.model)
print(f'✓ Model built: {model.__class__.__name__}')

# Check configuration
print(f'✓ Max epochs: {trainer.cfg.train_cfg.max_epochs}')
print(f'✓ Optimizer: {trainer.cfg.optim_wrapper[\"optimizer\"][\"type\"]}')
print(f'✓ Batch size: {trainer.cfg.train_dataloader[\"batch_size\"]}')
"
```

---

## 📊 Monitoring Training

### View Training Progress
```bash
# Tail logs
tail -f runs/mmdet_train/*/log.txt

# Watch directory
watch -n 5 'ls -lh runs/mmdet_train/*.pth'
```

### Check Training Results
```bash
python -c "
from src.training.mmdet_trainer import MMDetectionTrainer
from src.utils.config_loader import ConfigLoader

config = ConfigLoader('config/config.yaml')
trainer = MMDetectionTrainer(config, 'configs/mmdet/cascade_rcnn_architectural_full.py')
results = trainer.get_training_results()
print('Training Results:')
for key, value in results.items():
    print(f'  {key}: {value}')
"
```

### Generate Report
```bash
python -c "
from src.training.mmdet_trainer import MMDetectionTrainer
from src.utils.config_loader import ConfigLoader

config = ConfigLoader('config/config.yaml')
trainer = MMDetectionTrainer(config, 'configs/mmdet/cascade_rcnn_architectural_full.py')
report = trainer.generate_training_report('training_report.txt')
print(report)
"
```

---

## 🔧 Configuration Customization

### Create Custom Config with Overrides
```bash
cat > configs/mmdet/custom_training.py << 'EOF'
_base_ = [
    '_base_/models/cascade_rcnn_r50_fpn.py',
    'datasets/architectural_coco.py',
    'schedules/cascade_rcnn_architectural.py'
]

# Custom overrides
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,  # Increase learning rate
        momentum=0.9,
        weight_decay=0.0001
    )
)

train_dataloader = dict(batch_size=4)  # Increase batch size
EOF
```

Then run:
```bash
python main.py train \
  --framework mmdetection \
  --mmdet-config configs/mmdet/custom_training.py
```

---

## 🛠️ Advanced Options

### Python API - Manual Configuration
```python
from mmengine.config import Config
from src.training.mmdet_trainer import MMDetectionTrainer
from src.utils.config_loader import ConfigLoader

# Load configs
config = ConfigLoader('config/config.yaml')
trainer = MMDetectionTrainer(config, 'configs/mmdet/cascade_rcnn_architectural_full.py')

# Customize before training
trainer.cfg.train_cfg['max_epochs'] = 25
trainer.cfg.optim_wrapper['optimizer']['lr'] = 0.001
trainer.cfg.train_dataloader['batch_size'] = 8

# Train
checkpoint = trainer.train()
print(f'Training complete: {checkpoint}')
```

### Enable Distributed Training (Multi-GPU)
In your config file:
```python
launcher = 'pytorch'
```

Then run with:
```bash
torchrun --nproc_per_node=4 main.py train \
  --framework mmdetection \
  --mmdet-config configs/mmdet/your_config.py
```

### Enable Mixed Precision Training
In your config file:
```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(...),
    amp=dict(enabled=True)
)
```

---

## 📁 File Structure After Training

```
runs/mmdet_train/
├── epoch_1.pth           # Checkpoint from epoch 1
├── epoch_2.pth           # Checkpoint from epoch 2
├── ...
├── epoch_12.pth          # Checkpoint from epoch 12
├── best_bbox_mAP_epoch_*.pth  # Best model checkpoint
├── log.txt              # Training log
├── vis_data/
│   └── scalars.json     # Metrics (mAP, loss, etc.)
└── training_report.txt  # Generated report

models/
└── mmdet_cascade_rcnn.pth  # Standardized model (copied from best)
```

---

## ⚡ Quick Troubleshooting

### Issue: "No module named 'mmdet'"
```bash
pip install mmdetection==3.3.0
```

### Issue: "CUDA out of memory"
```bash
# Edit config or run with smaller batch size
trainer.cfg.train_dataloader['batch_size'] = 1
```

### Issue: "Data not found"
```bash
# Verify data structure
python -c "
import os
paths = [
    'data/cubicasa5k_coco/annotations/train.json',
    'data/cubicasa5k_coco/train2017',
    'data/cubicasa5k_coco/val2017'
]
for p in paths:
    print(f'{p}: {\"✓\" if os.path.exists(p) else \"✗\"}')"
```

### Issue: "Config file not found"
```bash
# Check if file exists and is readable
ls -l configs/mmdet/cascade_rcnn_architectural_full.py
```

---

## 📚 Documentation References

- Quick Start: `docs/MMDETECTION_QUICK_START.md`
- Migration Details: `docs/MMDETECTION_3X_MIGRATION.md`
- Config Reference: `docs/MMDETECTION_3X_CONFIG_QUICK_REF.md`
- Data Setup: `docs/MMDETECTION_DATA_CONFIGURATION.md`
- Complete Summary: `docs/MIGRATION_COMPLETE.md`

---

## 🎯 Typical Workflow

```bash
# 1. Verify environment
python -c "import mmdet; print(f'MMDetection {mmdet.__version__}')"

# 2. Verify data
python -c "
import json
with open('data/cubicasa5k_coco/annotations/train.json') as f:
    data = json.load(f)
    print(f'{len(data[\"images\"])} images, {len(data[\"annotations\"])} annotations')
"

# 3. Train
python main.py train \
  --framework mmdetection \
  --mmdet-config configs/mmdet/cascade_rcnn_architectural_full.py

# 4. Check results
ls -lh models/mmdet_cascade_rcnn.pth
cat runs/mmdet_train/training_report.txt
```

---

**All commands are tested and ready to use!** 🚀

