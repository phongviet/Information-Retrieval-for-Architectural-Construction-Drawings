# Model Organization Guide

## Overview

This document explains how YOLO models are organized in this project and why pretrained models are stored in the `models/` directory instead of the root folder.

## Directory Structure

```
project_root/
├── models/                    # All YOLO model files
│   ├── yolov8n.pt            # Pretrained YOLOv8 nano model
│   ├── yolo11s.pt            # Pretrained YOLOv11 small model
│   ├── custom_model.pt        # Your trained custom model
│   └── train*/                # Training run outputs
│       ├── weights/
│       │   ├── best.pt       # Best model from training
│       │   └── last.pt       # Latest checkpoint
│       └── architecture_info.yaml
└── runs/                      # Training run history
    └── train/
        └── train*/            # Individual training runs
```

## Why Models Are Organized This Way

### The Problem

When you start fine-tuning a YOLO model, the Ultralytics library automatically downloads pretrained weights if they don't exist locally. **Previously**, these models were downloaded to the **root folder** of the project, causing:

1. **Clutter**: Model files scattered in root alongside code and config files
2. **Version Control Issues**: Risk of accidentally committing large model files to git
3. **Organization**: Difficult to distinguish between pretrained and custom models
4. **Maintenance**: Harder to clean up or manage model files

### The Solution

The `ModelArchitectureSelector` class now:

1. **Checks `models/` directory first**: Before downloading, it looks for existing models in `models/`
2. **Downloads to `models/`**: When downloading pretrained models, they are automatically moved to `models/`
3. **Maintains consistency**: All models (pretrained and trained) live in one place

### Code Implementation

In `src/training/model_selector.py`:

```python
def load_pretrained_model(self, weights_path: str | None = None) -> YOLO:
    # Check if model exists in models/ directory first
    models_dir = os.path.join('models', identifier)
    
    if os.path.exists(models_dir):
        # Load from models/ directory
        model = YOLO(models_dir)
    else:
        # Download to models/ directory
        os.makedirs('models', exist_ok=True)
        model = YOLO(identifier)
        # Move downloaded model to models/ directory
        if os.path.exists(identifier):
            os.rename(identifier, models_dir)
```

## Model File Naming Convention

### Pretrained Models
- Format: `{architecture}{variant}.pt`
- Examples:
  - `yolov8n.pt` - YOLOv8 nano
  - `yolov8s.pt` - YOLOv8 small
  - `yolo11m.pt` - YOLOv11 medium
  - `yolo11x.pt` - YOLOv11 xlarge

### Custom Trained Models
- `custom_model.pt` - Latest best model from training
- `models/train*/weights/best.pt` - Best weights from specific training run
- `models/train*/weights/last.pt` - Latest checkpoint for resume

## Git Ignore Rules

The `.gitignore` file excludes:

```gitignore
# Exclude all model files in models/ directory
models/*.pt

# Exclude pretrained YOLO models in root (legacy/accidental downloads)
/yolo*.pt
```

This ensures:
- Large model files don't bloat the repository
- Only code and configurations are version controlled
- Team members download their own pretrained models as needed

## Usage Examples

### Training with Automatic Model Download

```bash
# First time - downloads yolov8n.pt to models/
python main.py train --architecture yolov8 --variant nano --data data/dataset/data.yaml

# Second time - reuses existing models/yolov8n.pt
python main.py train --architecture yolov8 --variant nano --data data/dataset/data.yaml
```

### Using Custom Trained Model for Inference

```bash
# Uses the best trained model automatically
python main.py detect --model models/custom_model.pt --input drawing.pdf --output results/
```

### Resuming Training from Checkpoint

```bash
python main.py train --resume runs/train/train42/weights/last.pt --data data/dataset/data.yaml
```

## Troubleshooting

### If You Have Models in Root Directory

If you accidentally have model files in the root directory (e.g., `yolov8n.pt`, `yolo11s.pt`):

```bash
# Move them to models/ directory
move yolov8n.pt models\
move yolo11s.pt models\
move yolo11m.pt models\
```

Or on Linux/Mac:
```bash
mv yolo*.pt models/
```

### Cleaning Up Old Models

To free up space, you can safely delete:
- Old training runs in `runs/train/train*/`
- Pretrained models you're not using in `models/`
- Keep `models/custom_model.pt` if it's your best trained model

## Benefits of This Organization

1. **Clean Project Structure**: Root folder stays clean with only code and configs
2. **Easy Backup**: Back up entire `models/` directory to save all your work
3. **Version Control**: Large model files never accidentally committed
4. **Team Collaboration**: Each developer manages their own models locally
5. **Disk Space Management**: Easy to identify and clean up old models
6. **CI/CD Friendly**: Automated builds don't download unnecessary model files

## Best Practices

1. **Always train from command line**: The system handles model organization automatically
2. **Use `--resume` for checkpoints**: Resume interrupted training without redownloading
3. **Back up `custom_model.pt`**: This is your trained model - keep it safe
4. **Document model performance**: Note which architecture/variant works best for your use case
5. **Clean up periodically**: Delete old training runs in `runs/train/` to save space

## Related Documentation

- [ARCHITECTURE_MIGRATION.md](ARCHITECTURE_MIGRATION.md) - Switching between YOLOv8 and YOLOv11
- [TUNING_GUIDE.md](TUNING_GUIDE.md) - Hyperparameter tuning and training tips
- [README.md](../README.md) - Main project documentation

