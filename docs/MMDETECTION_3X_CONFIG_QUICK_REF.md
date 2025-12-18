# MMDetection 3.x Config Migration - Quick Reference

## Before & After Config Structure

### Old Format (MMDetection 2.x)
```python
# cfg.runner - how to run training
runner = dict(type='EpochBasedRunner', max_epochs=12)

# cfg.optimizer - optimizer configuration
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# cfg.lr_config - learning rate scheduling
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11],
    gamma=0.1
)

# cfg.data - training/val/test data
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(type='CocoDataset', ...),
    val=dict(type='CocoDataset', ...),
    test=dict(type='CocoDataset', ...)
)

# cfg.log_config - logging hooks
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
```

### New Format (MMDetection 3.x)
```python
# train_cfg - how to run training loop
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12)

# optim_wrapper - unified optimizer configuration
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

# param_scheduler - learning rate and other param scheduling
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1
    )
]

# train_dataloader - training data loader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='CocoDataset', ...)
)

# val_dataloader - validation data loader
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', ...)
)

# test_dataloader - test data loader
test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', ...)
)

# val_cfg - how to run validation loop
val_cfg = dict(type='ValLoop')

# default_hooks - replacement for log_config
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='SamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# custom_hooks - additional hooks
custom_hooks = [
    dict(type='NumClassCheckHook')
]

# test_evaluator - evaluation metric
test_evaluator = dict(type='CocoMetric', metric='bbox')
```

## Automatic Migration in Code

The `MMDetectionTrainer._inject_default_schedules()` method **automatically handles this conversion**:

1. **If your config has only the model** (like `_base_/models/cascade_rcnn_r50_fpn.py`):
   - ✅ Injects `train_cfg` with 12 epochs
   - ✅ Injects `optim_wrapper` with SGD optimizer
   - ✅ Injects `param_scheduler` with warmup + multi-step decay
   - ✅ Converts legacy `data.train` → `train_dataloader`
   - ✅ Converts legacy `data.val` → `val_dataloader`
   - ✅ Injects default hooks and evaluator

2. **If your config has schedules** (like `schedules/cascade_rcnn_architectural.py`):
   - ✅ Uses your custom configuration
   - ✅ Converts legacy format keys to new format if needed

## Code Examples

### Building Model (Changed)
```python
# Before (2.x)
from mmdet.apis import build_detector
model = build_detector(cfg.model)

# After (3.x)
from mmdet.registry import MODELS
model = MODELS.build(cfg.model)
```

### Accessing Training Parameters (Changed)
```python
# Before (2.x)
max_epochs = cfg.runner.max_epochs
learning_rate = cfg.optimizer.lr
batch_size = cfg.data.samples_per_gpu

# After (3.x)
max_epochs = cfg.train_cfg.max_epochs
learning_rate = cfg.optim_wrapper['optimizer']['lr']
batch_size = cfg.train_dataloader['batch_size']
```

### Initializing Runner (Changed)
```python
# Before (2.x)
runner = Runner(
    model=model,
    work_dir=cfg.work_dir,
    train_dataloader=cfg.data.train,
    optim_wrapper=dict(optimizer=cfg.optimizer),
    param_scheduler=cfg.lr_config,
    train_cfg=cfg.get('train_cfg', dict()),
)

# After (3.x)
runner = Runner(
    model=model,
    work_dir=cfg.work_dir,
    train_dataloader=cfg.train_dataloader,
    val_dataloader=cfg.val_dataloader,
    train_cfg=cfg.train_cfg,
    val_cfg=cfg.val_cfg,
    optim_wrapper=cfg.optim_wrapper,
    param_scheduler=cfg.param_scheduler,
    default_hooks=cfg.default_hooks,
    custom_hooks=cfg.custom_hooks,
    launcher='none',
    env_cfg=dict(cudnn_benchmark=True),
    test_evaluator=cfg.test_evaluator,
)
```

## Default Values Injected by Trainer

```python
# Optimizer (if missing)
optim_wrapper = {
    'type': 'OptimWrapper',
    'optimizer': {
        'type': 'SGD',
        'lr': 0.02,          # Can override via config.training.learning_rate
        'momentum': 0.9,
        'weight_decay': 0.0001
    },
    'clip_grad': None
}

# Training (if missing)
train_cfg = {
    'type': 'EpochBasedTrainLoop',
    'max_epochs': 12        # Can override via config.training.epochs
}

# Learning rate schedule (if missing)
param_scheduler = [
    {
        'type': 'LinearLR',
        'start_factor': 0.001,
        'by_epoch': False,
        'begin': 0,
        'end': 500
    },
    {
        'type': 'MultiStepLR',
        'by_epoch': True,
        'milestones': [8, 11],
        'gamma': 0.1
    }
]

# Hooks (if missing)
default_hooks = {
    'timer': {'type': 'IterTimerHook'},
    'logger': {'type': 'LoggerHook', 'interval': 50},
    'param_scheduler': {'type': 'ParamSchedulerHook'},
    'checkpoint': {'type': 'CheckpointHook', 'interval': 1, 'save_best': 'auto'},
    'sampler_seed': {'type': 'SamplerSeedHook'},
    'visualization': {'type': 'DetVisualizationHook'}
}
```

## Environment Variables Handled

| Variable | Old (2.x) | New (3.x) | Purpose |
|---|---|---|---|
| `cfg.gpu_ids` | `[0]` | `launcher='none'` | Single GPU training |
| `cfg.launcher` | N/A | `'none'` | No distributed training |
| `cfg.env_cfg` | N/A | `dict(cudnn_benchmark=True)` | GPU optimization |
| `cfg.log_level` | `'INFO'` | `'INFO'` | Logging verbosity |

## Testing Your Config

```bash
# Verify config can be loaded
python -c "
from mmengine.config import Config
cfg = Config.fromfile('your_config.py')
print('Config loaded successfully')
print(f'Has train_cfg: {hasattr(cfg, \"train_cfg\")}')
print(f'Has optim_wrapper: {hasattr(cfg, \"optim_wrapper\")}')
"

# Verify trainer can initialize
python -c "
from src.training.mmdet_trainer import MMDetectionTrainer
trainer = MMDetectionTrainer(None, 'your_config.py')
print('Trainer initialized with injected defaults')
print(f'Max epochs: {trainer.cfg.train_cfg.max_epochs}')
"
```

## Key Differences Summary

| Aspect | 2.x | 3.x |
|---|---|---|
| **Model Building** | `build_detector()` | `MODELS.build()` |
| **Config Keys** | `runner`, `optimizer`, `lr_config` | `train_cfg`, `optim_wrapper`, `param_scheduler` |
| **Data Format** | `cfg.data.train` | `cfg.train_dataloader` |
| **Logging** | `log_config` with hooks | `default_hooks` dict |
| **GPU Selection** | `gpu_ids` | `launcher` |
| **Runner Params** | Limited | Extensive (30+ params) |
| **Registry** | `mmdet.apis.build_*` | `mmdet.registry.REGISTRY_NAME.build()` |

All these changes are **automatically handled** by the refactored `MMDetectionTrainer` class!

