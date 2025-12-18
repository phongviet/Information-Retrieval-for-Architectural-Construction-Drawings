# Dataset configuration for architectural element detection
# COCO format with 4 classes: door, window, wall, object

dataset_type = 'CocoDataset'
data_root = 'data/architectural_coco/'  # Path to converted COCO dataset

# Class names (must match COCO JSON categories)
classes = ('door', 'window', 'wall', 'object')

# Image normalization (ImageNet statistics)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

# Validation pipeline
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])
]

# Test pipeline (same as validation)
test_pipeline = val_pipeline

# Dataset configuration
data = dict(
    samples_per_gpu=2,  # Batch size per GPU
    workers_per_gpu=2,  # Data loading workers per GPU
    persistent_workers=True,  # Keep workers alive between epochs

    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline
    ),

    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=val_pipeline
    ),

    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_test2017.json',
        img_prefix=data_root + 'test2017/',
        pipeline=test_pipeline
    )
)

# Evaluation metric
evaluation = dict(
    interval=1,  # Evaluate every epoch
    metric='bbox',  # Bounding box mAP
    save_best='auto'  # Automatically save best model
)
