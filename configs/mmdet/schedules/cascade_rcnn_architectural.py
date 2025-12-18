# Training schedule for Cascade R-CNN on architectural drawings
# Optimized for fine-tuning pretrained COCO model on architectural domain

# Optimizer configuration
optimizer = dict(
    type='SGD',
    lr=0.02,  # Base learning rate for batch size 2
    momentum=0.9,
    weight_decay=0.0001
)

optimizer_config = dict(grad_clip=None)

# Learning rate schedule
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11],  # Decay at epochs 8 and 11
    gamma=0.1  # LR decay factor
)

# Total epochs for fine-tuning
runner = dict(type='EpochBasedRunner', max_epochs=12)

# Checkpoint configuration
checkpoint_config = dict(interval=1)  # Save every epoch

# Logging configuration
log_config = dict(
    interval=50,  # Log every 50 iterations
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

# Custom hooks
custom_hooks = [dict(type='NumClassCheckHook')]

# Distribution configuration
dist_params = dict(backend='nccl')

# Logging level
log_level = 'INFO'

# Load pretrained weights
load_from = None  # Path to pretrained checkpoint (optional)
resume_from = None  # Path to checkpoint for resuming (optional)

# Workflow
workflow = [('train', 1)]  # Train for 1 epoch per iteration

# Disable opencv multithreading
opencv_num_threads = 0

# Set multi-processing start method
mp_start_method = 'fork'

# Auto-scaling learning rate
auto_scale_lr = dict(enable=False, base_batch_size=16)
