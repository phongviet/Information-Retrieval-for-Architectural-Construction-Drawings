# Composite MMDetection configuration for Architectural Drawing Detection
# Combines model architecture, dataset configuration, and training schedules
#
# This config can be used directly for training:
# python main.py train --framework mmdetection --mmdet-config configs/mmdet/cascade_rcnn_architectural_full.py

_base_ = [
    '_base_/models/cascade_rcnn_r50_fpn.py',
    'datasets/architectural_coco.py',
    'schedules/cascade_rcnn_architectural.py'
]

# Optional: Custom overrides
# Uncomment to customize:

# model = dict(
#     backbone=dict(
#         depth=50,
#         frozen_stages=1,
#         init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
#     )
# )

# train_cfg = dict(
#     type='EpochBasedTrainLoop',
#     max_epochs=20  # Increase from default 12 to 20
# )

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='SGD',
#         lr=0.01,  # Adjust learning rate
#         momentum=0.9,
#         weight_decay=0.0001
#     )
# )

# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.001,
#         by_epoch=False,
#         begin=0,
#         end=500
#     ),
#     dict(
#         type='MultiStepLR',
#         by_epoch=True,
#         milestones=[15, 18],  # Decay at epochs 15 and 18
#         gamma=0.1
#     )
# ]

