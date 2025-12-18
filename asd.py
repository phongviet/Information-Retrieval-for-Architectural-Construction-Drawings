import torch
from src.training.mmdet_trainer import MMDetectionTrainer

# 1. Define the correct configuration path
# (Make sure the file 'config/cascade_rcnn_cubicasa.py' contains the code you just pasted)
config_path = r"configs/cascade_rcnn_cubicasa.py"

# 2. Define a simple config wrapper
class TrainingConfig:
    data_root = r'data/cubicasa5k_coco/'
    # We don't need other params for pure testing

# 3. Initialize the Trainer
print("Initializing Trainer with corrected config...")
trainer = MMDetectionTrainer(TrainingConfig(), config_path)

# 4. Load your trained weights (Epoch 11)
# CHECK THIS PATH: Make sure it points to your actual .pth file
checkpoint_path = 'runs/mmdet_train/epoch_11.pth'
print(f"Loading weights from: {checkpoint_path}")
trainer.cfg.load_from = checkpoint_path

# 5. Run Validation Only
print("Starting Evaluation...")
trainer.train()