"""Manual test script for training workflow.

This is NOT a pytest test - it's a manual script to be run directly:
    python tests/test_training_manual.py

Do not run this with pytest as it will execute during collection.
"""
import tempfile
import yaml
from pathlib import Path
from src.utils.config_loader import ConfigLoader
from src.training.trainer import Trainer

def main():
    """Run the manual training test."""
    # Create temporary config (model_path not required for training)
    config_data = {
        'system': {'gpu_id': 0, 'seed': 42, 'log_level': 'INFO'},
        'model': {'confidence_threshold': 0.5},  # model_path not needed for training
        'training': {
            'model_architecture': 'yolov8',
            'model_variant': 'nano',
            'epochs': 1,  # Minimal for quick testing
            'batch_size': 2,
            'img_size': 640
        },
        'inference': {'dpi': 150, 'slice_height': 640, 'slice_width': 640}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    print(f"Config created at: {config_path}")

    # Load config
    config = ConfigLoader(config_path)
    print("Config loaded successfully")

    # Create trainer
    trainer = Trainer(config, architecture='yolov8', variant='nano')
    print("Trainer created successfully")

    # Run training
    dataset_path = '../data/floortest3.1.v1-data.yolov8/data.yaml'
    print(f"Starting training with dataset: {dataset_path}")
    print("This will take a few minutes...")

    try:
        results = trainer.train(dataset_path)
        print("Training completed successfully!")

        # Check for architecture_info.yaml
        train_base = Path('../runs/train')
        train_dirs = [d for d in train_base.iterdir() if d.is_dir() and d.name.startswith('train')]
        if train_dirs:
            latest_train_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
            print(f"Latest training directory: {latest_train_dir}")

            arch_info_path = latest_train_dir / 'architecture_info.yaml'
            if arch_info_path.exists():
                print(f"✓ architecture_info.yaml created at: {arch_info_path}")
                with open(arch_info_path, 'r') as f:
                    arch_info = yaml.safe_load(f)
                print(f"  Architecture: {arch_info.get('architecture')}")
                print(f"  Variant: {arch_info.get('variant')}")
                print(f"  Identifier: {arch_info.get('identifier')}")
            else:
                print("✗ architecture_info.yaml NOT created")

            weights_path = latest_train_dir / 'weights' / 'best.pt'
            if weights_path.exists():
                print(f"✓ Model weights saved at: {weights_path}")
            else:
                print("✗ Model weights NOT saved")

        # Check processing.log
        if Path('../processing.log').exists():
            with open('../processing.log', 'r') as f:
                log_content = f.read()
            if 'Architecture: yolov8' in log_content:
                print("✓ Architecture log message found in processing.log")
            else:
                print("✗ Architecture log message NOT found in processing.log")
                print("  Searching for any architecture mentions...")
                for line in log_content.split('\n'):
                    if 'architecture' in line.lower():
                        print(f"    {line}")

    except Exception as e:
        print(f"Training failed with error: {e}")


if __name__ == '__main__':
    main()
    import traceback
    traceback.print_exc()

