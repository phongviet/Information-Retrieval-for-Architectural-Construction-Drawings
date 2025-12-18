"""
Script to modify YOLO checkpoint epoch target for extended training.
This allows resuming training with preserved optimizer state.
"""
import torch
import sys

def modify_checkpoint_epochs(checkpoint_path: str, new_epochs: int, current_epoch: int = None):
    """
    Modify the epoch target in a YOLO checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file (e.g., 'runs/train/train3/weights/last.pt')
        new_epochs: New target epoch count (e.g., 1000)
        current_epoch: Current epoch to set (optional, will auto-detect if not provided)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Display current checkpoint info
    old_epoch = checkpoint.get('epoch', -1)
    current_best_fitness = checkpoint.get('best_fitness', None)

    print(f"\n📊 Current Checkpoint Info:")
    print(f"   Current epoch: {old_epoch}")
    print(f"   Best fitness: {current_best_fitness}")

    # Check if train_args exists
    if 'train_args' in checkpoint:
        old_epochs = checkpoint['train_args'].get('epochs', 'Not set')
        print(f"   Current epoch target: {old_epochs}")

        # If current_epoch not provided and old_epoch is -1, try to infer from old_epochs
        if current_epoch is None:
            if old_epoch == -1 and isinstance(old_epochs, int) and old_epochs > 0:
                # Training was completed, so epoch should be old_epochs - 1
                current_epoch = old_epochs - 1
                print(f"\n⚠️  Warning: Epoch was -1 but training target was {old_epochs}")
                print(f"   Auto-setting current epoch to {current_epoch} (assuming training completed)")
            else:
                current_epoch = old_epoch

        # Modify the epoch field and target
        checkpoint['epoch'] = current_epoch
        checkpoint['train_args']['epochs'] = new_epochs

        print(f"\n✅ Updated epoch: {old_epoch} → {current_epoch}")
        print(f"✅ Updated epoch target: {old_epochs} → {new_epochs}")
        print(f"\n   Training will resume from epoch {current_epoch + 1} to epoch {new_epochs - 1}")
        print(f"   That's {new_epochs - current_epoch - 1} additional epochs")
    else:
        print("⚠️  Warning: 'train_args' not found in checkpoint")
        return False

    # Save the modified checkpoint
    backup_path = checkpoint_path.replace('.pt', '_backup.pt')
    print(f"\n💾 Creating backup: {backup_path}")
    torch.save(torch.load(checkpoint_path, map_location='cpu', weights_only=False), backup_path)

    print(f"💾 Saving modified checkpoint: {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)

    print(f"\n✨ Success! You can now resume training with:")
    print(f"   python main.py train --resume {checkpoint_path} --data <your_data.yaml>")
    print(f"\n   Training will continue from epoch {current_epoch + 1} to epoch {new_epochs}")
    print(f"   Adam optimizer state will be preserved!")

    return True

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python modify_checkpoint.py <checkpoint_path> <new_epochs> [current_epoch]")
        print("\nExamples:")
        print("  # Auto-detect current epoch:")
        print("  python modify_checkpoint.py runs/train/train3/weights/last.pt 1000")
        print("\n  # Manually set current epoch (if you know you trained for 100 epochs):")
        print("  python modify_checkpoint.py runs/train/train3/weights/last.pt 400 99")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    new_epochs = int(sys.argv[2])
    current_epoch = int(sys.argv[3]) if len(sys.argv) == 4 else None

    modify_checkpoint_epochs(checkpoint_path, new_epochs, current_epoch)

