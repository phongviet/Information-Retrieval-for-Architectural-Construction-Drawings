"""YOLO model trainer with architecture selection support."""

import logging
import os
import shutil
import yaml
from pathlib import Path
from typing import Optional
from ultralytics import YOLO
from src.utils.config_loader import ConfigLoader
from src.training.model_selector import ModelArchitectureSelector


class Trainer:
    """YOLO model trainer with architecture selection support.

    Supports multiple YOLO architectures (v8, v11) via architecture parameter.
    Provides configuration-driven training with checkpoint resume and model standardization.

    Attributes:
        config: ConfigLoader instance for configuration access.
        architecture: Selected YOLO architecture (e.g., 'yolov8').
        variant: Model size variant (e.g., 'nano').
        selector: ModelArchitectureSelector for model initialization.
        logger: Configured logger for training events.

    Examples:
        >>> config = ConfigLoader('config/config.yaml')
        >>> trainer = Trainer(config, architecture='yolo11', variant='small')
        >>> trainer.train('data/floortest3.1-data.yolov8/data.yaml')
    """

    def __init__(self, config: ConfigLoader, architecture: str = 'yolov8', variant: str = 'nano') -> None:
        """Initialize the trainer with configuration and architecture selection.

        Args:
            config: ConfigLoader instance with loaded configuration.
            architecture: YOLO architecture name (case-insensitive).
            variant: Model size variant.

        Raises:
            ValueError: If architecture or variant is unsupported.
        """
        self.config = config
        self.architecture = architecture.lower()
        self.variant = variant
        self.logger = logging.getLogger(__name__)

        # Initialize architecture selector
        try:
            self.selector = ModelArchitectureSelector(self.architecture, self.variant)
        except ValueError as e:
            raise ValueError(f"Invalid architecture configuration: {e}") from e

        self.logger.info(f"Trainer initialized with architecture: {self.architecture}{self.variant}")

    def _epoch_end_callback(self, trainer) -> None:
        """Callback for logging per-epoch training metrics."""
        try:
            epoch = trainer.epoch + 1
            loss = trainer.loss if hasattr(trainer, 'loss') else 'N/A'
            self.logger.debug(f"Epoch {epoch} completed - Loss: {loss}")
        except Exception:
            # Silently skip if metrics not available
            pass

    def train(self, data_path: str, epochs: Optional[int] = None,
              batch_size: Optional[int] = None, img_size: Optional[int] = None, resume: Optional[str] = None):
        """Execute YOLO model training with configured parameters.

        Args:
            data_path: Path to data.yaml dataset configuration file.
            epochs: Number of training epochs (overrides config if provided).
            batch_size: Training batch size (overrides config if provided).
            img_size: Image size (overrides config if provided).
            resume: Path to checkpoint file to resume training (e.g., runs/train/trainN/weights/last.pt).

        Returns:
            Training results object from ultralytics.

        Raises:
            FileNotFoundError: If data.yaml or checkpoint file not found.
            RuntimeError: If CUDA out of memory or other runtime errors.
        """
        # Validate dataset file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset configuration not found: {data_path}")

        # Extract training parameters from config (support both flat and nested structure)
        training_config = self.config.config.get('training', {})
        hyperparams = training_config.get('hyperparameters', training_config)

        epochs = epochs or hyperparams.get('epochs', training_config.get('epochs', 10))
        batch_size = batch_size or hyperparams.get('batch_size', training_config.get('batch_size', 16))
        img_size = img_size or hyperparams.get('img_size', training_config.get('img_size', 640))

        # Extract hyperparameters
        learning_rate = hyperparams.get('learning_rate', training_config.get('learning_rate', 0.01))
        optimizer = hyperparams.get('optimizer', training_config.get('optimizer', 'SGD'))

        # Get device configuration
        system_config = self.config.config.get('system', {})
        device = system_config.get('gpu_id', 0)

        self.logger.info(f"Starting training: {epochs} epochs, batch size {batch_size}, "
                        f"image size {img_size}, model architecture {self.architecture}, dataset {data_path}")

        try:
            # Log architecture for tracking
            self.logger.info(f"Architecture: {self.architecture}")

            # Initialize YOLO model - either from checkpoint or pretrained
            if resume:
                if not os.path.exists(resume):
                    raise FileNotFoundError(f"Checkpoint file not found: {resume}")
                self.logger.info(f"Resuming training from checkpoint: {resume}")
                model = YOLO(resume)
            else:
                # Initialize YOLO model using architecture selector
                self.logger.info(f"Initializing YOLO model with architecture: {self.architecture}{self.variant}")
                model = self.selector.load_pretrained_model()

            # Prepare training arguments
            train_kwargs = {
                'data': data_path,
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': img_size,
                'lr0': learning_rate,
                'optimizer': optimizer,
                'device': device,
                'project': 'runs/train/',
            }

            # Add callback for per-epoch logging
            model.add_callback('on_train_epoch_end', self._epoch_end_callback)

            # Execute training
            self.logger.info("Training started...")
            results = model.train(**train_kwargs)

            # Get the latest training directory
            train_base = Path('runs/train')
            train_dirs = [d for d in train_base.iterdir() if d.is_dir() and d.name.startswith('train')]
            if train_dirs:
                latest_train_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)

                # Save architecture metadata
                arch_info = self.selector.get_architecture_info()
                arch_info_path = latest_train_dir / 'architecture_info.yaml'
                with open(arch_info_path, 'w') as f:
                    yaml.dump(arch_info, f, default_flow_style=False)
                self.logger.info(f"Architecture metadata saved to {arch_info_path}")

                # Standardize model location
                best_path = latest_train_dir / 'weights' / 'best.pt'
                if best_path.exists():
                    target_path = Path('models/custom_model.pt')
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(best_path), str(target_path))
                    self.logger.info(f"Best model copied to {target_path}")

            self.logger.info("Training completed successfully")
            return results

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self.logger.error("CUDA out of memory. Try reducing batch_size or img_size.")
            else:
                self.logger.error(f"Training failed: {e}")
            raise
        except Exception as e:
            self.logger.exception("Unexpected error during training")
            raise

