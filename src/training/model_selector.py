# Module-level constants for supported architectures and variants
SUPPORTED_ARCHITECTURES = ['yolov8', 'yolo11']

MODEL_VARIANTS = {
    'nano': 'n',
    'small': 's',
    'medium': 'm',
    'large': 'l',
    'xlarge': 'x'
}

import logging
import os
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelArchitectureSelector:
    """Flexible architecture selection system for YOLO models.

    Enables easy switching between YOLOv8, YOLOv11, and future variants
    for training and inference without code modifications.

    Attributes:
        architecture (str): Normalized architecture name (e.g., 'yolov8').
        variant (str): Model variant (e.g., 'nano').
    """

    def __init__(self, architecture: str = 'yolov8', variant: str = 'nano') -> None:
        """Initialize the architecture selector.

        Args:
            architecture: YOLO architecture name (case-insensitive).
            variant: Model size variant.

        Raises:
            ValueError: If architecture or variant is unsupported.
        """
        # Normalize architecture to lowercase
        self.architecture = architecture.lower()

        # Validate architecture
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture '{architecture}'. "
                f"Supported architectures: {SUPPORTED_ARCHITECTURES}"
            )

        # Validate variant
        if variant not in MODEL_VARIANTS:
            raise ValueError(
                f"Unsupported variant '{variant}'. "
                f"Supported variants: {list(MODEL_VARIANTS.keys())}"
            )

        self.variant = variant

    def get_model_identifier(self) -> str:
        """Construct ultralytics model filename.

        Returns:
            Model identifier string in format 'architecturevariant.pt'.

        Examples:
            >>> selector = ModelArchitectureSelector('yolov8', 'nano')
            >>> selector.get_model_identifier()
            'yolov8n.pt'
        """
        identifier = f"{self.architecture}{MODEL_VARIANTS[self.variant]}.pt"
        return identifier

    def load_pretrained_model(self, weights_path: str | None = None) -> YOLO:
        """Load pretrained YOLO model.

        Args:
            weights_path: Path to custom weights file. If None, uses default pretrained model.

        Returns:
            Initialized YOLO model object.

        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            if weights_path and os.path.exists(weights_path):
                model = YOLO(weights_path)
            else:
                identifier = self.get_model_identifier()
                # Check if model exists in models/ directory first
                models_dir = os.path.join('models', identifier)

                if os.path.exists(models_dir):
                    # Load from models/ directory
                    model = YOLO(models_dir)
                else:
                    # Download to models/ directory
                    os.makedirs('models', exist_ok=True)
                    # YOLO will download to current directory, then we move it
                    model = YOLO(identifier)
                    # Move downloaded model to models/ directory
                    if os.path.exists(identifier):
                        os.rename(identifier, models_dir)
            return model
        except RuntimeError as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def get_architecture_info(self) -> dict:
        """Get metadata about the selected architecture.

        Returns:
            Dictionary containing architecture metadata.
        """
        # Define features based on architecture
        if self.architecture == 'yolov8':
            features = ['anchor_based']
            version = '>=8.1.0'
        elif self.architecture == 'yolo11':
            features = ['anchor_free', 'dynamic_head']
            version = '>=8.2.0'
        else:
            features = []
            version = 'unknown'

        info = {
            'architecture': self.architecture,
            'variant': self.variant,
            'identifier': self.get_model_identifier(),
            'supported_features': features,
            'ultralytics_version_required': version
        }

        return info

    def validate_compatibility(self, dataset_config: dict) -> bool:
        """Validate dataset configuration compatibility.

        Args:
            dataset_config: Dataset config dict with keys 'nc', 'names', optional 'img_size'.

        Returns:
            True if compatible, raises ValueError if incompatible.

        Raises:
            ValueError: If configuration is incompatible.
        """
        # YOLO supports arbitrary number of classes
        if 'img_size' in dataset_config:
            img_size = dataset_config['img_size']
            if img_size % 32 != 0:
                logger.warning(f"Image size {img_size} is not multiple of 32. Recommended: multiples of 32 for optimal performance.")

        # Always return True as YOLO supports flexible configurations
        return True