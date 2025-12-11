import yaml
import logging
from src.training.model_selector import ModelArchitectureSelector, SUPPORTED_ARCHITECTURES

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self.system = self.config.get('system', {})
        self.model = self.config.get('model', {})
        self.inference = self.config.get('inference', {})
        self.training = self.config.get('training', {})

        # Extract architecture parameters with backwards compatibility fallback
        training_config = self.config.get('training', {})
        self.model_architecture = training_config.get('model_architecture', 'yolov8')
        self.model_variant = training_config.get('model_variant', 'nano')

        # Set defaults in config if missing (for backwards compatibility)
        if 'model_architecture' not in training_config:
            self.config.setdefault('training', {})['model_architecture'] = 'yolov8'
        if 'model_variant' not in training_config:
            self.config.setdefault('training', {})['model_variant'] = 'nano'

        # Validate architecture using ModelArchitectureSelector
        try:
            # Instantiate selector to trigger validation
            selector = ModelArchitectureSelector(self.model_architecture, self.model_variant)
            logging.info(f"Model architecture configured: {self.model_architecture}{self.model_variant}")
        except ValueError as e:
            logging.error(
                f"Unsupported model architecture '{self.model_architecture}' in config. "
                f"Supported: {SUPPORTED_ARCHITECTURES}"
            )
            raise ValueError(
                f"Invalid architecture configuration. {str(e)}"
            ) from e

        self._validate()

    def get(self, section: str, key: str, default=None):
        """Get a configuration value from a specific section.

        Args:
            section: The configuration section (e.g., 'training', 'model').
            key: The key within the section.
            default: Default value if key not found.

        Returns:
            The configuration value or default.
        """
        section_data = self.config.get(section, {})
        return section_data.get(key, default)

    def _validate(self):
        # Validate confidence_threshold
        conf = self.model.get('confidence_threshold', 0.5)
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {conf}")

        # Note: model_path is optional - only required for inference mode, not for training
        # Training uses pretrained weights via ModelArchitectureSelector
        # Validation of model_path existence is deferred to inference pipeline

        # Validate positive numeric parameters
        dpi = self.inference.get('dpi', 300)
        if dpi <= 0:
            raise ValueError(f"dpi must be positive, got {dpi}")

        slice_height = self.inference.get('slice_height', 640)
        if slice_height <= 0:
            raise ValueError(f"slice_height must be positive, got {slice_height}")

        slice_width = self.inference.get('slice_width', 640)
        if slice_width <= 0:
            raise ValueError(f"slice_width must be positive, got {slice_width}")

        max_megapixels = self.inference.get('max_megapixels', 200)
        if max_megapixels <= 0:
            raise ValueError(f"max_megapixels must be positive, got {max_megapixels}")

        # Validate overlap_ratio
        overlap_ratio = self.inference.get('overlap_ratio', 0.2)
        if not (0.0 <= overlap_ratio <= 0.5):
            raise ValueError(f"overlap_ratio must be between 0.0 and 0.5, got {overlap_ratio}")

        # Validate training parameters
        epochs = self.training.get('hyperparameters', {}).get('epochs', 50)
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")

        batch_size = self.training.get('hyperparameters', {}).get('batch_size', 16)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        img_size = self.training.get('hyperparameters', {}).get('img_size', 640)
        if img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")

        learning_rate = self.training.get('hyperparameters', {}).get('learning_rate', 0.001)
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        patience = self.training.get('hyperparameters', {}).get('patience', 10)
        if patience < 0:
            raise ValueError(f"patience must be non-negative, got {patience}")

        # Validate augmentation parameters (ranges 0.0-1.0 for probabilities, appropriate for others)
        aug = self.training.get('augmentation', {})
        hsv_h = aug.get('hsv_h', 0.015)
        if not (0.0 <= hsv_h <= 1.0):
            raise ValueError(f"hsv_h must be between 0.0 and 1.0, got {hsv_h}")

        hsv_s = aug.get('hsv_s', 0.7)
        if not (0.0 <= hsv_s <= 1.0):
            raise ValueError(f"hsv_s must be between 0.0 and 1.0, got {hsv_s}")

        hsv_v = aug.get('hsv_v', 0.4)
        if not (0.0 <= hsv_v <= 1.0):
            raise ValueError(f"hsv_v must be between 0.0 and 1.0, got {hsv_v}")

        degrees = aug.get('degrees', 0.0)
        if degrees < 0:
            raise ValueError(f"degrees must be non-negative, got {degrees}")

        translate = aug.get('translate', 0.1)
        if not (0.0 <= translate <= 1.0):
            raise ValueError(f"translate must be between 0.0 and 1.0, got {translate}")

        scale = aug.get('scale', 0.5)
        if not (0.0 <= scale <= 1.0):
            raise ValueError(f"scale must be between 0.0 and 1.0, got {scale}")

        shear = aug.get('shear', 0.0)
        if shear < 0:
            raise ValueError(f"shear must be non-negative, got {shear}")

        perspective = aug.get('perspective', 0.0)
        if not (0.0 <= perspective <= 1.0):
            raise ValueError(f"perspective must be between 0.0 and 1.0, got {perspective}")

        flipud = aug.get('flipud', 0.0)
        if not (0.0 <= flipud <= 1.0):
            raise ValueError(f"flipud must be between 0.0 and 1.0, got {flipud}")

        fliplr = aug.get('fliplr', 0.5)
        if not (0.0 <= fliplr <= 1.0):
            raise ValueError(f"fliplr must be between 0.0 and 1.0, got {fliplr}")

        mosaic = aug.get('mosaic', 1.0)
        if not (0.0 <= mosaic <= 1.0):
            raise ValueError(f"mosaic must be between 0.0 and 1.0, got {mosaic}")

        mixup = aug.get('mixup', 0.0)
        if not (0.0 <= mixup <= 1.0):
            raise ValueError(f"mixup must be between 0.0 and 1.0, got {mixup}")

    def get_model_architecture_config(self) -> dict:
        """Retrieve model architecture configuration for Trainer initialization.

        Returns:
            dict: Architecture configuration with keys:
                - 'architecture': str (validated architecture name)
                - 'variant': str (validated size variant)
                - 'identifier': str (model identifier for logging)

        Examples:
            >>> config = ConfigLoader('config.yaml')
            >>> arch_config = config.get_model_architecture_config()
            >>> arch_config['architecture']
            'yolov8'
        """
        logging.debug(f"Retrieving architecture config: {self.model_architecture}{self.model_variant}")

        # Construct identifier for logging
        from src.training.model_selector import MODEL_VARIANTS
        identifier = f"{self.model_architecture}{MODEL_VARIANTS[self.model_variant]}.pt"

        return {
            'architecture': self.model_architecture,
            'variant': self.model_variant,
            'identifier': identifier
        }
