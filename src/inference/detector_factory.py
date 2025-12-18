"""
Unified Detector Factory
Creates appropriate detector based on model file format
"""

import os
import torch
import logging
from typing import Union

logger = logging.getLogger(__name__)


class UnifiedDetectorFactory:
    """Factory for creating framework-agnostic detectors."""

    @staticmethod
    def detect_framework(model_path: str) -> str:
        """
        Detect framework from model file.

        Args:
            model_path: Path to model file

        Returns:
            Framework name: 'yolo' or 'mmdetection'
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Check file extension
        ext = os.path.splitext(model_path)[1].lower()

        if ext == '.pt':
            # YOLO model - verify by checking for ultralytics metadata
            try:
                checkpoint = torch.load(model_path, map_location='cpu')

                # Check for ultralytics-specific keys
                if isinstance(checkpoint, dict):
                    # Look for YOLO-specific keys
                    if 'model' in checkpoint or 'ema' in checkpoint:
                        logger.info(f"Detected YOLO model: {model_path}")
                        return 'yolo'

                # Default to YOLO for .pt files
                logger.info(f"Detected YOLO model (by extension): {model_path}")
                return 'yolo'

            except Exception as e:
                logger.warning(f"Could not inspect .pt file: {e}, assuming YOLO")
                return 'yolo'

        elif ext == '.pth':
            # MMDetection model
            logger.info(f"Detected MMDetection model: {model_path}")
            return 'mmdetection'

        else:
            raise ValueError(
                f"Unknown model format: {ext}. "
                f"Supported: .pt (YOLO), .pth (MMDetection)"
            )

    @staticmethod
    def create_detector(model_path: str, config=None):
        """
        Create detector based on model framework.

        Args:
            model_path: Path to model file
            config: ConfigLoader instance (optional)

        Returns:
            Detector instance (SAHIDetector or MMDetInferenceAdapter)
        """
        framework = UnifiedDetectorFactory.detect_framework(model_path)

        if framework == 'yolo':
            from src.inference.sahi_detector import SAHIDetector

            # Extract config parameters
            confidence_threshold = 0.5
            device = 'cuda:0'

            if config:
                if hasattr(config, 'inference'):
                    confidence_threshold = getattr(config.inference, 'confidence_threshold', 0.5)
                if hasattr(config, 'system'):
                    gpu_id = getattr(config.system, 'gpu_id', 0)
                    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

            logger.info(f"Creating YOLO detector: {model_path}")
            detector = SAHIDetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device=device
            )

            return detector

        elif framework == 'mmdetection':
            from src.inference.mmdet_adapter import MMDetInferenceAdapter

            # Extract config parameters
            confidence_threshold = 0.5
            device = 'cuda:0'
            config_path = None

            if config:
                if hasattr(config, 'inference'):
                    confidence_threshold = getattr(config.inference, 'confidence_threshold', 0.5)
                if hasattr(config, 'system'):
                    gpu_id = getattr(config.system, 'gpu_id', 0)
                    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
                if hasattr(config, 'mmdet_config_path'):
                    config_path = config.mmdet_config_path

            # Default config path if not specified
            if not config_path:
                # Try to find config in standard locations
                import glob
                configs = glob.glob('configs/mmdet/**/*.py', recursive=True)
                if configs:
                    config_path = configs[0]
                    logger.info(f"Using default MMDetection config: {config_path}")
                else:
                    raise ValueError(
                        "MMDetection config path not found. "
                        "Specify in config.yaml or pass config_path"
                    )

            logger.info(f"Creating MMDetection detector: {model_path}")
            detector = MMDetInferenceAdapter(
                model_path=model_path,
                config_path=config_path,
                device=device,
                confidence_threshold=confidence_threshold
            )

            return detector

        else:
            raise NotImplementedError(f"Framework not supported: {framework}")
