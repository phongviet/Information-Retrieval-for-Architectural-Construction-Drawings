"""
MMDetection Inference Adapter
Wraps MMDetection inference APIs to provide unified interface
"""

import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class MMDetInferenceAdapter:
    """Adapter for MMDetection model inference."""

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = 'cuda:0',
        confidence_threshold: float = 0.5
    ):
        """
        Initialize MMDetection inference adapter.

        Args:
            model_path: Path to trained model checkpoint (.pth)
            config_path: Path to MMDetection config file
            device: Device for inference ('cuda:0', 'cpu', etc.)
            confidence_threshold: Minimum confidence for detections
        """
        from mmdet.apis import init_detector

        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Initialize model
        logger.info(f"Loading MMDetection model: {model_path}")
        self.model = init_detector(config_path, model_path, device=device)

        # Extract class names from config
        if hasattr(self.model.cfg, 'classes'):
            self.class_names = self.model.cfg.classes
        elif hasattr(self.model.cfg.data, 'test') and hasattr(self.model.cfg.data.test, 'classes'):
            self.class_names = self.model.cfg.data.test.classes
        else:
            # Default for architectural dataset
            self.class_names = ['door', 'window', 'wall', 'object']

        logger.info(f"MMDetection model loaded: {len(self.class_names)} classes, "
                   f"device={device}, threshold={confidence_threshold}")

        # Framework identifier
        self.framework = 'mmdetection'

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run inference on image.

        Args:
            image: Image as numpy array (H, W, 3) in RGB

        Returns:
            List of detection dictionaries in unified format
        """
        # Validate input
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Image must be numpy array, got {type(image)}")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be (H, W, 3), got shape {image.shape}")

        # Run inference
        try:
            from mmdet.apis import inference_detector

            result = inference_detector(self.model, image)

            # Parse results and convert to unified format
            detections = self._parse_mmdet_result(result, image.shape[:2])

            logger.debug(f"MMDetection inference: {len(detections)} detections")
            return detections

        except RuntimeError as e:
            logger.error(f"MMDetection inference failed: {e}")
            if 'out of memory' in str(e).lower():
                logger.error("OOM error - try reducing image size or using CPU")
            raise

    def _parse_mmdet_result(self, result, img_shape: tuple) -> List[Dict]:
        """
        Parse MMDetection result to unified format.

        Args:
            result: MMDetection inference result (list of arrays per class)
            img_shape: Image shape (H, W)

        Returns:
            List of detection dictionaries
        """
        detections = []

        # MMDetection result is list of arrays, one per class
        # Each array has shape (N, 5) with [x1, y1, x2, y2, confidence]
        for class_id, class_detections in enumerate(result):
            if len(class_detections) == 0:
                continue

            # Filter by confidence threshold
            for detection in class_detections:
                confidence = float(detection[4])

                if confidence >= self.confidence_threshold:
                    # Extract bbox
                    x1, y1, x2, y2 = detection[:4]

                    # Ensure bbox is within image bounds
                    x1 = max(0, min(x1, img_shape[1]))
                    y1 = max(0, min(y1, img_shape[0]))
                    x2 = max(0, min(x2, img_shape[1]))
                    y2 = max(0, min(y2, img_shape[0]))

                    # Add to detections in unified format
                    detections.append({
                        'bbox': (float(x1), float(y1), float(x2), float(y2)),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': self.class_names[class_id],
                        'framework': 'mmdetection'
                    })

        # Sort by confidence descending
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        return detections

    def get_class_name(self, class_id: int) -> str:
        """Get class name for class ID."""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f'class_{class_id}'

    def get_detector_info(self) -> dict:
        """Get detector configuration and architecture information.

        Returns:
            Dictionary containing detector configuration and architecture info.
        """
        return {
            'model_path': self.model_path,
            'config_path': self.config_path,
            'framework': self.framework,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'class_names': self.class_names
        }
