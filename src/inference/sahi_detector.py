import torch
import numpy as np
import logging
import yaml
import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def detect_model_architecture(weights_path: str) -> dict:
    """Detect model architecture from weights file and metadata.

    Attempts to detect architecture through multiple strategies:
    1. Parse architecture_info.yaml from weights directory or training directory
    2. Infer from filename patterns (yolo11, yolov8)
    3. Default to yolov8 for backwards compatibility

    Args:
        weights_path: Path to the model weights file.

    Returns:
        Dictionary with architecture information containing keys:
        - architecture: str (e.g., 'yolov8', 'yolo11')
        - variant: str (e.g., 'nano', 'small', or 'unknown')
        - identifier: str (model identifier or 'unknown')
        - detected_from: str ('metadata_file', 'filename_inference', or 'default_assumption')
    """
    weights_dir = os.path.dirname(weights_path)
    filename = os.path.basename(weights_path).lower()

    # Strategy 1: Check for architecture_info.yaml in weights directory
    arch_info_path = os.path.join(weights_dir, 'architecture_info.yaml')
    if os.path.exists(arch_info_path):
        try:
            with open(arch_info_path, 'r') as f:
                arch_data = yaml.safe_load(f)
            return {
                'architecture': arch_data.get('architecture', 'unknown'),
                'variant': arch_data.get('variant', 'unknown'),
                'identifier': arch_data.get('identifier', 'unknown'),
                'detected_from': 'metadata_file'
            }
        except Exception:
            pass  # Fall through to next strategy

    # Strategy 2: Check parent training directory (e.g., runs/train/trainN/weights/best.pt -> runs/train/trainN/)
    parent_dir = os.path.dirname(weights_dir)
    if os.path.basename(weights_dir) == 'weights' and os.path.exists(parent_dir):
        arch_info_path = os.path.join(parent_dir, 'architecture_info.yaml')
        if os.path.exists(arch_info_path):
            try:
                with open(arch_info_path, 'r') as f:
                    arch_data = yaml.safe_load(f)
                return {
                    'architecture': arch_data.get('architecture', 'unknown'),
                    'variant': arch_data.get('variant', 'unknown'),
                    'identifier': arch_data.get('identifier', 'unknown'),
                    'detected_from': 'metadata_file'
                }
            except Exception:
                pass  # Fall through to next strategy

    # Strategy 3: Filename pattern inference
    if 'yolo11' in filename:
        return {
            'architecture': 'yolo11',
            'variant': 'unknown',
            'identifier': 'unknown',
            'detected_from': 'filename_inference'
        }
    elif 'yolov8' in filename or 'yolo8' in filename:
        return {
            'architecture': 'yolov8',
            'variant': 'unknown',
            'identifier': 'unknown',
            'detected_from': 'filename_inference'
        }

    # Strategy 4: Default fallback for backwards compatibility
    return {
        'architecture': 'yolov8',
        'variant': 'unknown',
        'identifier': 'unknown',
        'detected_from': 'default_assumption'
    }


class SAHIDetector:
    """A detector class for performing sliced inference on large images using SAHI and YOLOv8.

    This class integrates SAHI (Slicing Aided Hyper Inference) with YOLOv8 models to detect
    small architectural elements on large floor plans through overlapping tile-based inference,
    solving the downscale data loss problem for high-resolution drawings.
    """

    def __init__(self, model_path: str, config: dict) -> None:
        """Initialize the SAHIDetector with YOLOv8 model and SAHI configuration.

        Args:
            model_path: Path to the trained YOLOv8 model weights file.
            config: Configuration dictionary containing detection parameters.
        """
        self.logger = logging.getLogger(__name__)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=model_path,
            confidence_threshold=config['confidence'],
            device=device
        )

        # Detect model architecture
        self.architecture_info = detect_model_architecture(model_path)

        # Log architecture detection
        if self.architecture_info['detected_from'] == 'metadata_file':
            self.logger.info(f"Detected model architecture: {self.architecture_info['architecture']}{self.architecture_info['variant']} from architecture_info.yaml")
        elif self.architecture_info['detected_from'] == 'filename_inference':
            self.logger.info(f"Inferred model architecture: {self.architecture_info['architecture']} from filename (metadata not found)")
        elif self.architecture_info['detected_from'] == 'default_assumption':
            self.logger.warning(f"Architecture metadata not found for model {model_path}. Assuming YOLOv8 for backwards compatibility. Train new models with Phase 7 for accurate metadata.")

        # Architecture-specific inference parameter adjustment (placeholder for future extensibility)
        if self.architecture_info['architecture'] == 'yolo11':
            # Placeholder for v11-specific optimizations if needed
            pass
        elif self.architecture_info['architecture'] == 'yolov9':
            # Future architecture support
            pass
        # Preserve existing SAHI parameters from config.yaml as primary source

        # Extract slice parameters from config
        self.slice_height = config['slice_height']
        self.slice_width = config['slice_width']
        self.overlap_height_ratio = config['overlap_height_ratio']
        self.overlap_width_ratio = config['overlap_width_ratio']
        # Validate parameters
        if not (isinstance(self.slice_height, int) and self.slice_height > 0):
            raise ValueError(f"slice_height must be a positive integer, got {self.slice_height}")
        if not (isinstance(self.slice_width, int) and self.slice_width > 0):
            raise ValueError(f"slice_width must be a positive integer, got {self.slice_width}")
        if not (0.0 <= self.overlap_height_ratio <= 0.5):
            raise ValueError(f"overlap_height_ratio must be between 0.0 and 0.5, got {self.overlap_height_ratio}")
        if not (0.0 <= self.overlap_width_ratio <= 0.5):
            raise ValueError(f"overlap_width_ratio must be between 0.0 and 0.5, got {self.overlap_width_ratio}")
        # Tile dimensions match YOLO input size (typically 640x640) and overlap prevents split object issues

    def detect(self, image: np.ndarray) -> list[dict]:
        """Execute sliced inference on the input image using SAHI.

        Args:
            image: NumPy array of the image in (height, width, 3) RGB format.

        Returns:
            List of dictionaries containing detection results.
        """
        # SAHI automatically handles tile generation, overlap management, and initial NMS
        result = get_sliced_prediction(
            image=image,
            detection_model=self.model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio
        )
        # Extract detection data from PredictionResult
        detections = []
        for det in result.object_prediction_list:
            bbox = tuple(det.bbox.to_xyxy())
            detection = {
                'bbox': bbox,
                'class_id': det.category.id,
                'class_name': det.category.name,
                'confidence': det.score.value
            }
            detections.append(detection)
        # Log detection count
        self.logger.debug(f"Extracted {len(detections)} detections from sliced inference")
        return detections

    def get_detector_info(self) -> dict:
        """Get detector configuration and architecture information.

        Returns:
            Dictionary containing detector configuration including:
            - model_path: str
            - architecture: str
            - variant: str
            - confidence_threshold: float
            - slice_height: int
            - slice_width: int
            - overlap_height_ratio: float
            - overlap_width_ratio: float
        """
        return {
            'model_path': self.model.model_path if hasattr(self.model, 'model_path') else 'unknown',
            'architecture': self.architecture_info['architecture'],
            'variant': self.architecture_info['variant'],
            'confidence_threshold': self.model.confidence_threshold if hasattr(self.model, 'confidence_threshold') else 'unknown',
            'slice_height': self.slice_height,
            'slice_width': self.slice_width,
            'overlap_height_ratio': self.overlap_height_ratio,
            'overlap_width_ratio': self.overlap_width_ratio
        }
