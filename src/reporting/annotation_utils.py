"""
Utility functions for annotating images with detection bounding boxes and labels.

This module provides stateless functions for drawing bounding boxes, labels with backgrounds,
and complete annotation workflows for object detection results. Used by the inference pipeline
to generate visual outputs with class-specific colors and confidence scores.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('processing.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


def get_default_class_colors() -> Dict[str, Tuple[int, int, int]]:
    """
    Returns default color mapping for architectural classes in BGR format for OpenCV.

    Colors chosen for visual differentiation:
    - door: green (0,255,0) for entry/exit points
    - window: red (255,0,0) for openings
    - wall: blue (0,0,255) for structural elements
    - object: yellow (0,255,255) for furniture/fixtures

    Returns:
        Dictionary mapping class names to BGR color tuples.
    """
    return {
        'door': (0, 255, 0),    # Green for doors
        'window': (255, 0, 0),  # Red for windows
        'wall': (0, 0, 255),    # Blue for walls
        'object': (0, 255, 255) # Yellow for objects
    }


def draw_bounding_box(image: np.ndarray, bbox: Tuple[float, float, float, float], color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    """
    Draws a bounding box rectangle on the image.

    Args:
        image: Input image as NumPy array.
        bbox: Bounding box coordinates as (x1, y1, x2, y2).
        color: Color of the bounding box in BGR format.
        thickness: Thickness of the rectangle lines.

    Returns:
        Modified image with bounding box drawn.
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


def draw_label_with_background(image: np.ndarray, text: str, position: Tuple[int, int], color: Tuple[int, int, int], bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Draws text with a background rectangle for readability.

    Args:
        image: Input image as NumPy array.
        text: Text to draw.
        position: Bottom-left position of the text as (x, y).
        color: Color of the text in BGR format.
        bg_color: Color of the background rectangle.

    Returns:
        Modified image with label and background drawn.
    """
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = position
    # Draw background rectangle
    cv2.rectangle(image, (x, y - text_height - 5), (x + text_width, y), bg_color, -1)
    # Draw text
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


def annotate_detections(image: np.ndarray, detections: List[Dict], class_colors: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    """
    Annotates an image with bounding boxes and labels for all detections.

    Args:
        image: Input image as NumPy array.
        detections: List of detection dictionaries, each containing 'bbox', 'class_name', and 'confidence'.
        class_colors: Dictionary mapping class names to BGR colors.

    Returns:
        Annotated image with bounding boxes and labels.

    Example:
        colors = get_default_class_colors()
        annotated = annotate_detections(image, detections, colors)
    """
    logger.debug(f"Annotating {len(detections)} detections on image")
    annotated = image.copy()
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        color = class_colors.get(class_name, (0, 255, 0))  # Default to green
        # Draw bounding box
        draw_bounding_box(annotated, bbox, color)
        # Format label
        label = f"{class_name} {confidence:.2f}"
        # Draw label above bbox
        x1, y1, _, _ = bbox
        draw_label_with_background(annotated, label, (int(x1), int(y1)), color)
    return annotated
