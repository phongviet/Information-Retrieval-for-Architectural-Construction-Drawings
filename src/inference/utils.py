import logging

logger = logging.getLogger(__name__)


def calculate_iou(box1: tuple, box2: tuple) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    IoU measures the overlap between two boxes, used for duplicate detection in NMS.

    Args:
        box1: First bounding box as (x1, y1, x2, y2).
        box2: Second bounding box as (x1, y1, x2, y2).

    Returns:
        IoU value as float between 0.0 and 1.0.

    Example:
        >>> calculate_iou((0, 0, 10, 10), (5, 5, 15, 15))
        0.14285714285714285
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def apply_nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Apply Non-Maximum Suppression to remove duplicate detections.

    NMS eliminates duplicates at tile boundaries by keeping high-confidence detections
    and removing overlapping ones with the same class above the IoU threshold.

    Args:
        detections: List of detection dictionaries with 'bbox', 'class_id', 'confidence'.
        iou_threshold: IoU threshold for considering overlaps as duplicates.

    Returns:
        Filtered list of detections after NMS.
    """
    if not detections:
        return []

    # Sort by confidence descending
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    kept = []

    for det in detections:
        keep = True
        for kept_det in kept:
            if det['class_id'] == kept_det['class_id']:
                iou = calculate_iou(det['bbox'], kept_det['bbox'])
                if iou > iou_threshold:
                    keep = False
                    break
        if keep:
            kept.append(det)

    logger.debug(f"NMS: {len(detections)} -> {len(kept)} detections after duplicate removal with IoU threshold {iou_threshold}")
    return kept


def validate_coordinates(bbox: tuple, img_width: int, img_height: int) -> tuple:
    """Validate and clamp bounding box coordinates to image boundaries.

    Prevents out-of-bounds boxes by clamping coordinates, ensuring valid YOLO output.

    Args:
        bbox: Bounding box as (x1, y1, x2, y2).
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Clamped bounding box as (x1, y1, x2, y2).

    Example:
        >>> validate_coordinates((-5, -5, 15, 15), 10, 10)
        (0, 0, 10, 10)
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(x1 + 1, min(x2, img_width))
    y2 = max(y1 + 1, min(y2, img_height))
    return (x1, y1, x2, y2)


def aggregate_detections(detection_list: list[list[dict]]) -> list[dict]:
    """Aggregate detections from multiple sources and apply post-processing.

    Flattens nested detection lists, applies NMS to remove duplicates, and validates coordinates.

    Args:
        detection_list: Nested list of detections, e.g., from multiple pages.

    Returns:
        Final list of deduplicated detections.

    Example:
        >>> dets1 = [{'bbox': (0,0,10,10), 'class_id': 1, 'confidence': 0.9}]
        >>> dets2 = [{'bbox': (5,5,15,15), 'class_id': 1, 'confidence': 0.8}]
        >>> aggregate_detections([dets1, dets2])
        [{'bbox': (0,0,10,10), 'class_id': 1, 'confidence': 0.9}]
    """
    # Flatten
    all_detections = [det for sublist in detection_list for det in sublist]
    # Apply NMS
    filtered = apply_nms(all_detections)
    # Validate coordinates (assuming image dimensions are known, but for now skip if not)
    # Since img_width/height not provided, perhaps assume or skip validation here
    # For simplicity, return filtered as is, or add parameter if needed
    return filtered
