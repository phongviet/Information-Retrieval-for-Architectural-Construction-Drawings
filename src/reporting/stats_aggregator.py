import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('processing.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


def aggregate_counts(results: Dict[int, Dict]) -> Dict[int, Dict[str, int]]:
    """
    Aggregates detection counts per page per class from the inference results.

    Args:
        results: Dictionary with page_num as key and page data as value.
                 Page data contains 'detections' list or 'error' string.

    Returns:
        Nested dictionary {page_id: {class_name: count}} for pages with detections.
    """
    logger.info("Starting statistics aggregation from inference results")
    page_results = {}
    for page_num, page_data in results.items():
        if 'detections' in page_data:
            detections = page_data['detections']
            counts = {}
            for det in detections:
                class_name = det['class_name']
                counts[class_name] = counts.get(class_name, 0) + 1
            page_results[page_num] = counts
            logger.debug(f"Page {page_num}: processed detections, counts: {counts}")
        else:
            logger.warning(f"Page {page_num} skipped due to error: {page_data['error']}")
    total_detections = sum(sum(counts.values()) for counts in page_results.values())
    page_count = len(page_results)
    class_count = len(set(class_name for counts in page_results.values() for class_name in counts))
    logger.debug(f"Aggregated {total_detections} detections across {page_count} pages for {class_count} classes")
    return page_results


def calculate_confidence_stats(detections: List[Dict]) -> Tuple[float, float, float]:
    """
    Calculates confidence statistics (average, min, max) for a list of detections.

    Args:
        detections: List of detection dictionaries for a specific class on a page.

    Returns:
        Tuple of (avg_confidence, min_confidence, max_confidence).
        Returns (0.0, 0.0, 0.0) if detections list is empty.
    """
    if not detections:
        return (0.0, 0.0, 0.0)
    confidences = [d['confidence'] for d in detections]
    avg_conf = float(np.mean(confidences))
    min_conf = float(min(confidences))
    max_conf = float(max(confidences))
    return (avg_conf, min_conf, max_conf)


def handle_empty_pages(page_results: Dict[int, Dict[str, int]], total_pages: int, class_names: List[str]) -> Dict[int, Dict[str, int]]:
    """
    Ensures all pages (1 to total_pages) and all class_names are represented in page_results,
    filling missing entries with zero counts.

    Args:
        page_results: Dictionary {page_id: {class_name: count}} from aggregate_counts.
        total_pages: Total number of pages in the document.
        class_names: List of expected class names.

    Returns:
        Complete page_results dict with all pages and classes.
    """
    complete_results = {}
    for page_num in range(1, total_pages + 1):
        if page_num in page_results:
            counts = page_results[page_num].copy()
        else:
            counts = {}
        for class_name in class_names:
            if class_name not in counts:
                counts[class_name] = 0
        complete_results[page_num] = counts
    return complete_results


def structure_for_csv(results: Dict[int, Dict], class_names: Optional[List[str]] = None) -> List[Dict]:
    """
    Structures the inference results into a list of dictionaries suitable for CSV output.

    Each row represents a page-class combination with count and confidence statistics.

    Args:
        results: Dictionary with page_num as key and page data as value.
        class_names: List of class names to include, even if zero detections. Defaults to ['door', 'object', 'wall', 'window'].

    Returns:
        List of dictionaries, each with keys: 'Page_ID', 'Class', 'Count',
        'Confidence_Avg', 'Min_Confidence', 'Max_Confidence'.

    Example:
        [{'Page_ID': 1, 'Class': 'door', 'Count': 2, 'Confidence_Avg': 0.85, ...}, ...]
    """
    if class_names is None:
        class_names = ['door', 'object', 'wall', 'window']
    # First, group detections per page per class
    page_class_detections = {}
    for page_num, page_data in results.items():
        if 'detections' in page_data:
            detections = page_data['detections']
            class_dets = {}
            for det in detections:
                class_name = det['class_name']
                if class_name not in class_dets:
                    class_dets[class_name] = []
                class_dets[class_name].append(det)
            page_class_detections[page_num] = class_dets
        else:
            # For error pages, no detections
            page_class_detections[page_num] = {}

    # Determine total_pages as max page_num
    total_pages = max(results.keys()) if results else 0

    # Now, build the list
    rows = []
    for page_num in range(1, total_pages + 1):
        class_dets = page_class_detections.get(page_num, {})
        for class_name in class_names:
            detections = class_dets.get(class_name, [])
            count = len(detections)
            avg_conf, min_conf, max_conf = calculate_confidence_stats(detections)
            row = {
                'Page_ID': page_num,
                'Class': class_name,
                'Count': count,
                'Confidence_Avg': avg_conf,
                'Min_Confidence': min_conf,
                'Max_Confidence': max_conf
            }
            rows.append(row)
    return rows
