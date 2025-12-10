import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def save_tile_debug(tile_img: np.ndarray, detections: list[dict], tile_id: str, output_dir: str) -> None:
    """Save debug visualization for an individual tile with bounding boxes and labels.

    This helps identify per-tile detection coverage and quality.

    Args:
        tile_img: Tile image as NumPy array.
        detections: List of detections for this tile.
        tile_id: Unique identifier for the tile.
        output_dir: Base output directory.
    """
    img_copy = tile_img.copy()
    for det in detections:
        bbox = det['bbox']
        class_name = det['class_name']
        confidence = det['confidence']
        # Draw bounding box
        cv2.rectangle(img_copy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        # Draw label background
        label = f"{class_name} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_copy, (int(bbox[0]), int(bbox[1]) - text_height - 5), (int(bbox[0]) + text_width, int(bbox[1])), (255, 255, 255), -1)
        # Draw label text
        cv2.putText(img_copy, label, (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    tiles_dir = os.path.join(output_dir, 'tiles')
    os.makedirs(tiles_dir, exist_ok=True)
    output_path = os.path.join(tiles_dir, f'tile_{tile_id}.jpg')
    cv2.imwrite(output_path, img_copy, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logger.debug(f"Tile debug visualization saved: {output_path}")


def save_stitched_debug(full_image: np.ndarray, detections: list[dict], page_num: int, output_dir: str) -> None:
    """Save debug visualization for the full stitched page with final detections.

    This shows final detection coverage after NMS for quality assessment.

    Args:
        full_image: Full page image as NumPy array.
        detections: Final list of detections after NMS.
        page_num: Page number.
        output_dir: Base output directory.
    """
    img_copy = full_image.copy()
    for det in detections:
        bbox = det['bbox']
        class_name = det['class_name']
        confidence = det['confidence']
        # Draw bounding box
        cv2.rectangle(img_copy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        # Draw label background
        label = f"{class_name} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_copy, (int(bbox[0]), int(bbox[1]) - text_height - 5), (int(bbox[0]) + text_width, int(bbox[1])), (255, 255, 255), -1)
        # Draw label text
        cv2.putText(img_copy, label, (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Add metadata
    metadata = f"Page {page_num} - {len(detections)} detections"
    cv2.putText(img_copy, metadata, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    debug_dir = os.path.join(output_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    output_path = os.path.join(debug_dir, f'page_{page_num}_stitched.jpg')
    cv2.imwrite(output_path, img_copy, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logger.info(f"Stitched debug visualization saved: {output_path}")


def visualize_overlap_regions(image: np.ndarray, slice_params: dict, output_dir: str, page_num: int) -> None:
    """Visualize tile boundaries and overlap regions on the full image.

    This helps users understand the tiling strategy and identify split objects at boundaries.

    Args:
        image: Full image as NumPy array.
        slice_params: Dict with slice_height, slice_width, overlap_height_ratio, overlap_width_ratio.
        output_dir: Base output directory.
        page_num: Page number.
    """
    img_copy = image.copy()
    h, w = img_copy.shape[:2]
    slice_h = slice_params['slice_height']
    slice_w = slice_params['slice_width']
    overlap_h_ratio = slice_params['overlap_height_ratio']
    overlap_w_ratio = slice_params['overlap_width_ratio']

    # Calculate step sizes
    step_h = int(slice_h * (1 - overlap_h_ratio))
    step_w = int(slice_w * (1 - overlap_w_ratio))

    # Draw tile boundaries
    y = 0
    tile_count_h = 0
    tile_count_w = 0
    while y < h:
        x = 0
        tile_count_w_inner = 0
        while x < w:
            cv2.rectangle(img_copy, (x, y), (min(x + slice_w, w), min(y + slice_h, h)), (100, 100, 100), 2)
            x += step_w
            tile_count_w_inner += 1
        tile_count_w = max(tile_count_w, tile_count_w_inner)
        y += step_h
        tile_count_h += 1

    # Draw overlap regions (simplified, draw semi-transparent overlays)
    # For simplicity, draw overlap areas with different color
    y = 0
    while y < h:
        x = 0
        while x < w:
            # Overlap region
            overlap_x = max(0, x + slice_w - int(slice_w * overlap_w_ratio))
            overlap_y = max(0, y + slice_h - int(slice_h * overlap_h_ratio))
            cv2.rectangle(img_copy, (overlap_x, overlap_y), (min(x + slice_w, w), min(y + slice_h, h)), (0, 255, 255), -1)
            x += step_w
        y += step_h

    debug_dir = os.path.join(output_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    output_path = os.path.join(debug_dir, f'page_{page_num}_tiles.jpg')
    cv2.imwrite(output_path, img_copy, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logger.debug(f"Tile grid: {tile_count_h}x{tile_count_w} tiles with {overlap_h_ratio*100:.0f}% vertical, {overlap_w_ratio*100:.0f}% horizontal overlap. Saved: {output_path}")
