"""
Module for generating JSON metadata files documenting detection runs.

This module creates comprehensive metadata JSON files that capture run details
for audit trails, reproducibility, and analysis. Metadata includes timestamps,
configuration parameters, detection summaries, and output file paths.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('processing.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


def generate_metadata(input_pdf: str, config: Dict, detection_summary: Dict, output_dir: str, architecture_info: dict = None) -> str:
    """
    Generates a JSON metadata file documenting the detection run.

    Args:
        input_pdf: Full path to the input PDF file.
        config: Dictionary containing key configuration parameters (e.g., DPI, SAHI settings, confidence_threshold).
        detection_summary: Dictionary with detection statistics (e.g., total_pages, class_counts, error_pages).
        output_dir: Directory where the metadata file will be saved.
        architecture_info: Optional; Dictionary with information about the model architecture used for detection.

    Returns:
        Path to the generated metadata JSON file.

    Raises:
        OSError: If unable to create output directory or write file.
        ValueError: If metadata cannot be serialized to JSON.

    Example:
        metadata = {
            "timestamp": "2023-12-03T10:00:00",
            "input_pdf": "/path/to/input.pdf",
            "configuration": {"dpi": 300, "confidence_threshold": 0.5},
            "detection_summary": {"total_pages": 10, "class_counts": {"door": 5}},
            "output_files": ["/path/to/report.csv"]
        }
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build metadata dictionary
    metadata = {
        "timestamp": datetime.now().isoformat(),  # ISO 8601 for run identification
        "input_pdf": input_pdf,  # Full path for traceability
        "configuration": config,  # Key settings for reproducibility
        "detection_summary": detection_summary,  # Results overview
        "output_files": []  # List of generated outputs (to be populated by caller)
    }

    # Add architecture information if provided
    if architecture_info:
        metadata["model_architecture"] = {
            'architecture': architecture_info.get('architecture', 'unknown'),
            'variant': architecture_info.get('variant', 'unknown'),
            'identifier': architecture_info.get('identifier', 'unknown'),
            'detected_from': architecture_info.get('detected_from', 'unknown')
        }

    # Serialize and write to file
    metadata_path = os.path.join(output_dir, 'run_metadata.json')
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Run metadata saved: {metadata_path}")
        return metadata_path
    except (OSError, ValueError) as e:
        logger.exception(f"Failed to generate metadata: {e}")
        raise OSError(f"Unable to write metadata to {metadata_path}: {e}") from e
