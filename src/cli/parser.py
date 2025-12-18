import argparse
import os
from src.training.model_selector import ModelArchitectureSelector, SUPPORTED_ARCHITECTURES, MODEL_VARIANTS


def validate_file_exists(filepath: str) -> str:
    """Validates that a file exists at the given filepath.

    Args:
        filepath: Path to the file to check.

    Returns:
        The filepath if it exists.

    Raises:
        argparse.ArgumentTypeError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise argparse.ArgumentTypeError(f"File '{filepath}' does not exist.")
    return filepath


def validate_positive_int(value: str) -> int:
    """Validates that a string represents a positive integer.

    Args:
        value: String value to convert and validate.

    Returns:
        The integer value if valid.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise ValueError
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a positive integer.")


def validate_architecture_args(args: argparse.Namespace) -> None:
    """Validates architecture-related CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Raises:
        ValueError: If architecture or variant arguments are invalid.
    """
    # Only validate if architecture/variant attributes exist (train command only)
    if hasattr(args, 'architecture') and args.architecture and args.architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Invalid architecture '{args.architecture}'. Supported: {SUPPORTED_ARCHITECTURES}")

    if hasattr(args, 'variant') and args.variant and args.variant not in MODEL_VARIANTS:
        raise ValueError(f"Invalid variant '{args.variant}'. Supported: {list(MODEL_VARIANTS.keys())}")


def validate_framework_args(args):
    """
    Validate framework-specific arguments.

    Args:
        args: Parsed command-line arguments

    Raises:
        SystemExit: If validation fails
    """
    import sys
    import logging

    logger = logging.getLogger(__name__)

    # Check if MMDetection is requested
    if args.framework == 'mmdetection':
        # Check MMDetection installation
        try:
            import mmdet
            import mmcv
            import mmengine
            logger.info(f"MMDetection framework validated: mmdet {mmdet.__version__}")
        except ImportError as e:
            logger.error("MMDetection framework not installed!")
            logger.error(f"Error: {e}")
            logger.error("Install MMDetection: pip install -r requirements_mmdet.txt")
            sys.exit(1)

        # For training, require --mmdet-config
        if args.command == 'train' and not args.mmdet_config:
            logger.error("--mmdet-config is required when using --framework=mmdetection for training")
            logger.error("Example: --mmdet-config configs/mmdet/_base_/models/cascade_rcnn_r50_fpn.py")
            sys.exit(1)

        # Validate config file exists
        if args.mmdet_config:
            import os
            if not os.path.exists(args.mmdet_config):
                logger.error(f"MMDetection config not found: {args.mmdet_config}")
                sys.exit(1)

    # YOLO validation
    elif args.framework == 'yolo':
        try:
            import ultralytics
            logger.info(f"YOLO framework validated: ultralytics {ultralytics.__version__}")
        except ImportError:
            logger.error("Ultralytics YOLO not installed!")
            logger.error("Install YOLO: pip install ultralytics")
            sys.exit(1)

        # YOLO training requires --data argument
        if args.command == 'train' and not args.data:
            logger.error("--data is required when using --framework=yolo for training")
            logger.error("Example: --data data/floortest3.1.v1-data.yolov8/data.yaml")
            sys.exit(1)

    logger.info(f"Framework validation passed: {args.framework}")


def create_parser() -> argparse.ArgumentParser:
    """Creates CLI parser with train/detect subcommands.

    Returns:
        Configured ArgumentParser with subcommands for training and detection.
    """
    parser = argparse.ArgumentParser(
        description="Architectural Drawing Detection Pipeline - Multi-framework detection for architectural elements",
        add_help=True,
        epilog="""
Examples:
  YOLO Training:       python main.py train --framework yolo --data dataset.yaml --epochs 50
  MMDetection Training: python main.py train --framework mmdetection --mmdet-config configs/mmdet/_base_/models/cascade_rcnn_r50_fpn.py --epochs 12
  Auto-detect Detection: python main.py detect --input floor_plan.pdf --output results/
  Explicit YOLO:        python main.py detect --framework yolo --input plan.pdf
  Debug mode:           python main.py detect --input plan.pdf --debug
  Train with architecture: python main.py train --data data.yaml --architecture yolo11 --variant small --epochs 50
"""
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train subcommand
    train_parser = subparsers.add_parser(
        'train',
        help='Train object detection model using YOLO or MMDetection framework',
        description='Trains object detection model on architectural drawing dataset with configurable architecture and framework (YOLO or MMDetection). Supports checkpoint resume for interrupted training.'
    )
    # Framework selection
    train_parser.add_argument(
        '--framework',
        type=str,
        choices=['yolo', 'mmdetection'],
        default='yolo',
        help="Detection framework to use. "
             "'yolo' for Ultralytics YOLOv8/v11 (fast, simple). "
             "'mmdetection' for MMDetection Cascade R-CNN (higher accuracy, complex scenes). "
             "Default: yolo"
    )
    # Architecture selection (before data for logical grouping)
    train_parser.add_argument(
        '--architecture',
        type=str,
        default=None,
        choices=SUPPORTED_ARCHITECTURES,
        help="YOLO architecture variant (overrides config.yaml). Choices: yolov8 (anchor-based, faster), yolo11 (anchor-free, higher accuracy). Default: from config.yaml"
    )
    train_parser.add_argument(
        '--variant',
        type=str,
        default=None,
        choices=list(MODEL_VARIANTS.keys()),
        help="Model size variant (overrides config.yaml). Affects speed/accuracy tradeoff. Choices: nano (fastest), small, medium, large, xlarge (most accurate). Default: from config.yaml"
    )
    # Data file path (required for YOLO, optional for MMDetection)
    train_parser.add_argument('--data', type=validate_file_exists, required=False, help="Path to YOLO dataset data.yaml file (required for YOLO framework, optional for MMDetection)")
    # Optional epochs override
    train_parser.add_argument('--epochs', type=validate_positive_int, help="Number of training epochs (overrides config.yaml value)")
    # Optional batch size override
    train_parser.add_argument('--batch-size', type=validate_positive_int, help="Training batch size (overrides config.yaml value)")
    # Resume from checkpoint
    train_parser.add_argument('--resume', type=str, help="Path to checkpoint file (e.g., runs/train/trainN/weights/last.pt) to resume training")
    # MMDetection-specific arguments
    train_parser.add_argument(
        '--mmdet-config',
        type=str,
        help="Path to MMDetection config file (required if --framework=mmdetection). "
             "Example: configs/mmdet/_base_/models/cascade_rcnn_r50_fpn.py"
    )
    train_parser.add_argument(
        '--pretrained',
        type=str,
        help='Path to pretrained checkpoint for transfer learning (MMDetection)'
    )
    # Config file path
    train_parser.add_argument('--config', type=str, default='config/config.yaml', help="Path to configuration file")

    # Detect subcommand
    detect_parser = subparsers.add_parser(
        'detect',
        help='Run detection on PDF using trained model (auto-detects framework)',
        description='Runs detection on PDF floor plans using trained model. Auto-detects framework from model file extension. Generates CSV statistics and annotated images with bounding boxes.'
    )
    # Framework selection (optional - auto-detects from model file)
    detect_parser.add_argument(
        '--framework',
        type=str,
        choices=['yolo', 'mmdetection'],
        help="Detection framework (optional - auto-detected from model file). "
             "Use to override auto-detection if needed."
    )
    # Input PDF file (required)
    detect_parser.add_argument('--input', type=validate_file_exists, required=True, help="Path to input PDF file for detection")
    # Optional model override
    detect_parser.add_argument('--model', type=str, help="Path to trained model weights .pt file (overrides config.yaml model_path)")
    # Output directory
    detect_parser.add_argument('--output', type=str, default='data/output', help="Output directory for CSV reports and annotated images with bounding boxes")
    # Debug flag
    detect_parser.add_argument('--debug', action='store_true', help="Enable debug mode for SAHI tile visualization and overlap region analysis")
    # MMDetection-specific
    detect_parser.add_argument(
        '--mmdet-config',
        type=str,
        help='Path to MMDetection config file (for MMDetection models)'
    )
    # Config file path
    detect_parser.add_argument('--config', type=str, default='config/config.yaml', help="Path to configuration file")

    return parser
