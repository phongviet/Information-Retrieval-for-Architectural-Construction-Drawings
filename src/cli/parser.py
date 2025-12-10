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


def create_parser() -> argparse.ArgumentParser:
    """Creates CLI parser with train/detect subcommands.

    Returns:
        Configured ArgumentParser with subcommands for training and detection.
    """
    parser = argparse.ArgumentParser(
        description="Architectural Drawing Detection Pipeline - YOLO-based detection for architectural elements",
        add_help=True,
        epilog="""
Examples:
  Train model:    python main.py train --data dataset.yaml --epochs 50
  Detect objects: python main.py detect --input floor_plan.pdf --output results/
  Debug mode:     python main.py detect --input plan.pdf --debug
  Train with architecture: python main.py train --data data.yaml --architecture yolo11 --variant small --epochs 50
"""
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train subcommand
    train_parser = subparsers.add_parser(
        'train',
        help='Train YOLO model on architectural drawing dataset with configurable architecture',
        description='Trains YOLO model on architectural drawing dataset with configurable architecture (v8, v11, etc.). Supports checkpoint resume for interrupted training.'
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
    # Data file path (required)
    train_parser.add_argument('--data', type=validate_file_exists, required=True, help="Path to YOLO dataset data.yaml file")
    # Optional epochs override
    train_parser.add_argument('--epochs', type=validate_positive_int, help="Number of training epochs (overrides config.yaml value)")
    # Optional batch size override
    train_parser.add_argument('--batch-size', type=validate_positive_int, help="Training batch size (overrides config.yaml value)")
    # Resume from checkpoint
    train_parser.add_argument('--resume', type=str, help="Path to checkpoint file (e.g., runs/train/trainN/weights/last.pt) to resume training")
    # Config file path
    train_parser.add_argument('--config', type=str, default='config/config.yaml', help="Path to configuration file")

    # Detect subcommand
    detect_parser = subparsers.add_parser(
        'detect',
        help='Detect architectural elements in PDF floor plans',
        description='Detects doors, windows, walls, and objects in PDF floor plans. Generates CSV statistics and annotated images with bounding boxes.'
    )
    # Input PDF file (required)
    detect_parser.add_argument('--input', type=validate_file_exists, required=True, help="Path to input PDF file for detection")
    # Optional model override
    detect_parser.add_argument('--model', type=str, help="Path to trained model weights .pt file (overrides config.yaml model_path)")
    # Output directory
    detect_parser.add_argument('--output', type=str, default='data/output', help="Output directory for CSV reports and annotated images with bounding boxes")
    # Debug flag
    detect_parser.add_argument('--debug', action='store_true', help="Enable debug mode for SAHI tile visualization and overlap region analysis")
    # Config file path
    detect_parser.add_argument('--config', type=str, default='config/config.yaml', help="Path to configuration file")

    return parser
