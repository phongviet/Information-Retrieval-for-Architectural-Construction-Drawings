import os
import logging
import torch
from .config_loader import ConfigLoader

def setup_logging():
    """Configure dual-channel logging: console INFO, file DEBUG.

    This function is idempotent - safe to call multiple times.
    Only configures logging once, subsequent calls are no-ops.
    """
    logger = logging.getLogger()

    # Check if logging is already configured (has handlers)
    if logger.hasHandlers():
        return

    logger.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (DEBUG and above)
    file_handler = logging.FileHandler('processing.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def validate_environment():
    """
    Perform comprehensive pre-flight validation before pipeline execution.

    Checks GPU availability, critical paths, configuration parameters, and write permissions.
    Raises exceptions with descriptive messages if validation fails.
    """
    # Setup logging if not already configured
    if not logging.getLogger().hasHandlers():
        setup_logging()

    logger = logging.getLogger(__name__)

    # 1. Check GPU availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA available: GPU {device_name}")
    else:
        logger.warning("Running on CPU - training will be slow")

    # 2. Verify critical paths
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found at {config_path}. Create configuration file or specify path with --config")

    models_dir = "models/"
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"models/ directory not found. Run from project root or create directory")

    output_dir = "data/output/"
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"data/output/ directory not found. Run from project root or create directory")

    logger.info("Critical paths verified")

    # 3. Validate configuration parameters
    try:
        config_loader = ConfigLoader(config_path)
        logger.debug(f"Configuration loaded: system={config_loader.system}, model={config_loader.model}, inference={config_loader.inference}")
        logger.info("Configuration parameters validated")
    except (ValueError, FileNotFoundError) as e:
        raise RuntimeError(f"Configuration validation failed during pre-flight check: {e}")

    # 4. Check write permissions
    dirs_to_check = [models_dir, output_dir]
    for dir_path in dirs_to_check:
        if not os.access(dir_path, os.W_OK):
            raise PermissionError(f"No write permission for {dir_path}. Check directory permissions or run with appropriate user privileges")

    logger.debug("Write permissions verified for output directories")
    logger.info("Pre-flight environment validation completed successfully")
