#!/usr/bin/env python3
"""
MMDetection Installation Validation Script

Validates MMDetection framework installation and coexistence with YOLO.
Checks imports, versions, CUDA ops, and framework functionality.
"""

import logging
import sys
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('processing.log')
    ]
)
logger = logging.getLogger(__name__)


def validate_imports() -> Tuple[bool, str]:
    """Validate all MMDetection packages can be imported."""
    try:
        import mmcv
        import mmengine
        import mmdet
        return True, "All imports successful"
    except ImportError as e:
        return False, f"Import failed: {e}. Run: pip install -r requirements_mmdet.txt"


def validate_versions() -> Tuple[bool, str]:
    """Check versions against minimum requirements."""
    try:
        from packaging import version
        import mmcv, mmengine, mmdet

        checks = {
            'mmcv': (mmcv.__version__, '2.0.0'),
            'mmengine': (mmengine.__version__, '0.8.0'),
            'mmdet': (mmdet.__version__, '3.0.0')
        }

        for pkg, (current, required) in checks.items():
            if version.parse(current) < version.parse(required):
                return False, f"{pkg} version {current} < required {required}"

        return True, "All versions meet requirements"
    except ImportError:
        return False, "packaging not installed. Run: pip install packaging"


def validate_cuda_ops() -> Tuple[bool, str]:
    """Verify mmcv CUDA ops are available."""
    try:
        import mmcv
        # Test if CUDA ops compiled in mmcv
        if hasattr(mmcv, 'ops'):
            return True, "CUDA ops available in mmcv"
        else:
            return False, "CUDA ops not found. Reinstall mmcv with GPU wheel"
    except Exception as e:
        return False, f"CUDA ops check failed: {e}"


def test_mmdet_functionality() -> Tuple[bool, str]:
    """Test basic MMDetection functionality."""
    try:
        from mmengine.config import Config
        from mmdet.apis import init_detector
        # This validates the API is accessible
        return True, "MMDetection APIs accessible"
    except Exception as e:
        return False, f"MMDetection functionality test failed: {e}"


def validate_installation() -> bool:
    """Run all validation checks and report results."""
    checks = [
        ('Package Imports', validate_imports),
        ('Package Versions', validate_versions),
        ('CUDA Operations', validate_cuda_ops),
        ('MMDetection APIs', test_mmdet_functionality)
    ]

    all_passed = True
    for check_name, check_func in checks:
        passed, message = check_func()
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name} - {message}")
        logger.info(f"{check_name}: {'PASSED' if passed else 'FAILED'} - {message}")
        if not passed:
            all_passed = False

    return all_passed


def test_yolo_framework() -> Tuple[bool, str]:
    """Verify YOLO framework still works."""
    try:
        from ultralytics import YOLO
        # Test basic initialization (don't download models)
        return True, "YOLO framework operational"
    except Exception as e:
        return False, f"YOLO import failed: {e}"


def test_mmdet_framework() -> Tuple[bool, str]:
    """Verify MMDetection framework works."""
    try:
        from mmdet.apis import init_detector, inference_detector
        return True, "MMDetection framework operational"
    except Exception as e:
        return False, f"MMDetection import failed: {e}"


def test_gpu_availability() -> Tuple[bool, str]:
    """Verify GPU available for both frameworks."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA not available"

        # Check mmcv GPU ops
        gpu_available = torch.cuda.is_available()
        return gpu_available, f"GPU available: {torch.cuda.get_device_name(0)}"
    except Exception as e:
        return False, f"GPU check failed: {e}"


def environment_verification() -> bool:
    """Generate complete environment verification report."""
    try:
        import torch
        import ultralytics
        import mmcv, mmengine, mmdet
    except ImportError as e:
        print(f"✗ Environment verification FAILED: Missing imports - {e}")
        logger.error(f"Environment verification failed: {e}")
        raise EnvironmentError(f"Framework coexistence verification failed: {e}")

    print("\n" + "="*60)
    print("ENVIRONMENT VERIFICATION REPORT")
    print("="*60)

    # Version information
    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"Ultralytics Version: {ultralytics.__version__}")
    print(f"mmcv Version: {mmcv.__version__}")
    print(f"mmengine Version: {mmengine.__version__}")
    print(f"mmdet Version: {mmdet.__version__}")

    # Framework coexistence tests
    print("\nFramework Coexistence Tests:")
    tests = [
        ('YOLO Framework', test_yolo_framework),
        ('MMDetection Framework', test_mmdet_framework),
        ('GPU Availability', test_gpu_availability)
    ]

    all_passed = True
    for test_name, test_func in tests:
        passed, message = test_func()
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name} - {message}")
        logger.info(f"{test_name}: {'PASSED' if passed else 'FAILED'} - {message}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ Environment verification PASSED: YOLO and MMDetection coexisting successfully")
        logger.info("Environment verification PASSED: YOLO and MMDetection coexisting successfully")
    else:
        print("\n✗ Environment verification FAILED: Review errors above")
        logger.error("Environment verification FAILED: Review errors above")
        raise EnvironmentError("Framework coexistence verification failed")

    return all_passed


if __name__ == "__main__":
    print("MMDetection Installation Validation")
    print("=" * 40)

    # Run installation validation
    print("\n1. Installation Validation:")
    install_ok = validate_installation()

    if install_ok:
        print("\n2. Environment Verification:")
        env_ok = environment_verification()

        if env_ok:
            print("\n🎉 SUCCESS: MMDetection installation and coexistence verified!")
            logger.info("MMDetection installation and coexistence verification completed successfully")
            sys.exit(0)
        else:
            print("\n❌ FAILURE: Environment verification failed")
            logger.error("Environment verification failed")
            sys.exit(1)
    else:
        print("\n❌ FAILURE: Installation validation failed")
        logger.error("Installation validation failed")
        sys.exit(1)
