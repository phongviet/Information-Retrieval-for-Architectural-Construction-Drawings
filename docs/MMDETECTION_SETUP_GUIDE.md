# MMDetection Setup Guide

## Overview

This guide provides step-by-step instructions for installing and configuring the MMDetection framework alongside the existing YOLO setup. MMDetection enables Cascade R-CNN support for higher accuracy object detection on architectural drawings.

## Prerequisites

- **Conda Environment:** GR2 with Python 3.12
- **PyTorch:** 2.0+ with CUDA 11.8 (installed from Phase 1)
- **Existing Setup:** YOLO framework functional
- **Disk Space:** ~2GB for MMDetection and dependencies

## Installation Steps

### 1. Activate Conda Environment
```bash
conda activate GR2
```

### 2. Verify PyTorch with CUDA
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```
**Expected output:** `PyTorch: 2.x.x+cu118, CUDA: True`

### 3. Install mmcv with CUDA 11.8 Support
```bash
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```
**CRITICAL:** This step must be done BEFORE installing other MMDetection packages. It installs the GPU-enabled version of mmcv.

### 4. Install MMDetection Dependencies
```bash
pip install -r requirements_mmdet.txt
```

### 5. Verify Installation
```bash
python src/utils/validate_mmdet_install.py
```
**Expected output:** All checks should PASS with no errors.

## Troubleshooting

### ImportError: mmcv not found
**Cause:** mmcv not installed or wrong version
**Solution:**
```bash
pip uninstall mmcv mmcv-full  # Remove any existing versions
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

### RuntimeError: CUDA ops not available in mmcv
**Cause:** CPU-only mmcv installed instead of GPU version
**Solution:** Reinstall mmcv using the GPU wheel from OpenMMLab index (see Step 3)

### Version conflict between mmcv and PyTorch
**Cause:** Incompatible mmcv version for PyTorch
**Solution:** Check [compatibility matrix](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) and install matching version

### YOLO framework stops working after MMDetection install
**Cause:** Dependency conflict
**Solution:** Rollback and reinstall carefully:
```bash
pip uninstall mmcv mmengine mmdet
pip install ultralytics  # Reinstall YOLO
# Then retry MMDetection installation following steps above
```

### GPU memory errors during installation
**Cause:** Insufficient GPU memory for compilation
**Solution:** Close other GPU applications and retry, or use CPU-only installation (not recommended)

## Verification

After installation, verify both frameworks work:

### Quick Framework Tests
```bash
# Test YOLO
python -c "from ultralytics import YOLO; print('YOLO OK')"

# Test MMDetection
python -c "from mmdet.apis import init_detector; print('MMDetection OK')"
```

### Full Verification
```bash
python src/utils/validate_mmdet_install.py
```

**Expected Results:**
- All validation checks PASS
- Both frameworks operational
- GPU available for both
- No import conflicts

## Rollback Procedures

If installation fails and you need to restore the original YOLO-only environment:

### Complete Rollback
```bash
# Remove MMDetection packages
pip uninstall mmcv mmengine mmdet openmim matplotlib pycocotools albumentations

# Reinstall YOLO if needed
pip install ultralytics>=8.1.0

# Verify YOLO still works
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

### Partial Rollback
If only some packages are problematic, uninstall selectively and reinstall.

## Additional Resources

- [MMDetection Documentation](https://mmdetection.readthedocs.io/)
- [MMCV Installation Guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
- [OpenMMLab Compatibility Matrix](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#compatibility)

## Support

If installation issues persist:
1. Check the troubleshooting section above
2. Run the validation script for detailed error messages
3. Verify CUDA 11.8 compatibility with your GPU
4. Check PyTorch version compatibility

---

**Installation completed successfully when:** All validation checks pass and both YOLO and MMDetection frameworks are operational.
