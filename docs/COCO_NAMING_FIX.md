# Quick Fix: COCO Dataset File Naming Issue

## Problem
Your COCO annotation files are named with the `instances_` prefix:
- `instances_train2017.json` (instead of `train.json`)
- `instances_val2017.json` (instead of `val.json`)
- `instances_test2017.json` (instead of `test.json`)

## Solution Applied ✅

I've updated the configuration files to use the correct file names:

### 1. **configs/mmdet/datasets/architectural_coco.py** - UPDATED
Changed the annotation file references from:
```python
ann_file=data_root + 'annotations/train.json'
```

To:
```python
ann_file=data_root + 'annotations/instances_train2017.json'
```

The same fix was applied to `val` and `test` datasets.

### 2. **src/training/mmdet_trainer.py** - UPDATED
Updated the placeholder dataset configuration to use:
```python
ann_file='data/cubicasa5k_coco/annotations/instances_train2017.json'
```

### 3. **Data Validation Error Message** - IMPROVED
Updated error messages to clearly show the correct COCO file naming convention.

---

## ✅ What You Can Do Now

### Quick Test
Verify the data is now accessible:
```bash
python -c "
import json
with open('data/cubicasa5k_coco/annotations/instances_train2017.json') as f:
    data = json.load(f)
    print(f'✓ Train: {len(data[\"images\"])} images, {len(data[\"annotations\"])} annotations')
    print(f'✓ Classes: {[c[\"name\"] for c in data[\"categories\"]]}')
"
```

### Run Training
Now you can train with:
```bash
python main.py train \
  --framework mmdetection \
  --mmdet-config configs/mmdet/cascade_rcnn_architectural_full.py
```

---

## 📋 Your COCO Dataset Structure

```
data/cubicasa5k_coco/
├── annotations/
│   ├── instances_train2017.json    ✅ Correct name
│   ├── instances_val2017.json      ✅ Correct name
│   └── instances_test2017.json     ✅ Correct name (optional)
├── train2017/                       ✅ Training images
│   ├── image_001.jpg
│   └── ...
├── val2017/                         ✅ Validation images
│   └── ...
└── test2017/                        ✅ Test images (optional)
    └── ...
```

---

## 🎯 Next Steps

1. **Verify your data** (optional test above)
2. **Run training**:
   ```bash
   python main.py train \
     --framework mmdetection \
     --mmdet-config configs/mmdet/cascade_rcnn_architectural_full.py
   ```
3. **Monitor progress** in `runs/mmdet_train/`
4. **Check results** in `models/mmdet_cascade_rcnn.pth`

---

## 📝 Files Modified

| File | Change | Status |
|------|--------|--------|
| `configs/mmdet/datasets/architectural_coco.py` | Updated annotation file names to `instances_*2017.json` | ✅ Done |
| `src/training/mmdet_trainer.py` | Updated placeholder config + improved error messages | ✅ Done |

---

## ⚡ COCO Naming Convention

Standard COCO dataset file naming:
- **Standard names**: `annotations/train.json`, `annotations/val.json`
- **With year suffix**: `annotations/instances_train2017.json`, `annotations/instances_val2017.json`
- **With split suffix**: `annotations/train2017/`, `annotations/val2017/`

Your dataset uses the **year-suffixed naming** format (like official COCO), which is now properly configured! 🎉

---

**Ready to train! Just run the command above.** 🚀

