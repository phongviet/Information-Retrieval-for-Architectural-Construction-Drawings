# SAHI Tuning Guide for Architectural Drawing Detection

## Introduction

### Problem Statement
Architectural PDFs are large format (A0/A1: 36x48 inches). When rasterized at 300 DPI, this creates images approximately 10,000x13,000 pixels (~130 megapixels), far exceeding standard computer vision model input sizes.

### Why Standard YOLO Fails
Standard YOLOv8 models resize input images to 640x640 pixels, resulting in a loss of 99% of the original pixel data. Small architectural elements like doors and windows (typically 50-100 pixels in the original image) become sub-pixel features after downscaling, making them undetectable by the model.

### SAHI Solution
SAHI (Slicing Aided Hyper Inference) addresses this by dividing large images into overlapping 640x640 pixel tiles. YOLOv8 runs inference on each tile at full resolution, preserving fine details. Results are then stitched together using Non-Maximum Suppression (NMS) to eliminate duplicate detections across tile boundaries.

### Split Object Problem
Objects positioned exactly on tile boundaries can be problematic:
- **Duplicate counting**: An object may be detected in multiple adjacent tiles, leading to the same element counted twice.
- **Split detection**: An object may be partially visible in one tile and partially in another, resulting in incomplete or fragmented bounding boxes.

Proper overlap ensures each object appears completely within at least one tile, preventing these issues.

**Example Diagram:**
```
Tile 1 (640x640)          Tile 2 (640x640)
+-------------------+     +-------------------+
|                   |     |                   |
|   +-----------+   |     |   +-----------+   |
|   |   Door    |---|---  |   |           |   |
|   +-----------+   |     |   +-----------+   |
|                   |     |                   |
+-------------------+     +-------------------+
        Overlap Zone (e.g., 128 pixels)
```
Without overlap: Door split across tiles → partial detections.
With 20% overlap: Door fully visible in Tile 1 → single detection.

## Parameter Optimization

The following table compares SAHI configurations for different slice sizes and overlap ratios. Choose based on your drawing complexity and performance requirements.

| Slice Size | Overlap 0.1 (10%) | Overlap 0.2 (20%) | Overlap 0.3 (30%) |
|------------|-------------------|-------------------|-------------------|
| 512x512   | Fast processing (more tiles), may miss small objects (<50 pixels), higher risk of split detections | Moderate speed, better coverage for small elements, still some boundary issues | Slower, excellent for dense plans with many tiny features, minimal splits |
| 640x640   | **RECOMMENDED Starting Point** - Balanced quality and speed for most architectural plans, matches YOLO training size, handles most boundary cases | Best overall for typical floor plans: good coverage, reasonable speed, low split risk | Highest quality for complex drawings, 30% slower due to overlap, near-zero split issues |
| 1024x1024 | Fewer tiles (faster overall), may miss very small details (<40 pixels), better for large-scale plans with bigger elements | Good for site plans: fast processing, adequate for medium objects, some small element misses | Slower but comprehensive, reduces misses for detailed large-scale drawings |

### Recommendations
- **Starting point**: `slice_height=640, slice_width=640, overlap_ratio=0.2` for typical architectural floor plans.
- **Increase overlap to 0.3** if debug mode reveals split doors/windows at tile boundaries.
- **Use 512x512** for very dense plans with many small elements (e.g., detailed mechanical drawings).
- **Use 1024x1024** for large-scale site plans with bigger architectural elements (e.g., building footprints).

## Confidence Threshold Tuning

### Precision vs Recall Trade-off
- **High threshold (>0.6)**: Fewer false positives, cleaner results, but may miss real objects (lower recall).
- **Low threshold (<0.4)**: Detects more objects (higher recall), but includes noise/false positives (lower precision).
- **Balanced (0.5)**: Middle ground, recommended starting point for tuning.

### Architectural Drawing Noise Challenge
Grid lines, dimension text, hatching patterns, and title blocks can trigger false detections, requiring careful threshold adjustment.

### Drawing Type Considerations
- **Clean vector PDFs (CAD exports)**: Can use lower thresholds (0.4-0.5) due to minimal noise.
- **Scanned blueprints or raster PDFs**: Need higher thresholds (0.6-0.7) to filter scanning artifacts and noise.

### Tuning Workflow
1. Start with `confidence_threshold=0.5` in `config.yaml`.
2. Run detection on sample pages: `python main.py detect --input sample.pdf --output results/`.
3. Review outputs for false positives (e.g., text as windows, grid lines as walls): increase to 0.6-0.7.
4. If missing obvious doors/windows: decrease to 0.3-0.4.
5. Use debug mode to visually assess: `python main.py detect --input sample.pdf --output results/ --debug`.
6. Adjust iteratively until satisfactory balance.

### Class-Specific Considerations
- **Windows**: May need lower threshold (more variation in appearance).
- **Walls**: May need higher threshold (similar to structural lines).

## Debug Mode Usage

### Debug Flag Functionality
The `--debug` flag enables visualization outputs for parameter tuning:
- **Individual tile detections**: Saved to `data/output/tiles/tile_*.jpg`, showing per-tile detections before stitching.
- **Full-page stitched views**: Saved to `data/output/debug/page_X_stitched.jpg`, showing final results after NMS.
- **Tile boundary visualization**: Saved to `data/output/debug/page_X_tiles.jpg`, showing grid and overlap regions.

### Step-by-Step Debugging Workflow
1. Run detection with debug: `python main.py detect --input plan.pdf --output results/ --debug`.
2. Review `page_X_stitched.jpg` for overall detection quality and coverage.
3. Examine `tiles/` folder for tile boundaries and split objects crossing edges.
4. Check `page_X_tiles.jpg` for tile grid layout and overlap zones.
5. If split detections at boundaries: increase `overlap_ratio` in `config.yaml` from 0.2 to 0.3.
6. If duplicate detections: NMS may need adjustment (code-level, not config).
7. Re-run and iterate until satisfactory.

### Visual Inspection Checklist
- Split objects (partial bounding boxes at edges).
- Duplicate detections (same object multiple times).
- Missed objects in overlap zones.
- False positives in tile corners.

## Troubleshooting Scenarios

### Double Counting Issue
**Problem**: Same door detected twice with slightly different bounding boxes.  
**Solution**: Increase `overlap_ratio` from 0.2 to 0.3 to ensure objects fully visible in one tile, or check NMS IoU threshold in code (currently 0.5).

### Missed Small Objects
**Problem**: Tiny door symbols (<50 pixels) not detected.  
**Solution**: Decrease `confidence_threshold` from 0.5 to 0.3-0.4, increase DPI from 300 to 400 for finer detail, consider 512x512 tiles for denser coverage.

### False Positives
**Problem**: Grid lines as walls, dimension text as windows, hatching as objects.  
**Solution**: Increase `confidence_threshold` from 0.5 to 0.6-0.7 to filter noise, check training data quality and augmentation, retrain model with better negative examples.

### Split Objects at Boundaries
**Problem**: Doors partially visible at tile edges causing incomplete detections.  
**Solution**: Increase `overlap_ratio` to 0.3, verify boundaries with debug visualization, ensure objects appear completely in at least one tile.

### Memory Errors with Large Drawings
**Problem**: Auto-downscale threshold exceeded for A0+ drawings.  
**Solution**: System handles automatically (square root DPI adjustment); if issues persist, reduce `max_megapixels` from 200 to 150 in `config.yaml`.

### Performance Too Slow
**Problem**: Processing takes minutes per page.  
**Solution**: Reduce DPI from 300 to 200, reduce `overlap_ratio` from 0.3 to 0.2, use 1024x1024 tiles for fewer tiles.
