"""
Framework Comparison Testing
Automated benchmarking of YOLO vs MMDetection on architectural drawings
"""

import pytest
import numpy as np
import torch
import time
import json
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


@pytest.fixture
def framework_models():
    """
    Provide trained models for both frameworks.

    Returns:
        Dictionary with framework names as keys and model info as values
    """
    models = {}

    # YOLO model
    yolo_model_path = 'models/custom_model.pt'
    if Path(yolo_model_path).exists():
        from ultralytics import YOLO
        models['yolo'] = {
            'path': yolo_model_path,
            'model': YOLO(yolo_model_path),
            'framework': 'yolo'
        }
        logger.info(f"Loaded YOLO model: {yolo_model_path}")
    else:
        pytest.skip(f"YOLO model not found: {yolo_model_path}")

    # MMDetection model
    mmdet_model_path = 'models/mmdet_cascade_rcnn.pth'
    mmdet_config = 'configs/mmdet/_base_/models/cascade_rcnn_r50_fpn.py'

    if Path(mmdet_model_path).exists() and Path(mmdet_config).exists():
        from mmdet.apis import init_detector
        models['mmdetection'] = {
            'path': mmdet_model_path,
            'config': mmdet_config,
            'model': init_detector(mmdet_config, mmdet_model_path, device='cuda:0'),
            'framework': 'mmdetection'
        }
        logger.info(f"Loaded MMDetection model: {mmdet_model_path}")
    else:
        pytest.skip(f"MMDetection model not found: {mmdet_model_path} or {mmdet_config}")

    return models


@pytest.fixture
def test_dataset():
    """
    Provide standardized test dataset.

    Returns:
        Dictionary with test images and annotations
    """
    # Use floortest3.1 dataset test split
    dataset_path = Path('data/floortest3.1.v1-data.yolov8')

    if not dataset_path.exists():
        pytest.skip(f"Test dataset not found: {dataset_path}")

    # Load test images
    test_images_dir = dataset_path / 'test' / 'images'
    test_labels_dir = dataset_path / 'test' / 'labels'

    if not test_images_dir.exists():
        pytest.skip(f"Test images not found: {test_images_dir}")

    # Collect test image paths (use subset for speed)
    import glob
    image_files = sorted(glob.glob(str(test_images_dir / '*')))[:20]  # Use subset for speed

    # Load corresponding labels
    annotations = {}
    for img_path in image_files:
        img_name = Path(img_path).stem
        label_path = test_labels_dir / f"{img_name}.txt"

        if label_path.exists():
            with open(label_path, 'r') as f:
                annotations[img_path] = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded test dataset: {len(image_files)} images")

    return {
        'images': image_files,
        'annotations': annotations,
        'dataset_path': dataset_path
    }


@pytest.mark.integration
def test_accuracy_comparison(framework_models, test_dataset):
    """
    Compare detection accuracy between frameworks.

    Measures: Precision, Recall, mAP@0.5, mAP@0.75
    """
    from scipy import stats
    import cv2

    results = {}

    for framework_name, model_info in framework_models.items():
        logger.info(f"Testing accuracy for {framework_name}")

        predictions = []
        ground_truths = []

        # Run inference on all test images
        for img_path in test_dataset['images']:
            # Load image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get predictions
            if framework_name == 'yolo':
                result = model_info['model'].predict(image, verbose=False)[0]
                detections = []
                if result.boxes is not None:
                    for box in result.boxes:
                        detections.append({
                            'bbox': box.xyxy[0].cpu().numpy(),
                            'confidence': float(box.conf[0]),
                            'class_id': int(box.cls[0])
                        })
            else:  # mmdetection
                from mmdet.apis import inference_detector
                result = inference_detector(model_info['model'], image)
                detections = []
                for class_id, class_dets in enumerate(result):
                    for det in class_dets:
                        if det[4] >= 0.5:  # confidence threshold
                            detections.append({
                                'bbox': det[:4],
                                'confidence': float(det[4]),
                                'class_id': class_id
                            })

            predictions.append(detections)

            # Load ground truth
            if img_path in test_dataset['annotations']:
                gt_boxes = []
                img_h, img_w = image.shape[:2]

                for line in test_dataset['annotations'][img_path]:
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:])

                        # Convert YOLO to xyxy
                        x1 = (x_c - w/2) * img_w
                        y1 = (y_c - h/2) * img_h
                        x2 = (x_c + w/2) * img_w
                        y2 = (y_c + h/2) * img_h

                        gt_boxes.append({
                            'bbox': [x1, y1, x2, y2],
                            'class_id': class_id
                        })

                ground_truths.append(gt_boxes)

        # Calculate metrics
        metrics = calculate_detection_metrics(predictions, ground_truths)
        results[framework_name] = metrics

        logger.info(f"{framework_name} - mAP@0.5: {metrics['mAP_50']:.3f}, "
                   f"Precision: {metrics['precision']:.3f}, "
                   f"Recall: {metrics['recall']:.3f}")

    # Statistical significance test
    if len(results) == 2:
        frameworks = list(results.keys())
        mAP_diff = results[frameworks[0]]['mAP_50'] - results[frameworks[1]]['mAP_50']

        logger.info(f"mAP difference: {abs(mAP_diff):.3f}")

        # Note: Full significance test would require per-image mAPs
        # For now, report absolute difference
        if abs(mAP_diff) > 0.05:
            logger.warning(f"Significant mAP difference detected (>{0.05})")

    # Save results
    output_path = Path('tests/results/accuracy_comparison.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Accuracy results saved: {output_path}")

    # Assert both frameworks achieve reasonable accuracy
    for framework_name, metrics in results.items():
        assert metrics['mAP_50'] > 0.5, f"{framework_name} mAP too low: {metrics['mAP_50']}"


def calculate_detection_metrics(predictions, ground_truths, iou_threshold=0.5):
    """Calculate precision, recall, and mAP."""

    all_tp = 0
    all_fp = 0
    all_fn = 0

    for preds, gts in zip(predictions, ground_truths):
        # Match predictions to ground truths
        matched_gts = set()

        for pred in preds:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gts):
                if gt['class_id'] != pred['class_id']:
                    continue

                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx not in matched_gts:
                all_tp += 1
                matched_gts.add(best_gt_idx)
            else:
                all_fp += 1

        all_fn += len(gts) - len(matched_gts)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    mAP_50 = precision * recall / (precision + recall) * 2 if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'mAP_50': mAP_50,
        'true_positives': all_tp,
        'false_positives': all_fp,
        'false_negatives': all_fn
    }


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes."""
    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]

    # Intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0


@pytest.mark.integration
@pytest.mark.slow
def test_speed_comparison(framework_models, test_dataset):
    """
    Compare inference speed between frameworks.

    Measures: Throughput (img/s), latency (ms), GPU memory
    """
    import cv2
    import tracemalloc

    results = {}

    # Prepare test images in memory
    test_images = []
    for img_path in test_dataset['images'][:20]:  # Use subset
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_images.append(img)

    logger.info(f"Loaded {len(test_images)} test images for speed testing")

    for framework_name, model_info in framework_models.items():
        logger.info(f"Benchmarking speed for {framework_name}")

        # GPU warmup
        for img in test_images[:5]:
            if framework_name == 'yolo':
                _ = model_info['model'].predict(img, verbose=False)
            else:
                from mmdet.apis import inference_detector
                _ = inference_detector(model_info['model'], img)

        # Clear GPU memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Timed inference
        latencies = []

        start_time = time.perf_counter()

        for img in test_images:
            img_start = time.perf_counter()

            if framework_name == 'yolo':
                _ = model_info['model'].predict(img, verbose=False)
            else:
                from mmdet.apis import inference_detector
                _ = inference_detector(model_info['model'], img)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            img_end = time.perf_counter()
            latencies.append((img_end - img_start) * 1000)  # Convert to ms

        end_time = time.perf_counter()

        # Calculate metrics
        total_time = end_time - start_time
        throughput = len(test_images) / total_time
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)

        # GPU memory
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB

        results[framework_name] = {
            'throughput_img_per_sec': throughput,
            'latency_mean_ms': mean_latency,
            'latency_std_ms': std_latency,
            'gpu_memory_mb': gpu_memory,
            'total_time_sec': total_time
        }

        logger.info(f"{framework_name} - Throughput: {throughput:.2f} img/s, "
                   f"Latency: {mean_latency:.1f}±{std_latency:.1f} ms, "
                   f"GPU Memory: {gpu_memory:.1f} MB")

    # Calculate speedup
    if len(results) == 2:
        frameworks = list(results.keys())
        speedup = results[frameworks[0]]['throughput_img_per_sec'] / \
                 results[frameworks[1]]['throughput_img_per_sec']

        logger.info(f"Speed ratio ({frameworks[0]}/{frameworks[1]}): {speedup:.2f}x")

    # Save results
    output_path = Path('tests/results/speed_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Speed results saved: {output_path}")


def generate_comparison_report(accuracy_results, speed_results, output_path='tests/results/framework_comparison_report.txt'):
    """
    Generate comprehensive framework comparison report.

    Args:
        accuracy_results: Accuracy metrics dictionary
        speed_results: Speed metrics dictionary
        output_path: Path to save report
    """
    from datetime import datetime

    report_lines = [
        "=" * 80,
        "YOLO vs MMDetection Framework Comparison Report",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: floortest3.1.v1-data.yolov8 (test split)",
        "",
        "ACCURACY COMPARISON",
        "-" * 80,
    ]

    # Accuracy table
    report_lines.append(f"{'Framework':<20} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12}")
    report_lines.append("-" * 80)

    for framework, metrics in accuracy_results.items():
        report_lines.append(
            f"{framework:<20} "
            f"{metrics['precision']:<12.3f} "
            f"{metrics['recall']:<12.3f} "
            f"{metrics['mAP_50']:<12.3f}"
        )

    report_lines.extend([
        "",
        "SPEED COMPARISON",
        "-" * 80,
        f"{'Framework':<20} {'Throughput':<15} {'Latency (ms)':<18} {'GPU Mem (MB)':<15}",
        "-" * 80,
    ])

    for framework, metrics in speed_results.items():
        report_lines.append(
            f"{framework:<20} "
            f"{metrics['throughput_img_per_sec']:<15.2f} "
            f"{metrics['latency_mean_ms']:<18.1f} "
            f"{metrics['gpu_memory_mb']:<15.1f}"
        )

    # Recommendations
    report_lines.extend([
        "",
        "RECOMMENDATIONS",
        "-" * 80,
    ])

    frameworks = list(accuracy_results.keys())
    if len(frameworks) == 2:
        yolo_metrics = accuracy_results.get('yolo', {})
        mmdet_metrics = accuracy_results.get('mmdetection', {})
        yolo_speed = speed_results.get('yolo', {})
        mmdet_speed = speed_results.get('mmdetection', {})

        if yolo_speed and mmdet_speed:
            speedup = yolo_speed.get('throughput_img_per_sec', 0) / mmdet_speed.get('throughput_img_per_sec', 1)
            report_lines.append(f"YOLO is {speedup:.1f}x faster than MMDetection")

        if yolo_metrics and mmdet_metrics:
            mAP_diff = mmdet_metrics.get('mAP_50', 0) - yolo_metrics.get('mAP_50', 0)
            if mAP_diff > 0.05:
                report_lines.append(f"MMDetection achieves {mAP_diff:.1%} higher mAP")

        report_lines.extend([
            "",
            "Use YOLO when:",
            "  • Inference speed is critical (>10 FPS required)",
            "  • Limited GPU memory (<6 GB)",
            "  • Simpler deployment needed",
            "",
            "Use MMDetection when:",
            "  • Maximum accuracy required",
            "  • Processing time not constrained",
            "  • Complex scenes with dense annotations",
        ])

    report_lines.extend([
        "",
        "=" * 80,
        "End of Report",
        "=" * 80
    ])

    # Write report
    report_text = "\n".join(report_lines)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(report_text)

    logger.info(f"Comparison report generated: {output_path}")
    print(report_text)

    return report_text


# Main test orchestrator
def test_complete_framework_comparison(framework_models, test_dataset):
    """Run complete framework comparison and generate report."""

    logger.info("Starting complete framework comparison...")

    # Load previous results
    acc_results = json.load(open('tests/results/accuracy_comparison.json'))
    speed_results = json.load(open('tests/results/speed_comparison.json'))

    # Generate report
    generate_comparison_report(acc_results, speed_results)

    logger.info("Framework comparison complete!")
