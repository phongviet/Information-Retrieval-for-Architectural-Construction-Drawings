"""
Accuracy Assessment Script for Detection Evaluation

This module provides utilities for evaluating detection accuracy by comparing predictions
against ground truth annotations. It calculates precision, recall, F1 metrics per class,
generates confusion matrices, and produces comprehensive accuracy reports.

Usage:
    python tests/test_accuracy.py --predictions results.json --ground-truth annotations.json --output report.txt

Or programmatically:
    predictions = load_detection_results('results.json')
    ground_truth = load_ground_truth('annotations.json')
    metrics = calculate_metrics(predictions, ground_truth)
    cm = generate_confusion_matrix(predictions, ground_truth, class_names)
    generate_accuracy_report(metrics, cm, class_names, 'report.txt')
"""

import json
import os
import glob
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import logging
from src.inference.utils import calculate_iou

logger = logging.getLogger(__name__)


def load_ground_truth(annotations_path: str, image_dimensions: Dict[str, Tuple[int, int]] = None) -> Dict[int, List[Dict[str, Any]]]:
    """
    Load ground truth annotations from YOLO label files or JSON format.

    Supports:
    - YOLO format: Directory with page_*.txt files containing normalized coordinates
    - JSON format: Single file with {"page_num": [{"bbox": [x1,y1,x2,y2], "class_id": int, "class_name": str}, ...]}

    Args:
        annotations_path: Path to annotations (directory for YOLO, file for JSON)
        image_dimensions: Optional dict mapping page names to (width, height) for YOLO conversion

    Returns:
        Dictionary: {page_num: [{'bbox': (x1,y1,x2,y2), 'class_id': int, 'class_name': str}, ...]}

    Raises:
        ValueError: If format is unsupported or files malformed
    """
    if os.path.isdir(annotations_path):
        # YOLO format
        return _load_yolo_ground_truth(annotations_path, image_dimensions)
    elif annotations_path.endswith('.json'):
        # JSON format
        return _load_json_ground_truth(annotations_path)
    else:
        raise ValueError(f"Unsupported annotation format: {annotations_path}")


def _load_yolo_ground_truth(annotations_dir: str, image_dimensions: Dict[str, Tuple[int, int]] = None) -> Dict[int, List[Dict[str, Any]]]:
    """Load ground truth from YOLO label files."""
    ground_truth = {}
    label_files = glob.glob(os.path.join(annotations_dir, "*.txt"))

    for label_file in label_files:
        page_name = os.path.splitext(os.path.basename(label_file))[0]
        page_num = int(page_name.split('_')[1]) if '_' in page_name else 0

        if image_dimensions and page_name in image_dimensions:
            img_width, img_height = image_dimensions[page_name]
        else:
            # Assume default or skip conversion
            img_width, img_height = 1000, 1000  # Placeholder

        annotations = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert normalized to absolute
                x1 = (x_center - width / 2) * img_width
                y1 = (y_center - height / 2) * img_height
                x2 = (x_center + width / 2) * img_width
                y2 = (y_center + height / 2) * img_height

                # Class names mapping (assuming standard)
                class_names = {0: 'door', 1: 'object', 2: 'wall', 3: 'window'}
                class_name = class_names.get(class_id, f'class_{class_id}')

                annotations.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_id': class_id,
                    'class_name': class_name
                })

        ground_truth[page_num] = annotations

    return ground_truth


def _load_json_ground_truth(json_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """Load ground truth from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    ground_truth = {}
    for page_str, annotations in data.items():
        page_num = int(page_str)
        ground_truth[page_num] = annotations

    return ground_truth


def load_detection_results(results_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Load detection results from JSON format.

    Args:
        results_path: Path to JSON file with detection results in same format as ground truth

    Returns:
        Dictionary: {page_num: [{'bbox': (x1,y1,x2,y2), 'class_id': int, 'class_name': str, 'confidence': float}, ...]}
    """
    if results_path.endswith('.json'):
        with open(results_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported results format: {results_path}")


def calculate_metrics(predictions: Dict[int, List[Dict[str, Any]]],
                     ground_truth: Dict[int, List[Dict[str, Any]]],
                     iou_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Calculate precision, recall, and F1 metrics per class and overall.

    Matching algorithm:
    - For each prediction, find best matching ground truth with IoU > threshold and same class
    - Matched pairs are True Positives
    - Unmatched predictions are False Positives
    - Unmatched ground truth are False Negatives

    Args:
        predictions: {page_num: [{'bbox': (x1,y1,x2,y2), 'class_id': int, 'class_name': str}, ...]}
        ground_truth: Same format as predictions
        iou_threshold: Minimum IoU for matching

    Returns:
        {'per_class': {class_name: {'precision': float, 'recall': float, 'f1': float, 'support': int}},
         'overall': {'precision': float, 'recall': float, 'f1': float}}
    """
    # Initialize counters per class
    class_counters = {}  # class_name: {'tp': int, 'fp': int, 'fn': int}

    all_pages = set(predictions.keys()) | set(ground_truth.keys())

    for page_num in all_pages:
        pred_list = predictions.get(page_num, [])
        gt_list = ground_truth.get(page_num, [])

        # Track matched ground truth
        matched_gt = set()

        for pred in pred_list:
            pred_bbox = pred['bbox']
            pred_class = pred['class_name']
            best_iou = 0.0
            best_gt_idx = -1

            for idx, gt in enumerate(gt_list):
                if idx in matched_gt:
                    continue
                if gt['class_name'] != pred_class:
                    continue
                iou = calculate_iou(pred_bbox, gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= iou_threshold:
                # True Positive
                if pred_class not in class_counters:
                    class_counters[pred_class] = {'tp': 0, 'fp': 0, 'fn': 0}
                class_counters[pred_class]['tp'] += 1
                matched_gt.add(best_gt_idx)
            else:
                # False Positive
                if pred_class not in class_counters:
                    class_counters[pred_class] = {'tp': 0, 'fp': 0, 'fn': 0}
                class_counters[pred_class]['fp'] += 1

        # False Negatives: unmatched ground truth
        for idx, gt in enumerate(gt_list):
            if idx not in matched_gt:
                gt_class = gt['class_name']
                if gt_class not in class_counters:
                    class_counters[gt_class] = {'tp': 0, 'fp': 0, 'fn': 0}
                class_counters[gt_class]['fn'] += 1

    # Calculate metrics
    per_class = {}
    total_precision = []
    total_recall = []
    total_f1 = []

    for class_name, counts in class_counters.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }

        total_precision.append(precision)
        total_recall.append(recall)
        total_f1.append(f1)

    overall = {
        'precision': np.mean(total_precision) if total_precision else 0.0,
        'recall': np.mean(total_recall) if total_recall else 0.0,
        'f1': np.mean(total_f1) if total_f1 else 0.0
    }

    return {'per_class': per_class, 'overall': overall}


def generate_confusion_matrix(predictions: Dict[int, List[Dict[str, Any]]],
                             ground_truth: Dict[int, List[Dict[str, Any]]],
                             class_names: List[str],
                             iou_threshold: float = 0.5) -> np.ndarray:
    """
    Generate confusion matrix showing classification patterns.

    Matrix[i][j] = count of true class i predicted as class j
    Diagonal elements are correct classifications.

    Args:
        predictions: Detection results
        ground_truth: Ground truth annotations
        class_names: List of class names in order
        iou_threshold: IoU threshold for matching

    Returns:
        NxN numpy array where N = len(class_names)
    """
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    all_pages = set(predictions.keys()) | set(ground_truth.keys())

    for page_num in all_pages:
        pred_list = predictions.get(page_num, [])
        gt_list = ground_truth.get(page_num, [])

        matched_gt = set()

        for pred in pred_list:
            pred_bbox = pred['bbox']
            pred_class = pred['class_name']
            best_iou = 0.0
            best_gt_idx = -1
            best_gt_class = None

            for idx, gt in enumerate(gt_list):
                if idx in matched_gt:
                    continue
                iou = calculate_iou(pred_bbox, gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
                    best_gt_class = gt['class_name']

            if best_iou >= iou_threshold and best_gt_class in class_to_idx and pred_class in class_to_idx:
                true_idx = class_to_idx[best_gt_class]
                pred_idx = class_to_idx[pred_class]
                cm[true_idx][pred_idx] += 1
                matched_gt.add(best_gt_idx)

    return cm


def generate_accuracy_report(metrics: Dict[str, Any],
                           confusion_matrix: np.ndarray,
                           class_names: List[str],
                           output_path: str) -> None:
    """
    Generate comprehensive accuracy report.

    Args:
        metrics: From calculate_metrics
        confusion_matrix: From generate_confusion_matrix
        class_names: List of class names
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("Accuracy Assessment Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Overall statistics
        f.write("Overall Statistics\n")
        f.write("-" * 20 + "\n")
        total_pages = len(set(metrics.get('per_class', {}).keys()))  # Approximate
        total_gt = sum(cls['support'] for cls in metrics['per_class'].values())
        total_pred = sum(cls['tp'] + cls['fp'] for cls in metrics['per_class'].values())
        f.write(f"Total pages evaluated: {total_pages}\n")
        f.write(f"Total ground truth objects: {total_gt}\n")
        f.write(f"Total predicted objects: {total_pred}\n")
        f.write(f"Overall Precision: {metrics['overall']['precision']:.3f}\n")
        f.write(f"Overall Recall: {metrics['overall']['recall']:.3f}\n")
        f.write(f"Overall F1-Score: {metrics['overall']['f1']:.3f}\n\n")

        # Per-class metrics
        f.write("Per-Class Metrics\n")
        f.write("-" * 20 + "\n")
        f.write("Class      | Precision | Recall | F1-Score | Support\n")
        f.write("-" * 50 + "\n")
        for class_name in class_names:
            if class_name in metrics['per_class']:
                cls = metrics['per_class'][class_name]
                f.write(f"{class_name:<10} | {cls['precision']:.3f}     | {cls['recall']:.3f}  | {cls['f1']:.3f}    | {cls['support']}\n")
            else:
                f.write(f"{class_name:<10} | 0.000     | 0.000  | 0.000    | 0\n")
        f.write("\n")

        # Confusion Matrix
        f.write("Confusion Matrix\n")
        f.write("-" * 20 + "\n")
        header = "True\\Pred | " + " | ".join(f"{name[:5]:>5}" for name in class_names)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for i, true_class in enumerate(class_names):
            row = f"{true_class[:5]:>10} | " + " | ".join(f"{confusion_matrix[i][j]:>5}" for j in range(len(class_names)))
            f.write(row + "\n")
        f.write("\n")

        # Interpretation
        f.write("Interpretation Notes\n")
        f.write("-" * 20 + "\n")
        f.write("Precision: Of all detections for this class, what percentage were correct?\n")
        f.write("Recall: Of all true instances, what percentage were detected?\n")
        f.write("F1-Score: Harmonic mean balancing precision and recall\n\n")

        f.write("Common Issues\n")
        f.write("-" * 15 + "\n")
        f.write("- Low precision: Many false positives (noise, misclassifications)\n")
        f.write("- Low recall: Missed detections (high confidence threshold, small objects)\n")
        f.write("- Confusion between classes: Check training data or augmentation\n")

    logger.info(f"Accuracy report saved: {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate detection accuracy')
    parser.add_argument('--predictions', required=True, help='Path to detection results JSON')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth annotations')
    parser.add_argument('--output', default='accuracy_report.txt', help='Output report path')
    args = parser.parse_args()

    predictions = load_detection_results(args.predictions)
    ground_truth = load_ground_truth(args.ground_truth)
    metrics = calculate_metrics(predictions, ground_truth)
    cm = generate_confusion_matrix(predictions, ground_truth, ['door', 'object', 'wall', 'window'])
    generate_accuracy_report(metrics, cm, ['door', 'object', 'wall', 'window'], args.output)
