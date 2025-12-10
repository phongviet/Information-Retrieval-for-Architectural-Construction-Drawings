import yaml
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def validate_dataset(data_yaml_path: str):
    """
    Validate YOLOv8 dataset structure integrity.
    
    Args:
        data_yaml_path: Path to data.yaml file
    """
    data_dir = Path(data_yaml_path).parent
    
    # Load data.yaml
    try:
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"data.yaml not found: {data_yaml_path}")
        return
    
    logging.info(f"Loaded data.yaml from {data_yaml_path}")
    
    # Validate class configuration
    nc = data.get('nc')
    names = data.get('names')
    
    if nc != 4:
        logging.error(f"Expected nc=4, got {nc}")
    else:
        logging.info("Number of classes validated: nc=4")
    
    expected_names = ['door', 'object', 'wall', 'window']
    if names != expected_names:
        logging.error(f"Expected names={expected_names}, got {names}")
    else:
        logging.info(f"Class names validated: {names}")
    
    # Validate splits
    splits = ['train', 'val', 'test']
    for split in splits:
        img_path_rel = data.get(split)
        if not img_path_rel:
            logging.error(f"No {split} path specified in data.yaml")
            continue
        
        # Resolve relative path
        img_path = (data_dir / img_path_rel).resolve()
        if not img_path.exists():
            logging.error(f"{split} images directory does not exist: {img_path}")
            continue
        
        logging.info(f"{split} images directory exists: {img_path}")
        
        # Check labels directory (parallel to images)
        labels_path = img_path.parent / 'labels'
        if not labels_path.exists():
            logging.error(f"{split} labels directory does not exist: {labels_path}")
            continue
        
        logging.info(f"{split} labels directory exists: {labels_path}")
        
        # Check file pairing
        img_files = list(img_path.glob('*.jpg'))
        img_basenames = {f.stem for f in img_files}
        
        label_files = list(labels_path.glob('*.txt'))
        label_basenames = {f.stem for f in label_files}
        
        matched = img_basenames & label_basenames
        img_only = img_basenames - label_basenames
        label_only = label_basenames - img_basenames
        
        if img_only:
            logging.error(f"{split}: Images without corresponding labels ({len(img_only)} files): {sorted(list(img_only)[:5])}{'...' if len(img_only) > 5 else ''}")
        
        if label_only:
            logging.error(f"{split}: Labels without corresponding images ({len(label_only)} files): {sorted(list(label_only)[:5])}{'...' if len(label_only) > 5 else ''}")
        
        logging.info(f"{split}: {len(matched)} matched pairs, {len(img_files)} total images, {len(label_files)} total labels")
    
    logging.info("Dataset validation completed")

if __name__ == "__main__":
    # Default path for the floortest3.1 dataset
    data_yaml = "data/floortest3.1.v1-data.yolov8/data.yaml"
    validate_dataset(data_yaml)
