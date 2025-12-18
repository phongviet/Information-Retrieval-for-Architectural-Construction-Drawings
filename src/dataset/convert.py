import os
import json
import shutil
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_ROOT = "..\..\data\cubicasa5k"
OUTPUT_DIR = "..\..\data\cubicasa5k_coco"

# Set this to True if you want to re-save images to remove libpng warnings.
# False = Faster (copy files directly).
# True  = Slower (reads and re-writes images), but fixes "iCCP" warnings.
CLEAN_IMAGES = True

CATEGORIES = [
    {"id": 1, "name": "Wall"},
    {"id": 2, "name": "Window"},
    {"id": 3, "name": "Door"},
    {"id": 4, "name": "Stairs"},
]
CAT_NAME_TO_ID = {cat["name"]: cat["id"] for cat in CATEGORIES}

SVG_KEYWORD_MAPPING = {
    "wall": "Wall", "window": "Window", "door": "Door", "stairs": "Stairs", "stair": "Stairs"
}


def parse_svg_poly(svg_path):
    if not os.path.exists(svg_path): return []
    try:
        tree = ET.parse(svg_path)
    except ET.ParseError:
        return []

    root = tree.getroot()
    annotations = []

    # helper to check if a specific node has a class/id we care about
    def get_category(element):
        ident = element.get('id', '').lower()
        cls = element.get('class', '').lower()
        for keyword, cat_name in SVG_KEYWORD_MAPPING.items():
            if keyword in ident or keyword in cls:
                return CAT_NAME_TO_ID[cat_name]
        return None

    # REVISED TRAVERSAL: Pass 'parent_cat_id' down to children
    def traverse(node, parent_cat_id=None):
        # 1. Does this specific node have a new category? (e.g. <g id="wall">)
        current_cat_id = get_category(node)

        # If no new category found, stick with the parent's category
        if current_cat_id is None:
            current_cat_id = parent_cat_id

        # 2. If we have a valid category (either from this node or inherited), check for geometry
        if current_cat_id is not None:
            points = []
            # Check for Polygon / Polyline
            if node.tag.endswith('polygon') or node.tag.endswith('polyline'):
                points_str = node.get('points')
                if points_str:
                    nums = points_str.replace(',', ' ').split()
                    points = [float(n) for n in nums]
            # Check for Rect
            elif node.tag.endswith('rect'):
                x = float(node.get('x', 0))
                y = float(node.get('y', 0))
                w = float(node.get('width', 0))
                h = float(node.get('height', 0))
                points = [x, y, x + w, y, x + w, y + h, x, y + h]
            # Check for Path (Simple conversion for straight lines only)
            elif node.tag.endswith('path'):
                # Note: Full path parsing is complex. CubiCasa mainly uses polygons.
                # We skip complex paths to avoid errors, as walls are usually polygons.
                pass

            # 3. If valid geometry found, save annotation
            if len(points) >= 6:
                poly_np = np.array(points).reshape(-1, 2)
                x_min, y_min = np.min(poly_np, axis=0)
                x_max, y_max = np.max(poly_np, axis=0)
                w, h = x_max - x_min, y_max - y_min

                if w > 1 and h > 1:
                    annotations.append({
                        'category_id': current_cat_id,
                        'segmentation': [points],
                        'bbox': [x_min, y_min, w, h],
                        'area': w * h,
                        'iscrowd': 0
                    })

        # 4. Recurse into children, passing the determined category
        for child in node:
            traverse(child, current_cat_id)

    traverse(root)
    return annotations

def process_split(split_name, txt_file_path):
    print(f"Processing {split_name}...")

    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

    coco_data = {
        "info": {"description": f"CubiCasa5k {split_name}"},
        "licenses": [], "images": [], "annotations": [], "categories": CATEGORIES
    }

    dest_img_dir = os.path.join(OUTPUT_DIR, f"{split_name}2017")
    os.makedirs(dest_img_dir, exist_ok=True)

    global_ann_id = 1

    for idx, line in tqdm(enumerate(lines), total=len(lines)):
        rel_path = line.strip().lstrip('/').lstrip('\\')
        if not rel_path: continue

        folder_path = os.path.join(DATASET_ROOT, rel_path)

        # Candidate images
        image_candidates = ["F1_original.png", "F1_scaled.png", "original.jpg"]
        src_img_path = None
        for cand in image_candidates:
            p = os.path.join(folder_path, cand)
            if os.path.exists(p):
                src_img_path = p
                break

        src_svg_path = os.path.join(folder_path, "model.svg")

        if not src_img_path: continue

        # New filename
        new_filename = rel_path.replace("/", "_").replace("\\", "_") + ".png"
        dest_img_path = os.path.join(dest_img_dir, new_filename)

        # --- IMAGE PROCESSING ---
        if CLEAN_IMAGES:
            # Slower: Loads image and saves it again (Strips bad metadata/warnings)
            img = cv2.imread(src_img_path)
            if img is None: continue
            cv2.imwrite(dest_img_path, img)
            height, width = img.shape[:2]
        else:
            # Faster: Copies file directly (Keeps warnings)
            shutil.copy2(src_img_path, dest_img_path)
            # We still need to read it to get the size
            img = cv2.imread(src_img_path)
            if img is None: continue
            height, width = img.shape[:2]

        image_id = idx + 1
        coco_data["images"].append({
            "id": image_id, "width": width, "height": height, "file_name": new_filename
        })

        # --- ANNOTATION PROCESSING ---
        # For Test set, we only generate annotations if the SVG exists.
        # Often test sets don't have labels, but if CubiCasa provides them, we use them.
        if os.path.exists(src_svg_path):
            anns = parse_svg_poly(src_svg_path)
            for ann in anns:
                ann['image_id'] = image_id
                ann['id'] = global_ann_id
                coco_data["annotations"].append(ann)
                global_ann_id += 1

    ann_dir = os.path.join(OUTPUT_DIR, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    json_path = os.path.join(ann_dir, f"instances_{split_name}2017.json")

    with open(json_path, 'w') as f:
        json.dump(coco_data, f)

    print(f"Saved {json_path}. Images: {len(coco_data['images'])}, Annotations: {len(coco_data['annotations'])}")


def main():
    splits = [
        ("train", "train.txt"),
        ("val", "val.txt"),
        ("test", "test.txt")  # <--- Added Test Set here
    ]

    for split_name, txt_name in splits:
        txt_path = os.path.join(DATASET_ROOT, txt_name)
        if os.path.exists(txt_path):
            process_split(split_name, txt_path)
        else:
            print(f"Warning: {txt_name} not found (Skipping)")


if __name__ == "__main__":
    main()