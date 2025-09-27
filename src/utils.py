import os
import random
import cv2
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

def ccpd_to_yolo_bbox(x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert CCPD bounding box format to YOLO format.
    
    CCPD: (x1, y1, x2, y2) - absolute coordinates
    YOLO: (x_center, y_center, width, height) - normalized coordinates
    """
    # Calculate center point
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    
    # Calculate width and height
    width = x2 - x1
    height = y2 - y1
    
    # Normalize coordinates
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm

def parse_ccpd_filename(filename: str) -> Optional[Tuple[int, int, int, int]]:
    """Extract bounding box coordinates from CCPD filename."""
    try:
        parts = Path(filename).stem.split('-')
        if len(parts) < 3:
            return None
            
        bbox_str = parts[2]
        coords = bbox_str.split('_')
        
        x1, y1 = map(int, coords[0].split('&'))
        x2, y2 = map(int, coords[1].split('&'))
        
        return x1, y1, x2, y2
    except (ValueError, IndexError):
        return None

def create_dataset_structure(output_dir: str):
    """Create YOLO dataset directory structure."""
    base_path = Path(output_dir)
    
    # Create main directories
    for split in ['train', 'val', 'test']:
        (base_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (base_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print(f"Created dataset structure in {output_dir}")

def split_dataset(image_files: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict:
    """Split dataset into train/validation/test sets."""
    random.shuffle(image_files)
    
    total_files = len(image_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    splits = {
        'train': image_files[:train_count],
        'val': image_files[train_count:train_count + val_count],
        'test': image_files[train_count + val_count:]
    }
    
    return splits

def convert_ccpd_to_yolo(ccpd_dir: str, output_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Convert CCPD dataset to YOLO format.
    
    Args:
        ccpd_dir: Path to CCPD dataset directory
        output_dir: Path to output YOLO dataset directory  
        train_ratio: Ratio of training data (default: 0.8)
        val_ratio: Ratio of validation data (default: 0.1)
    """
    ccpd_path = Path(ccpd_dir)
    output_path = Path(output_dir)
    
    # Find all CCPD images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(ccpd_path.rglob(ext)))
    
    print(f"Found {len(image_files)} CCPD images")
    
    if not image_files:
        print("No images found in CCPD directory")
        return
    
    # Create output directory structure
    create_dataset_structure(output_dir)
    
    # Split dataset
    dataset_splits = split_dataset([str(f) for f in image_files], train_ratio, val_ratio)
    
    # Process each split
    converted_count = 0
    failed_count = 0
    
    for split_name, file_list in dataset_splits.items():
        print(f"\nProcessing {split_name} split: {len(file_list)} images")
        
        images_dir = output_path / split_name / 'images'
        labels_dir = output_path / split_name / 'labels'
        
        for img_path in file_list:
            img_path = Path(img_path)
            
            # Parse CCPD filename for bounding box
            bbox_coords = parse_ccpd_filename(img_path.name)
            if not bbox_coords:
                failed_count += 1
                continue
            
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                failed_count += 1
                continue
                
            img_height, img_width = img.shape[:2]
            
            # Convert to YOLO format
            x1, y1, x2, y2 = bbox_coords
            yolo_bbox = ccpd_to_yolo_bbox(x1, y1, x2, y2, img_width, img_height)
            
            # Copy image file
            new_img_name = f"ccpd_{converted_count:06d}.jpg"
            shutil.copy2(str(img_path), str(images_dir / new_img_name))
            
            # Create YOLO annotation file
            label_file = labels_dir / f"ccpd_{converted_count:06d}.txt"
            with open(label_file, 'w') as f:
                # Class 0 for license_plate, followed by normalized bbox coordinates
                f.write(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
            
            converted_count += 1
            
            if converted_count % 100 == 0:
                print(f"Converted {converted_count} images...")
    
    print(f"\n=== Conversion Complete ===")
    print(f"Successfully converted: {converted_count} images")
    print(f"Failed conversions: {failed_count} images")
    print(f"Train: {len(dataset_splits['train'])} images")
    print(f"Val: {len(dataset_splits['val'])} images") 
    print(f"Test: {len(dataset_splits['test'])} images")
    
    
    create_dataset_yaml(output_dir)

def create_dataset_yaml(output_dir: str):
    """Create dataset.yaml configuration file for YOLOv8."""
    yaml_content = f"""# CCPD License Plate Dataset Configuration
path: {output_dir}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images      # val images (relative to 'path') 
test: test/images    # test images (relative to 'path')

# Classes
nc: 1  # number of classes
names: ['license_plate']  # class names
"""
    
    yaml_path = Path(output_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset.yaml at {yaml_path}")


if __name__ == "__main__":
    # CCPD dataset to YOLO format
    ccpd_directory = "D:/Project_Computer_vision/ALPR/data/raw"
    output_directory = "D:/Project_Computer_vision/ALPR/data/annotations"
    
    convert_ccpd_to_yolo(
        ccpd_dir=ccpd_directory,
        output_dir=output_directory,
        train_ratio=0.8,  
        val_ratio=0.1     
    )
