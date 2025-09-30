import os
import random
import cv2
import shutil
import numpy as np
import glob
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
    """Convert CCPD dataset to YOLO format."""
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
path: {output_dir}  """
    
    yaml_path = Path(output_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset.yaml at {yaml_path}")


def create_test_video(output_path: str = 'test_video.mp4', duration: int = 5, fps: int = 10) -> str:
   
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = 640, 480
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    
    total_frames = duration * fps
    print(f"Creating test video: {output_path} ({duration}s, {fps} FPS)")
    
    for frame_num in range(total_frames):
        # Create background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50
        
        # Moving car simulation
        car_x = 50 + (frame_num * 8) % (width - 200)
        cv2.rectangle(frame, (car_x, 200), (car_x + 150, 280), (80, 80, 80), -1)
        
        # License plate on car
        plate_x = car_x + 40
        plate_y = 235
        cv2.rectangle(frame, (plate_x, plate_y), (plate_x + 70, plate_y + 20), (255, 255, 255), -1)
        
        # Alternate between different plate numbers for testing
        plate_texts = ["ABC123", "XYZ789", "DEF456"]
        current_text = plate_texts[(frame_num // 10) % len(plate_texts)]
        
        cv2.putText(frame, current_text, (plate_x + 2, plate_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created: {output_path}")
    return output_path


def get_video_properties(video_path: str) -> Optional[dict]:
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None
    
    properties = {
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration_sec': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(cap.get(cv2.CAP_PROP_FPS), 1)
    }
    
    cap.release()
    
    print(f"Video Properties: {video_path}")
    print(f"  Resolution: {properties['width']}x{properties['height']}")
    print(f"  FPS: {properties['fps']}, Duration: {properties['duration_sec']:.1f}s")
    print(f"  Total Frames: {properties['total_frames']}")
    
    return properties


def extract_frames(video_path: str, output_dir: str, frame_interval: int = 30) -> List[str]:
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return []
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    extracted_frames = []
    
    print(f"Extracting frames from {video_path} (every {frame_interval} frames)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Save frame
            frame_filename = f"frame_{frame_count:06d}.jpg"
            frame_path = output_path / frame_filename
            cv2.imwrite(str(frame_path), frame)
            extracted_frames.append(str(frame_path))
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames, extracted {len(extracted_frames)} frames")
    
    cap.release()
    print(f"Extracted {len(extracted_frames)} frames to {output_dir}")
    
    return extracted_frames


def create_video_from_frames(frame_dir: str, output_path: str, fps: int = 10, pattern: str = "*.jpg") -> str:
    
    frame_path = Path(frame_dir)
    frame_files = sorted(glob.glob(str(frame_path / pattern)))
    
    if not frame_files:
        print(f"No frames found in {frame_dir} matching pattern {pattern}")
        return ""
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Cannot read first frame: {frame_files[0]}")
        return ""
    
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    
    print(f"Creating video from {len(frame_files)} frames...")
    
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        if frame is not None:
            out.write(frame)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(frame_files)} frames")
    
    out.release()
    print(f"Video created: {output_path}")
    
    return output_path


def test_video_utilities():
    """Test video utility functions"""
    print("=== Testing Video Utilities ===")
    
    # Test video creation
    test_video = create_test_video("sample_test.mp4", duration=3, fps=10)
    
    # Test video properties
    props = get_video_properties(test_video)
    
    # Test frame extraction
    if props:
        extract_frames(test_video, "extracted_frames", frame_interval=5)
        print("Frame extraction complete!")
    
    print("Video utilities testing complete!")


if __name__ == "__main__":
    print("ALPR Utilities")
    print("1. Convert CCPD to YOLO format")
    print("2. Test video utilities")
    
    choice = input("Choose option (1 or 2): ")
    
    if choice == "1":
        # Your existing CCPD conversion
        ccpd_directory = "D:/Project_Computer_vision/ALPR/data/raw"
        output_directory = "D:/Project_Computer_vision/ALPR/data/annotations"
        
        convert_ccpd_to_yolo(
            ccpd_dir=ccpd_directory,
            output_dir=output_directory,
            train_ratio=0.8,  
            val_ratio=0.1     
        )
    
    elif choice == "2":
        # New video utilities testing
        test_video_utilities()
    
    else:
        print("Invalid choice")
