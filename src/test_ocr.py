import cv2
import os
import numpy as np
from ultralytics import YOLO
from ocr_reader import LicensePlateOCR

# Global variables to avoid reloading models multiple times
model = None
ocr = None

def load_models_once():
    """Load models once and reuse them - saves energy"""
    global model, ocr
    
    if ocr is None:
        print("Loading OCR...")
        ocr = LicensePlateOCR(gpu=False)
        print("OCR ready!")
    
    if model is None:
        print("Loading YOLOv8...")
        model_paths = [
            '../models/license_plate_detection/yolov8_ccpd_optimized/weights/last.pt',
            '../models/license_plate_detection/yolov8_ccpd_optimized/weights/best.pt'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = YOLO(path)
                    print(f"Model loaded from {os.path.basename(path)}")
                    break
                except:
                    continue
    
    return model is not None and ocr is not None

def quick_ocr_test():
    """Quick test without creating unnecessary images"""
    print("Quick OCR test...")
    
    if not load_models_once():
        return False
    
    # Simple white image with black text
    test_img = np.full((60, 200, 3), 255, dtype=np.uint8)
    cv2.putText(test_img, "TEST123", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    text, conf = ocr.extract_text(test_img)
    if text:
        print(f"OCR works: '{text}' ({conf:.2f})")
        return True
    else:
        print("OCR test failed")
        return False

def test_full_pipeline():
    """Test complete detection + OCR pipeline"""
    print("Testing full pipeline...")
    
    if not load_models_once():
        print("Can't load models")
        return
    
    # Create test scene efficiently
    img = np.full((250, 500, 3), 60, dtype=np.uint8)  # Gray background
    
    # Simple car shape
    cv2.rectangle(img, (100, 70), (400, 180), (90, 90, 90), -1)
    
    # License plate area
    cv2.rectangle(img, (200, 130), (350, 160), (255, 255, 255), -1)
    cv2.putText(img, "ABC123", (210, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Run detection
    results = model(img, verbose=False) 
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        print(f"Found {len(results[0].boxes)} detections")
        
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cropped = img[y1:y2, x1:x2]
            
            text, conf = ocr.extract_text(cropped)
            if text:
                print(f"Plate {i+1}: '{text}' ({conf:.2f})")
    else:
        print("No plates detected, testing OCR directly...")
        plate_area = img[130:160, 200:350]
        text, conf = ocr.extract_text(plate_area)
        if text:
            print(f"Direct OCR: '{text}' ({conf:.2f})")

def test_with_real_image():
    """Test with real CCPD image if available"""
    print("Looking for real test image...")
    
    if not load_models_once():
        return
    
    # Check common data locations
    data_dirs = ['../data/raw/', '../data/']
    test_file = None
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            
            for file in os.listdir(data_dir):
                if file.lower().endswith('.jpg'):
                    test_file = os.path.join(data_dir, file)
                    break
            if test_file:
                break
    
    if not test_file:
        print("No real images found, skipping")
        return
    
    print(f"Testing with: {os.path.basename(test_file)}")
    
    # Load and process
    img = cv2.imread(test_file)
    if img is None:
        return
    
    results = model(img, verbose=False)
    
    if results[0].boxes is not None:
        print(f"Detected {len(results[0].boxes)} plates")
        
        
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cropped = img[y1:y2, x1:x2]
        
        text, conf = ocr.extract_text(cropped)
        if text:
            print(f"Real plate: '{text}' ({conf:.2f})")
        else:
            print("Couldn't read the plate")
    else:
        print("No plates found in real image")

def run_all_tests():
    """Run tests in order of complexity"""
    print("Starting ALPR tests...\n")
    
    
    if not quick_ocr_test():
        print("Basic test failed - stopping")
        return
    
    print("\n" + "="*30 + "\n")
    
    
    test_full_pipeline()
    
    print("\n" + "="*30 + "\n")
    
    
    test_with_real_image()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    run_all_tests()
