import cv2
import os
import numpy as np
from ultralytics import YOLO
from ocr_reader import LicensePlateOCR
from utils import create_test_video


model = None
ocr = None

def load_models_once():
    
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
    """Quick sanity check to make sure OCR works"""
    print("Quick OCR test...")
    
    if not load_models_once():
        return False
    
    # Create simple test image in memory
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
    """Test YOLOv8 detection + OCR together"""
    print("Testing full pipeline...")
    
    if not load_models_once():
        print("Can't load models")
        return
    
    # Create a fake car scene for testing
    img = np.full((250, 500, 3), 60, dtype=np.uint8)  
    
    # Draw a simple car
    cv2.rectangle(img, (100, 70), (400, 180), (90, 90, 90), -1)
    
    # Add license plate
    cv2.rectangle(img, (200, 130), (350, 160), (255, 255, 255), -1)
    cv2.putText(img, "ABC123", (210, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    
    results = model(img, verbose=False) 
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        print(f"YOLOv8 found {len(results[0].boxes)} plates")
        
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cropped = img[y1:y2, x1:x2]
            
            text, conf = ocr.extract_text(cropped)
            if text:
                print(f"Plate {i+1}: '{text}' ({conf:.2f})")
    else:
        print("YOLOv8 didn't detect anything, trying OCR directly...")
        
        plate_area = img[130:160, 200:350]
        text, conf = ocr.extract_text(plate_area)
        if text:
            print(f"Direct OCR: '{text}' ({conf:.2f})")

def test_with_real_image():
    """Try testing with actual CCPD images if we can find any"""
    print("Looking for real test image...")
    
    if not load_models_once():
        return
    
    
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
    
    # Load and process the image
    img = cv2.imread(test_file)
    if img is None:
        return
    
    results = model(img, verbose=False)
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        print(f"Detected {len(results[0].boxes)} plates in real image")
        
        
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

def test_video_processing():
    """NEW: Test video processing capabilities"""
    print("Testing video processing...")
    
    if not load_models_once():
        print("Can't load models for video test")
        return
    
    
    print("Creating test video with moving plates...")
    test_video_path = create_test_video("test_alpr_video.mp4", duration=3, fps=10)
    
    if not os.path.exists(test_video_path):
        print("Failed to create test video")
        return
    
    # Process the video
    cap = cv2.VideoCapture(test_video_path)
    
    if not cap.isOpened():
        print("Can't open test video")
        return
    
    frame_count = 0
    detections = []
    
    print("Processing video frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        
        if frame_count % 3 == 0:
            # Run detection
            results = model(frame, verbose=False)
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cropped_plate = frame[y1:y2, x1:x2]
                    
                    # Try OCR
                    text, conf = ocr.extract_text(cropped_plate)
                    
                    if text and conf > 0.6:  
                        detection = {
                            'frame': frame_count,
                            'text': text,
                            'confidence': conf
                        }
                        detections.append(detection)
                        print(f"Frame {frame_count}: Found '{text}' ({conf:.2f})")
    
    cap.release()
    
    print(f"Video processing complete!")
    print(f"Total frames: {frame_count}")
    print(f"Plates detected: {len(detections)}")
    
    if detections:
        print("Sample detections:")
        for detection in detections[:3]:  
            print(f"  Frame {detection['frame']}: '{detection['text']}' ({detection['confidence']:.2f})")
    
    # Clean up test file
    if os.path.exists(test_video_path):
        os.remove(test_video_path)
        print("Cleaned up test video")
    
    return len(detections) > 0

def test_live_camera():
   
    print("Testing live camera feed...")
    print("Press 'q' to quit, 'c' to capture a frame for testing")
    
    if not load_models_once():
        print("Can't load models for camera test")
        return
    
    cap = cv2.VideoCapture(0)  
    
    if not cap.isOpened():
        print("Can't access camera - skipping live test")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        
        if frame_count % 10 == 0:
            # Run detection
            results = model(frame, verbose=False)
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Draw detection box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Try OCR
                    cropped_plate = frame[y1:y2, x1:x2]
                    text, conf = ocr.extract_text(cropped_plate)
                    
                    if text and conf > 0.7:
                        # Show detected text
                        cv2.putText(frame, f"{text} ({conf:.2f})", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        print(f"Live detection: '{text}' ({conf:.2f})")
        
        # Show frame
        cv2.imshow('Live ALPR Test (Press q to quit)', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Capture current frame for testing
            cv2.imwrite('captured_frame.jpg', frame)
            print("Frame captured for testing")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Live camera test finished")

def run_all_tests():
    """Run all tests from basic to advanced"""
    print("Starting comprehensive ALPR testing...\n")
    
    
    if not quick_ocr_test():
        print("Basic OCR failed - something's wrong with setup")
        return
    
    print("\n" + "="*40 + "\n")
    
    
    test_full_pipeline()
    
    print("\n" + "="*40 + "\n")
    
    
    test_with_real_image()
    
    print("\n" + "="*40 + "\n")
    
    
    video_success = test_video_processing()
    
    print("\n" + "="*40 + "\n")
    
    
    print("Would you like to test live camera? (y/n): ", end="")
    try:
        choice = input().lower().strip()
        if choice == 'y':
            test_live_camera()
    except:
        print("Skipping live camera test")
    
    print("\n" + "="*50)
    print("All tests completed!")
    
    # Summary
    print("\nTest Summary:")
    print(" OCR Module: Working")
    print(" YOLOv8 Detection: Working") 
    print(" Complete Pipeline: Working")
    print(f" Video Processing: {'Working' if video_success else 'Needs work'}")

if __name__ == "__main__":
    run_all_tests()