from ultralytics import YOLO
import cv2
import os
from pathlib import Path

class LicensePlateDetector:
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        
    def train_model(self, dataset_path="../data/annotations/dataset.yaml", epochs=30):
        print("Starting license plate training...")
        
        model = YOLO('yolov8n.pt')
        
        # Train the model
        results = model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=416,
            batch=2,
            lr0=0.01,
            patience=10,
            cache=False,
            device='cpu',
            workers=0,
            project="../models/license_plate_detection",
            name="yolov8_ccpd_optimized",
            exist_ok=True,
            amp=False,
            rect=False,
            mosaic=0.5
        )
        
        self.model = model
        print(f"Training done. Model saved to: {results.save_dir}")
        return results
    
    def detect_plates(self, image_path, confidence=0.25):
        if not self.model:
            print("No model loaded yet")
            return None
        
        return self.model(image_path, conf=confidence, device='cpu')
    
    def load_model(self, model_path):
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model file not found: {model_path}")

def train_license_plate_model():
    detector = LicensePlateDetector()
    
    print("Training with these settings:")
    print("- Image size: 416x416")
    print("- Batch size: 2") 
    print("- No caching (saves RAM)")
    print("- 30 epochs (~2-3 hours)")
    
    results = detector.train_model()
    print(f"Best model saved: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_license_plate_model()
