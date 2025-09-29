import easyocr
import cv2
import numpy as np
import re
from typing import List, Tuple, Optional

class LicensePlateOCR:
    
    def __init__(self, languages=['en', 'ch_sim'], gpu=False):
        print("Loading EasyOCR model...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.min_confidence = 0.6
        print("OCR ready!")
    
    def preprocess_image(self, image):
        # Convert to grayscale 
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhance contrast for better character recognition
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Reduce noise while keeping edges sharp
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        filtered = cv2.bilateralFilter(blurred, 9, 75, 75)
        
        return filtered
    
    def extract_text(self, cropped_plate):
        try:
            # Clean up the image first
            processed = self.preprocess_image(cropped_plate)
            
            # Run OCR
            results = self.reader.readtext(processed)
            
            if not results:
                return None, 0.0
            
            # Filter by confidence and clean text
            valid_results = []
            for (bbox, text, confidence) in results:
                if confidence >= self.min_confidence:
                    clean_text = self.clean_text(text)
                    if clean_text:
                        valid_results.append((clean_text, confidence))
            
            if not valid_results:
                return None, 0.0
            
            # Return the best result
            best_text, best_conf = max(valid_results, key=lambda x: x[1])
            return best_text, best_conf
            
        except Exception as e:
            print(f"OCR error: {str(e)}")
            return None, 0.0
    
    def clean_text(self, text):
        # Remove spaces and make uppercase
        cleaned = re.sub(r'\s+', '', text.upper())
        
        # Fix common OCR mistakes for license plates
        if any(char.isdigit() for char in cleaned):
            # Only fix character confusions if we see numbers
            fixes = {'O': '0', 'I': '1', 'S': '5', 'Z': '2'}
            for old, new in fixes.items():
                cleaned = cleaned.replace(old, new)
        
        return cleaned
    
    def batch_extract_text(self, plate_images):
        results = []
        for i, image in enumerate(plate_images):
            print(f"Processing plate {i+1}/{len(plate_images)}...")
            text, confidence = self.extract_text(image)
            results.append((text, confidence))
            
            if text:
                print(f"Found: '{text}' ({confidence:.2f})")
            else:
                print("No text found")
        
        return results
    
    def set_confidence_threshold(self, threshold):
        # Keep threshold between 0 and 1
        self.min_confidence = max(0.0, min(1.0, threshold))
        print(f"Confidence threshold: {self.min_confidence}")

def test_ocr():
    print("Testing OCR module...")
    ocr = LicensePlateOCR(gpu=False)
    print("OCR module ready for use!")
    print("Usage: text, conf = ocr.extract_text(plate_image)")
    return ocr

if __name__ == "__main__":
    test_ocr()
