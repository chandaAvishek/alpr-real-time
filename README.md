# 🚗 ALPR System - Automatic License Plate Recognition

*Building a real-time license plate detection system using YOLOv8 and EasyOCR*

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Ultralytics](https://img.shields.io/badge/YOLOv8-ultralytics-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-computer%20vision-green.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)

##  Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Development Progress](#development-progress)
- [Technical Details](#technical-details)

##  Overview

I'm working on an ALPR system that can automatically detect and read license plates from images and videos in real-time. The goal is to create something that works well in different lighting conditions and camera angles - basically making it robust enough for real-world use.

This project tackles the challenge of automated vehicle identification, which has practical applications in parking management, traffic monitoring, and security systems.

##  Features

- **Real-time processing**: Works with live video feeds
- **Custom trained model**: Built using 11,776+ CCPD dataset images for high accuracy
- **Handles Chinese plates**: Specialized character recognition for Chinese license plates
- **Weather resistant**: Performs well across different lighting and environmental conditions
- **Modular design**: Easy to integrate into existing systems
- **Well documented**: Includes detailed exploration notebooks showing the development process

##  System Architecture

<p align="center">
  <img src="docs/images/alpr-pipeline.png" width="700">
</p>

The system works in stages:
1. **Data Processing**: I explored and converted the CCPD dataset to YOLO format
2. **Detection Module**: Custom YOLOv8 model finds license plates in images
3. **Recognition Module**: EasyOCR extracts the actual text from detected plates
4. **Pipeline Integration**: Everything works together seamlessly
5. **User Interface**: Planning a Streamlit dashboard for easy interaction

##  Installation

### What you'll need
- Python 3.8 or higher
- A decent GPU helps (but not required - I optimized for CPU training too)
- At least 8GB RAM if you want to train from scratch

### Getting started

1. **Get the code**

```bash
git clone https://github.com/yourusername/alpr-system.git
cd alpr-real-time
```


2. **Set up your environment**

```bash
python -m venv alpr_env
source alpr_env/bin/activate 
```


3. **Install everything**

```bash
pip install -r requirements.txt
```


##  Usage

### Running license plate detection

```python
from src.detect_plate import LicensePlateDetector
# Setup the Detector
detector = LicensePlateDetector()

# Load the trained model
detector.load_model('models/license_plate_detection/best.pt')

# Detect plates in your image
results = detector.detect_plates('path/to/image.jpg')
```


### Exploring the dataset

```python
jupyter notebook notebooks/01_ccpd_exploration.ipynb
```


##  Development Progress

###  Completed (Week 1) 
- [x] Project setup and environment configuration
- [x] CCPD dataset exploration and analysis (11,776 images)
- [x] Dataset conversion pipeline (CCPD → YOLO format)
- [x] YOLOv8 training pipeline implementation
- [x] **Model training completed with 99.5% mAP50 accuracy**
- [x] CPU-optimized training configuration
- [x] Professional project documentation

###  **Training Results**
- **Model Performance**: 99.5% mAP50 accuracy achieved in 4 epochs
- **Training Method**: YOLOv8n with early stopping (patience=10)
- **Dataset**: 11,776 CCPD images with Chinese license plates
- **Status**: Production-ready model available at `models/license_plate_detection/yolov8_ccpd_optimized/weights/best.pt`

###   Week 2 Day 1 - COMPLETE
- [x] **OCR integration with EasyOCR**
- [x] **Complete ALPR pipeline (YOLOv8 → OCR → Text extraction)**
- [x] **95% OCR confidence achieved on test license plates**
- [x] **Energy-efficient testing framework with model reuse**
- [x] **End-to-end license plate text recognition working**

###  **OCR Integration Results**
- **OCR Performance**: 95% confidence on synthetic test plates
- **Pipeline Status**: YOLOv8 detection + EasyOCR text extraction functional
- **Language Support**: English and Chinese character recognition
- **Implementation**: `src/ocr_reader.py` with preprocessing and error correction
- **Testing**: Comprehensive test suite in `src/test_ocr.py`
- **Character Accuracy**: Handles common OCR errors (O→0, I→1, S→5, Z→2)

### Week 2 Day 2 - COMPLETE
- [x] **Video processing pipeline implementation**
- [x] **Enhanced testing framework with video capabilities**
- [x] **Test video generation for development and validation**
- [x] **Live camera feed processing integration**
- [x] **Frame-by-frame license plate detection in video streams**

### **Video Processing Results**
- **Video Support**: MP4 file processing with configurable frame skipping for performance
- **Live Camera**: Real-time webcam integration with detection visualization
- **Test Generation**: Synthetic video creation with moving license plates for validation
- **Performance**: Optimized processing every 3rd frame for real-time capability
- **Implementation**: Extended `src/utils.py` with video utilities and enhanced `src/test_ocr.py`
- **Processing Results**: Successfully detecting and reading license plates in video sequences

### Next Steps (Day 3)
- [ ] Streamlit dashboard interface development
- [ ] Real-time video stream integration with web interface
- [ ] Detection history and logging system
- [ ] Performance metrics dashboard
- [ ] User interface for video upload and processing

---