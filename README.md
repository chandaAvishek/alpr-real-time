# 🚗 Real-Time ALPR (Automatic License Plate Recognition)

An **end-to-end real-time ALPR system** that detects vehicles, extracts license plates, applies OCR to recognize plate numbers, and visualizes results live through a dashboard.

<p align="center">
  <img src="docs/images/alpr-pipeline.png" width="700">
</p>

---

## ✨ Features
- 🚙 **Vehicle Detection:** YOLOv8-based detector for cars and motorcycles  
- 🏷️ **License Plate Detection:** Fine-tuned YOLOv8 model for plate localization  
- 🔤 **OCR Recognition:** EasyOCR pipeline to read license numbers  
- 📊 **Dashboard UI:** Streamlit interface to display live results and logs  
- 🎥 **Real-Time Inference:** Works on live webcam or video streams  
- 📁 **Logging:** Saves detections with timestamps and exports to CSV  

---

## 🗂 Dataset
This project uses the **[CCPD2020 Dataset](https://github.com/detectRecog/CCPD)**, a large-scale benchmark for license plate detection and recognition.  
- 📸 **200K+ images** of vehicles from real-world scenarios  
- 🎯 Bounding box annotations for license plates  
- 🌙 Covers day/night, different weather conditions  

---

## 🛠️ Tech Stack
- **Deep Learning:** [YOLOv8](https://github.com/ultralytics/ultralytics) for vehicle & plate detection  
- **OCR:** [EasyOCR](https://github.com/JaidedAI/EasyOCR) for plate text recognition  
- **Computer Vision:** OpenCV for preprocessing & visualization  
- **Dashboard:** Streamlit for real-time monitoring  
- **Data Science:** Pandas, Matplotlib, Seaborn for analysis  

---

## 🚀 Quick Start

### 1️⃣ Clone the repository
```bash
git clone https://github.com/chandaAvishek/alpr-real-time.git
cd alpr-real-time
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### Download the dataset
```bash
python scripts/download_ccpd.py
```

## 3️⃣ Run data exploration
```bash
jupyter notebook notebooks/data_exploration.ipynb

```

## 4️⃣ Launch real time ALPR
```bash
python src/main.py
```

## Results
Results (Coming Soon)

✅ Vehicle detection examples

✅ License plate detection visualization

✅ OCR accuracy metrics

## Future Improvements

 Multi-camera support

 Support for EU/German plates

 Plate tracking across frames

 Deploy as Docker + API for production


## Author

Avishek Chanda
🎓 Master's Student in Information & Communication Engineering, TU Darmstadt
📌 Interested in Computer Vision, AI for Traffic Management, and Edge AI Applications

