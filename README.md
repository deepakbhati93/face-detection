# 🤖 Face Detection System

A modern, high-accuracy AI Face Detection web application built with **Python**, **Streamlit**, and **OpenCV**. 

This system supports dual face-detection engines: **YuNet ONNX Deep Learning** for near 100% accuracy on complex group photos, and **Haar Cascade** with interactive parameter tuning.

---

## ✨ Features

- **🤖 Deep Learning Engine (YuNet)**: Powered by OpenCV's official `FaceDetectorYN` ONNX model. Accurately detects small, tilted, shadowed, or occluded faces in crowded photos with confidence score badges.
- **🧠 Haar Cascade Engine**: Pre-trained frontal face XML classifier with built-in **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to enhance dark or shadowed areas.
- **⚙️ Interactive Parameter Tuning**: Adjust Confidence Thresholds, Scale Factor, and Min Neighbors in real-time via the Streamlit sidebar.
- **📤 Image Upload & Visualization**: Drag and drop `PNG`, `JPG`, or `JPEG` images to compare original and annotated face-bounding-box outputs side by side.
- **📊 Real-Time Metrics**: Instant status metrics displaying total face count and active model details.
- **📁 File Storage Helper**: Automatically records uploaded originals to `uploads/` and annotated images to `outputs/`.

---

## 📁 Project Structure

```text
Face_Detection_System/
├── app.py                              # Streamlit web application frontend
├── train_model.py                      # Backend AI face detection module (YuNet & Haar)
├── requirements.txt                    # Project Python dependencies
├── README.md                           # Project documentation
├── .gitignore                          # Git ignore rules for clean repo tracking
├── .streamlit/
│   └── config.toml                     # Streamlit layout & theme styling
├── assets/
│   └── logo.png                        # Application logo
├── models/
│   ├── face_detection_yunet_2023mar.onnx   # YuNet Deep Learning ONNX model (~230 KB)
│   └── haarcascade_frontalface_default.xml # OpenCV Haar Cascade classifier XML
├── uploads/
│   └── .gitkeep                        # Uploaded images directory
└── outputs/
    └── .gitkeep                        # Annotated output images directory
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Face_Detection_System.git
cd Face_Detection_System
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Launch the Streamlit web server:

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 🎮 Usage Guide

1. **Select Detection Engine**: Use the sidebar radio buttons to choose between:
   - **Deep Learning (YuNet - Highly Accurate)** *(Recommended for group photos)*
   - **Haar Cascade (Classic OpenCV)**
2. **Adjust Sensitivity**:
   - For YuNet, adjust the **Confidence Threshold** slider (default: `0.65`).
   - For Haar Cascade, tune **Scale Factor** (e.g. `1.05`), **Min Neighbors** (e.g. `4`), and toggle **CLAHE Contrast Enhancement**.
3. **Upload an Image**: Drag and drop any group or single-face image into the upload box.
4. **View Results**: Compare original and detected images side-by-side with real-time face metrics.

---

## 📦 Technologies Used

- **Language**: Python
- **Frontend Framework**: [Streamlit](https://streamlit.io/)
- **Computer Vision**: [OpenCV](https://opencv.org/) (`cv2`)
- **AI Models**: YuNet Deep Learning ONNX (`cv2.FaceDetectorYN`) & OpenCV Haar Cascade
- **Data Processing**: NumPy & Pillow

---

## 👨‍💻 Developer

Developed by **Dhruv Kumar**
