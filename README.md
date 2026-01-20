# 🚨 Suspicious Activity Detection in Video Reconnaissance Framework

## 📌 Project Overview

This project presents an **AI-based intelligent surveillance system** capable of detecting suspicious activities in both **live video streams and uploaded videos**. The system integrates **object detection, action recognition, facial analysis, and real-time alerting** into a single framework to enhance public safety and reduce human dependency on manual surveillance.

The application is built using **Computer Vision & Deep Learning** techniques and provides a simple web interface for users to start live monitoring or analyze recorded videos.

---

## 🎯 Key Features

* Real-time **object detection** using YOLOv5
* **Suspicious action recognition** (Fighting, Theft, Robbery, etc.)
* **Face detection** with age, gender, and mask status analysis
* **Live webcam surveillance**
* **Video upload & offline analysis**
* **Instant Telegram alerts** with snapshots and activity details
* GPU-accelerated inference using **CUDA**

---

## 🧠 Algorithms & Models Used

* **YOLOv5** – Object and weapon detection
* **SlowFast (3D-CNN)** – Action recognition from video frames
* **CNN (Keras)** – Mask detection
* **OpenCV DNN (TensorFlow & Caffe)** – Face, age, and gender detection

---

## 🏗️ System Architecture (High Level)

1. Video input (Live webcam / Uploaded video)
2. Frame extraction
3. Object detection using YOLOv5
4. Action recognition using SlowFast 3D-CNN
5. Face, age, gender, and mask analysis
6. Decision logic for suspicious activity
7. Real-time alert via Telegram
8. Output display (popup window / saved video)

---

## 🖥️ Tech Stack

### Software

* Python 3.9+
* OpenCV
* PyTorch
* TensorFlow / Keras
* Flask
* SQLite
* Streamlit (optional UI)
* Telegram Bot API

### Hardware

* NVIDIA GPU (CUDA supported – recommended)
* Webcam / CCTV camera
* Minimum 8 GB RAM

---

## 📦 Pre-trained Models Used

This project uses multiple **pre-trained deep learning models** for different tasks:

| Task               | Model Type              | Files                                                         |
| ------------------ | ----------------------- | ------------------------------------------------------------- |
| Object Detection   | YOLOv5 (PyTorch)        | `yolov5s.pt`                                                  |
| Action Recognition | SlowFast (3D-CNN)       | `model_new.h5`                                                |
| Mask Detection     | CNN (Keras)             | `mask_detector.h5`                                            |
| Face Detection     | TensorFlow (OpenCV DNN) | `opencv_face_detector_uint8.pb`, `opencv_face_detector.pbtxt` |
| Age Detection      | Caffe                   | `age_net.caffemodel`, `age_deploy.prototxt`                   |
| Gender Detection   | Caffe                   | `gender_net.caffemodel`, `gender_deploy.prototxt`             |

> ⚠️ **Important Note**
> Some model files are **not included in this repository** due to GitHub’s file size limitations (25 MB).
>
> Please **download them manually** and place them in the project root directory before running the system.

---

## 📥 Model Download Instructions

You must download the following files manually:

* `yolov5s.pt` → from official YOLOv5 repository
* `model_new.h5` → trained SlowFast action recognition model
* `mask_detector.h5` → trained mask detection model
* Caffe models (`age`, `gender`) → OpenCV DNN models
* Face detector (`.pb`, `.pbtxt`) → OpenCV TensorFlow models

Place all models in the **project root directory**.

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Flask Web App

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

### 3️⃣ Live Surveillance

* Login
* Click **Start Streaming**
* Webcam feed opens in a popup window

### 4️⃣ Video Analysis

* Upload a video file
* Action recognition popup will appear
* Telegram alert sent if suspicious activity is detected

---

## 🧪 Test Video

The `test/` folder contains sample videos used for **action analysis testing**.

---

## 🗄️ Database Management

* Database: **SQLite**
* File: `user_data.db`
* Stores user login and registration details

---

## 🚀 Results

* Accurate detection of suspicious activities
* Real-time alerts reduce response time
* Works efficiently on live and recorded videos
* Achieves higher efficiency compared to traditional CNN-only models

---

## 🔮 Future Enhancements

* CCTV / IP camera (RTSP) integration
* Multi-camera tracking with DeepSORT / ByteTrack
* Crowd behavior & panic detection
* Audio-based threat detection
* Cloud dashboard & analytics
* Mobile app with push notifications

---

## 📌 Project Domain

**Computer Vision & Deep Learning**

---

## 👨‍💻 Author

**Final Year Project – AI-based Surveillance System**
