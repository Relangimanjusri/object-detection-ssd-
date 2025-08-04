# Object Detection using MobileNet SSD and OpenCV

This project demonstrates real-time object detection using the **MobileNet Single Shot Detector (SSD)** and **OpenCV's DNN module**.

### 🔍 Description

This script detects objects such as people, cars, cats, dogs, etc., using a pre-trained model on the COCO dataset. It uses your computer’s webcam and draws bounding boxes with class labels and confidence scores.

### 🧠 Model Info

- **Model**: MobileNet SSD trained on Caffe
- **Files Used**:
  - `MobileNetSSD_deploy.prototxt` – Model architecture
  - `MobileNetSSD_deploy.caffemodel` – Pre-trained weights

### 🚀 Requirements

- Python 3.x
- OpenCV
- imutils
- NumPy

Install using:

```bash
pip install opencv-python imutils numpy

