
import numpy as np
import imutils
import cv2
import time

# ---------------------------------------------
# File paths for model files
prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"
# Confidence threshold
confThresh = 0.2
# ---------------------------------------------
# Class labels MobileNet SSD was trained on
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
]
# Random colors for bounding boxes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# ---------------------------------------------
# Load the model from disk
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print("[INFO] Model loaded successfully!")
# ---------------------------------------------
# Initialize video capture (webcam)
print("[INFO] Starting video stream...")
cv = cv2.VideoCapture(0)  # 0 = default webcam
time.sleep(2.0)  # Warm-up
while True:
    ret, frame = cv.read()
    if not ret or frame is None:
        continue

    # Resize frame for faster processing
    frame = imutils.resize(frame, width=800)
    (h, w) = frame.shape[:2]

    # Convert frame to a blob for DNN module
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,  # scale factor
        (300, 300),  # size
        127.5  # mean subtraction
    )

    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confThresh:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)

            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Show output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key
        break

# Clean up
cv.release()
cv2.destroyAllWindows()
