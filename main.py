import cv2
import numpy as np
import time

# Load pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Assign a unique color for each class
np.random.seed(42)  # Ensures the same colors every time
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype="uint8")

# Load the video
video = cv2.VideoCapture("Test2.mp4")

# FPS calculation variables
prev_time = 0

while True:
    ret, frame = video.read()
    if not ret:
        break  # Exit loop if video ends

    frame = cv2.resize(frame, (640, 480))
    (h, w) = frame.shape[:2]

    # Prepare input blob for the network
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.3:  # Lower confidence threshold to detect smaller objects
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box
            color = [int(c) for c in COLORS[class_id]]
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
            cv2.rectangle(frame, (startX, startY - 25), (endX, startY), color, -1)

            # Label text
            label = f"{CLASSES[class_id]}: {confidence:.2f}"
            cv2.putText(frame, label, (startX + 5, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Display FPS on frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


