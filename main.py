import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import numpy as np


capture = cv2.VideoCapture(0)

previousTime = 0
currentTime = 0

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

while capture.isOpened():
    ret, frame = capture.read()

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(frame))

    recognition_result = recognizer.recognize(image)
    # top_gesture = recognition_result.gestures[0][0]
    hand_landmarks = recognition_result.hand_landmarks
    cv2.imshow("Facial and Hand Landmarks", mp.Image.numpy_view(image))
    # results.append((top_gesture, hand_landmarks))
# eqfae