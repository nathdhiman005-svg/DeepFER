"""
webcam.py

Real-time facial emotion recognition using the trained DeepFER model.
"""

import cv2
import numpy as np

from utils import load_trained_model
from preprocess import preprocess_single_image

# ==========================================================
# Configuration
# ==========================================================

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ==========================================================
# Face Detector
# ==========================================================

def load_face_detector():
    """
    Loads the Haar Cascade face detector.
    """

    print("\n==========================")
    print("LOADING FACE DETECTOR")
    print("==========================")

    detector = cv2.CascadeClassifier(CASCADE_PATH)

    if detector.empty():
        raise RuntimeError("Failed to load Haar Cascade classifier.")

    print("\nFace Detector Loaded Successfully.")
    print("\n==========================")

    return detector


# ==========================================================
# Emotion Prediction
# ==========================================================

def predict_emotion(model, face_image):
    """
    Predicts the emotion of a cropped face.
    """

    processed_image = preprocess_single_image(face_image)

    prediction = model.predict(
        processed_image,
        verbose=0,
    )[0]

    predicted_index = np.argmax(prediction)

    emotion = CLASS_NAMES[predicted_index]

    confidence = prediction[predicted_index] * 100

    return emotion, confidence


# ==========================================================
# Webcam
# ==========================================================

def start_webcam(model, detector):
    """
    Starts the webcam and performs real-time emotion recognition.
    """

    print("\n==========================")
    print("STARTING WEBCAM")
    print("==========================")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Unable to open webcam.")

    while True:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60),
        )

        for (x, y, w, h) in faces:

            face = frame[
                y:y+h,
                x:x+w,
            ]

            face_rgb = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB,
            )

            emotion, confidence = predict_emotion(
                model,
                face_rgb,
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            label = f"{emotion.title()} ({confidence:.1f}%)"

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow(
            "DeepFER - Real-Time Emotion Recognition",
            frame,
        )

        key = cv2.waitKey(1)

        if key & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


# ==========================================================
# Main
# ==========================================================

def main():

    model = load_trained_model()

    detector = load_face_detector()

    start_webcam(
        model,
        detector,
    )


if __name__ == "__main__":
    main()