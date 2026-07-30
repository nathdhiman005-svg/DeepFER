import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_trained_model
from preprocess import preprocess_single_image

# ==========================================================
# Configuration
# ==========================================================
MODEL_PATH = "models/trained_models/best_model.keras"

IMAGE_PATH = "assets/demo/angry.jpg"

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def load_image(image_path):
    """
    Loads an image from disk.
    """

    print("\n==========================")
    print("LOADING IMAGE")
    print("==========================")

    print(f"Loading image from: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    print("\nImage Loaded Successfully.")

    print("\n==========================")

    return image


def predict_emotion(model, image):
    """
    Predicts the emotion present in the image.
    """

    print("\n==========================")
    print("PREDICTING EMOTION")
    print("==========================")

    prediction = model.predict(
        image,
        verbose=0,
    )[0]

    for emotion, prob in zip(CLASS_NAMES, prediction):
        print(f"{emotion:<10}: {prob*100:.2f}%")

    predicted_index = np.argmax(
        prediction,
    )

    confidence = prediction[predicted_index] * 100

    predicted_emotion = CLASS_NAMES[
        predicted_index
    ]

    print("\nPrediction Completed.")

    print("\n==========================")

    return predicted_emotion, confidence

def display_prediction(emotion, confidence):
    """
    Displays the prediction result.
    """

    print("\n==========================")
    print("PREDICTION RESULT")
    print("==========================")

    print(f"\nPredicted Emotion : {emotion.title()}")

    print(f"Confidence        : {confidence:.2f}%")

    print("\n==========================")

def show_prediction(image, emotion, confidence):
    """
    Displays the predicted image.
    """

    plt.figure(figsize=(6, 6))

    plt.imshow(image)

    plt.title(
        f"{emotion.title()} ({confidence:.2f}%)"
    )

    plt.axis("off")

    plt.show()

def main():

    # ==========================================================
    # 1. Load Trained Model
    # ==========================================================

    model = load_trained_model()

    # ==========================================================
    # 2. Load Image
    # ==========================================================

    image = load_image(
        IMAGE_PATH,
    )

    # ==========================================================
    # 3. Preprocess Image
    # ==========================================================

    processed_image = preprocess_single_image(
        image,
    )

    # ==========================================================
    # 4. Predict Emotion
    # ==========================================================

    emotion, confidence = predict_emotion(
        model,
        processed_image,
    )

    # ==========================================================
    # 5. Display Prediction
    # ==========================================================

    display_prediction(
        emotion,
        confidence,
    )

    show_prediction(
        image,
        emotion,
        confidence,
    )


if __name__ == "__main__":
    main()