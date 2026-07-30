import os
import matplotlib.pyplot as plt
import tensorflow as tf


MODEL_PATH = "models/trained_models/best_model.keras"

def plot_training_history(history):
    """
    Plot and save the training and validation accuracy and loss curves.
    """

    # Create the output directory if it doesn't exist
    os.makedirs("outputs/plots", exist_ok=True)

    # ==========================
    # Accuracy Plot
    # ==========================
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        "outputs/plots/training_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ==========================
    # Loss Plot
    # ==========================
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        "outputs/plots/training_loss.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\nTraining graphs saved successfully!")

def load_trained_model():
    """
    Loads the trained DeepFER model from disk.
    """
    print("\n==========================")
    print("LOADING MODEL")
    print("==========================")

    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"\nModel Loaded Successfully:")
    print(MODEL_PATH)

    print("\n==========================")

    return model