from tensorflow.keras import losses
from utils import load_trained_model

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    # ConfusionMatrixDisplay,
)

from preprocess import (
    get_dataset_paths,
    load_tf_test_dataset,
    preprocess_dataset,
)

# ==========================================================
# Evaluation Configuration
# ==========================================================

OUTPUT_DIR = Path("outputs/evaluation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
EVALUATION_SUMMARY_PATH = OUTPUT_DIR / "evaluation_summary.txt"




def evaluate_model(model, test_ds):
    """
    Evaluates the trained model on the test dataset.
    """
    print("\n==========================")
    print("MODEL EVALUATION")
    print("==========================")

    loss, accuracy = model.evaluate(
        test_ds,
        verbose=1,
    )

    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    print("\n==========================")

    return loss, accuracy

def generate_predictions(model, test_ds):
    """
    Generates predictions and collects the true labels.
    """

    print("\n==========================")
    print("GENERATING PREDICTIONS")
    print("==========================")

    y_true = []
    y_pred = []

    for images, labels in test_ds:

        predictions = model.predict(
            images,
            verbose=0,
        )

        predicted_labels = np.argmax(
            predictions,
            axis=1,
        )

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"\nTotal Test Samples: {len(y_true)}")

    print("\nPredictions Generated Successfully.")

    print("\n==========================")

    return y_true, y_pred

def generate_classification_report(y_true, y_pred, class_names):
    """
    Generates and saves the classification report.
    """

    print("\n==========================")
    print("CLASSIFICATION REPORT")
    print("==========================")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )
   
    with open(CLASSIFICATION_REPORT_PATH, "w") as file:
        file.write(report)

    print(report)

    print(f"\nClassification Report Saved To:")
    print(CLASSIFICATION_REPORT_PATH)

    print("\n==========================")

def generate_confusion_matrix(y_true, y_pred, class_names):
    """
    Generates and saves the confusion matrix.
    """

    print("\n==========================")
    print("CONFUSION MATRIX")
    print("==========================")

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()


    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print(f"\nConfusion Matrix Saved To:")
    print(CONFUSION_MATRIX_PATH)

    print("\n==========================")

def save_evaluation_summary(loss, accuracy):
    """
    Saves the overall evaluation summary.
    """

    print("\n==========================")
    print("SAVING EVALUATION SUMMARY")
    print("==========================")


    with open(EVALUATION_SUMMARY_PATH, "w") as file:

        file.write("DeepFER Model Evaluation\n")
        file.write("=============================\n\n")

        file.write(f"Test Loss: {loss:.4f}\n")
        file.write(f"Test Accuracy: {accuracy:.4f}\n")

    print(f"\nEvaluation Summary Saved To:")
    print(EVALUATION_SUMMARY_PATH)

    print("\n==========================")


def main():

    # ==========================================================
    # 1. Load Test Dataset
    # ==========================================================

    _, test_dir = get_dataset_paths()

    test_ds = load_tf_test_dataset(test_dir)

    class_names = test_ds.class_names

    test_ds_processed = preprocess_dataset(test_ds)

    print("\n==========================")
    print("TEST DATASET")
    print("==========================")

    print(f"\nClasses: {len(class_names)}")
    print(f"Class Names: {class_names}")
    print(f"Batches: {len(test_ds_processed)}")

    print("\n==========================")

    # ==========================================================
    # 2. Load Trained Model
    # ==========================================================

    model = load_trained_model()

    # ==========================================================
    # 3. Evaluate Model
    # ==========================================================

    loss, accuracy = evaluate_model(
        model,
        test_ds_processed,
    )

    # ==========================================================
    # 4. Generate Predictions
    # ==========================================================

    y_true, y_pred = generate_predictions(
        model,
        test_ds_processed,
    )

    # ==========================================================
    # 5. Generate Evaluation Reports
    # ==========================================================

    generate_classification_report(
        y_true,
        y_pred,
        class_names,
    )

    generate_confusion_matrix(
        y_true,
        y_pred,
        class_names,
    )

    save_evaluation_summary(
        loss,
        accuracy,
    )

if __name__ == "__main__":
    main()