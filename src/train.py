import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from model import build_model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from preprocess import (
    get_dataset_paths,
    load_tf_train_datasets,
    load_tf_test_dataset,
    preprocess_dataset,
    print_preprocessing_summary,
    display_processed_sample,
)
from utils import plot_training_history

# ==========================================================
# Training Configuration
# ==========================================================

EPOCHS = 20

PATIENCE = 5

LEARNING_RATE_FACTOR = 0.2

MIN_LEARNING_RATE = 1e-6

MODEL_SAVE_PATH = "models/trained_models/best_model.keras"


def create_callbacks():
    """
    Creates the callbacks used during model training.

    Returns
    -------
    list
        List of TensorFlow callbacks.
    """

    checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=LEARNING_RATE_FACTOR,
        patience=2,
        min_lr=MIN_LEARNING_RATE,
        verbose=1,
    )

    return [
        checkpoint,
        early_stopping,
        reduce_lr,
    ]



def main():
    """
    Main execution block that coordinates the complete DeepFER pipeline.
    """

    # ==========================================================
    # 1. Get Dataset Paths
    # ==========================================================

    train_dir, test_dir = get_dataset_paths()

    # ==========================================================
    # 2. Load TensorFlow Datasets
    # ==========================================================

    train_ds, val_ds = load_tf_train_datasets(train_dir)
    test_ds = load_tf_test_dataset(test_dir)

    # Save class names before preprocessing
    class_names = train_ds.class_names

    # ==========================================================
    # 3. Apply Data Preprocessing
    # ==========================================================

    train_ds_processed = preprocess_dataset(train_ds)
    val_ds_processed = preprocess_dataset(val_ds)
    test_ds_processed = preprocess_dataset(test_ds)

    # ==========================================================
    # 4. Print Preprocessing Summary
    # ==========================================================

    print_preprocessing_summary(
        train_ds_processed,
        val_ds_processed,
        test_ds_processed,
    )

    # ==========================================================
    # 5. Display Sample Image
    # ==========================================================

    display_processed_sample(train_ds_processed, class_names)

    # ==========================================================
    # 6. Build the Model
    # ==========================================================

    model = build_model()

    print("\n==========================")
    print("MODEL SUMMARY")
    print("==========================\n")

    model.summary()

    # ==========================================================
    # 7. Create Training Callbacks
    # ==========================================================

    callbacks = create_callbacks()

    print("\n==========================")
    print("TRAINING CALLBACKS")
    print("==========================")

    print(f"\nEpochs: {EPOCHS}")
    print(f"Early Stopping Patience: {PATIENCE}")
    print(f"Learning Rate Factor: {LEARNING_RATE_FACTOR}")
    print(f"Minimum Learning Rate: {MIN_LEARNING_RATE}")
    print(f"Best Model Path: {MODEL_SAVE_PATH}")

    print("\nCallbacks Loaded:")
    print("- ModelCheckpoint")
    print("- EarlyStopping")
    print("- ReduceLROnPlateau")

    print("\n==========================")

    # ==========================================================
    # 8. Train the Model
    # ==========================================================

    print("\n==========================")
    print("MODEL TRAINING")
    print("==========================")
    print("\nTraining has started...\n")

    history = model.fit(
        train_ds_processed,
        validation_data=val_ds_processed,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # ==========================================================
    # 9. Training Summary
    # ==========================================================

    print("\n==========================")
    print("TRAINING COMPLETED")
    print("==========================")

    print(f"\nTotal Epochs Completed: {len(history.history['loss'])}")
    print(f"\nFinal Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"\nFinal Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

    print("\nBest Model Saved To:")
    print(MODEL_SAVE_PATH)

    print("\n==========================")

    # Save the training graphs
    plot_training_history(history)
    
if __name__ == "__main__":
    main()
