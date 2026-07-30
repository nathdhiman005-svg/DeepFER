"""
model.py

Builds and compiles the DeepFER emotion recognition model using
MobileNetV2 and Transfer Learning.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# ==========================================================
# Model Constants
# ==========================================================

IMAGE_SIZE = (224, 224)
NUM_CHANNELS = 3
NUM_CLASSES = 7

LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.30


# ==========================================================
# Build Model
# ==========================================================

def build_model():
    """
    Builds and compiles the DeepFER model using MobileNetV2.

    Architecture:
        Input
            ↓
        MobileNetV2 (ImageNet pretrained)
            ↓
        GlobalAveragePooling2D
            ↓
        Dropout
            ↓
        Dense (128, ReLU)
            ↓
        Dropout
            ↓
        Dense (7, Softmax)

    Returns:
        tf.keras.Model
    """

    # ------------------------------------------------------
    # Load MobileNetV2 without its original classifier
    # ------------------------------------------------------

    base_model = MobileNetV2(
        input_shape=IMAGE_SIZE + (NUM_CHANNELS,),
        include_top=False,
        weights="imagenet",
    )

    # Freeze the pretrained layers
    base_model.trainable = False

    # ------------------------------------------------------
    # Build Classification Head
    # ------------------------------------------------------

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (NUM_CHANNELS,))

    x = base_model(inputs, training=False)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(DROPOUT_RATE)(x)

    x = Dense(
        128,
        activation="relu",
        name="dense_128",
    )(x)

    x = Dropout(DROPOUT_RATE)(x)

    outputs = Dense(
        NUM_CLASSES,
        activation="softmax",
        name="emotion_output",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="DeepFER_MobileNetV2",
    )

    # ------------------------------------------------------
    # Compile Model
    # ------------------------------------------------------

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ==========================================================
# Test Model
# ==========================================================

if __name__ == "__main__":
    model = build_model()
    model.summary()