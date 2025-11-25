"""
models.py

U-Net architecture for the iSeg 2017 brain segmentation project.

- Input:    (H, W, 2)  → channels = [T1, T2]
- Output:   (H, W, 4)  → 4 classes:
    0: background
    1: CSF
    2: gray matter
    3: white matter

We use softmax + sparse_categorical_crossentropy, so the ground truth
should be integer class indices 0..3 with shape (H, W, 1).

Compatible with:
- TensorFlow 2.20.0
- Keras 3.x via tf.keras
"""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Two Conv2D + BN + ReLU layers."""
    x = layers.Conv2D(filters, (3, 3), padding="same", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU(name=f"{name}_relu2")(x)
    return x


def build_unet(
    input_shape: Tuple[int, int, int],
    num_classes: int = 4,
    base_filters: int = 16,
    learning_rate: float = 1e-3,
    name: str = "unet",
    use_dropout: bool = False,
) -> tf.keras.Model:
    """Build and compile a 2D U-Net for multi-class segmentation."""
    inputs = layers.Input(shape=input_shape, name="input")

    # Encoder
    c1 = conv_block(inputs, base_filters, "enc1")
    p1 = layers.MaxPooling2D((2, 2), name="enc1_pool")(c1)

    c2 = conv_block(p1, base_filters * 2, "enc2")
    p2 = layers.MaxPooling2D((2, 2), name="enc2_pool")(c2)

    c3 = conv_block(p2, base_filters * 4, "enc3")
    p3 = layers.MaxPooling2D((2, 2), name="enc3_pool")(c3)

    c4 = conv_block(p3, base_filters * 8, "enc4")
    p4 = layers.MaxPooling2D((2, 2), name="enc4_pool")(c4)

    # Bottleneck
    bn = conv_block(p4, base_filters * 16, "bottleneck")
    if use_dropout:
        bn = layers.Dropout(0.5, name="bottleneck_dropout")(bn)

    # Decoder
    u4 = layers.Conv2DTranspose(
        base_filters * 8, (2, 2), strides=(2, 2), padding="same", name="dec4_up"
    )(bn)
    u4 = layers.Concatenate(axis=-1, name="dec4_concat")([u4, c4])
    c5 = conv_block(u4, base_filters * 8, "dec4")

    u3 = layers.Conv2DTranspose(
        base_filters * 4, (2, 2), strides=(2, 2), padding="same", name="dec3_up"
    )(c5)
    u3 = layers.Concatenate(axis=-1, name="dec3_concat")([u3, c3])
    c6 = conv_block(u3, base_filters * 4, "dec3")

    u2 = layers.Conv2DTranspose(
        base_filters * 2, (2, 2), strides=(2, 2), padding="same", name="dec2_up"
    )(c6)
    u2 = layers.Concatenate(axis=-1, name="dec2_concat")([u2, c2])
    c7 = conv_block(u2, base_filters * 2, "dec2")

    u1 = layers.Conv2DTranspose(
        base_filters, (2, 2), strides=(2, 2), padding="same", name="dec1_up"
    )(c7)
    u1 = layers.Concatenate(axis=-1, name="dec1_concat")([u1, c1])
    c8 = conv_block(u1, base_filters, "dec1")

    # Multi-class output
    outputs = layers.Conv2D(
        num_classes, (1, 1), activation="softmax", name="output"
    )(c8)

    model = models.Model(inputs=inputs, outputs=outputs, name=name)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_unet_stage1(
    input_shape: Tuple[int, int, int] = (144, 192, 2),
    num_classes: int = 4,
    base_filters: int = 16,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Convenience wrapper for the Stage 1 model."""
    return build_unet(
        input_shape=input_shape,
        num_classes=num_classes,
        base_filters=base_filters,
        learning_rate=learning_rate,
        name="unet_stage1",
        use_dropout=True,
    )


if __name__ == "__main__":
    # Simple build sanity check
    m = build_unet_stage1()
    m.summary()
