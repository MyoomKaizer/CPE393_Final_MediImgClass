from tensorflow.keras import layers, models, optimizers


def _conv_block(x, n_filters):
    x = layers.Conv2D(n_filters, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(n_filters, 3, activation="relu", padding="same")(x)
    return x


def unet(input_shape, lr=1e-3):
    inputs = layers.Input(shape=input_shape)

    c1 = _conv_block(inputs, 16)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = _conv_block(p1, 32)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = _conv_block(p2, 64)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = _conv_block(p3, 128)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = _conv_block(p4, 256)

    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.Concatenate()([u6, c4])
    c6 = _conv_block(u6, 128)

    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = _conv_block(u7, 64)

    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = _conv_block(u8, 32)

    u9 = layers.UpSampling2D((2, 2))(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = _conv_block(u9, 16)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
