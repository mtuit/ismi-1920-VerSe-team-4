import tensorflow as tf


def get_model(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(), metrics=tf.keras.metrics.MeanSquaredError()):
    model = unet3D_model()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def unet3D_model(input_shape=(128, 64, 64), num_classes=25, num_layers=3, filters=32, output_activation="sigmoid"):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.expand_dims(inputs, -1)

    down_layers = []
    for i in range(num_layers):
        x, filters = conv3d_block(inputs=x, filters=filters, mode="down")
        down_layers.append(x)
        x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    x, filters = conv3d_block(inputs=x, filters=filters, mode="down")

    for conv in reversed(down_layers):
        x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)
        x = tf.keras.layers.concatenate([conv, x], axis=4)

        x, filters = conv3d_block(inputs=x, filters=filters, mode="up")

    outputs = tf.keras.layers.Conv3D(num_classes, (1, 1, 1), padding="same", activation=output_activation)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def conv3d_block(inputs, filters, mode, padding="same", activation="relu"):
    x = tf.keras.layers.Conv3D(filters, (3, 3, 3), padding=padding, activation=activation)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    if mode == "down":
        filters *= 2
    elif mode == "up":
        filters //= 2
    
    x = tf.keras.layers.Conv3D(filters, (3, 3, 3), padding=padding, activation=activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    return x, filters