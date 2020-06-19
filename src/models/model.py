import tensorflow.keras as keras
import tensorflow as tf


def get_model(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.MeanSquaredError()]):
    model = unet3D_model()
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
    model.summary()
    return model


def unet3D_model(input_shape=(128, 64, 64), num_classes=25, num_layers=3, filters=8, output_activation="sigmoid"):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.expand_dims(inputs, -1)

    # encode
    down_layers = []
    for i in range(num_layers):
        x, filters = conv3d_block(inputs=x, filters=filters, mode="down")
        down_layers.append(x)
        x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    
    # bottom layer
    x, filters = conv3d_block(inputs=x, filters=filters, mode="down")

    # decode
    for conv in reversed(down_layers):
        x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)
        x = tf.keras.layers.concatenate([conv, x], axis=4)

        x, filters = conv3d_block(inputs=x, filters=filters, mode="up")

    outputs = tf.keras.layers.Conv3D(num_classes, (1, 1, 1), padding="same", activation=output_activation)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

    

def conv3d_block(inputs, filters, mode, padding="same", activation="relu"):
    if mode == "up":
        filters //= 2
    
    x = tf.keras.layers.Conv3D(filters, (3, 3, 3), padding=padding, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    if mode == "down":
        filters *= 2
    
    
    x = tf.keras.layers.Conv3D(filters, (3, 3, 3), padding=padding, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    return x, filters
