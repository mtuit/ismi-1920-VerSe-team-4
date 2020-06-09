# create 3d u-net
# input = image
# output = x*y grid, each containing:
# one centroid location with x,y,w,h and the class (1-25) with a certainty score of: c

import tensorflow as tf


def U_net_3D_model():
    inputs = tf.keras.Input(shape=(None, 32, 64, 64))

    # block 1
    x = tf.keras.layers.Conv3D(32, (3, 3, 3), padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)#(x)

    x = tf.keras.layers.Conv3D(64, (3, 3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # shortcut to s1
    s1 = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)#(x)

    x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(s1)

    # block 2
    x = tf.keras.layers.Conv3D(64, (3, 3, 3), padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)#(x)

    x = tf.keras.layers.Conv3D(128, (3, 3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # shortcut to s2
    s2 = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)#(x)

    x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(s2)

    # block 3
    x = tf.keras.layers.Conv3D(128, (3, 3, 3), padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)#(x)

    x = tf.keras.layers.Conv3D(256, (3, 3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # shortcut to s3
    s3 = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)#(x)

    x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(s3)

    # bottom
    x = tf.keras.layers.Conv3D(256, (3, 3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)#(x)

    x = tf.keras.layers.Conv3D(512, (3, 3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)#(x)

    # block 4
    x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)
    x = tf.keras.layers.concatenate([s3, x], axis=4)

    x = tf.keras.layers.Conv3D(256, (3, 3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)#(x)

    x = tf.keras.layers.Conv3D(256, (3, 3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)#(x)

    # block 5
    # x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)
    x = tf.keras.layers.concatenate([s2, x], axis=4)

    x = tf.keras.layers.Conv3D(128, (3, 3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)#(x)

    x = tf.keras.layers.Conv3D(128, (3, 3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)#(x)

    # block 6
    # x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)
    x = tf.keras.layers.concatenate([s1, x], axis=4) # was s2

    x = tf.keras.layers.Conv3D(64, (3, 3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)#(x)

    x = tf.keras.layers.Conv3D(64, (3, 3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)#(x)

    # output layer
    outputs = tf.keras.layers.Conv3D(25, (1, 1, 1), padding="same")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # TODO: add loss(cross entropy?) add metrics
    model.summary()  # als dit kan?

    return model
