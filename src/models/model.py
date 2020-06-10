import tensorflow as tf
import numpy as np

from tensorflow import keras


def get_model(input_shape, loss=keras.loss.MeanSquaredError(), optimizer=keras.optimizers.Adam(), metrics=keras.metrics.MeanSquaredError()):
    model = U_net_3D_model(input_shape=input_shape)
    return model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

def U_net_3D_model(input_shape=(None, 32, 64, 64)):
    inputs = tf.keras.Input(shape=input_shape)
    
    #block 1
    x = tf.keras.layers.Conv3D(32, (3,3,3) , padding="same") (inputs)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.activations.relu(alpha=0.0, max_value=None, threshold=0) (x)
    
    x = tf.keras.layers.Conv3D(64, (3,3,3) , padding="same") (x)
    x = tf.keras.layers.BatchNormalization()
    s1 = tf.keras.activations.relu(alpha=0.0, max_value=None, threshold=0) (x)
    
    x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(s1)
    
    #block 2
    x = tf.keras.layers.Conv3D(64, (3,3,3) , padding="same") (inputs)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.activations.relu(alpha=0.0, max_value=None, threshold=0) (x)
    
    x = tf.keras.layers.Conv3D(128, (3,3,3) , padding="same") (x)
    x = tf.keras.layers.BatchNormalization()
    s2 = tf.keras.activations.relu(alpha=0.0, max_value=None, threshold=0) (x)
    
    x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(s2)
    
    #block 3
    x = tf.keras.layers.Conv3D(128, (3,3,3) , padding="same") (inputs)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.activations.relu(alpha=0.0, max_value=None, threshold=0) (x)
    
    x = tf.keras.layers.Conv3D(256, (3,3,3) , padding="same") (x)
    x = tf.keras.layers.BatchNormalization()
    s3 = tf.keras.activations.relu(alpha=0.0, max_value=None, threshold=0) (x)
    
    x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(s3)
    
    #bottom
    x = tf.keras.layers.Conv3D(256, (3,3,3) , padding="same") (x)
    x = tf.keras.layers.BatchNormalization()
    x = tf.keras.activations.relu(alpha=0.0, max_value=None, threshold=0) (x)
    
    x = tf.keras.layers.Conv3D(512, (3,3,3) , padding="same") (x)
    x = tf.keras.layers.BatchNormalization()
    x = tf.keras.activations.relu(alpha=0.0, max_value=None, threshold=0) (x)
    
    # block 4
    x = tf.keras.layers.UpSampling3D(size = (2,2,2)) (x)
    x = np.concatenate([s3,x], axis = 4)
    
    x = tf.keras.layers.Conv3D(256, (3,3,3) , padding="same") (x)
    x = tf.keras.layers.BatchNormalization()
    x = tf.keras.activations.relu(alpha=0.0, max_value=None, threshold=0) (x)
    
    x = tf.keras.layers.Conv3D(256, (3,3,3) , padding="same") (x)
    x = tf.keras.layers.BatchNormalization()
    x = tf.keras.activations.relu(alpha=0.0, max_value=None, threshold=0) (x)
    
    # block 5
    x = tf.keras.layers.UpSampling3D(size = (2,2,2)) (x)
    x = np.concatenate([s2,x], axis = 4)
    
    x = tf.keras.layers.Conv3D(128, (3,3,3) , padding="same") (x)
    x = tf.keras.layers.BatchNormalization()
    x = tf.keras.activations.relu(alpha=0.0, max_value=None, threshold=0) (x)
    
    x = tf.keras.layers.Conv3D(128, (3,3,3) , padding="same") (x)
    x = tf.keras.layers.BatchNormalization()
    x = tf.keras.activations.relu(alpha=0.0, max_value=None, threshold=0) (x)
    
    # block 6
    x = tf.keras.layers.UpSampling3D(size = (2,2,2)) (x)
    x = np.concatenate([s2,x], axis = 4)
    
    x = tf.keras.layers.Conv3D(64, (3,3,3) , padding="same") (x)
    x = tf.keras.layers.BatchNormalization()
    x = tf.keras.activations.relu(alpha=0.0, max_value=None, threshold=0) (x)
    
    x = tf.keras.layers.Conv3D(64, (3,3,3) , padding="same") (x)
    x = tf.keras.layers.BatchNormalization()
    x = tf.keras.activations.relu(alpha=0.0, max_value=None, threshold=0) (x)
    
    # output layer
    outputs = tf.keras.layers.Conv3D(25, (1,1,1) , padding="same") (x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)