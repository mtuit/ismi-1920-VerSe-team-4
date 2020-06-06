import model
from model import U_net_3D_model
import tensorflow as tf


# fit model
def train_u_net(epochs, train_gen, val_gen, epochs):
    weight_path = 'models/temp/model_{epoch:02d}-{val_loss:.2f}.h5' # TODO: needs change
    optimizer = tf.keras.optimizers.Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1E-08)
    u_model = model.U_net_3D_model()
    # TODO: optional use keras model loader
    # model = tf.keras.models.load_model(model_path)

    # compile model
    loss_fn = tf.keras.losses.mean_squared_error(y_pred=None, y_pred=None)
    # TODO: make loss
    u_model.compile(loss=loss_fn,
                  optimizer=optimizer)
                  # metrics=YOUR METRIC

    # create callback list
    earlystopper = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_path, verbose=1, save_best_only=True)
    # tensorboard stuff
    # TBD
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs',
                                                 update_freq="epoch")
    callback_list = [checkpoint, earlystopper, tensorboard]

    # TODO: make it possible to apply custom model name
    results = model.fit(train_gen,
                        epochs=epochs,
                        vallideation_data=val_gen,
                        callbacks=callback_list)
