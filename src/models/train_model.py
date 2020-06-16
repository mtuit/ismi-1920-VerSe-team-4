from src.models.model import get_model
import tensorflow as tf

def train_u_net(train_gen, val_gen, epochs, steps_per_epoch):
    weight_path = 'models/temp/model_{epoch:02d}-{val_loss:.2f}.h5'  # TODO: needs change
    
    u_model = get_model()

    earlystopper = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_path, verbose=1, save_weights_only=True, save_best_only=True)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs',
                                                 update_freq="epoch")
    
    callback_list = [checkpoint, earlystopper, tensorboard]

    # TODO: make it possible to apply custom model name
    results = u_model.fit(train_gen,
                          epochs=epochs,
                          validation_data=val_gen,
                          callbacks=callback_list,
                          steps_per_epoch=steps_per_epoch,
                          verbose=1)
