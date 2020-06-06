import model

import numpy as np
import tensorflow as tf

from tensorflow import keras


def train(model, train_dataset, validation_dataset, callbacks=callbacks, epochs=10):
    history = model.fit(x=train_dataset, epochs=epochs, callbacks=callbacks, validation_data=validation_dataset)

    # model.fit(train_dataset, epochs=, validation_data=validation_dataset)

    pass

if __name__ == "__main__":
    model_save_path = 'models/temp/model_{epoch:02d}-{val_loss:.2f}.h5'

    model = model.get_model() # Later kan hier een model ingeladen worden via keras model_load() functionaliteit
    train_dataset = None # TODO insert datasets here
    validation_dataset = None # TODO insert datasets here
    epochs = 10
    
    model_check_point = keras.callbacks.ModelCheckPoint(model_save_path, monitor='val_loss')
    early_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    callbacks = [model_check_point, ]
    train(model=model, train_dataset=train_dataset, validation_dataset=validation_dataset, callback=callbacks, epochs=epochs)
