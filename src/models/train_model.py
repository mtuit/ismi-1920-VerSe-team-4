from src.models.model import get_model
import tensorflow as tf
from src.data.postprocessing import visualize
import os
import SimpleITK as sitk
import numpy as np
from skimage import transform

    
class PostprocessingCallback(tf.keras.callbacks.Callback):  
    
    def __init__(self, validation_set, size=2):
        super().__init__()
        self.my_validation_set = validation_set
        
    def on_epoch_end(self, epoch, logs={}):
        for filename in self.my_validation_set: 
            itk_img = sitk.ReadImage(os.path.join(os.path.join('data/processed/normalized-images', 'images'), filename))
            itk_img_mask = sitk.ReadImage(os.path.join(os.path.join('data/processed/normalized-images', 'centroid_masks'), filename))
            itk_img_arr = np.array(sitk.GetArrayFromImage(itk_img))
            
            itk_img_arr_resize = transform.resize(itk_img_arr, (128,64,64))
            y_pred = self.model.predict(np.expand_dims(itk_img_arr_resize, 0))
            
            visualize(itk_img_arr, y_pred, threshold = 0.1)
        
            
    

def train_u_net(train_gen, val_gen, epochs, steps_per_epoch, validation_set):
    weight_path = 'models/temp/model_{epoch:02d}-{val_loss:.2f}.h5'  # TODO: needs change
    
    u_model = get_model()

    earlystopper = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_path, verbose=1, save_weights_only=True, save_best_only=True)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs',
                                                 update_freq="epoch")
    postprocessing = PostprocessingCallback(validation_set)
    
    # limiting validation step:
    
    callback_list = [checkpoint, earlystopper, tensorboard, postprocessing]

    # TODO: make it possible to apply custom model name
    results = u_model.fit(train_gen,
                          epochs=epochs,
                          validation_data=val_gen,
                          callbacks=callback_list,
                          steps_per_epoch=steps_per_epoch,
                          verbose=1)
    
    return u_model