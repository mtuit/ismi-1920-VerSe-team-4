import os

import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from skimage import transform
from src.data.postprocessing import visualize, _get_locations_output
from src.data.preprocessing import generate_heatmap
from src.utils.utils import save_prediction

def predict_test_set(model, test_set, base_path='data/processed/normalized-images/images', input_shape=(128, 64, 64), output_path='models/predictions'):
    """
    Predict the whole test set. Should be run from main.py context
    
    Args:
        model (keras.Model): Model to perform predictions on
        test_set (list): Python list containing the names of the images in strings, eg. 'verse001.mha'
        base_path (str): Path where to load the images from
        input_shape (tuple): Shape in which the input for the model should be in
        output_path (str): Path where the predictions for the image should be saved
    """
    
    for file in test_set:
        img = sitk.ReadImage(os.path.join(base_path, file))
        img_array = sitk.GetArrayFromImage(img)
        img_resized = transform.resize(img_array, input_shape)
        y_true = sitk.ReadImage(os.path.join('../../data/processed/normalized-images/centroid_masks', file))

        y_true = sitk.GetArrayFromImage(y_true)
        y_true = generate_heatmap(y_true, y_true.shape, n_classes=25, debug=True)
        y_true = np.expand_dims(y_true, 0)
        
        y_pred = model.predict(np.expand_dims(img_resized, 0))

        visualize(img_array, y_true, threshold=0.5)

        locations = _get_locations_output(y_pred, img_array.shape, threshold=0.5, return_labels=True)
        
        save_prediction(locations, file, output_path)