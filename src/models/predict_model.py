import os

import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from skimage import transform
from src.models.model import get_model
from src.data.postprocessing import visualize, _get_locations_output
from src.data.preprocessing import generate_heatmap, histogram_match
from src.utils.utils import save_prediction


def predict(model, to_predict, base_path='data/processed/normalized-images', input_shape=(128, 64, 64), output_path='models/predictions'):
    """
    Predict the whole test set. Should be run from main.py context
    
    Args:
        model (keras.Model): Model to perform predictions on
        to_predict (list): Python list containing the names of the images in strings, eg. 'verse001.mha'
        base_path (str): Path where to load the images from
        input_shape (tuple): Shape in which the input for the model should be in
        output_path (str): Path where the predictions for the image should be saved
    """

    if type(to_predict) is not list: 
        to_predict = [to_predict]
    
    ref_img = sitk.ReadImage(os.path.join(base_path, 'images', 'verse004.mha'))
    ref_img_arr = np.array(sitk.GetArrayFromImage(ref_img))
    
    for file in to_predict:
        img = sitk.ReadImage(os.path.join(base_path, 'images', file))
        img_array = sitk.GetArrayFromImage(img)
        hist_img = histogram_match(img_array, ref_img_arr)
        img_resized = transform.resize(hist_img, input_shape)
        
        y_pred = model.predict(np.expand_dims(img_resized, 0))
        
        visualize(img_array, y_pred, threshold=0.5)
        locations = _get_locations_output(y_pred[0], img_array.shape, threshold=0.5, return_labels=True)
        save_prediction(locations, file, output_path)


if __name__ == "__main__":
    WEIGHT_PATH = '../../models/temp/model_05.h5'

    # Create a base model and load the weights
    model = get_model()
    weights = model.load_weights(WEIGHT_PATH)

    predict(model, 'verse016.mha', base_path='../../data/processed/normalized-images', output_path='../../models/predictions')
