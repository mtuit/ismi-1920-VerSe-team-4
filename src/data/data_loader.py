import os, random

import numpy as np
import tensorflow as tf
import SimpleITK as sitk

#from src.models.train_model import train_u_net
from src.data.preprocessing import generate_heatmap

from skimage import transform
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class VerseDataset():
    """ class to get images and labels

    Attributes:
    
    """
    def __init__(self, base_path, shape=(128, 64, 64), n_classes=25, seed=2020, split=0.8):
        """ initialize the dataset """
        self.base_path = base_path
        self.input_shape = shape
        self.output_shape = shape + (n_classes,)
        self.n_classes = n_classes
        self.split = split
        self.seed = seed
        
        # Get a list of all images and centroids
        self.images = os.listdir(os.path.join(self.base_path, 'images'))
        self.centroids = os.listdir(os.path.join(self.base_path, 'centroid_masks'))

        # Randomize using seed
        random.seed(self.seed)
        random.shuffle(self.images)
        
        self.test = []
        self.filtered_images = []
        self.train, self.validation, self.test = self._train_validation_test_split(self.images, self.centroids)
        

    def get_dataset(self, purpose):
        if purpose == 'train':
            return self._get_dataset_from_generator(self._get_train_generator)
        elif purpose == 'validation':
            return self._get_dataset_from_generator(self._get_validation_generator)
        elif purpose == 'test':
            return self._get_dataset_from_generator(self._get_train_generator)
        else:
            return ValueError('Purpose should be either \'train\', \'validation\', or \'test\'')
        
        
    def _get_train_generator(self):
        """
            Get a generator for the training set
        """
        for data_element in self.train:
            print("training on: {}".format(data_element))
            image, heatmap = self._generate_input_tuple(data_element)
            print("yields: {}".format(data_element))
            yield (image, heatmap)


    def _get_validation_generator(self):
        """
            Get a generator for the validation set
        """
        for data_element in self.validation:
            print("validating on: {}".format(data_element))
            image, heatmap = self._generate_input_tuple(data_element)
            
            yield (image, heatmap)


    def _get_test_generator(self):
        """
            Get a generator for the test set
        """
        for data_element in self.test:
            image, heatmap = self._generate_input_tuple(data_element)
        
            yield (image, heatmap)
            
            
    def _generate_input_tuple(self, data_element):
        itk_img = sitk.ReadImage(os.path.join(os.path.join(self.base_path, 'images'), data_element))
        itk_centroid = sitk.ReadImage(os.path.join(os.path.join(self.base_path, 'centroid_masks'), data_element))
    
        itk_img_arr = np.array(sitk.GetArrayFromImage(itk_img))
        itk_centroid_arr = sitk.GetArrayFromImage(itk_centroid)
        
        # Image is resized here, but heatmaps are resized to corresponding shape since the resize messes with the label values
        itk_img_arr_resize = transform.resize(itk_img_arr, self.input_shape)
    
        heatmap = generate_heatmap(itk_centroid_arr, self.input_shape, self.n_classes, debug=False)
        heatmap = np.moveaxis(heatmap, 0, -1)
        
        return (itk_img_arr_resize, heatmap)
        
        
    def _get_dataset_from_generator(self, data_generator):
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            (tf.float64, tf.int32),
            (tf.TensorShape(self.input_shape), tf.TensorShape(self.output_shape)))
        return dataset

    def _train_validation_test_split(self, images, centroids):
        """ Generates a train/validation/test split for the images

        Args:
            images (list): paths to images
            centroids (list): paths to centroid masks
        """
   
        # Get test images from images list. Test images are images for which a centroid mask does NOT exist
        test = []
        filtered_images = []
    
        # split images in test and train/val set
        for image in images:
            if image not in centroids:
                test.append(image)
            else:
                filtered_images.append(image)
    
        # Generate train/validation split based on remaining images and centroids
        train, validation = train_test_split(filtered_images, train_size=self.split, random_state=self.seed)
    
        return train, validation, test


if __name__ == '__main__':
    # set paths
    BASE_PATH = 'data/raw/training_data'
    BASE_PATH_NORMALIZED = 'data/processed/normalized-images'
    
    # generate training and validation data
    verse_dataset = VerseDataset(base_path=BASE_PATH_NORMALIZED)
    
    training_dataset = verse_dataset.get_dataset('train')
    validation_dataset = verse_dataset.get_dataset('validation')
    
    #train_u_net(training_dataset, validation_dataset, 1)
    
    print("--DONE--")
