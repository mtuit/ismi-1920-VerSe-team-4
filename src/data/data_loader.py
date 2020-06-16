import os, random

import numpy as np
import tensorflow as tf
import SimpleITK as sitk

from src.models.train_model import train_u_net
from src.data.preprocessing import generate_heatmap

from skimage import transform
from sklearn.model_selection import train_test_split


class VerseDataset():
    """ class to get images and labels

    Attributes:
    
    """
    def __init__(self, shape=(128, 64, 64), n_classes=25, seed=2020, split=0.8):
        """ initialize the dataset """
        self.input_shape = shape
        self.output_shape = shape + (n_classes,)
        self.n_classes = n_classes
        
        # Get a list of all images and centroids
        self.images = os.listdir(os.path.join(BASE_PATH_NORMALIZED, 'images'))
        self.centroids = os.listdir(os.path.join(BASE_PATH_NORMALIZED, 'centroid_masks'))

        # Randomize using seed
        random.seed(seed)
        random.shuffle(self.images)
        
        self.test = []
        self.filtered_images = []
        
        self.train, self.validation, self.test = self._train_validation_test_split(self.images, self.centroids, seed, split)
        # split images in test and train/val set
        for image in self.images:
            if image not in self.centroids:
                self.test.append(image)
            else:
                self.filtered_images.append(image)

        # Generate train/validation split based on remaining images and centroids
        self.train, self.validation = train_test_split(self.filtered_images, train_size=split, random_state=seed)
    
    
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
            image, heatmap = self._generate_input_tuple(data_element)
            
            yield (image, heatmap)


    def _get_validation_generator(self):
        """
            Get a generator for the training set
        """
        for data_element in self.validation:
            image, heatmap = self._generate_input_tuple(data_element)
        
            yield (image, heatmap)


    def _get_test_generator(self):
        """
            Get a generator for the training set
        """
        for data_element in self.test:
            image, heatmap = self._generate_input_tuple(data_element)
        
            yield (image, heatmap)
            
            
    def _generate_input_tuple(self, data_element):
        itk_img = sitk.ReadImage(os.path.join(os.path.join(BASE_PATH_NORMALIZED, 'images'), data_element))
        itk_centroid = sitk.ReadImage(os.path.join(os.path.join(BASE_PATH_NORMALIZED, 'centroid_masks'), data_element))
    
        itk_img_arr = np.array(sitk.GetArrayFromImage(itk_img))
        itk_centroid_arr = sitk.GetArrayFromImage(itk_centroid)
    
        # Image is resized here, but heatmaps are resized to corresponding shape since the resize messes with the label values
        itk_img_arr_resize = transform.resize(itk_img_arr, self.input_shape)
    
        heatmap = generate_heatmap(itk_centroid_arr, self.input_shape, self.n_classes, debug=True)
        heatmap = np.moveaxis(heatmap, 0, -1)
        
        return (itk_img_arr_resize, heatmap)
        
        
    def _get_dataset_from_generator(self, data_generator):
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            (tf.float64, tf.int32),
            (tf.TensorShape(self.input_shape), tf.TensorShape(self.output_shape)))
        return dataset

    def _train_validation_test_split(self, images, centroids, seed, split):
        """ Generates a train/validation/test split for the images

        Args:
            images (list): paths to images
            centroids (list): paths to centroid masks
            seed (int): the seed for the random suffle
            split (float): train/test split, first fraction is assigned to train set
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
        train, validation = train_test_split(filtered_images, train_size=split, random_state=seed)
    
        return train, validation, test


if __name__ == '__main__':
    BASE_PATH = 'data/raw/training_data'
    BASE_PATH_NORMALIZED = 'data/processed/normalized-images'
    os.chdir('../..')

    split = 0.8
    # print('Generating train validation test split....')
    # generate_train_validation_test_split(2020, split)
    
    versedataset = VerseDataset()

    # train_generator = versedataset.get_data_gen('train')
    train_dataset = versedataset.get_dataset('train').batch(1)
    validation_dataset = versedataset.get_dataset('validation').batch(1)
    
    train_u_net(train_dataset, validation_dataset, 1)