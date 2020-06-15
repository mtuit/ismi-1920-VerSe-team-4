import os, random, json
import timeit
from typing import List

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy
from src.data.preprocessing import resize
import itertools
from skimage import transform

# import model stuff
from src.models.train_model import train_u_net

from glob import glob
from scipy import signal
from sklearn.model_selection import train_test_split


class VerseDataset():
    """ class to get images and labels

    Attributes:
        images =
    """

    def __init__(self, seed, split=0.8):
        """ initialize the dataset """

        # Get a list of all images and centroids

        self.images = os.listdir(os.path.join(BASE_PATH_NORMALIZED, 'images'))
        self.centroids = os.listdir(os.path.join(BASE_PATH_NORMALIZED, 'centroid_masks'))

        # Randomize using seed
        random.seed(seed)
        random.shuffle(self.images)
        self.test = []
        self.filtered_images = []

        # split images in test and train/val set
        for image in self.images:
            if image not in self.centroids:
                self.test.append(image)
            else:
                self.filtered_images.append(image)

        # Generate train/validation split based on remaining images and centroids
        self.train, self.validation = train_test_split(self.filtered_images, train_size=split, random_state=seed)

    def get_data_gen(self, purpose):
        """ get the specified dataset"""

        if purpose == 'train':
            data = self.train
        elif purpose == 'validation':
            data = self.validation
        elif purpose == 'test':
            data = self.test
        #
        for i in itertools.count(1):
            path = random.sample(data, 1)
            # Training data generator object (non-exhaustive)
            # Generator object which yields tuples of the original image and the 25 label heatmaps

            itk_img = sitk.ReadImage(os.path.join(os.path.join(BASE_PATH_NORMALIZED, 'images'), path[0]))
            itk_centroid = sitk.ReadImage(os.path.join(os.path.join(BASE_PATH_NORMALIZED, 'centroid_masks'), path[0]))

            itk_img_arr = np.array(sitk.GetArrayFromImage(itk_img))
            itk_centroid_arr = sitk.GetArrayFromImage(itk_centroid)
            itk_img_arr_resize = resize(itk_img_arr, (128, 64, 64))

            # could be that when the array is resized, values change
            # if this happens, we should take another approach
            itk_centroid_arr_resize = resize(itk_centroid_arr, (128, 64, 64))
            heatmap = generate_heatmap(itk_centroid_arr_resize, 3.0)

            yield (itk_img_arr_resize, heatmap)

    def get_dataset(self):
        aa = tf.data.Dataset.from_generator(
            self.get_data_gen('train'),
            (tf.float64, tf.int32),
            (tf.TensorShape([128, 64, 64]), tf.TensorShape([25, 128, 64, 64])))
        return aa

        # def get_dataset_from_generator(self):
        #     s = ['train', 'validation', 'test']
        #     generators = []
        #     for st in s:
        #         datagen = self.get_data_gen(st)
        #
        #         generators.append(tf.data.Dataset.from_generator(
        #                             datagen,
        #                             (tf.float64, tf.int32),
        #                             (tf.TensorShape([128, 64, 64]), tf.TensorShape([25, 128, 64, 64]))))

        return generators


# obso
def generate_train_validation_test_split(seed, split):
    """ returns file ids for trainset and test set

    Args:
        seed (int): the seed for the random suffle
        split (float): train/test split, first fraction is assigned to train set

    """

    # Get a list of all images and centroids
    path = 'data/processed/normalized-images'
    images = os.listdir(os.path.join(path, 'images'))
    centroids = os.listdir(os.path.join(path, 'centroid_masks'))

    # Randomize using seed
    random.seed(seed)
    random.shuffle(images)

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

    print(f'Saving {len(train)} files to train.txt')
    np.savetxt('data/processed/train.txt', np.array(train), fmt='%s', newline='\n')

    print(f'Saving {len(validation)} files to train.txt')
    np.savetxt('data/processed/validation.txt', np.array(validation), fmt='%s', newline='\n')

    print(f'Saving {len(test)} files to train.txt')
    np.savetxt('data/processed/test.txt', np.array(test), fmt='%s', newline='\n')


# obso
def load_txt_to_nparray(path):
    text_file = open(path, "r")
    x = text_file.read().split('\n')
    text_file = text_file.close()
    return x


"""
    Changes one heatmap with 25 different integers into 25 one-hot heatmaps
"""


def generate_heatmap(centroid_array, sigma, debug=False):
    heatmap = []
    number_of_vertebrae = 25

    if debug:
        print(f'Generating heatmaps of vertebraes...')
    	start = timeit.timeit()
    for i in range(1, number_of_vertebrae + 1):
    # for i in tqdm(1, number_of_vertebrae + 1):
        if debug:
            print("Generating heatmaps of vertebra {}".format(i))
        centroid_array_one_hot = np.where(centroid_array == i, 1, 0)
        
        # if no centroid found just return an empty array (to prevent unneccesary computations):
        if (np.max(centroid_array_one_hot) < 0.01):
            heatmap.append(np.zeros((128, 64, 64)))
        else:         
            x = np.arange(sigma * -2.5, sigma * 3, 1)
            y = np.arange(sigma * -2.5, sigma * 3, 1)
            z = np.arange(sigma * -2.5, sigma * 3, 1)
            xx, yy, zz = np.meshgrid(x, y, z)
            kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))
            # convolve changes all the values of the heatmap (with tiny amounts) but most values should remain 0
            # duurt lang!!!!!!!:
            location = np.argmax(centroid_array_one_hot)
            filtered = scipy.signal.convolve(centroid_array_one_hot, kernel, mode="same") 
            # duurt lang!!!!!!!:
            filtered_resize = transform.resize(filtered, (128, 64, 64))
            heatmap.append(filtered_resize)
      
    if debug:
	end = timeit.timeit()
        print(f'Return 25 heatmaps {}'.format(end - start)

    return np.array(heatmap)
""" =============== afkomstig uit merge, ik vermoed dat dit oud is, graag weghalen indien correct   
    for i in range(1, number_of_vertebrae + 1):
        centroid_array_one_hot = np.where(centroid_array == i, 1, 0)

        x = np.arange(sigma * -2.5, sigma * 3, 1)
        y = np.arange(sigma * -2.5, sigma * 3, 1)
        z = np.arange(sigma * -2.5, sigma * 3, 1)

        xx, yy, zz = np.meshgrid(x, y, z)

        kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))

        # convolve changes all the values of the heatmap (with tiny amounts) but most values should remain 0
        filtered = scipy.signal.convolve(centroid_array_one_hot, kernel, mode="same")
        heatmap.append(filtered)
        
>>>>>>> bec03efd658ad9e9ad8f85bcaac8c46123ee4344
"""


"""
    Training data generator object (non-exhaustive)
    Generator object which yields tuples of the original image and the 25 label heatmaps
"""


# obso
def training_data_generator():
    data = load_txt_to_nparray('data/processed/train.txt')
    source_path = 'data/processed/normalized-images'
    for i in itertools.count(1):
        path = random.sample(data, 1)

        itk_img = sitk.ReadImage(os.path.join(os.path.join(source_path, 'images'), path[0]))
        itk_centroid = sitk.ReadImage(os.path.join(os.path.join(source_path, 'centroid_masks'), path[0]))

        itk_img_arr = np.array(sitk.GetArrayFromImage(itk_img))
        itk_centroid_arr = sitk.GetArrayFromImage(itk_centroid)
# <<<<<<< HEAD
        # itk_img_arr_resize = transform.resize(itk_img_arr, (128, 64, 64))

        # heatmap = generate_heatmap(itk_centroid_arr, 3.0, debug=True)
# =======
        itk_img_arr_resize = transform.resize(itk_img_arr, (128, 64, 64))

        # could be that when the array is resized, values change
        # if this happens, we should take another approach
        itk_centroid_arr_resize = transform.resize(itk_centroid_arr, (128, 64, 64))

        heatmap = generate_heatmap(itk_centroid_arr_resize, 3.0, debug=True)
# >>>>>>> bec03efd658ad9e9ad8f85bcaac8c46123ee4344
        heatmap = np.moveaxis(heatmap, 0, -1)

        yield (itk_img_arr_resize, heatmap)


def general_data_generator(purpose_data):
    data = purpose_data
    for i in itertools.count(1):
        path = random.sample(data, 1)

        itk_img = sitk.ReadImage(os.path.join(os.path.join(BASE_PATH_NORMALIZED, 'images'), path[0]))
        itk_centroid = sitk.ReadImage(os.path.join(os.path.join(BASE_PATH_NORMALIZED, 'centroid_masks'), path[0]))

        itk_img_arr = np.array(sitk.GetArrayFromImage(itk_img))
        itk_centroid_arr = sitk.GetArrayFromImage(itk_centroid)
        itk_img_arr_resize = resize(itk_img_arr, (128, 64, 64))

        # could be that when the array is resized, values change
        # if this happens, we should take another approach
        itk_centroid_arr_resize = resize(itk_centroid_arr, (128, 64, 64))
        heatmap = generate_heatmap(itk_centroid_arr_resize, 3.0)
        heatmap = np.moveaxis(heatmap, 0, -1)

        yield (itk_img_arr_resize, heatmap)


"""
    Validation data generator object (exhaustive)
    Generator object which yields tuples of the original image and the 25 label heatmaps
"""


# obso
def validation_data_generator():
    data = load_txt_to_nparray('data/processed/validation.txt')
    source_path = 'data/processed/normalized-images'
    for v in data:
        itk_img = sitk.ReadImage(os.path.join(os.path.join(source_path, 'images'), v))
        itk_centroid = sitk.ReadImage(os.path.join(os.path.join(source_path, 'centroid_masks'), v))

        itk_img_arr = np.array(sitk.GetArrayFromImage(itk_img))
        itk_centroid_arr = sitk.GetArrayFromImage(itk_centroid)
        itk_img_arr_resize = resize(itk_img_arr, (128, 64, 64))

        # could be that when the array is resized, values change
        # if this happens, we should take another approach
        itk_centroid_arr_resize = resize(itk_centroid_arr, (128, 64, 64))
        heatmap = generate_heatmap(itk_centroid_arr_resize, 3.0)
        heatmap = np.moveaxis(heatmap, 0, -1)
        yield (itk_img_arr_resize, heatmap)


"""
    Testing data generator object (exhaustive)
    Generator object which yields tuples of the original image and the 25 label heatmaps
"""


# obso
def testing_data_generator():
    data = load_txt_to_nparray('data/processed/test.txt')
    source_path = 'data/processed/normalized-images'
    for v in data:
        itk_img = sitk.ReadImage(os.path.join(os.path.join(source_path, 'images'), v))
        itk_centroid = sitk.ReadImage(os.path.join(os.path.join(source_path, 'centroid_masks'), v))

        itk_img_arr = np.array(sitk.GetArrayFromImage(itk_img))
        itk_centroid_arr = sitk.GetArrayFromImage(itk_centroid)
        itk_img_arr_resize = resize(itk_img_arr, (128, 64, 64))

        # could be that when the array is resized, values change
        # if this happens, we should take another approach
        itk_centroid_arr_resize = resize(itk_centroid_arr, (128, 64, 64))
        heatmap = generate_heatmap(itk_centroid_arr_resize, 3.0)
        heatmap = np.moveaxis(heatmap, 0, -1)
        yield (itk_img_arr_resize, heatmap)


"""
    returns a dataset from the right generator
    input and output shapes are fixed
    inputs must be a generator function (!)
"""


def get_dataset_from_generator(data_generator):
    return tf.data.Dataset.from_generator(
        data_generator,
        (tf.float64, tf.int32),
        (tf.TensorShape([128, 64, 64]), tf.TensorShape([128, 64, 64, 25])))


if __name__ == '__main__':
    BASE_PATH = 'data/raw/training_data'
    BASE_PATH_NORMALIZED = 'data/processed/normalized-images'
    os.chdir('../..')

    split = 0.8
    # print('Generating train validation test split....')
    # generate_train_validation_test_split(2020, split)
    
    versedataset = VerseDataset(2020, split)

    # pseudocode oude manier:
    # training_dataset = get_dataset_from_generator(training_data_generator)
    # validation_dataset = get_dataset_from_generator(validation_data_generator)
    # testing_dataset = get_dataset_from_generator(testing_data_generator)
    #
    # # call the model
    # training_dataset = training_dataset.batch(1)
    # validation_dataset = validation_dataset.batch(1)
    #
    # train_u_net(training_dataset, validation_dataset, 5)
    #
    # print("--DONE--")

    # pseudocode with class:
    # DEBUG, want itertools laten werken met een class instance is nog niet zo evident
    tst = training_data_generator()
    tst_cls = versedataset.get_data_gen('train')
    dfg = tf.data.Dataset.from_generator(versedataset.get_data_gen('train'),
                                         (tf.float64, tf.int32),
                                         tf.TensorShape([128, 64, 64]), tf.TensorShape([25, 128, 64, 64]))

    training_dataset_v = get_dataset_from_generator(versedataset.get_data_gen('train'))
    validation_dataset_v = get_dataset_from_generator(versedataset.get_data_gen('validation'))
    testing_dataset_v = get_dataset_from_generator(versedataset.get_data_gen('test'))
