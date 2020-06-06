import os, random, json
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy

from glob import glob
from scipy import signal
from sklearn.model_selection import train_test_split

def generate_train_validation_test_split(seed, split):
    """ returns file ids for trainset and test set

    Args:
        seed (int): the seed for the random suffle
        split (float): train/test split, first fraction is assigned to train set

    """

    # Get a list of all images and centroids
    images = os.listdir(os.path.join(BASE_PATH_NORMALIZED, 'images'))
    centroids = os.listdir(os.path.join(BASE_PATH_NORMALIZED, 'centroid_masks'))

    # Randomize using seed
    random.seed(seed)
    random.shuffle(images)

    # Get test images from images list. Test images are images for which a centroid mask does NOT exist
    test = []
    filtered_images = []

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


if __name__ == '__main__':
    BASE_PATH = 'data/raw/training_data'
    BASE_PATH_NORMALIZED  = 'data/processed/normalized-images'

    split = float(input("Input train/test split ratio. First fraction is assigned to train set. Input as float number."))

    print('Generating train validation test split....')

    generate_train_validation_test_split(2020, split)

    print("--DONE--")
