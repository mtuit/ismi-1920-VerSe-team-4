import os, random, json

import numpy as np
import SimpleITK as sitk
import scipy

from scipy import signal
from skimage.transform import rescale, resize, downscale_local_mean

def get_centroids_of_image(image_path):
    """
    Gets the centroid locations of vertebrae from a centroids mask

    Args:
        image_path (str): Path to the centroids mask.

    Returns:
        A dictionary containing labels as keys and a dictionary of coordinates as values in the following format:

        {
            1: {
                'x': 0,
                'y': 0,
                'z': 0,
            },
        }
    """

    img_centroids_mask = sitk.ReadImage(image_path)
    np_centroids_mask = sitk.GetArrayFromImage(img_centroids_mask)

    # Value of voxels correspond to label numbers, therefore we retrieve all unique values in the mask
    labels = np.unique(np_centroids_mask)

    # We have to delete 0 values since these are padding and not centroid locations
    labels_without_zeros = np.delete(labels, 0)

    centroids = {}

    for label in labels_without_zeros:
        z, y, x = np.where(
            np_centroids_mask == label)  # Axis are reversed when converting from sitk image to numpy array, therefore z, y, x
        centroids[label] = {'x': x.item(), 'y': y.item(), 'z': z.item()}

    return centroids


def generate_heatmap_target(heatmap_size, centroids, sigma=3.0):
    """
    Generates a heatmap images in the corresponding dimensions with regard to centroids using a Gaussian kernel.
    Args:
        heatmap_size (tuple): The size of the image.
        centroids (dict): Dict containing all the centroid locations [x, y, z] in the corresponding heatmap.
        sigma (float): Sigma value used in by the Gaussian Kernel.

    Returns:
        A numpy.ndarray containing heatmaps of centroids with same dimension as heatmap_size.
    """

    heatmap = np.zeros(heatmap_size)

    # Read all coordinates and change value in the heatmap to 1 in those locations
    for label, coordinates in centroids.items():
        x, y, z = coordinates['x'], coordinates['y'], coordinates['z']
        heatmap[x, y, z] = 1

    # Create Gaussian kernel
    x = np.arange(sigma * -2.5, sigma * 3)
    y = np.arange(sigma * -2.5, sigma * 3)
    z = np.arange(sigma * -2.5, sigma * 3)

    X, Y, Z = np.meshgrid(x, y, z)

    gaussian_kernel = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * sigma ** 2))

    # Convolve the Gaussian Kernel on the heatmap where 1 corresponds to centroid location, generating heatmaps
    heatmap = scipy.signal.convolve(heatmap, gaussian_kernel, mode="same")

    return heatmap

def resize_padded(img, new_shape, fill_cval=0, order=1):
    ratio = np.min([n / i for n, i in zip(new_shape, img.shape)])
    interm_shape = np.rint([s * ratio for s in img.shape]).astype(np.int)
    interm_img = resize(img, interm_shape, order=order, cval=fill_cval)

    new_img = np.empty(new_shape, dtype=interm_img.dtype)
    new_img.fill(fill_cval)

    pad = [(n - s) >> 1 for n, s in zip(new_shape, interm_shape)]
    new_img[[slice(p, -p, None) if 0 != p else slice(None, None, None) 
             for p in pad]] = interm_img

    return new_img


def myresize(image, new_shape):
    """
    Resize an image to desired new size.


    Args:
        image (numpy.ndarray): Image array .
        new_shape (tuple): Shape to which the image array should be resized.

    Returns:
        A resized numpy.ndarray of the original image.
    """
    
    # old:
    #assert (image.ndim == len(new_shape))

    #x = np.random.randint(0, image.shape[0] - new_shape[0])
    #y = np.random.randint(0, image.shape[1] - new_shape[1])
    #z = np.random.randint(0, image.shape[2] - new_shape[2])

    #reshaped_image = image[x:x + new_shape[0], y:y + new_shape[1], z:z + new_shape[2]]

    # new:
    new = resize_padded(image, new_shape)
    
    
    
    
    return new
