import numpy as np
import SimpleITK as sitk
import scipy
import time

from scipy import signal
from skimage.transform import resize


def generate_heatmap(centroid_array, heatmap_size, n_classes, sigma=3.0, debug=False):
    heatmap = []
    number_of_vertebrae = n_classes
    
    if debug:
        print(f'Generating heatmaps of vertebraes...')
        start = time.time()
    
    for i in range(1, number_of_vertebrae + 1):
        # for i in tqdm(1, number_of_vertebrae + 1):
        if debug:
            print("Generating heatmaps of vertebra {}".format(i))
        
        centroid_array_one_hot = np.where(centroid_array == i, 1, 0)
        
        # if no centroid found just return an empty array (to prevent unneccesary computations):
        if (np.max(centroid_array_one_hot) < 0.01):
            if debug:
                print('Adding empty heatmap!')
            heatmap.append(np.zeros(heatmap_size))
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
            filtered_resize = resize(filtered, heatmap_size)
            heatmap.append(filtered_resize)
    
    if debug:
        end = time.time()
        print('Return 25 heatmaps {}'.format(end - start))
    
    return np.array(heatmap)

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
    """ ===== leftover from commit: remove if intended
    new = resize_padded(image, new_shape)
    
    
    
    
    return new
    """

    assert (image.ndim == len(new_shape))

    x = np.random.randint(0, image.shape[0] - new_shape[0])
    y = np.random.randint(0, image.shape[1] - new_shape[1])
    z = np.random.randint(0, image.shape[2] - new_shape[2])

    reshaped_image = image[x:x + new_shape[0], y:y + new_shape[1], z:z + new_shape[2]]
    return reshaped_image
