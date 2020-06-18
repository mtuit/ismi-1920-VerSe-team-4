import numpy as np
import scipy
import time
import random

from scipy import signal, ndimage, misc
from skimage.transform import resize
import copy


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


# thresholding
def thresh_lowering(x, thresh=0, thresh2=10000):
    """ Changes all pixel values outside of the thresholds to 0
    default settings are used in picture 2 as seen in whatsapp
    picture 3 is the results of default settings minus thresh=800, thresh2=1600
    """

    x[x < thresh] = 0
    x[x > thresh2] = 0

    return x


def thresh_lowering_minus(x, thresh1=(0, 10000), thresh2=(800, 1600)):
    """ Changing one of the pics permanently does so, deepcopy is required to overcome this when
    subtracting two different occasions of thresh_lowering on the same picture
    default settings result into picture 3 as seen in whatsapp
    """

    x1 = copy.deepcopy(x)
    x2 = copy.deepcopy(x)

    return thresh_lowering(x1, thresh1[0], thresh1[1]) - thresh_lowering(x2, thresh2[0], thresh2[1])


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


def preproces_caroursel(input, seed):
    rotate_range = 20
    shift_range = 3

    output = augment_rotate(input, 0, rotate_range, choose_random=True, seed=seed)
    output = augment_shift(output, 0, shift_range, choose_random=True, seed=seed)
    return output


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
    # assert (image.ndim == len(new_shape))

    # x = np.random.randint(0, image.shape[0] - new_shape[0])
    # y = np.random.randint(0, image.shape[1] - new_shape[1])
    # z = np.random.randint(0, image.shape[2] - new_shape[2])

    # reshaped_image = image[x:x + new_shape[0], y:y + new_shape[1], z:z + new_shape[2]]

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


# not used yet
def augment_flip(image):
    """
    Create an augmented image by mirroring in sagittal plane.

    Args:
        image (numpy.ndarray): Image array .

    Returns:
        A numpy.ndarray with first dimension (or x, sagittal) of the original image inverted.
    """

    flipped_image = np.flip(image, 0)

    return flipped_image


# not used yet
def augment_rotate(image, axis, degrees, choose_random=False, seed=0):
    """
    Create a rotated image, the image corners are padded with values of the image edge

    Args:
        image (numpy.ndarray): Image array .
        axis (int 1,2,3): Axis around which the image will be rotated (from center)
        degrees (int): Amount of rotation in degrees (can be negative), if choose_random=True, than a random value
        will be picked to rotate
        choose_random (bool):if choose_random=True, than a random value will be picked to rotate
        seed (int): seed for the random value

    Returns:
        An anticlockwise rotated numpy.ndarray of the original image
    """

    ax1 = (axis + 1) % 3
    ax2 = (axis + 2) % 3

    ## TODO: do: how to pad
    if choose_random:
        assert seed <= 0
        random.seed(seed)
        degrees = random.randint(- abs(degrees), abs(degrees))

    rotated_image = ndimage.rotate(image, degrees, axes=(ax1, ax2), reshape=False, mode='nearest')

    return rotated_image


# not used yet
def augment_shift(image, ax, distance, choose_random=False, seed=0):
    """
    Create an image that is moved by some pixels, padding the empty side with edge values

    Args:
        image (numpy.ndarray): Image array .
        ax (int 1,2,3): Axis along which the image will be shifted
        distance (int): Amount of shift in pixels (can be negative
        choose_random (bool):if choose_random=True, than a random value will be picked to rotate
        seed (int): seed for the random value)

    Returns:
        A shifted numpy.ndarray of the original image.
    """

    shape = np.shape(image)
    dim_length = shape[ax]

    if choose_random:
        random.seed(seed)
        distance = random.randint(- abs(distance), abs(distance))

    assert (distance < dim_length)

    rolled_image = np.roll(image, distance, axis=ax)

    delete_start = 0
    delete_end = distance

    if (distance < 0):
        delete_start = dim_length + distance
        delete_end = dim_length

    deletables = list(range(delete_start, delete_end))

    cropped_image = np.delete(rolled_image, deletables, axis=ax)

    pad0 = (0, 0)
    pad1 = (0, 0)
    pad2 = (0, 0)

    if (ax == 0):
        if (distance < 0):
            pad0 = (0, -distance)
        else:
            pad0 = (distance, 0)
    if (ax == 1):
        if (distance < 0):
            pad1 = (0, -distance)
        else:
            pad1 = (distance, 0)
    if (ax == 2):
        if (distance < 0):
            pad2 = (0, -distance)
        else:
            pad2 = (distance, 0)

    pad_loc = (pad0, pad1, pad2)

    shifted_image = np.pad(cropped_image, pad_loc, 'edge')

    return shifted_image
