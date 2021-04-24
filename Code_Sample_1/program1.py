import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from math import floor
import skimage


# ************************************
# ********* HELPER FUNCTIONS *********
# ************************************

def equalize_calculations(hist_orig):
    """
    The function calculates the equalization formulas on an image.
    :param hist_orig: The image to be equalized.
    :return: a list containing the original histogram of the image, the histogram of the equalized image, and
            the lookup table to create the equalized new image.
    """
    sum_hist = np.cumsum(hist_orig)
    resolution = sum_hist[-1]
    first_nonzero = np.flatnonzero(sum_hist).min()

    lookup_table = np.zeros_like(hist_orig)

    lookup_table[first_nonzero:] = \
        np.round(((sum_hist[first_nonzero:] - sum_hist[first_nonzero]) / (resolution - sum_hist[first_nonzero])) * 255)

    return lookup_table


def quantize_calculation(im_orig, n_quant, n_iter):
    """
    Implements the quantize calculations on the original image.
    :param im_orig: The original image
    :param n_quant: The number of colors for the quantized-image
    :param n_iter: The maximum number of iterations for the quantize algorithm.
    :return: list contains: (1) array of the errors recorded in each iteration.
                            (2) the z's division.
                            (3) the q's values matching to the divisions (q[i] matches the range(z[i],z[i+1]).
    """
    # initiate resources about the image:
    hist_orig = np.histogram(im_orig, bins=256, range=(0, 255))[0]
    cumsum_hist = np.cumsum(hist_orig)  # the sum histogram.
    hist_orig_prob = hist_orig / cumsum_hist[-1]

    # make the z-division array, initiate new q-values array:
    z_division = np.linspace(0, cumsum_hist[-1], num=(n_quant + 1))
    z_division = np.fromiter(map(lambda element: np.searchsorted(cumsum_hist, element, side="left"), z_division),
                             dtype=np.int)
    z_division[0] = -1

    q_array = np.zeros(n_quant, dtype=np.int)
    old_z_div = z_division.copy()

    # start improving q's and z's:
    errors = []
    for k in range(n_iter):
        # calculate new 'q':
        for i in range(n_quant):
            start = floor(z_division[i]) + 1
            end = floor(z_division[i + 1])
            q_array[i] = np.round(np.average(np.arange(start, end + 1), weights=hist_orig_prob[start:end + 1]))

        # calculate new 'z':
        for j in range(1, n_quant):
            z_division[j] = ((q_array[j - 1] + q_array[j]) / 2)

        # calculate general_error:
        general_error = 0
        for i in range(n_quant):
            g = np.arange(floor(z_division[i]) + 1, floor(z_division[i+1]) + 1)
            current_q = np.full_like(g, fill_value=q_array[i])
            segment_error = (((current_q - g) ** 2) * hist_orig_prob[g])
            general_error += np.sum(segment_error)
        errors.append(general_error)

        if np.array_equal(old_z_div, z_division):
            break
        else:
            old_z_div = z_division.copy()

    return [errors, z_division, q_array]


def create_image_from_z_q(im_orig255, z_division, q_array):
    """
    Gets ideal z's and q's - and creating a new image with q_array-length colors.
    :param im_orig255: original image, pixels values represented in [0,255] range.
    :param z_division: the z's divisions of the image.
    :param q_array: the matching q's values of the image.
    :return: a brand new quantized image, scaled in the range of [0,1].
    """
    im_orig255_copy = im_orig255.copy()
    for i in range(z_division.size - 1):
        im_orig255[np.logical_and((z_division[i] <= im_orig255_copy), (im_orig255_copy <= z_division[i + 1]))] = \
            q_array[i]

    return im_orig255 / 255


# *********************************
# ********* API FUNCTIONS *********
# *********************************

def read_image(filename, representation):
    """
    This function returns an image (RGB or grayscale) from filename and representation.
    :param filename the filename of an image on disk (RGB or grayscale)
    :param representation representation code of the output, either 1 (grayscale img) or 2 (RGB img)
    :return an image (np.float64 type) with intensities, normalized to the range [0,1]
    """
    # Get original image, and store the 'float64' one:
    image = imageio.imread(filename)
    img = image.astype('float64')

    # Normalize to [0,1]:
    img /= 255.0
    if representation == 1:
        img = rgb2gray(img)

    return img


def imdisplay(filename, representation):
    """
    This function displays an image (RGB or grayscale) from filename and representation.
    :param filename the filename of an image on disk (RGB or grayscale)
    :param representation representation code of the image-to-be-showed, either 1 (grayscale img) or 2 (RGB img)
    """
    img = read_image(filename, representation)
    plt.figure()

    if representation == 1:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)

    plt.axis('off')
    plt.show()


def rgb2yiq(imRGB):
    """
    The function takes an image represented in RGB, and converts it to YIQ.
    :param imRGB: RGB imaged, normalized to [0,1] scale.
    :return: a brand new YIQ image.
    """
    rgb2yiq_matrix = np.array([0.299, 0.587, 0.114,
                               0.596, -0.275, -0.321,
                               0.212, -0.523, 0.311]).reshape(3, 3)
    yiq_matrix = np.dot(imRGB, rgb2yiq_matrix.T)
    return yiq_matrix


def yiq2rgb(imYIQ):
    """
    The function takes an image represented in YIQ, and converts it to RGB.
    :param imYIQ: YIQ imaged, normalized to [0,1] scale (Y channel), and I,Q channels with [-1,1] values.
    :return: a brand new YIQ image.
    """
    yiq2rgb_matrix = np.linalg.inv(np.array([0.299, 0.587, 0.114,
                                             0.596, -0.275, -0.321,
                                             0.212, -0.523, 0.311]).reshape(3, 3))

    rgb_matrix = np.dot(imYIQ, yiq2rgb_matrix.T)
    return rgb_matrix


def histogram_equalize(im_orig):
    """
    The function implements equalization on an image.
    :param im_orig: The image to be equalized.
    :return: a list containing the new image the original histogram of the image, the histogram of the equalized image.
    """
    if im_orig.ndim == 2:
        # Grayscale - init resources:
        im_orig255 = np.round(im_orig.copy() * 255).astype(np.int)
        hist_orig = np.histogram(im_orig255, bins=256, range=(0, 255))[0]

        # Calculate histograms and lookup-table; create new image:
        lookup_table = equalize_calculations(hist_orig)
        im_eq = lookup_table[im_orig255] / 255

        # create new histogram for the image:
        hist_eq = np.histogram(im_eq, bins=256, range=(0, 255))[0]

    else:
        # RGB - init resources:
        yiq_im = rgb2yiq(im_orig)
        y_channel_img = yiq_im[:, :, 0].reshape(yiq_im.shape[0], yiq_im.shape[1])
        im_orig255 = np.round(y_channel_img.copy() * 255.0).astype(np.int)
        hist_orig = np.histogram(im_orig255, bins=256, range=(0, 255))[0]

        # Calculate histograms and lookup-table
        lookup_table = equalize_calculations(hist_orig)

        # update the Y channel; create new image:
        yiq_im[:, :, 0] = lookup_table[im_orig255] / 255
        im_eq = yiq2rgb(yiq_im)

        # create new histogram for the image (out of the Y channel):
        hist_eq = np.histogram(yiq_im[:, :, 0], bins=256, range=(0, 255))[0]

    return [im_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    Implements quantize on an image, according to the given-input arguments. Can get a grayscale OR RGB image.
    :param im_orig: the input grayscale or RGB image to be quantized.
    :param n_quant: the number of intesities the output im_quant image should have.
    :param n_iter: the maximum number of iterations of the optimization procedure.
    :return: list - [im_quant, error]
    """

    if im_orig.ndim == 2:
        # Grayscale - init resources:
        im_orig255 = np.round(im_orig.copy() * 255).astype(np.int)  # img with original intensities.

        # calculate z,q and errors:
        error, z_division, q_array = quantize_calculation(im_orig255, n_quant, n_iter)

        # create new image:
        im_quant = create_image_from_z_q(im_orig255, z_division, q_array)
    else:
        # RGB - init resources:
        yiq_im = rgb2yiq(im_orig)
        y_channel_img = yiq_im[:, :, 0].reshape(yiq_im.shape[0], yiq_im.shape[1])
        im_orig255 = np.round((y_channel_img.copy() * 255.0)).astype(np.int)

        # calculate z,q and errors:
        error, z_division, q_array = quantize_calculation(im_orig255, n_quant, n_iter)

        # update the 'y' channel; create new image:
        yiq_im[:, :, 0] = create_image_from_z_q(im_orig255, z_division, q_array)
        im_quant = yiq2rgb(yiq_im)

    return [im_quant, error]
