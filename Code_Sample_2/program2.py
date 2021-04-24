from skimage.color import rgb2gray
import imageio
import numpy as np
import scipy.io.wavfile as wf
from math import floor
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from scipy.signal import convolve2d


# *******************************
# ************* DFT *************
# *******************************

# ************* HELPERS *************

def make_dft_trans_matrix(n):
    """
    :param n: The signal's shape.
    :return: The dft trnasformation matrix
    """
    if n == 0:
        return np.empty((0, 0))
    cols, rows = np.meshgrid(np.arange(n), np.arange(n))
    transform_matrix = (-2 * np.pi * complex(0, 1) / n) * (cols * rows)
    transform_matrix = np.exp(transform_matrix)
    return transform_matrix


# ************* 1D-DFT *************

def DFT(signal):
    """
    transform a 1D discrete signal to its Fourier representation.
    :param signal: A signal (1D)
    :return: A matching Fourier representation
    """
    n = signal.shape[0]
    transform_matrix = make_dft_trans_matrix(n)
    complex_fsignal = np.dot(transform_matrix, signal)
    return complex_fsignal


def IDFT(fourier_signal):
    """
     transform a Fourier representation to its 1D discrete signal
    :param fourier_signal: The Fourier signal.
    :return: 1D discrete signal
    """
    n = fourier_signal.shape[0]
    transform_matrix = np.linalg.inv(make_dft_trans_matrix(n))
    complex_signal = np.dot(transform_matrix, fourier_signal)

    return complex_signal


# ************* 2D-DFT *************

def DFT2(image):
    """
    convert a 2D discrete signal to its Fourier representation
    :param image: grayscale image of dtype float64
    :return: 2D array of dtype complex128 (?)
    """
    original_dim = image.ndim
    if original_dim == 3:
        image = image.reshape(image.shape[0], image.shape[1])

    out = DFT(DFT(image).T).T

    if original_dim == 3:
        out = out.reshape(out.shape[0], out.shape[1], 1)
    return out


def IDFT2(fourier_image):
    """
    convert a Fourier representation to its 2D discrete signal.
    :param fourier_image: 2D array of dtype complex128
    :return:
    """
    original_dim = fourier_image.ndim
    if original_dim == 3:
        fourier_image = fourier_image.reshape(fourier_image.shape[0], fourier_image.shape[1])

    out = IDFT(IDFT(fourier_image).T).T

    if original_dim == 3:
        out = out.reshape(out.shape[0], out.shape[1], 1)
    return out


# ************* Fast Forward by Rate Change *************
def change_rate(filename, ratio):
    """
    changes the duration of an audio file by keeping the same samples, but changing the sample rate written in the
    file header.
    :param filename: a string representing the path to a WAV file
    :param ratio: is a positive float64 representing the duration change.
    :return:
    """

    wav_sample_rate, wav_content = wf.read(filename)
    wav_sample_rate *= ratio
    wf.write(filename='change_rate.wav', rate=int(wav_sample_rate), data=wav_content)


# *******************************************************
# ************* Fast Forward using Fourier **************
# *******************************************************

# ************* Helper Functions *************

def create_padding(data_len, ratio):
    """
    padding the data / rearranging the data size, preparing the data-set for the speed changing process.
    :param data_len: the current length of the data.
    :param ratio: the speed-change ratio.
    :return: the data set, with padding.
    """
    new_sample_num = int(data_len / ratio)
    if new_sample_num < data_len:
        padding = (data_len - new_sample_num)
    else:
        padding = new_sample_num - data_len
    if padding % 2 == 0:
        pad_left = pad_right = int(padding / 2)
    else:
        pad_left = int(floor(padding / 2))
        pad_right = int(floor(padding / 2) + 1)

    return pad_left, pad_right


def change_speed(data, ratio):
    """
    Changing the speed rate of given data according to a given ratio.
    :param data: the given data.
    :param ratio: the ratio. above 1: faster, between 0 to 1: slower.
    :return: the data frequencies, after transformation applied.
    """
    pad_left, pad_right = create_padding(len(data), ratio)
    data_frequencies = np.fft.fftshift(DFT(data))
    if ratio > 1:
        data_frequencies = IDFT(np.fft.ifftshift(data_frequencies[pad_left:-pad_right]))
    else:
        data_frequencies = IDFT(np.fft.ifftshift(np.pad(data_frequencies, (pad_right, pad_left), 'constant')))

    return data_frequencies


# ************* API Functions *************

def resize(data, ratio):
    """
    Using DFT and IDFT.
    :param data: 1D ndarray of dtype float64 or complex128, representing the original sample points.
    :param ratio:
    :return: 1D ndarray of type 'data', representing the new sample points.
    """
    if ratio == 1:
        return data
    else:
        new_data = change_speed(data, ratio).astype(data.dtype)

    return new_data


def change_samples(filename, ratio):
    """
    Change the wav smaple using DFT and IDFT.
    :param filename: a path to WAV file
    :param ratio: positive float64 representing duration change.
    :return: 1D ndarry of data type float64, representing the new sample points.
    """
    wav_sample_rate, wav_content = wf.read(filename)
    new_wav_content = resize(wav_content, ratio)
    wf.write(filename='change_samples.wav', rate=wav_sample_rate, data=new_wav_content)

    return new_wav_content.astype('float64')


# *******************************************************
# *********** Fast Forward using Spectrogram ************
# *******************************************************

def resize_spectrogram(data, ratio):
    """
    speeds up a WAV file, without changing the pitch, using spectrogram scaling.
    USE: stft, istft, win_length, hop_length.
    :param data: a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: a positive float64 representing the rate change of the WAV file (0.25 < ratio < 4)
    :return: the new sample points according to ratio with the same datatype as data
    """
    stft_results = stft(data)
    after_resize = np.vstack([resize(row, ratio) for row in stft_results])
    new_samples = istft(after_resize).astype(data.dtype)

    return new_samples


def resize_vocoder(data, ratio):
    """
    :param data: a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: a positive float64 representing the rate change of the WAV file (0.25 < ratio < 4)
    :return: the new sample points according to ratio with the same datatype as data
    """
    stft_results = stft(data)
    after_resize = phase_vocoder(stft_results, ratio)
    new_samples = istft(after_resize).astype(data.dtype)

    return new_samples


# *******************************************************
# ****************** Image Derivatives ******************
# *******************************************************
def magnitude(dx, dy):
    """
    calculates the magnitude of an image, using the derivatives (x and y directions).
    :param dx: derivative on the 'x' axis
    :param dy: derivative on the 'y' axis
    :return: the magnitude
    """
    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2).reshape((dx.shape[0], dy.shape[1]))


def conv_der(im):
    """
    computes the magnitude of image derivatives using convolution.
    :param im: grayscale images of type float64
    :return: grayscale images of type float64 - the magnitude of the derivative
    """

    derivative = np.fromiter((0.5, 0, -0.5), dtype='float64').reshape(1, 3)
    dx = convolve2d(im, derivative, mode="same")
    dy = convolve2d(im, derivative.T, mode="same")
    dxy = magnitude(dx, dy).astype('float64')

    return dxy


def fourier_der(im):
    """
    computes the magnitude of the image derivatives using Fourier transform.

    :param im: grayscale images of type float64
    :return: grayscale images of type float64 - the magnitude of the derivative
    """
    # Create the frequencies arrays, shifted to the middle:
    u, v = im.shape[0], im.shape[1]
    left_u = np.arange(u // 2)
    right_u = (np.arange(-(u // 2), 0))
    new_u = np.fft.fftshift(np.concatenate((left_u, right_u))).T
    left_v = np.arange(v // 2)
    right_v = (np.arange(-(v // 2), 0))
    new_v = np.fft.fftshift(np.concatenate((left_v, right_v))).T

    # derive the image on both axis, calculate magnitude:
    d_by_y = IDFT(np.fft.ifftshift((np.fft.fftshift((DFT(im.copy())).reshape(u, v))) * new_u[:, None])) * (
            2 * np.pi * 1J / u)
    d_by_x = IDFT(np.fft.ifftshift((np.fft.fftshift((DFT(im.copy().T)).reshape(v, u))) * new_v[:, None])).T * (
            2 * np.pi * 1J / v)
    dxy = magnitude(d_by_x, d_by_y)

    return dxy


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


# ****************************************************************
# ************* PROVIDED HELPERS BY THE COURSE STAFF *************
# ****************************************************************
def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec
