import numpy as np
import scipy.io as spio
import sys
from os import path
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from time import localtime, strftime


def create_gabor_filters(frequencies, thetas, sigmaX, sigmaY):
    """
    :param frequencies:
    :param thetas: Orientations
    :param sigmaX: Gaussian component sigma
    :param sigmaY: In another direction
    :return: A list of gabor kernels
    """
    kernels = []
    for frequency in frequencies:
        for theta in thetas:
            kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigmaX, sigma_y=sigmaY))
            kernels.append(kernel)
    return kernels


def compute_all_filter_responses(images, kernels):
    features = np.empty((images.shape[0], len(kernels), images.shape[1], images.shape[2]), dtype=np.double)
    for i, image in enumerate(images):
        # each feature in the features array has the number of kernels responses as images
        features[i] = convolve_filters(image, kernels)
    return features


def convolve_filters(image, kernels):
    feats = np.empty((len(kernels), image.shape[0], image.shape[1]), dtype=np.double)
    for k, kernel in enumerate(kernels):
        feats[k] = ndi.convolve(image, kernel, mode='nearest')
    return feats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print 'Usage: python filter_images.py path/to/working/folder'
        exit(1)

    # Load train and test images
    train_images = spio.loadmat(path.join(sys.argv[1], 'train_images.mat'))['images']
    test_images = spio.loadmat(path.join(sys.argv[1], 'test_images.mat'))['images']

    # Create a bank of Gabor filters
    # Wave length should be > 2 and < 1/5 of the image size (48x48) ,so I choose range(3, 15, 3)
    # Orientation is [0, pi) step pi/8
    # Sigmas set as pi
    # Totally 32 filters
    kernels = create_gabor_filters([1.0/x for x in range(3, 15, 3)], [x*np.pi*0.125 for x in range(8)], np.pi, np.pi)

    # Group every 1000 images, and persist the filter responses.
    # Doing this because the responses are very big and I difficulty in holding them in one gigantic array in memory
    # Here is the math, I have 32 filters per 48x48 floating point (float32) image. Totally about 30,000 images,
    # That can be 32 * 48 * 48 * 4 * 30,000 = 8847360000 bytes > 8G
    # The job is time consuming, print on screen for tracking progresses
    for i in [ x*1000 for x in range(train_images.shape[0]/1000+1)]:
        print "Computing training feature group " + str(i) + strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())
        features = compute_all_filter_responses(train_images[i:i+1000], kernels)
        spio.savemat(path.join(sys.argv[1], 'train_raw_features_set_'+ str(i) +'.mat'), {'features':features})

    # Doing the same operations on test images though the amount of images is small
    for i in [ x*1000 for x in range(test_images.shape[0]/1000+1)]:
        print "Computing testing feature group " + str(i) + strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())
        features = compute_all_filter_responses(test_images[i:i+1000], kernels)
        spio.savemat(path.join(sys.argv[1], 'test_raw_features_set_'+ str(i) +'.mat'), {'features':features})
