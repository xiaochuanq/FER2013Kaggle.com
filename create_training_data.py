import numpy as np
import scipy.io as spio
import sys
from os import path
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from time import localtime, strftime


IMAGE_SIZE = 48

"""
This script does two jobs for one purpose: To reduce dimension
    1) Downsample the original image filter responses. Gabor filter gave us 48 * 48 * 32 = 73728 dimensions,
       which is way to high. Even a PCA (by SVD) is not computable on this dimesion on my computer.
       Since Gabor filter has a gaussian component. So the response is already blurred. Downsampling makes sense
       for dimension reduction
    2) Afte downsampling, the # of features is still too high (up to a few thousands). So I did a Singular value
       decomposition to select features. There are other feature selection methods such as Information gain,
       Chi-square, may be worth of trying, either combined with this or separately.
"""

def create_training_features(raw_features, vector_size, sample_coord):
    """
    :param raw_features: an array of the banks of filter responses per image
    :param vector_size: the size of the descriptor used for training/testing
    :param sample_coord: at which coordinates to sample
    :return:
    """
    bank_size = raw_features[0].shape[0]  # number of filter responses of a filter bank.
    n = len(sample_coord)                 # down sample image width and height
    mesh = np.meshgrid(sample_coord, sample_coord);
    training_features = np.zeros((raw_features.shape[0], vector_size), dtype=np.float32) # allocate buffer for all training features

    for i, images in enumerate(raw_features):
        # reshape all downsampled responses as a number_of_features by 1 vector
        training_features[i] = down_sample_filter_responses(images, bank_size, n, mesh).reshape(vector_size)

    return training_features


def down_sample_filter_responses(images, bank_size, n, meshgrids):
    # bank_size == images.shape[0]; length = bank_size * n * n; meshgrid should match n
    # precompute these variables to save time
    new_images = np.zeros((bank_size, n, n), dtype=np.double)
    for i, image in enumerate(images):
        new_images[i] = image[meshgrids]

    return new_images


def batch_downsample_and_vectorize_filter_responses(filter_response_path_prefix, nsamples, vector_size, sample_coord, dtype=np.float32):
    X = np.zeros((nsamples, vector_size), dtype=dtype)
    i = 0
    for batch in [x *1000 for x in range(nsamples/1000 + 1)]:
        features = spio.loadmat(filter_response_path_prefix + str(batch) + '.mat')['features']
        batch_size = features.shape[0]
        print "Downsampling batch " + str(i) + '-->' + str(i+batch_size)
        X[range(i, i+batch_size)] = create_training_features(features, vector_size, sample_coord)
        i = i + batch_size

    return X


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print 'Usage: python create_training_data.py path/to/working/folder size_of_filter_bank'
        exit(1)

    size_of_filter_bank = int(sys.argv[2])
    sample_coord = range(3,IMAGE_SIZE,4)
    n_components = 100;

    # Get the training and test sample sizes
    n_train = len(spio.loadmat(path.join(sys.argv[2], 'train_labels.mat'))['labels'][0]) # need specify [0] to access the 'Y' vector
    n_test = len(spio.loadmat(path.join(sys.argv[2], 'test_labels.mat'))['labels'][0])

    # Load and downsample grouped filter responses for train and test dataset respectively
    vector_size = size_of_filter_bank * len(sample_coord) * len(sample_coord)
    Xtrain = batch_downsample_and_vectorize_filter_responses(
        path.join(sys.argv[1], "train_raw_features_set_"),
        n_train,
        vector_size,
        sample_coord)

    # same thing for test data
    Xtest = batch_downsample_and_vectorize_filter_responses(
        path.join(sys.argv[1], "test_raw_features_set_"),
        n_test,
        vector_size,
        sample_coord)

    print 'Starting SVD decomposition @' + strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())
    pca = PCA(n_components)
    pca.fit(Xtrain)
    joblib.dump(path.join(sys.argv[1], 'pca.svd')) # Persis it. SVD computation takes quite a while
    print 'Completed @' + strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())

    Xtrain = pca.transform(Xtrain)
    Xtest = pca.transform(Xtest)

    # Persist training and testing dataset
    spio.savemat(path.join(sys.argv[1], "xtrain.mat"), {'X': Xtrain})
    spio.savemat(path.join(sys.argv[1], "xtest.mat"), {'X': Xtest})