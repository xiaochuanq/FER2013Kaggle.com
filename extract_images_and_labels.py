import csv
import numpy as np
import scipy.io as spio
import sys
from os import path
from skimage import exposure


FACE_IMAGE_SIZE_WIDTH = 48
FACE_IMAGE_SIZE_HEIGHT = 48
TRAINING = "training"

"""
This script extract csv file and persist labels and images into MATLAB .mat files
"""

def parse_csv_data(data_path):
    raw_train_y = []
    raw_train_x = []
    raw_test_y = []
    raw_test_x = []
    with open(data_path, 'rb') as csvfile:
        data_reader = csv.reader(csvfile, delimiter=',')
        next(data_reader) # Skip the first row
        for row in data_reader:
            if row[2] == TRAINING:
                raw_train_y.append(int(row[0]))
                raw_train_x.append(row[1])
            else:
                raw_test_y.append(int(row[0]))
                raw_test_x.append(row[1])

    train_labels = np.array(raw_train_y, np.int)
    train_images = pixel_string_list_to_float_images(raw_train_x, FACE_IMAGE_SIZE_WIDTH, FACE_IMAGE_SIZE_HEIGHT)
    test_labels = np.array(raw_test_y, np.int)
    test_images = pixel_string_list_to_float_images(raw_test_x, FACE_IMAGE_SIZE_WIDTH, FACE_IMAGE_SIZE_HEIGHT)

    return train_labels, train_images, test_labels, test_images

"""
Floating number of 32 bit is used across all scripts in order to save memory. Precision should be enough
"""
def pixel_string_list_to_float_images(pixel_string_list, n_row, n_col):
    n = len(pixel_string_list)
    images = np.empty((n, n_row, n_col), np.float32)

    for i in range(n):
        images[i] = np.fromstring(pixel_string_list[i], dtype=np.float32, sep=' ').reshape((n_row, n_col))

    return images


def equalize_images(images):
    for i in range(images.shape[0]):
        images[i] = exposure.equalize_hist(images[i])



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print 'Usage: python extract_images_and_labels.py path/to/kaggle/fer2013.csv path/to/working/folder'
        exit(1)

    # Extract and convert strings in CSV file to images and labels
    train_labels, train_images, test_labels, test_images = parse_csv_data(sys.argv[1])

    # Histogram equlize of all images, in order to remove skin color and lighting impact
    equalize_images(train_images)
    equalize_images(test_images)

    # Persist all data in MATLAB format
    spio.savemat(path.join(sys.argv[2], 'train_images.mat'), {'images':train_images})
    spio.savemat(path.join(sys.argv[2], 'train_labels.mat'), {'labels':train_labels})
    spio.savemat(path.join(sys.argv[2], 'test_images.mat'), {'images':test_labels})
    spio.savemat(path.join(sys.argv[2], 'test_labels.mat'), {'labels':test_images})