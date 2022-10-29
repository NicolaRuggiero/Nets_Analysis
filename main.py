from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import glob
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub


def V3LargePredict(img, IMAGE_SHAPE=(224, 224)):
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5")
    ])
    model.build([None, 224, 224, 3])
    img = img.resize(IMAGE_SHAPE)
    img = np.array(img)

    result = model.predict(img[np.newaxis, ...])

    # print('result :' + str(result))
    # print('modulo :' + str(np.linalg.norm(result)))
    return result


def image_to_feature_vector(image):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return V3LargePredict(image)


def KNN(train_set, train_labels, test_set, test_true_labels, k):
    classifier = KNeighborsClassifier(n_neighbors=k)

    train_set = np.array(train_set)
    # print('shape of train_set :' + str(train_set.shape))
    nsamples, nx, ny = train_set.shape
    d2_train_set = train_set.reshape((nsamples, nx * ny))

    test_set = np.array(test_set)
    # print('shape of test_set :' + str(test_set.shape))
    nsamples, nx, ny = test_set.shape
    d2_test_set = test_set.reshape((nsamples, nx * ny))

    classifier.fit(d2_train_set, train_labels)
    results = classifier.score(d2_test_set, test_true_labels)
    print("mean accuracy :" + str(results) + ' con k = ' + str(k))


if __name__ == '__main__':
    test = []
    train = []
    train_labels = []
    test_true_labels = []
    k_values = [2, 3, 5, 7, 10, 20, 50]
    for filename in glob.glob('Test_images/' + '*jpg'):
        img = Image.open(filename)
        img = image_to_feature_vector(img)
        test.append(img)

        if 'partir' in filename:
            test_true_labels.append(1)


        elif 'autore' in filename:
            test_true_labels.append(2)


        elif 'porcellino' in filename:
            test_true_labels.append(3)

        else:
            test_true_labels.append(0)

    for filename in glob.glob('train_images/partir/' + '*jpg'):
        img = Image.open(filename)
        img = image_to_feature_vector(img)
        train.append(img)
        train_labels.append(1)
    print('Done Train,partir')

    for filename in glob.glob('train_images/calco/' + '*jpg'):
        img = Image.open(filename)
        img = image_to_feature_vector(img)
        train.append(img)
        train_labels.append(2)
    print('Done Train,calco')

    for filename in glob.glob('train_images/porcellino/' + '*jpg'):
        img = Image.open(filename)
        img = image_to_feature_vector(img)
        train.append(img)
        train_labels.append(3)

    for i in range(0, len(k_values)):
        KNN(train, train_labels, test, test_true_labels, k_values[i])

    # links : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html ,
    #         https://pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
