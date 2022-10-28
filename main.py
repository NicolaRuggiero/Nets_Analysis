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
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5",
                       trainable=True, arguments=dict(batch_norm_momentum=0.997))
    ])
    model.build([None, 224, 224, 3])
    img = img.resize(IMAGE_SHAPE)
    img = np.array(img) / 255.0

    result = model.predict(img[np.newaxis, ...])

    # print('result :' + str(result))
    # print('modulo :' + str(np.linalg.norm(result)))
    return result


def image_to_feature_vector(image):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return V3LargePredict(image)


def KNN(train_set, train_labels, test_set, test_true_labels, k):
    classifier = KNeighborsClassifier(n_neighbors=k, )
    classifier.fit(train_set, train_labels)
    results = classifier.score(test_set, test_true_labels)
    print("mean accuracy :" + str(results) + ' con k = ' + str(k))


if __name__ == '__main__':
    test = []
    train = []
    train_labels = []
    test_true_labels = []
    k_values = [2, 3, 5, 7, 10]
    for filename in glob.glob('Test_images/' + '*jpg'):
        img = Image.open(filename)
        img = image_to_feature_vector(img)
        test.append(img)
        if 'autore' in filename:
            test_true_labels.append(1)

        elif 'partir' in filename:
            test_true_labels.append(2)

        else:
            test_true_labels.append(0)

    print('Done Test')
    """
    for filename in glob.glob('train_images/partir/' + '*jpg'):
        img = Image.open(filename)
        img = image_to_feature_vector(img)
        train.append(img)
        if 'autore' in filename:
            train_labels.append(1)
        elif 'partir' in filename:
            train_labels.append(2)
    print('Done Train,partir')
    """
    for filename in glob.glob('train_images/calco/' + '*jpg'):
        img = Image.open(filename)
        img = image_to_feature_vector(img)
        train.append(img)
        if 'autore' in filename:
            train_labels.append(1)
        elif 'partir' in filename:
            train_labels.append(2)

    print('Done Train,calco')

    for i in range(0, len(k_values)):
        KNN(train, train_labels, test, test_true_labels, k_values[i])

    # links : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html ,
    #         https://pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
