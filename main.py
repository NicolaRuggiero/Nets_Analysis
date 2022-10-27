import os

import numpy as np

from PIL import Image
import glob
from sklearn.neighbors import KNeighborsClassifier
import pickle

"""
import tensorflow as tf
import tensorflow_hub as hub

def V3LargePredict(img, IMAGE_SHAPE):
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5",
                       trainable=True, arguments=dict(batch_norm_momentum=0.997))
    ])
    model.build([None, 224, 224, 3])
    img = img.resize(IMAGE_SHAPE)
    img = np.array(img) / 255.0

    result = model.predict(img[np.newaxis, ...])

    print('result :' + str(result))
    print('modulo :' + str(np.linalg.norm(result)))
    return result
    """


def KNN(train, train_labels, test, test_true_labels, k):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train, train_labels)
    results = classifier.score(test, test_true_labels)
    print("mean accuracy :" + str(results))


def _createTrainSet(path, l, labels, label):
    for filename in glob.glob(path + '/*.jpg'):  # assuming gif
        im = Image.open(filename)
        im = np.array(im,dtype=int)
        im.reshape(-1, 1)
        l.append(im)
        labels.append(label)


def createTrainSet(set, labels):
    subdirs = [x[0] for x in os.walk('train_images/')]
    print(subdirs)

    for i in range(1, len(subdirs)):
        _createTrainSet(subdirs[i], set, labels, i)


def createTestSet(l):
    for filename in glob.glob('test_images' + '/*.jpg'):
        im = Image.open(filename)
        im = np.array(im,dtype=int)
        im.reshape(-1, 1)
        l.append(im)
        print('test: ' + filename + 'salvato')


if __name__ == '__main__':
    train = []
    train_labels = []
    test = []

    createTrainSet(train, train_labels)
    createTestSet(test)
    test_true_labels = [3, 1, 1]

    KNN(train, train_labels, test, test_true_labels, 3)
