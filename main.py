import sys

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
import pickle
import tensorflow as tf
import tensorflow_hub as hub


def V3LargePredict(img, IMAGE_SHAPE=(224, 224)):
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5")
    ])
    model.build([None, 224, 224, 3])
    img = img.resize(IMAGE_SHAPE)
    img = np.array(img) / 255

    result = model.predict(img[np.newaxis, ...])

    # print('result :' + str(result.shape))
    # print('modulo :' + str(np.linalg.norm(result)))
    return result


def efficientPredict(img, IMAGE_SHAPE=(224, 224)):
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/lite4/classification/2")
    ])
    model.build([None, 224, 224, 3])
    img = img.resize(IMAGE_SHAPE)
    img = np.array(img) / 255

    result = model.predict(img[np.newaxis, ...])

    # print('result :' + str(result.shape))
    # print('modulo :' + str(np.linalg.norm(result)))
    return result


def image_to_feature_vector(image):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return efficientPredict(image)


def distanceDataset(train_set, arr):
    results = []
    for i in range(0, len(train_set)):
        d = np.linalg.norm(arr - train_set[i])
        if (d < 90.0):
            results.append(d)
        else:
            results.append(sys.maxsize)
    return results


def find_k_labels(index, train_labels):
    results = []
    for i in range(0, len(index)):
        results.append(train_labels[index[i]])
    return results


def most_frequent(List):
    return max(set(List), key=List.count)


def knn_byhand(train_set, train_labels, test_set, test_true_labels, k):
    results = []
    for i in range(0, len(test_set)):
        d = distanceDataset(train_set, test_set[i])
        #print('img : ' + str(i))
        #print('distanze: ' + str(d))
        k_labels_index = np.argpartition(d, k)
        k_labels_index = k_labels_index[0:k]
        #print('k_labels_index :' + str(k_labels_index))
        k_labels = find_k_labels(k_labels_index, train_labels)
        #print('k_labels:' + str(k_labels))
        w = most_frequent(k_labels)
        #print('w' + str(w))
        #print('test_true_label ' + str(test_true_labels[i]))

        if (w == test_true_labels[i]):
            results.append(1)
        else:
            results.append(0)

    score = sum(results) / len(results)
    print('mean accuracy per k = ' + str(k) + ' = ' + str(score))
    return score


def knn(train_set, train_labels, test_set, test_true_labels, k):
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

    """
    for filename in glob.glob('testKNN/train/' + '*jpg'):
        img = Image.open(filename)
        img = image_to_feature_vector(img)
        train.append(img)
        if 'partir' in filename:
            test_true_labels.append(1)


        elif 'autore' in filename:
            train_labels.append(2)


        elif 'porcellino' in filename:
            train_labels.append(3)

        else:
            train_labels.append(0)


    for filename in glob.glob('testKNN/train/' + '*jpg'):
        img = Image.open(filename)
        img = image_to_feature_vector(img)
        train.append(img)
        if 'partir' in filename:
            test_true_labels.append(1)


        elif 'autore' in filename:
            train_labels.append(2)


        elif 'porcellino' in filename:
            train_labels.append(3)

        else:
            train_labels.append(0)
    print('Done Train,calco')
    
    for filename in glob.glob('testKNN/train/' + '*jpg'):
        img = Image.open(filename)
        img = image_to_feature_vector(img)
        train.append(img)
        if 'partir' in filename:
            test_true_labels.append(1)


        elif 'autore' in filename:
            train_labels.append(2)


        elif 'porcellino' in filename:
            train_labels.append(3)

        else:
            train_labels.append(0)
    """
    train = pickle.load(open('train_dataset_efficient.p','rb'))
    train_labels = pickle.load(open('train_labels.p', 'rb'))
    print('len di train_dataset = ' + str(len(train)))
    print('len di train_dataset_labels = ' + str(len(train_labels)))
    for i in range(0, len(k_values)):
        knn_byhand(train, train_labels, test, test_true_labels, k_values[i])

    pickle.dump(train, open('train_dataset_efficient.p', 'wb'))
    pickle.dump(train_labels, open('train_labels.p', 'wb'))

# links : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html ,
#         https://pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
# reti :  efficient ,https://tfhub.dev/tensorflow/efficientnet/lite4/classification/2
#         V3large, https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5
