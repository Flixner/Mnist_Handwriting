import os
import time

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy as np

import matplotlib.pyplot as plt


def save_model(model, folder="results", path=os.getcwd(), name="model", timestamp=True, *args, **kwargs):
    name_ops = [name, "["]

    if timestamp:
        name_ops.append(str(int(time.time())))

    i = 0
    for key, val in kwargs.items():
        if timestamp and i == 0:
            name_ops.append(", ")

        name_ops.append(str(key))
        name_ops.append("_")
        name_ops.append(str(val))
        if i + 1 != len(kwargs):
            name_ops.append(", ")

        i += 1

    name_ops.append("]")
    name_ops.append(".h5")
    fullname = "".join(name_ops)



    dir_path = os.path.join(path, folder)

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        print("Create:", dir_path)

    path = os.path.join(dir_path, fullname)
    model.save(path)
    print("Saved trained model at {}".format(path))


def load_mnist_dataset(plot=False, printInfo=False):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if plot:
        fig = plt.figure()
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.tight_layout()
            plt.imshow(X_train[i], cmap='gray', interpolation='none')
            plt.title("Digit: {}".format(y_train[i]))
            plt.yticks([])
            plt.xticks([])

    if plot:
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(X_train[0], cmap='gray', interpolation='none')
        plt.title("Digit: {}".format(y_train[0]))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 1, 2)
        plt.hist(X_train[0].reshape(784))
        plt.title("Pixel Value Distribution")

    if printInfo:
        print("X_Train shape", X_train.shape)
        print("y_train shape", y_train.shape)
        print("X_test shape", X_test.shape)
        print("y_test shape", y_test.shape)

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    if printInfo:
        print("Train matrix shape", X_train.shape)
        print("Test matrix shape", X_test.shape)
        print(np.unique(y_train, return_counts=True))

    n_classes = 10
    if printInfo:
        print("Shape before one-hot encoding: ", y_train.shape)
    Y_train: np.ndarray = np_utils.to_categorical(y_train, n_classes)
    Y_test: np.ndarray = np_utils.to_categorical(y_test, n_classes)
    if printInfo:
        print("Shape after one-hot encoding: ", Y_train.shape)

    if plot:
        plt.show()

    return X_train, Y_train, X_test, Y_test


def plot_metrics(history):
    # Plot Metrics
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="lower right")

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")

    plt.tight_layout()


def build_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model