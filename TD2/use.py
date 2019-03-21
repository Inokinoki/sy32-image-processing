import numpy as np
from skimage import io, util
import os

import time

from sklearn.externals import joblib


def get_data(path, classes, img_width, img_height):
    pos_images = []
    neg_images = []

    #img_width = 24
    #img_height = 24

    n_pos = 0
    n_neg = 0

    for c in classes:
        for r, d, fs in os.walk(os.path.join(path, c)): #"train-data/train/pos"):
            for f in fs:
                pos_images.append(util.img_as_float(io.imread(r + "/" + f)))
        for r, d, fs in os.walk(os.path.join(path, c)):
            for f in fs:
                neg_images.append(util.img_as_float(io.imread(r + "/" + f)))

    n_pos = len(pos_images)
    n_neg = len(neg_images)

    # Init images np array
    train_images = np.zeros((n_pos + n_neg, img_width, img_height))

    for i in range(n_pos + n_neg):
        train_images[i, :, :] = (pos_images[i] if i<n_pos else neg_images[i - n_pos])

    # Init y
    train_y = np.concatenate((np.ones(n_pos), -np.ones(n_neg)))

    # reshape / ravel

    train_x = np.zeros((len(train_images) , img_width * img_height))

    for i in range(len(train_images)):
        train_x[i, :] = np.ravel(train_images[i, :, :])
    # np.ravel()

    return train_x, train_y

from sklearn.svm import SVC

m = joblib.load(".m")

test_x, test_y = get_data("train-data/test", ["pos", "neg"], 24, 24)

print(test_x.shape)

predit_y = m.predict(test_x)
print(test_y)
print(predit_y)

print("Taux d'erreur {}".format(np.mean(predit_y == test_y)))

joblib.dump(m, "{}.m".format(time.asctime()))


