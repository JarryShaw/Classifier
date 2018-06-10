import matplotlib.pyplot as plt
import numpy as np
import os


def ReadTiffData(path, rate=0.8):
    classes_path = [os.path.join(path, x) for x in os.listdir(path)]
    classes = [int(x.split('. ')[0]) for x in os.listdir(path)]

    imgs_train = []
    labels_train = []
    imgs_eval = []
    labels_eval = []
    for e in range(len(classes_path)):
        files = [os.path.join(classes_path[e], x) for x in os.listdir(classes_path[e])]
        for i in range(len(files)):
            tiff = plt.imread(files[i])
            if i < rate*len(files):
                imgs_train.append(tiff)
                labels_train.append(classes[e])
            else:
                imgs_eval.append(tiff)
                labels_eval.append(classes[e])

    imgs_train = np.asarray(imgs_train, np.float32)
    labels_train = np.asarray(labels_train, np.int32)
    shuffle = np.arange(len(imgs_train))
    np.random.shuffle(shuffle)
    imgs_train = imgs_train[shuffle]
    labels_train = labels_train[shuffle]

    imgs_eval = np.asarray(imgs_eval, np.float32)
    labels_eval = np.asarray(labels_eval, np.int32)
    shuffle = np.arange(len(imgs_eval))
    np.random.shuffle(shuffle)
    imgs_eval = imgs_eval[shuffle]
    labels_eval = labels_eval[shuffle]

    return imgs_train, labels_train, imgs_eval, labels_eval
