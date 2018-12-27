from __future__ import print_function

import keras
from keras import backend as K

import dataset
import numpy as np
import scipy, scipy.io
import IPython



class MNIST(dataset.Dataset):

    def __init__(self, flatten=False):
        self.rows, self.cols, self.num_classes = 28, 28, 10
        self.flatten = flatten
        super(MNIST, self).__init__()

    def _load(self):
        (ax, ay), (bx, by) = keras.datasets.mnist.load_data()
        self.data = {'train': {'x': ax, 'y': ay}, 'test': {'x': bx, 'y': by}}

    def _preprocess(self):
        if self.flatten:
            self._preprocess_mlp()
        else:
            self._preprocess_cnn()

    def _preprocess_cnn(self):
        r, c = self.rows, self.cols
        my_reshape = None
        if K.image_data_format() == 'channels_first':
            my_reshape = lambda d: d.reshape(d.shape[0], 1, r, c, order="A")
            self.input_shape = (1, r, c)
        else:
            my_reshape = lambda d: d.reshape(d.shape[0], r, c, 1, order="A")
            self.input_shape = (r, c, 1)
        for v1, d in self.data.iteritems():
            for v2, d2 in d.iteritems():
                if v2 == "x":
                    self.data[v1][v2] = my_reshape(d2).astype(np.float32) / 255
                elif v2 == "y":
                    # convert class vectors to binary class matrices
                    self.data[v1][v2] = keras.utils.to_categorical(d2,
                            self.num_classes)

    def _preprocess_mlp(self, is_matlab=False):
        train, test = self.data['train'], self.data['test']
        train['x'] = train['x'].reshape(train['x'].shape[0], 1, 28, 28, order="A")
        train['x'] = train['x'].reshape(train['x'].shape[0], 28 * 28)
        test['x'] = test['x'].reshape(test['x'].shape[0], 1, 28, 28, order="A")
        test['x'] = test['x'].reshape(test['x'].shape[0], 28 * 28)
        self.input_shape = 28 * 28
        #print(train['x'].shape)
        train['x'] = train['x'].astype(np.float32) / 255
        test['x'] = test['x'].astype(np.float32) / 255
        # convert class vectors to binary class matrices
        train['y'] = keras.utils.to_categorical(train['y'], self.num_classes)
        test['y'] = keras.utils.to_categorical(test['y'], self.num_classes)
        self.data['train'], self.data['test'] = train, test
        self.data['all'] = {}
        self.data['all']['x'] = np.concatenate((train['x'], test['x']))
        self.data['all']['y'] = np.concatenate((train['y'], test['y']))

    def show(self, category, ind1):
        if not self.plotter:
            self.plotter = Plotter()
        self._validate(category, "x")
        img = self.data[category]["x"][ind1].astype(np.float32) * 255
        img = img.reshape(self.rows, self.cols)
        self.plotter.add_subplot(img)



class EMNIST(MNIST):
    """ EMNIST Resources:
        - https://arxiv.org/pdf/1702.05373.pdf
        - https://github.com/j05t/emnist/blob/master/emnist.ipynb

    240,000 training set, 40,000 test set
    """

    def __init__(self, flatten=False):
        super(EMNIST, self).__init__(flatten=flatten)

    def _load(self, dataset_path="./datasets/emnist-digits.mat"):
        emnist = None
        try:
            emnist = scipy.io.loadmat(dataset_path)
        except:
            emnist = scipy.io.loadmat("./datasets/emnist-digits.mat")
        x_train = emnist["dataset"][0][0][0][0][0][0]
        x_train = x_train.astype(np.float32)
        y_train = emnist["dataset"][0][0][0][0][0][1]
        self.data['train'] = {'x': x_train, 'y': y_train}

        x_test = emnist["dataset"][0][0][1][0][0][0]
        x_test = x_test.astype(np.float32)
        y_test = emnist["dataset"][0][0][1][0][0][1]
        self.data['test'] = {'x': x_test, 'y': y_test}



