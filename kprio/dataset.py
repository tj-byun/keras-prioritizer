from __future__ import print_function

import keras
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Model

import sys
import os
import copy
import logging
import random
import collections
import cv2
import cPickle
import numpy as np
import scipy, scipy.io
import IPython
from multiprocessing import Pool

import matplotlib.pyplot as plt

logger = logging.getLogger('kprio')


class Plotter(object):
    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=(self.rows, self.cols))
        self.cnt = 0

    def add_subplot(self, fig):
        if self.cnt >= self.rows * self.cols:
            raise Exception('Figure is full')
        self.cnt += 1
        self.fig.add_subplot(self.rows, self.cols, self.cnt)
        plt.imshow(fig)

    def show(self):
        plt.show()


class Query(object):

    def __init__(self, category, variable, indices):
        self.category = self.cat = category
        self.variable = self.var = variable
        self.indices = self.ind = indices

    def copy(self, category=None, variable=None, indices=None):
        q = Query(self.category, self.variable, self.indices)
        if category:
            q.set_category(category)
        if variable:
            q.set_variable(variable)
        if indices:
            q.set_indices(indices)
        return q

    def set_category(self, cat):
        self.category = self.cat = cat

    def set_variable(self, var):
        self.variable = self.var = var

    def set_indices(self, ind):
        self.indices = self.ind = ind

    def map_indices(self, indices):
        """ map indices to indices (input ids) """
        return [self.indices[ind] for ind in indices]

    def to_dict(self):
        return {
                "category": self.category,
                "variable": self.variable,
                "indices": self.indices,
                }

    def __str__(self):
        return "<Query ('{}', '{}', {})>".\
                format(self.cat, self.var, self.ind)



class Dataset(object):

    def __init__(self):
        self.plotter = None
        self.data = {}
        self._load()
        self._preprocess()
        self._create_merged_set()

    def _create_merged_set(self):
        xs = np.concatenate([self.data[key]["x"] for key in self.data])
        ys = np.concatenate([self.data[key]["y"] for key in self.data])
        self.data["all"] = {"x": xs, "y": ys}

    def _load(self):
        raise NotImplemented("Abstract method `_load`")

    def _preprocess(self):
        raise NotImplemented("Abstract method `_load`")

    def show(self):
        raise NotImplemented("Abstract method `show`")

    def augment(self, cat, var, data):
        """ Augment """
        self.data[cat][var] = np.concatenate((self.data[cat][var], data), axis=0)

    def is_correct(self, x_query, y):
        """ Given an x_query (Query of input data) and a list of prediction
        outputs `y`, determine if the output is correct w.r.t.
        """
        x = self.gets_query(x_query)
        Y_query = x_query.copy()
        Y_query.set_variable("y")
        Y, _ = self.gets_query(Y_query)
        Y = np.array(map(np.argmax, Y))
        y = np.array(map(np.argmax, y))
        assert len(Y) == len(y), "{} != {}".format(len(Y), len(y))
        return Y == y

    def get_error(self, x_query, y):
        """ Get normalized error vector """
        yq = x_query.copy()
        yq.set_variable("y")
        x = self.gets_query(x_query)
        Y, _ = self.gets_query(yq)
        return np.abs(Y - y)

    def display(self, x_query, y, dims):
        raise NotImplementedError("Dataset::display(): unimplemented")
    
    def _validate(self, category, variable):
        if category not in self.data:
            raise Exception("invalid category {}".format(category))

    # Retrieve

    def get_length(self, category, variable):
        return len(self.data[category][variable])

    def gets_query(self, query):
        return self.gets(query.category, query.variable, query.indices)

    def gets_list(self, category, variable, indices):
        self._validate(category, variable)
        items = []
        for i in indices:
            items.append(self.data[category][variable][i:i+1])
        return items, indices

    def gets(self, category, variable, indices=None):
        self._validate(category, variable)
        items = []
        if indices is not None:
            for i in indices:
                items.append(self.data[category][variable][i:i+1])
            return np.concatenate(items, axis=0), indices
        else:
            items = self.data[category][variable][0:]
            indices = range(len(items))
            return items, indices

    def gets_random(self, category, variable, cnt):
        """ get `cnt` number of data in a random order. """
        inds = range(len(self.data[category][variable]))
        random.shuffle(inds)
        new_inds = inds[:cnt]
        return self.gets(category, variable, new_inds)

    def get_range(self, category, variable, ind_from, ind_to=None):
        self._validate(category, variable)
        return self.data[category][variable][ind_from:ind_to]
        #range(ind_from, ind_to if ind_to else len(self.data[category][variable]))

    def get(self, category, variable, ind_from, ind_to=None):
        raise Exception("Deprecated")
        return self.get_range(category, variable, ind_from, ind_to)

