# 8/2/18
# Taejoon Byun
from __future__ import print_function

import time

import keras
import numpy
from keras import backend as K
from keras.models import Model

from tqdm import tqdm
import numpy as np
import scipy, IPython, sys, os, deepdish, heapq, logging, itertools, math
import pandas as pd
import multiprocessing, joblib # parallelism

import matplotlib.pyplot as plt

logger = logging.getLogger('kprio')


class KerasCoverage():

    __logger = logging.getLogger('kprio')

    def __init__(self, model):
        self.model = model
        self.obligations = None
        self.ind = 0
        self.ind_to_input = {}
        
        self.df = None    # pd.DataFrame(columns=self.__get_obligations())
        self.coverage = {}  # crit -> Dataframe
        self.__measure_coverage = {}
        self.__measure_coverage['neuron'] = self.__measure_coverage_neuron
        self.__measure_coverage['k3'] = self.__measure_coverage_k3
        self.__measure_coverage['k11'] = self.__measure_coverage_k11

    def __log(self, msg):
        KerasCoverage.__logger.info(msg)


    def run(self, input_data, num_layers=None, is_from_last=False, \
            bucket_size=None, select_layer=None):
        ''' selecting a layer via select_layer overrules the layer selection
        made via num_layers and is_from_last '''
        ilen = -1
        if type(input_data) == list:
            ilen = len(input_data[0])
        elif type(input_data) == np.ndarray:
            ilen = input_data.shape[0]
        elif type(input_data) == tuple and type(input_data[0]) == np.ndarray:
            ilen = input_data[0].shape[0]
            input_data = input_data[0]
        assert (ilen > 0)
        KerasCoverage.__logger.debug('Running {} input data'.format(ilen))

        my_layers = self.model.layers[1:-1] # exclude i/o layers
        if num_layers:
            if is_from_last:
                my_layers = self.model.layers[-(num_layers+1):-1]
            else:
                my_layers = self.model.layers[1:num_layers+1]
        if select_layer and len(self.model.layers) > select_layer:
            my_layers = [self.model.layers[select_layer]]
        outputs = [layer.output for layer in my_layers]

        # When the length of the `input_data` is larger than given
        # `bucket_size`, run the K.function multiple times in order to prevent
        # the GPU memory from overflowing.
        ats = None
        def get_flat_shape(arr):
            return reduce(lambda x, y: x * y, arr.shape[1:])
        for i in range(len(input_data) / bucket_size + 1):
            ind_a = i * bucket_size
            ind_b = min(len(input_data), (i + 1) * bucket_size)
            input_bucket = input_data[ind_a:ind_b]
            # at: activation trace per layer. List of `np.array`s
            at = K.function([self.model.input], outputs)([input_bucket])
            for i in range(len(at)):
                at[i] = np.array(at[i]).reshape(at[i].shape[0],
                        get_flat_shape(at[i]))
            if ats is None:
                ats = at
            else:
                for i in range(len(at)):
                    ats[i] = np.concatenate((ats[i], at[i]), axis=0)

        # transpose to [input x flat_actiation_trace]
        self.ATs = np.concatenate((ats), axis=1)
        # Commenting the below out to speed up. Not needed for the experiment.
        # self.df = pd.DataFrame(self.ATs, index=range(ilen))

        #KerasCoverage.__logger.debug('Converting to DataFrame')
        #TODO: Fixit
        #, columns=self.__get_obligations(neuron_outs))
        # ilen, dsa_vector = self.calculate_dsa(self.df, input_data, True, 2)
        return self.ATs


    def load_data(self, filename):
        KerasCoverage.__logger.debug('Loading data from {}'.format(filename))
        self.df = pd.read_csv(filename)

    
    def save_header(self, filename):
        self.df = pd.DataFrame(columns=self.__get_obligations())
        self.df.to_csv(filename, index=False)


    def save_data(self, filename):
        KerasCoverage.__logger.info('Saving outs')
        self.df = self.df.round(4)
        KerasCoverage.__logger.debug('writing to {}'.format(filename))
        self.df.to_csv(filename, index=False, header=False)

    def __measure_coverage_neuron(self):
        df = self.df.copy()
        func = lambda x: 1 if x > 0 else 0
        #df[col] = swifter.swiftapply(df[col], func)
        return df.applymap(func)

    def __measure_coverage_k11(self):
        def foo(x):
            if x <= 0:
                return 0
            elif x >= 10:
                return 10
            else:
                int(math.floor(x)) + 1
        return self.df.applymap(foo)

    def __measure_coverage_k3(self):
        def foo(x):
            if x <= 0:
                return 0
            elif x < 1:
                return 1
            else:
                return 2
        return self.df.applymap(foo)


    def measure_coverage(self, crit):
        KerasCoverage.__logger.debug('Measuring {} coverage'.format(crit))
        if len(self.df) == 0:
            raise Exception("No input had been provided")
        if crit not in self.__measure_coverage:
            raise Exception("Coverage criterion {} is not supported".format(crit))
        elif crit in self.coverage:
            return  # already measured
        else:
            self.coverage[crit] = self.__measure_coverage[crit]()
        #self.coverage[crit].to_csv('kakaka')
        return self.coverage[crit]


    def __get_obligations(self, neuron_outs):
        """
        :return: a list of coverage obligations.
                 i.e. ['x_1_1', 'x_1_2', ... , 'x_7_10']
        """
        if self.obligations:
            return self.obligations
        self.obligations = []
        l = [[np.reshape(each_input, (-1)) for each_input in layer] \
                for layer in neuron_outs]
        l = map(list, zip(*l))
        for i, layer in enumerate(l[0]):
            if i == 0 or i == len(l[0]) - 1:
                # skip input / output layer
                continue
            for j, _ in enumerate(layer):
                self.obligations.append('x_{}_{}'.format(i, j))
        return self.obligations


