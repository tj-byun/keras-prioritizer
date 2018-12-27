from __future__ import print_function

import keras
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Model

import numpy as np
import scipy, scipy.io, IPython, sys, os, deepdish, heapq, glob, logging, joblib, shutil, tqdm
import tensorflow as tf
import multiprocessing

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets

from kerascov.coverage import KerasCoverage
from kerascov.entropy import KerasEntropy

logger = None

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

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

def get_dataset(model, dataset_name):
    logger.debug("Loading dataset")
    flat = len(model._layers[0].batch_input_shape) == 2
    if dataset_name.upper() == "EMNIST":
        return Datasets.EMNIST(flatten=flat)
    else:
        return Datasets.MNIST(flatten=flat)


def exp_run(dataset_name, model_path):
    """ For a given (training) dataset and a model, calculate its activation
    pattern frequency """

    model = keras.models.load_model(model_path)
    dataset = get_dataset(model, dataset_name)
    
    kcovs = []
    N, I, J = 60000, 1, 60
    increment = N / J
    for i in range(I, J):
        logger.info("Initializing KerasCoverage")
        kcov = KerasCoverage(model, dataset)
        if i == 0:
            kcov.save_header(filename="frequency/header.csv")
        i_from, i_to = i * increment, i * increment + increment
        src = '/media/ramdisk/tmp'
        dst = "frequency/{}_{}_{}_{}.csv".format(dataset_name, model_path.split('/')[-1], i_from, i_to)
        logger.info("Measuring coverage from {} to {}".format(i_from, i_to))
        # run all training data
        kcov.run([dataset.get_range("train", "x", i_from, i_to)])
        kcov.save_outs(filename=src)
        shutil.move(src, dst)
        #kcovs.append(kcov)
    #for kcov in kcovs:
        #kcov.save_outs(filename=fname)


def run_and_measure(model_path, dataset, criterion, inc, i):
    a, b = i * inc, (i + 1) * inc
    msg = "{:3d}: run_and_measure from {} to {}".format(i, a, b)
    logger.info(msg)
    model = keras.models.load_model(model_path)
    kcov = KerasCoverage(model, dataset)
    kcov.run([dataset.get_range("train", "x", a, b)])
    kcov.measure_coverage(criterion)
    entropy = KerasEntropy()
    entropy.measure(kcov.coverage[criterion])
    logging.info("{:3d}: done".format(i))
    return entropy


def exp_entropy(dataset_name, criterion, model_path, i, j):
    model = keras.models.load_model(model_path)
    dataset = get_dataset(model, dataset_name)
    inc = 1000
    N = len(dataset.get_range("train", "x", 0))   # # of training data: 40k or 240k
    I, J = int(i), int(j)
    logger.info("Running parallel jobs")
    entropies = joblib.Parallel(n_jobs=10)\
            (joblib.delayed(run_and_measure)\
            (model_path, dataset, criterion, inc, i,)\
            for i in tqdm.tqdm(range(I, J)))
    logger.info("Parallel jobs are completed")
    entropy = None
    for e in entropies:
        if entropy is None:
            entropy = e
        else:
            entropy = entropy.merge(e)
    entropy.save_data("entropy_{}_{}_{}-{}.csv".format(criterion, I * inc, J * inc, model_path.split('/')[-1]))
    return


def exp_uncertainty(dataset_name, model_path):
    model = keras.models.load_model(model_path)
    dataset = get_dataset(model, dataset_name)

    kcov = KerasCoverage(model, dataset)
    entropy = KerasEntropy()
    logger.info("loading entropy data")
    entropy.load_data("entropy_k3_0_60000-mnist_cnn_small_10.h5.csv", 60 * 1000)
    logger.info("running dataset")
    input_data = dataset.get_range("train", "x", 0, 600)
    kcov.run([input_data])
    logger.info("measuring coverage")
    df = kcov.measure_coverage('k3')

    logger.info("Calculating entropy")
    entropies = entropy.get_entropies(df)

    predictions = map(np.argmax, model.predict([input_data]))
    maxes = map(np.amax, model.predict([input_data]))
    labels = map(np.argmax, dataset.get_range("train", "y", 0, 600))
    dicts = [{"e": entropies[i], "p": predictions[i], "l": labels[i], \
            "m": maxes[i], "i": i} for i in range(600)]
    dicts = sorted(dicts, key=lambda d: d["e"])
    for d in dicts:
        print("[{:4d}] {:5s} {:.2f} {:6.1f}".format(d["i"], str(d["p"] == d["l"]), d["m"], d["e"]))


def exp_entropy_old(dataset_name, criterion, model_path):
    logger.info("Loading model {}".format(model_path))
    model = keras.models.load_model(model_path)
    flat = len(model._layers[0].batch_input_shape) == 2
    dataset = get_dataset(model, dataset_name)

    logger.info("Constructing a KerasCoverage and a KerasEntropy")
    header = KerasCoverage(model, dataset)
    header.load_data("frequency/header.csv")
    entropy = KerasEntropy(model, header.df)
    out_files = sorted([f for f in glob.glob("frequency/*.csv") if not 'header' in f])
    logger.info(out_files)
    for f_outs in out_files:
        logger.info("Measuring coverage & entropy from {}".format(f_outs))
        kcov = KerasCoverage(model, dataset)
        kcov.load_data(f_outs)
        kcov.measure_coverage("neuron")
        entropy.measure(kcov.coverage['neuron'])
        entropy.save_data("entropy_{}".format(f_outs))
    entropy.save_data("entropy.csv")


def exp_coverage(dataset_name, model_path):
    logger.info("Loading model {}".format(model_path))
    model = keras.models.load_model(model_path)
    flat = len(model._layers[0].batch_input_shape) == 2
    logger.info("Loading dataset")
    dataset = None
    if dataset_name.upper() == "EMNIST":
        dataset = Datasets.EMNIST(flatten=flat)
    else:
        dataset = Datasets.MNIST(flatten=flat)

    f_outs = "csvs/neuron_cov.csv"
    logger.info("Initializing KerasCoverage")
    kcov = KerasCoverage(model, dataset)
    logger.info("Measuring coverage")
    kcov.run([dataset.get_range('test', 'x', 0, 10000)])
    kcov.measure_coverage('neuron')
    kcov.save_outs(filename=f_outs)
    print(len(kcov.coverage['neuron'].columns))
    logger.info("Calculating entropy from {}".format(f_outs))
    entropy = KerasEntropy(kcov.coverage['neuron'])
    entropy.measure()


def find_low_amax(model, X, Y):
    results = []
    for i, logit in enumerate(model.predict(X)):
        maxv = np.max(logit)
        buggy = logit.argmax() != Y[i].argmax()
        results.append({'i': i, 'maxv': maxv, 'buggy': buggy, })
    results.sort(key=lambda x: x['maxv'])
    #for r in results:
    #    print("{:4d}, {:.8f}, {:1d}".format(r['i'], r['maxv'], r['buggy']))
    return results


def exp_low_amax():
    model = keras.models.load_model('mnist_models/mnist_cnn0.h5')
    dataset = Datasets.EMNIST(flatten=False)
    X = np.concatenate((dataset.get_range('test', 'x', 0), dataset.get_range('train', 'x', 0)), axis=0)
    Y = np.concatenate((dataset.get_range('test', 'y', 0), dataset.get_range('train', 'y', 0)), axis=0)
    results = find_low_amax(model, X, Y)


def exp_evaluate(model_path):
    model = keras.models.load_model(model_path)
    dataset = None
    if 'emnist' in model_path:
        dataset = dataset.MNIST(flatten=False)
    else:
        dataset = dataset.EMNIST(flatten=False)
    X = np.concatenate((dataset.get_range('test', 'x', 0), dataset.get_range('train', 'x', 0)), axis=0)
    Y = np.concatenate((dataset.get_range('test', 'y', 0), dataset.get_range('train', 'y', 0)), axis=0)
    print(model.evaluate(X, Y))


def exp_dropout(model_path):
    """ https://github.com/yaringal/DropoutUncertaintyExps/blob/master/net/net.py """
    model = keras.models.load_model(model_path)
    dataset = Datasets.MNIST(flatten=False)
    data_x, indices = dataset.gets('test', 'x', range(20))
    data_y, _ = dataset.gets('test', 'y', indices)

    y_train = dataset.get_range('train', 'y', 0, None)
    mean_y_train, std_y_train = np.mean(y_train), np.std(y_train)

    #standard_pred = model.predict(data_x, batch_size=500, verbose=1)
    #standard_pred = standard_pred * std_y_train + mean_y_train
    #rmse_standard_pred = np.mean((data_y.squeeze() - standard_pred.squeeze())**2.)**0.5
    T = 99
    pred = np.array([model.predict(data_x, batch_size=500, verbose=0) for _ in range(T)])
    #Yt_hat = Yt_hat * std_y_train + mean_y_train

    mean_pred = np.mean(pred, 0)
    std_pred = np.std(pred, 0)
    sum_std_pred = np.mean(std_pred, 1)
    print("mean_pred\n", mean_pred.shape, "\n", mean_pred, "\n")
    print("std_pred\n", std_pred.shape, "\n", std_pred, "\n")
    print("sum_std_pred\n", sum_std_pred.shape, "\n", sum_std_pred, "\n")
    # Mean squared error?
    mse = np.mean((data_y.squeeze() - mean_pred.squeeze())**2., 1)
    print("MSE\n", mse.shape, "\n", mse, "\n")


def exp_confusion():
    print("exp_confusion")
    for model_f in sorted(glob.glob("mnist_models/*.h5")):
        print('-'*80)
        print(model_f)
        model = keras.models.load_model(model_f)
        for dataset in [Datasets.MNIST(), Datasets.EMNIST()]:
            X = np.concatenate((dataset.get_range('test', 'x', 0),\
                    dataset.get_range('train', 'x', 0)), axis=0)
            Y = np.concatenate((dataset.get_range('test', 'y', 0),\
                    dataset.get_range('train', 'y', 0)), axis=0)
            print(model.evaluate(X, Y))
            Y_pred = map(np.argmax, model.predict(X))
            cm = confusion_matrix(map(np.argmax, Y), Y_pred)
            print(cm)

def create_logger(func_name):
    logger = logging.getLogger('kerascov')
    logger.setLevel(logging.DEBUG)
    handler_stream = logging.StreamHandler()
    handler_file = logging.FileHandler('{}.log'.format(func_name))
    formatter = logging.Formatter('%(asctime)-15s [%(levelname)s] %(message)s')
    handler_stream.setFormatter(formatter)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_stream)
    logger.addHandler(handler_file)
    return logger


def main():
    if len(sys.argv) < 2:
        print('invalid args')
        logger.error('Invalid args')
        return
    # call the specified function
    global logger
    logger = create_logger(sys.argv[1])
    logger.info("{}".format(str(sys.argv[1:])))
    globals()[sys.argv[1]](*sys.argv[2:])


if __name__ == '__main__':
    main()

