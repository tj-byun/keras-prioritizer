from __future__ import print_function

import keras
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Model

import numpy as np
import scipy, IPython, sys, os, deepdish
import tensorflow as tf
from multiprocessing import Pool

import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        self.rows = 10
        self.cols = 10
        self.fig = plt.figure(figsize=(self.rows, self.cols))
        self.cnt = 0

    def add_subplot(self, fig):
        if self.cnt >= 100:
            raise Exception('Figure is full')
        self.cnt += 1
        self.fig.add_subplot(self.rows, self.cols, self.cnt)
        plt.imshow(fig)

    def show(self):
        plt.show()


def load_and_preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    img_rows, img_cols = 28, 28
    num_classes = 10

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


def get_internal_neuron_output(model, input_data):
    """ Given a Keras model, iteratie through each layer, flatten the
    multi-dimensional arrays in each layer to a 1-dim vector, and eventually
    return a list of numpy array where each element is the output of each
    neuron.
    :return: a (flat) list of numpy.array
    """
    outputs = [layer.output for layer in model.layers]
    get_neuron_outputs = K.function([model.input], outputs)
    neuron_outs_by_layer = get_neuron_outputs([input_data])

    l = [[np.reshape(each_input, (-1)) for each_input in layer] \
            for layer in neuron_outs_by_layer]
    # transpose from [layer_id * input_id] to [input_id, layer_id]
    l = map(list, zip(*l))
    cnt = 0
    for _ in l[0]:
        cnt += len(_)
    return l, cnt


def measure_coverage(flat_neuron_output, is_covered):
    cov_bv = 0  # coverage bit-vector
    i = 0
    for out_by_input in flat_neuron_output:
        cov_list = []
        for layer in out_by_input:
            cov_list += map(is_covered, layer)
        bv = int(''.join(map(str, cov_list)), 2)
        if cov_bv == 0:
            # first input
            cov_bv = bv
        else:
            # bit-wise OR of the coverage bit-vectors
            cov_bv |= bv
        i += 1
        if i % 500 == 0:
            print('   {}/{}'.format(i, len(flat_neuron_output)))
    # count covered obligations
    covered_cnt = str(bin(cov_bv))[2:].count('1')
    return covered_cnt


def is_neuron_active(z):
    return 1 if z > 0 else 0


def get_activation_pattern(model, in_data, is_active):
    """ Returns an activation pattern """
    def flatten_pattern(l):
        return reduce(lambda l1, l2: l1 + l2, l)
    flat_neuron_output, _ = get_internal_neuron_output(model, in_data)
    pattern_list = []
    for out_by_input in flat_neuron_output:
        act_pattern = []
        for layer in out_by_input:
            act_pattern += map(is_active, layer)
        pattern_list.append(np.array(act_pattern))
    return pattern_list


def measure_for_each(model, data):
    print('\n# Measuring the individual coverage of {} test inputs'.format(len(data)))
    for i in range(n):
        neuron_outputs, tot = get_internal_neuron_output(model, data)
        cnt = measure_coverage(neuron_outputs, lambda x: 1 if x > 0 else 0)
        label = np.argmax(y_test[i])
        print('[%d] %.3f %% (%d / %d)' % (label, 100 * cnt / float(tot), cnt, tot))


def measure_batch(model, data):
    print('\n# Measuring the coverage over {} test inputs'.format(len(data)))
    neuron_outputs, tot = get_internal_neuron_output(model, data)
    cnt = measure_coverage(neuron_outputs, lambda x: 1 if x > 0 else 0)
    print('%.3f %% (%d / %d)' % (100 * cnt / float(tot), cnt, tot))


def exp_low_confidence(model, dataset):
    #TODO: totally messy
    act_pattern_list = get_activation_pattern(model, dataset['x'][:5000], lambda x: x)

    # Find training inputs with low confidence
    found = []
    for i, v1 in enumerate(act_pattern_list):
        if np.amax(model.predict(np.array([dataset['x'][i]]))) < 0.6:
            found.append((i, act_pattern_list[i]))
            predicted = model.predict(np.array([dataset['x'][i]]))
            print("index: {:4d}, label: [{:1d}], predicted: [{:1d}], confidence: {:.2f}"\
                    .format(i, y_train[i].argmax(), np.argmax(predicted), np.amax(predicted)))
        if cnt >= rows * cols:
            break
    print(found)

    # Find (unseen) test inputs that are similar to low-confidence inputs.
    act_pattern_list = get_activation_pattern(model, x_test[:10000], lambda x: x)
    for i, v1 in found:
        cnt += 1
        print("")
        for j, v2 in enumerate(act_pattern_list):
            d = scipy.spatial.distance.cosine(v1, v2)
            if d < 0.25:
                cnt += 1
                fig.add_subplot(rows, cols, cnt)
                plt.imshow(X_test[j])
                predicted = model.predict(np.array([x_test[j]]))
                if y_test[j].argmax() != np.argmax(predicted):
                    print("INTERE1TING!")
                print("({:1d}, {:1d}) index: {:4d}, label: [{:1d}], predicted: [{:1d}], confidence: {:.2f}, distance: {:.2f} from ({:1d})"\
                        .format(cnt/rows, cnt%rows, j, y_test[j].argmax(), np.argmax(predicted), np.amax(predicted), d, i))
            if cnt >= rows * cols:
                break
    plt.show()


def get_far_apart(model, dataset, measure='hidden', DATA_CNT=3000):
    hidden_outs = None
    if measure == 'hidden':
        hidden_outs = get_activation_pattern(model, dataset['x'][:DATA_CNT], lambda x: x)
    # found[label] = {d: distance, i: index1, j: index2, v1: out1, v2: out2}
    found = {}
    for i in range(10):
        found[i] = {
                'd': -1,        # distance
                'i': -1,        # index 1
                'j': -1,        # index 2
                'v1': None,     # (hidden output) vector 1
                'v2': None,     # vector 2
                }
    for i in range(DATA_CNT):
        for j in range(i + 1, DATA_CNT):
            if not dataset['y'][i].argmax() == dataset['y'][j].argmax():
                # (v1, v2) has to have the same label
                continue
            v1, v2 = None, None
            if measure == 'hidden':
                v1, v2 = hidden_outs[i], hidden_outs[j]
            elif measure == 'input':
                v1, v2 = dataset['x'][i].reshape(-1), dataset['x'][j].reshape(-1)
            elif measure == 'logit':
                logits = model.predict(np.array([dataset['x'][i], dataset['x'][j]]))
                v1, v2 = logits[0], logits[1]
            d = scipy.spatial.distance.cosine(v1, v2)
            label = dataset['y'][i].argmax()
            if d > found[label]['d']:
                # `d` shall be larger than the farthest distance found so far
                print("  found", label, d, i, j)
                found[label] = {'d': d, 'i': i, 'j': j, 'v1': v1, 'v2': v2}
    return found


def show_found_pairs(model, org_dataset, dataset, found):
    """
    :param found: found[label] = {d: distance, i: index1, j: index2, v1: out1, v2: out2}
    """
    plotter = Plotter()
    print("label, ( i,  j), confidence  , in_cosine")
    for label, tup in found.iteritems():
        d, i, j = tup['d'], tup['i'], tup['j']
        try:
            plotter.add_subplot(org_dataset['x'][i])
            plotter.add_subplot(org_dataset['x'][j])
        except Exception:
            break
        prediction = model.predict(np.array([dataset['x'][i], dataset['x'][j]]))
        logits = map(np.amax, prediction)
        print("[{:1d}], ({:4d}, {:4d}), [{:.2f}, {:.2f}], {:2.6f}".format(
            dataset['y'][i].argmax(), i, j, logits[0], logits[1], d))

def show_found(model, org_dataset, dataset, found):
    """
    :param found: found[label] = {d: distance, i: index1, j: index2, v1: out1, v2: out2}
    """
    plotter = Plotter()
    for label, tup in found.iteritems():
        d, i, j = tup['d'], tup['i'], tup['j']
        try:
            plotter.add_subplot(org_dataset['x'][j])
            print(';')
        except Exception as e:
            print(e)
            print("can't add more plots")
            break
        prediction = model.predict(np.array([dataset['x'][j]]))
        logits = map(np.amax, prediction)
        print("[{:1d}], ({:4d}), [{:.2f}], {:2.6f}".format(
            dataset['y'][j].argmax(), j, logits[0], d))


def find_similar(model, dataset, targets, measure='hidden', DATA_CNT=10000):
    """
    :param model: keras model
    :param targets: target[label] = {d: distance, i: index1, j: index2, v1: out1, v2: out2}
    """
    # Find (unseen) test inputs that are similar to low-confidence inputs.
    hidden_outs = get_activation_pattern(model, dataset['x'][:DATA_CNT], lambda x: x)

    target_vectors = []
    for _, tup in targets.iteritems():
        target_vectors += [tup['v1'], tup['v2']]

    similars = {}   # similars[index] = {d: distance, i, j}
    for i, v1 in enumerate(target_vectors):
        print(i)
        similars[i] = {'d': 1.0, 'i': -1, 'j': -1}
        for j in range(DATA_CNT):
            v2 = None
            if measure == 'hidden':
                v2 = hidden_outs[j]
            elif measure == 'input':
                v2 = dataset['x'][j].reshape(-1)
            elif measure == 'logit':
                v2 = model.predict(np.array([dataset['x'][j]]))[0]
            d = scipy.spatial.distance.cosine(v1, v2)
            if d < similars[i]['d']:
                similars[i] = {'d': d, 'i': i, 'j': j}
                print('.', end='')
    return similars


(x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

D_train = {'x': x_train, 'y': y_train}
D_test = {'x': x_test, 'y': y_test}
D_TRAIN = {'x': X_train, 'y': Y_train}
D_TEST = {'x': X_test, 'y': Y_test}


def main():
    if len(sys.argv) != 2:
        print('invalid #args')
        return

    model_path = sys.argv[1]
    model = keras.models.load_model(model_path)

    #exp_low_confidence(model)

    far_apart_f = 'far_apart.hd5'
    found = None
    if os.path.exists(far_apart_f):
        found = deepdish.io.load(far_apart_f)
        print('successfully read the file', far_apart_f)
    else:
        found = get_far_apart(model, D_train, measure='hidden')
        deepdish.io.save(far_apart_f, found)
    print(found)
    show_found_pairs(model, D_TRAIN, D_train, found)

    similar_f = 'similar.hd5'
    similars = None
    if os.path.exists(similar_f):
        similars = deepdish.io.load(similar_f)
        print('successfully read the file', similar_f)
    else:
        similars = find_similar(model, D_test, found, measure='hidden')
        deepdish.io.save(similar_f, similars)
    print(similars)

    show_found(model, D_TEST, D_test, similars)
    plt.show()

    #show_found_pairs(model, D_TRAIN, D_train, found)

    #measure_for_each(model, xtest[0:1])
    #measure_batch(model, x_test[0:10])


if __name__ == '__main__':
    main()

