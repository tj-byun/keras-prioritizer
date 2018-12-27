# 8/22/18
# 10/28/18
# Taejoon Byun
# Vaibhav Sharma
# Abhishek Vijayakumar

from __future__ import print_function

import logging
import IPython
import sys
import copy
import time
from scipy.stats import gaussian_kde
import math
import numba, numba.types
from numba import cuda
import itertools
import numpy
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

import os
import tensorflow as tf
import tensorflow_probability as tfp

import keras
from keras import backend as K

import dataset, coverage, coverage_matrix, entropy
import dsa

logger = logging.getLogger('kprio')


class TestSelector(object):
    """ A TestSelector gives a score to each input """

    def __init__(self, model, dataset, dquery):
        self.model = model
        self.dataset = dataset
        self.dquery = dquery
        self.input_data, _ = self.dataset.gets_query(dquery)
        self.current_index = 0
        self.sorted_indices = None


    def _prioritize(self):
        raise NotImplemented("Abstract method `TestSelector::_prioritize`")


    def _set_scores(self, scores):
        """ `scores` assigns positive real number to each input. A higher score
        represents a higher value (priority) """
        print(len(scores))
        print(len(self.dquery.ind))
        assert len(scores) == len(self.dquery.ind),\
                "{},{}".format(len(scores), len(self.dquery.ind))
        self.sorted_indices = np.array([self.dquery.ind[i] for i in
            np.argsort(scores)[::-1]])
        self.sorted_scores = np.sort(scores)[::-1]
        logger.debug("sorted_scores: " + str(self.sorted_scores))


    def get_scores(self):
        """ Get {input_index: score}"""
        f = lambda w: 'inf' if w == '--' else w
        return map(f, map(str, self.sorted_scores))


    def get_next(self, cnt):
        """ Get next `cnt` test input IDs """
        if self.sorted_indices is None:
            self._prioritize()
        assert(self.sorted_indices is not None)
        a = self.current_index
        b = a + cnt
        if b >= len(self.sorted_indices):
            #raise ValueError("Request exceeded the bound")
            b = None
        #self.current_index = b
        indices = self.sorted_indices[a:b]
        q = self.dquery.copy()
        #q.set_indices(self.d_query.map_indices(indices))
        q.set_indices(indices)
        return q


class ProbabilitySelector(TestSelector):

    def __init__(self, model, dataset, dquery, batch_size=1000):
        self.batch_size = batch_size
        self.verbose = 0
        super(ProbabilitySelector, self).__init__(model, dataset, dquery)
        self._prioritize()

    def _prioritize(self):
        predictions = self.model.predict([self.input_data], \
                batch_size=self.batch_size, verbose=self.verbose)
        maxes = map(np.amax, predictions)
        scores = [1.0 / m for m in maxes]
        self._set_scores(scores)



class SoftmaxEntropySelector(TestSelector):

    def __init__(self, model, dataset, dquery, batch_size=100):
        self.batch_size = batch_size
        self.verbose = 0
        super(SoftmaxEntropySelector, self).__init__(model, dataset, dquery)
        self._prioritize()

    def _prioritize(self):
        predictions = self.model.predict([self.input_data], \
                batch_size=self.batch_size, verbose=self.verbose)
        get_h = lambda p: - p * np.ma.log2(p)
        scores = np.sum(get_h(predictions), axis=1)
        self._set_scores(scores)



class DropoutSelector(TestSelector):
    """ Prioritize test inputs by uncertainty apprxoimated using Monte-Carlo
    Dropout.
    Ref: http://arxiv.org/abs/1703.04977 (eq.3)
    """

    def __init__(self, model, dataset, dquery, is_regression=False, repeat=100,
            batch_size=5000):
        """
        :param model: kprio.models.Model
        :param dataset: kprio.dataset.Dataset
        :param dquery: kprio.dataset.Query
        :param categorical: True if regression task, False if classification
        :param repeat: sampling repetition
        :param batch_size: input batch size used in prediction. Set a smaller
            value if crashes with OOM.
        """
        self.repeat = repeat
        self.batch_size = batch_size
        self.is_regression = is_regression
        super(DropoutSelector, self).__init__(model, dataset, dquery)
        self._prioritize()


    def demo(self):
        """ This function demos how the dropout prioritization works """
        # after sum
        a = np.array([
            [.9, .05, .05],
            [.3, .4, .3]
            ])
        print(a)
        get_h = lambda p: - p * np.log2(p)
        scores = np.sum(get_h(psum), axis=1)
        print(scores)
        # scores: array([0.39439769, 1.08889998])
        # the latter gets higher score because the averaged sum over samples
        # ([.3, .4, .3]) is more scattered (deemed uncertain).


    def __get_dropout_layer_id(self):
        for i, l in enumerate(self.model.layers):
            if type(model.layers[i]) == keras.layers.core.Dropout:
                return i


    def __predict_with_dropout(self, x):
        """ From: https://fairyonice.github.io/Measure-the-uncertainty-in-deep-learning-models-using-dropout.html"""
        sample = K.function([self.model.layers[0].input, K.learning_phase()],
                [self.model.layers[-1].output])
        return np.array([sample([x, 1])[0] for _ in range(self.repeat)])


    def _prioritize(self):
        # shape: (#repeats, #inputs, output_vector_length)
        #   e.g. (100, 1000, 10)
# (100, 1, 5000, 10)
        predictions = None
        for i in range(len(self.input_data) / self.batch_size + 1):
            ind_a = i * self.batch_size
            ind_b = min(len(self.input_data), (i + 1) * self.batch_size)
            logger.debug("Bucket {} {}".format(ind_a, ind_b))
            input_bucket = self.input_data[ind_a:ind_b]
            outs = self.__predict_with_dropout(input_bucket)
            if predictions is None:
                predictions = outs
                print(outs.shape)
            else:
                #print("dim", predictions.shape, outs.shape)
                predictions = np.concatenate((predictions, outs), axis=1)
        self._set_scores_multiple_runs(np.ma.array(predictions))


    def _set_scores_multiple_runs(self, predictions):
        # shape: (#repeats, #inputs, output_vector_length)
        if self.is_regression:
            # compute the predictive variance
            pvar = np.var(predictions, axis=0)  # variance across samples
            psum = np.sum(pvar, axis=1)         # sum across output vector
            self._set_scores(psum)
        else:
            # sum of softmaxes across samples:
            #     $ p(y=c | x, X, Y) \approx $
            #     $ \frac{1}{T} \sum_{t=1}^{T} {\it softmax}(f^{\hat{w_t}}(x)) $
            # shape: [input x output_vector]
            psum = np.sum(predictions, axis=0) / float(len(predictions))
            # $ H(p) = - \sum_{c=1}^{C} p_c \log p_c $
            get_h = lambda p: - p * np.ma.log2(p)
            # since log(0) is not defined, the value gets masked when p
            # equals 0. Fill the masked with 0.0 since the limit of p *
            # log(p) when p -> 0 is 0.0 .
            print(len(psum))
            print(psum.shape)
            scores = np.ma.filled(np.sum(get_h(psum), axis=1), 0.0)
            # the higher the mean is, the more unstable the prediction is
            self._set_scores(scores)



class CoverageSelector(TestSelector):

    def __init__(self, model, dataset, dquery, criterion, kcov=None):
        self.criterion = criterion
        self.kcov = kcov
        super(CoverageSelector, self).__init__(model, dataset, dquery)
        self._prioritize()
        self.increment = self.__get_increment(1.0)

    def __get_increment(self, percentage):
        assert(self.cmat)
        assert(percentage >= 0.0 and percentage <= 100.0)
        return (len(self.cmat.df.columns) / 100.0) * float(percentage)

    def _prioritize(self):
        if not self.kcov:
            self.kcov = coverage.KerasCoverage(self.model)
            self.kcov.run(self.input_data)
            self.kcov.measure_coverage(self.criterion)
        self.cmat = coverage_matrix.CoverageMatrix()
        self.cmat.df = self.kcov.coverage[self.criterion]
            #.reset_index(drop = True, inplace = True)
        self.gtor = coverage_matrix.SubsuiteGenerator()

    def get_next(self, cnt):
        suite = self.gtor.select_increased_by(self.cmat, self.increment)
        q = self.dquery.copy()
        #q.set_indices(self.d_query.map_indices(indices))
        q.set_indices(suite[:cnt])
        return q



class EntropySelector(TestSelector):
    #TODO (8/28/18): Implement

    def __init__(self, model, dataset, dquery, criterion, entropy_file,\
            entropy_len, kcov):
        self.criterion = criterion
        self.entropy_file = entropy_file
        self.entropy_len = entropy_len
        self.kcov = kcov
        super(EntropySelector, self).__init__(model, dataset, dquery)
        self._prioritize()

    def _prioritize(self):
        ke = entropy.KerasEntropy()

        #kcov = coverage.KerasCoverage(self.model)
        #kcov.run(self.input_data)
        #kcov.measure_coverage(self.criterion)
        entropies = ke.get_entropies(self.kcov.coverage[self.criterion])
        es = [(i, e) for i, e in enumerate(entropies)]
        es = sorted(es, key=lambda x: x[1], reverse=True)
        self.sorted_indices = [pair[0] for pair in es]



class LSASelector(TestSelector):

    def __init__(self, lsa_layer, threshold, model, dataset, train_data, test_query, is_from_last, n_layers):
        super(LSASelector, self).__init__(model, dataset, test_query)
        self.is_from_last = is_from_last
        self.n_layers = n_layers
        self.train_data = train_data
        self.lsa_layer = lsa_layer
        self.threshold = threshold
        self.test_data, _ = dataset.gets_query(test_query)
        self.bucket_size = 10000
        self._prioritize()


    def _prioritize(self):
        #logger.info('Running get_AT_likelihood-based surprise adequacy')
        self.train_Ys = map(np.argmax, self.model.predict(self.train_data))
        self.test_Ys = map(np.argmax, self.model.predict(self.test_data))
        # get ATs for only a single layer in self.lsa_layer
        # [input x activation_trace_vector (1d)]
        #TODO(12/10): Fix. Using deprecated method call
        train_ATs = self.model.get_activation_traces(self.train_data,
                self.n_layers, self.is_from_last, bucket_size=self.bucket_size,
                layer_to_select=self.lsa_layer)
        test_ATs = self.model.get_activation_traces(self.test_data,
                self.n_layers, self.is_from_last, bucket_size=self.bucket_size,
                layer_to_select=self.lsa_layer)
        assert len(self.test_data) == len(test_ATs)

        class_indices = {}

        for i, c in enumerate(self.train_Ys):
            if class_indices.has_key(c):
                class_indices[c].append(i)
            else:
                class_indices[c] = [i]

        # @numba.njit
        def for_each_test(a, b_s):
            return np.array([a - b_s[i] for i in range(b_s.shape[0])])

        # gives a LinAlgError because diff becomes a singular matrix and cannot be inverted during KDE
        # for indices in class_indices:
        #     training_ATs_c = train_ATs[indices]
        #     diff = for_all(training_ATs_c, training_ATs_c)
        #     kernels.append(gaussian_kde(np.asarray(diff), bw_method='scott'))

        def getd(As):
            d = np.array([0.00] * len(As)).reshape((len(As)))
            assert As.shape[0] == len(self.test_Ys)
            for i in numba.prange(As.shape[0]):
                training_ATs_c = train_ATs[class_indices[self.test_Ys[i]]]
                diff = for_each_test(As[i], training_ATs_c)
                for j in numba.prange(diff.shape[0]):
                    kernel = gaussian_kde(np.asarray(diff[j]), bw_method='scott')
                    IPython.embed()
                    # compute the KDE-based density for the test input at the ith position and normalize it
                    # using equation (1) in the Surprise Adequacy paper
                    d[i] += kernel.evaluate(diff[j])
                d[i] /= train_ATs.shape[0]

            return d

        def get_kernels(train_ATs, class_indices):
            kernels = []
            for c, indices in enumerate(class_indices):
                training_ATs_c = train_ATs[class_indices[c]]
                kernel = gaussian_kde(np.transpose(np.asarray(training_ATs_c)), bw_method='scott')
                kernels.append(kernel)
            return kernels

        def getd2(As, kernels, test_Ys):
            d = np.array([0.00] * len(As)).reshape((len(As)))
            assert As.shape[0] == len(test_Ys)
            for i in numba.prange(As.shape[0]):
                kernel = kernels[test_Ys[i]]
                # compute the KDE-based density for the test input at the ith position and normalize it
                # using equation (1) in the Surprise Adequacy paper
                this_estimate = kernel.evaluate(As[i])
                indices = class_indices[test_Ys[i]]
                for ind in indices:
                    training_estimate = kernel.evaluate(train_ATs[ind])
                    d[i] += (this_estimate[0] - training_estimate[0])
                d[i] /= test_ATs.shape[0]
            return d

        # filter out ATs that show variance below a threshold in self.threshold
        selector = VarianceThreshold(self.threshold)
        train_ATs = selector.fit_transform(train_ATs)
        test_ATs = selector.fit_transform(test_ATs)
        d = getd(test_ATs)
        # d = getd2(test_ATs, get_kernels(train_ATs, class_indices), self.test_Ys)

        # LSA vector will be negative log of the vector returned by getd as per equation (2) in Surprise Adequacy paper
        lsa_vector = -numpy.log(d)
        self._set_scores(lsa_vector)


class DSASelector(TestSelector):

    n_training_data = 50
    n_test_data = 50
    BIG_NUMBER = numpy.finfo('d').max

    def __init__(self, model, dataset, train_data, test_query, layer_to_select,
            bucket_size=0):
        super(DSASelector, self).__init__(model, dataset, test_query)
        self.train_data = train_data
        self.test_data, _ = dataset.gets_query(test_query)
        self.layer_to_select = layer_to_select
        self.bucket_size = 10000 if bucket_size == 0 else bucket_size
        self._prioritize()


    @staticmethod
    @numba.njit(fastmath=True)
    def get_l2_norm(at1, at2):
        """ get l2norm between two activation traces (in numpy.ndarray form)
        """
        assert len(at1) == len(at2)
        norm = 0.0
        for i in range(len(at1)):
            norm += ((at1[i] - at2[i]) * (at1[i] - at2[i]))
        return math.sqrt(norm)


    def __get_activation_traces(self):
        self.train_Ys = map(np.argmax, self.model.predict(self.train_data))
        self.test_Ys = map(np.argmax, self.model.predict(self.test_data))
        # [input x activation_trace_vector (1d)]
        self.train_ATs = self.model.get_activation_traces(self.train_data,
                self.layer_to_select, bucket_size=self.bucket_size)
        self.test_ATs = self.model.get_activation_traces(self.test_data,
                self.layer_to_select, bucket_size=self.bucket_size)
        self.train_AT_list = self.train_ATs.tolist()
        self.test_AT_list = self.test_ATs.tolist()
        self.vector_length = len(self.test_ATs[0])
        logger.debug("DSA activation trace length: " + str(self.vector_length))
        assert len(self.test_data) == len(self.test_ATs)


    def __get_inclass_min_distance(self):
        """ This method will skip AT pairs that correspond to inputs that have
        the same predicted class. returns list of training AT indices that are
        closest to each training AT (with diff. class)
        """
        logger.info("DSA norm computation - interclass")
        min_d = [-1] * len(self.test_ATs)
        min_d = dsa.get_test_train_dist(self.test_AT_list,\
                self.train_AT_list, self.test_Ys, self.train_Ys, min_d)
        assert len(min_d) == len(self.test_ATs)
        return min_d


    def __get_interclass_min_distance(self):
        """ This method will skip AT pairs that correspond to inputs that have
        different predicted classes. returns list of training AT indices that
        are closest to each test AT (with same class)
        """
        logger.info("DSA norm computation - inclass")
        min_d = [-1] * len(self.train_ATs)
        min_d = dsa.get_train_train_dist(self.train_AT_list, self.train_Ys, min_d)
        assert len(min_d) == len(self.train_ATs)
        return min_d


    def _prioritize(self):
        """ 'main' class for test prioritization """
        self.__get_activation_traces()
        interclass = self.__get_interclass_min_distance()
        inclass = self.__get_inclass_min_distance()

        scores = [None] * len(self.test_data)
        for i in range(len(inclass)):
            if inclass[i] == -1 or interclass[inclass[i]] == -1:
                scores[i] = self.BIG_NUMBER
            else:
                a = self.train_ATs[inclass[i]]
                dist_a = self.get_l2_norm(self.test_ATs[i], a)
                b = self.train_ATs[interclass[inclass[i]]]
                dist_b = self.get_l2_norm(a, b)
                scores[i] = dist_a / dist_b

        self._set_scores(np.ma.array(scores))



class BayesianSelector(DropoutSelector):

    def __init__(self, model, train_dataset, test_dataset, test_query,
            trained_model=None, is_regression=False, repeat=100,
            batch_size=256, learning_rate=0.001, max_steps=100):

        try:
            self.x_train, _ = train_dataset.gets("train", "x")
            self.y_train, _ = train_dataset.gets("train", "y")
        except:
            self.x_train, _ = train_dataset.gets("all", "x")
            self.y_train, _ = train_dataset.gets("all", "y")
    	self.x_val, _ = test_dataset.gets_query(test_query)
    	y_query = test_query.copy(variable="y")
    	self.y_val, _ = test_dataset.gets_query(y_query)

        self.bayesian_model = trained_model
    	self.learning_rate = learning_rate
    	self.max_steps = max_steps
        self.model_dir = os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                "logistic_regression/")

    	super(BayesianSelector, self).__init__(model, test_dataset, test_query,
    		is_regression=is_regression, repeat=repeat,
    		batch_size=batch_size)

    def get_trained_model(self):
        assert self.bayesian_model is not None
        return self.bayesian_model

    def _prioritize(self):
        logger.info("Loading data")
    	self.predictions_train, self.predictions_test = self.__load_data()
        logger.info("Training a Bayesian neural net")
        if self.bayesian_model is None:
            self.bayesian_model = self.__train()
        logger.info("Making predictions")
    	predictions = self.__predict(self.bayesian_model)
        self._set_scores_multiple_runs(predictions)


    def __train(self):
    	if tf.gfile.Exists(self.model_dir):
    	    tf.logging.warning("Warning: deleting old log directory at {}"\
    		    .format(self.model_dir))
    	    tf.gfile.DeleteRecursively(self.model_dir)
    	    tf.gfile.MakeDirs(self.model_dir)

        features, labels = self.predictions_train, self.y_train
    	#features, labels = self.__build_input_pipeline(self.predictions_train, self.y_train, self.batch_size)

    	#Define a bayesian neural network with three layers
        def dense_flipout(unit):
            return tfp.layers.DenseFlipout(units=unit,
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                activation='softsign')
    	with tf.name_scope("bayesian_neural_net", values=[features]):
            #neural_net=tf.keras.Sequential([dense_flipout(16),
            #    dense_flipout(8), dense_flipout(2)])
            neural_net=tf.keras.Sequential([dense_flipout(10)])
    	    output = neural_net(features)
    	    #Choice of optimal sigma unclear
    	    dist = tfp.distributions.Normal(output, 0.35)

    	# Compute the ELBO as the loss, averaged over the batch size.
        labels = tf.to_float(labels, name='ToFloat')
    	neg_log_likelihood = -tf.reduce_mean(dist.log_prob(labels))

    	kl = sum(neural_net.losses) / len(self.x_train)
    	elbo_loss = neg_log_likelihood + kl

        # Build metrics for evaluation. Predictions are formed from a single
        # forward pass of the probabilistic layers. They are cheap but noisy
        # predictions.
    	accuracy, accuracy_update_op = tf.metrics.mean_squared_error(
    	    labels=labels, predictions=output)

    	with tf.name_scope("train"):
    	    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    	    train_op = optimizer.minimize(elbo_loss)

    	init_op = tf.group(tf.global_variables_initializer(),
    			    tf.local_variables_initializer())

    	sess = tf.Session()
        sess.run(init_op)
        # Fit the model to data.
        for step in range(self.max_steps):
            _ = sess.run([train_op, accuracy_update_op])
            if step % 100 == 0 or step == self.max_steps - 1:
                loss_value, accuracy_value = sess.run([elbo_loss, accuracy])
                print("Step: {:>3d} Loss: {:.3f} MSE: {:.3f}".format(step,
                    loss_value, accuracy_value))
    	return neural_net


    def __predict(self, neural_net):
    	# Create a validation set and visualize the predictions
    	#features, labels = self.__build_input_pipeline(self.predictions_test, self.y_val, len(self.y_val))
        features, labels = self.predictions_train, self.y_train
    	predictions=[]
        with tf.Session() as sess:
            init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
            sess.run(init_op)
            for _ in range(self.repeat):
                with tf.name_scope("bayesian_neural_net", values=[features]):
                    predictions.append(neural_net(features).eval())
    	#TODO: Test
    	return np.asarray(predictions)


    def bucket_run(self, f, data, bucket_size=100):
        outs = []
        for i in range(len(data) / bucket_size + 1):
            ind_a = i * bucket_size
            ind_b = min(len(data), (i + 1) * bucket_size)
            bucket = data[ind_a:ind_b]
            outs.append(f(bucket))
        np.concatenate(outs, axis=0)


    def __load_data(self, scaler=1):
    	embedding_layer = self.model.get_feature_layer(offset=0)
    	embedding_space = K.function([self.model.layers[0].input],
    		[embedding_layer.output])

        predictions_train = [embedding_space([np.expand_dims(x/scaler,
            axis=0)])[0] for x in self.x_train]
    	predictions_train = np.squeeze(np.asarray(predictions_train))

        predictions_test = [embedding_space([np.expand_dims(x/scaler,
            axis=0)])[0] for x in self.x_val]
    	predictions_test = np.squeeze(np.asarray(predictions_test))
        return predictions_train, predictions_test


    def __build_input_pipeline(self, x, y, batch_size):
    	"""Build a Dataset iterator for supervised classification.

    	Args:
    	x: Numpy `array` of features, indexed by the first dimension.
    	y: Numpy `array` of labels, with the same first dimension as `x`.
    	batch_size: Number of elements in each training batch.

    	Returns:
    	batch_features: `Tensor` feed  features, of shape
    	  `[batch_size] + x.shape[1:]`.
    	batch_labels: `Tensor` feed of labels, of shape
    	  `[batch_size] + y.shape[1:]`.
    	"""
        logger.info("Building input pipeline")
    	training_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    	training_batches = training_dataset.repeat().batch(batch_size)
    	training_iterator = training_batches.make_one_shot_iterator()
    	batch_features, batch_labels = training_iterator.get_next()
    	return batch_features, batch_labels

