
import os
import tensorflow as tf
import coverage, dataset, selectors, models, evaluator
import mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

