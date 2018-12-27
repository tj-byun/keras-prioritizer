from __future__ import print_function

import sys
import copy
import pprint
import joblib
import logging
import random
import keras
import yaml
import numpy as np
import kprio as KP
from kprio import dataset, selectors, coverage_matrix, evaluator, models


def get_model_name(model_path):
    return model_path.split('/')[-1].split('-')[1]

def get_model_epochs(model_path):
    return model_path.split('/')[-1].split('.')[-2].split('-')[2]

def get_test_query(test_dataset, n_test, shuffle=False):
    ind = range(test_dataset.get_length("all", "x"))
    if shuffle:
        random.shuffle(ind)
    return KP.dataset.Query("all", "x", ind[:n_test])


def measure_accuracy(msg, model, xs, ys):
    assert len(xs) == len(ys)
    score = model.evaluate(xs, ys, verbose=0)
    expected = len(xs) * (1.0 - score[1])
    logger.info("[{}] scores: {}, expected_buggy: {}".format(msg, score,
        expected))
    return score[1]


def test_selector(func):
    """ A decorator for all test selecor functions. """
    def wrapper(*args, **kwargs):
        logger.info("Test selector: {}{}".format(func.__name__, args))
        name, selector = func(*args, **kwargs)
        selected = selector.get_next(config["n_test"])
        d = {"score": selector.get_scores()}
        if options["save"]:
            exp_evaluator.evaluate_and_save(name, selected, extra_data=d)
        if options["demo"]:
            exp_evaluator.display_selected(selected)
    return wrapper

@test_selector
def bayesian(repeat):
    #TODO (12/17): Test
    batch_size = options["batch_size"][TASK_NAME]
    steps = options["bayesian_training_steps"]
    global bayesian_model
    selector = KP.selectors.BayesianSelector(model,
            train_dataset, test_dataset, dquery,
            trained_model=bayesian_model,
            is_regression=is_regression, repeat=repeat,
            batch_size=batch_size, max_steps=steps)
    bayesian_model = selector.get_trained_model()
    return "bayesian-{}".format(repeat), selector


@test_selector
def dropout(repeat):
    batch_size = options["batch_size"][TASK_NAME]
    return "dropout-{}".format(repeat), \
            KP.selectors.DropoutSelector(model, test_dataset, dquery,
                    is_regression=is_regression,
                    repeat=repeat,
                    batch_size=batch_size)

@test_selector
def dsa(selection):
    train_data = None
    try:
        train_data, _ = train_dataset.gets("train", "x", range(config["n_train"]))
    except:
        train_data, _ = train_dataset.gets("", "x", range(config["n_train"]))
    bucket_size = options["bucket_size"][TASK_NAME]
    layer_selection = selection
    if selection == 'last1':
        layer_selection = slice(-1, None)
    elif selection == 'last2':
        layer_selection = slice(-2, None)
    elif selection == 'last3':
        layer_selection = slice(-3, None)
    return "dsa_{}".format(selection), \
            KP.selectors.DSASelector(model, test_dataset, train_data, dquery,
                    layer_selection, bucket_size=bucket_size)

@test_selector
def lsa_selector(n_layers, lsa_layer=5, reverse=True):
    print("\nLikelihood-based Surprise Adequacy selector")
    train_data, _ = train_dataset.gets("train", "x", range(config["n_train"]))
    return "lsa_{}".format(n_layers), \
            KP.selectors.LSASelector(lsa_layer, 0, model, test_dataset,
                    train_data, dquery, reverse, 1)

@test_selector
def probability_selector():
    return "probability", \
            KP.selectors.ProbabilitySelector(model, test_dataset,dquery)

@test_selector
def softmax_entropy():
    return "softmax_entropy", \
            KP.selectors.SoftmaxEntropySelector(model, test_dataset, dquery)

@test_selector
def _test_coverage_selector():
    print("\nTest coverage selector")
    return "coverage", \
            KP.selectors.CoverageSelector(model, test_dataset, dquery,
                    "neuron")


def create_logger():
    l = logging.getLogger('kprio')
    l.setLevel(logging.DEBUG)
    handler_stream = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)-15s [%(filename)s::%(levelname)s] %(message)s')
    handler_stream.setFormatter(formatter)
    l.addHandler(handler_stream)
    #l.addHandler(logging.FileHandler(evaluator.get_run_id() + ".log"))
    return l

def demo():
    exp_evaluator.demo()
    exp_evaluator.plot("demo")
    sys.exit()

def extract_last_dir(w):
    w = w[:-1] if w[-1] == '/' else w
    return w.split('/')[-1]


# Parse arguments & set up
options = None
with open('experiment.yaml', 'r') as f:
    options = yaml.load(f)

logger = create_logger()
if len(sys.argv) >= 3:
    TASK_NAME = sys.argv[1]
    MODEL_PATH = sys.argv[2]
else:
    print("Invalid #args")
    print("$ {} <task_name> <model_path>".format(sys.argv[0]))
    sys.exit()
config = {
        "task_name": TASK_NAME,
        "model_path": MODEL_PATH,
        "model_name": get_model_name(MODEL_PATH),
        "epochs": get_model_epochs(MODEL_PATH),
        }

if TASK_NAME == "mnist":
    is_regression = False
    flat = "fully" in MODEL_PATH
    train_dataset = KP.mnist.MNIST(flatten=flat)
    test_dataset = KP.mnist.EMNIST(flatten=flat)
    d = {"train_dataset": "mnist", "test_dataset": "emnist"}
    config.update(d)
elif "taxinet" in TASK_NAME:
    is_regression = True
    if len(sys.argv) != 5:
        print("TaxiNet: Invalid #args")
        print("$ {} <task_name> <model_path> <train_dir> <test_dir>".format(
            sys.argv[0]))
    tolerance = np.array(options["tolerance"][TASK_NAME])
    print("tolerance", tolerance)
    train_dataset = KP.taxinet.TaxiNet(sys.argv[3], tolerance)
    test_dataset = KP.taxinet.TaxiNet(sys.argv[4], tolerance)
    d = {"train_dataset": extract_last_dir(sys.argv[3]),
            "test_dataset": extract_last_dir(sys.argv[4])}
    config.update(d)

config["n_train"] = train_dataset.get_length("all", "x")
config["n_test"] = test_dataset.get_length("all", "x")
if config["n_test"] > options["test_limit"]:
    config["n_test"] = options["test_limit"]
model = models.load_model(TASK_NAME, MODEL_PATH)
dquery = get_test_query(test_dataset, config["n_test"], shuffle=True)
logger.info(config)

# Measure validation / test accuracy
try:
    xs = train_dataset.gets("test", "x")[0]
    ys = train_dataset.gets("test", "y")[0]
    if options["measure_accuracy"]:
        val_acc = measure_accuracy("Validation accuracy", model, xs, ys)
        config["val_acc"] = val_acc
except Exception as e:
    # when (validation) "test" dataset is not available, skip.
    pass
xs = test_dataset.gets("all", "x")[0]
ys = test_dataset.gets("all", "y")[0]
if options["measure_accuracy"]:
    test_acc = measure_accuracy("Test accuracy", model, xs, ys)
    config["test_acc"] = test_acc

# Run prioritization
exp_evaluator = KP.evaluator.Evaluator(model, test_dataset,
        options["db_name"], config)

bayesian_model = None

for tech in options["techniques"]:
    if "dropout" == tech[:7]:
        dropout(int(tech[8:]))
    elif "dsa" == tech[:3]:
        option = tech[4:]
        dsa(option)
    elif tech == "probability":
        probability()
    elif "bayesian" == tech[:8]:
        bayesian(int(tech[9:]))
    elif tech == "softmax":
        softmax_entropy()
    else:
        raise Exception("Invalid prioritization technique")


# Analyze & save the result
exp_evaluator.save()
analyzer = KP.evaluator.Analyzer(options["db_name"])
analyzer.analyze_current()
fig_fname = "{}-{}-{}-{}-{}k".format(TASK_NAME, config["model_name"],
        config["train_dataset"], config["test_dataset"],
        int(config["n_test"] / 1000))
analyzer.plot(fig_fname)

