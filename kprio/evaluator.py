# Taejoon Byun
# 10/18/18
# 10/19/18

from __future__ import print_function

import logging
import IPython
import sys

import itertools
import pandas as pd
import numpy as np
import tinydb
import datetime
import matplotlib
import matplotlib.pyplot as plt
import keras
import cv2

logger = logging.getLogger('kprio')
now = datetime.datetime.now()
run_id = now.strftime("%Y-%m-%d-%H-%M")

SAVE_LIMIT = 6000

def get_run_id():
    return run_id

"""
Data Structure

datetime
  year: int
  month: int
  day: int
  hour: int
  minute: int
config
  dmodel_path: str
  emodel_path: str
  epochs: int
  model_name: str
  n_select: int
  n_total: int
  n_train: int
  train_dataset: str
  test_dataset: str
ptech: str
buggy: array(int)
indices: array(int)
cnt: int

"""

class Analyzer(object):

    def __init__(self, db_name):
        self.db = tinydb.TinyDB(db_name)
        self.dfs = {}
        self.auc = {}


    def analyze(self, entries):
        self._calculate_and_update_auc(entries)


    def analyze_current(self):
        logger.info("Result: (date: " + str(now) + ")")
        query = tinydb.Query()
        entries = self.db.search(query.id == run_id)
        assert len(entries) > 0
        self.analyze(entries)


    def _calculate_and_update_auc(self, entries):
        query = tinydb.Query()
        for entry in entries:
            ptech = str(entry["ptech"])
            data = entry["buggy"]
            if "error" in entry and False:
                data = entry["error"]
            df = pd.DataFrame(data, columns=["buggy"],\
                    index=range(1, entry["cnt"]+1))
            df["cumsum"] = df["buggy"].cumsum()
            self.dfs[ptech] = df

            # Calculate the AUC of each prioritization technique
            err_cnt = df["cumsum"].max()
            max_auc = err_cnt * (entry["cnt"] - 0.5 * err_cnt)
            auc = 100.0 * df["cumsum"].sum() / max_auc
            q = (query.id == entry["id"]) & (query.ptech == entry["ptech"])
            #self.db.update({"auc": auc}, q)
            logger.info("ptech: {}, auc: {}".format(ptech, auc))
        self.dfs["ideal"] = self.__get_ideal_cumsum()


    def plot(self, fname):
        assert len(self.dfs) != 0
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 10}
        matplotlib.rc('font', **font)

        style = itertools.cycle(["-","--","-.",":"])
        lines = []
        int_locator = matplotlib.ticker.MaxNLocator(integer=True)
        plt.figure().gca().yaxis.set_major_locator(int_locator)
        plt.figure().gca().xaxis.set_major_locator(int_locator)
        for name, df in self.dfs.iteritems():
            l, = plt.plot(df.index.tolist(), df["cumsum"], alpha=0.7,
                    label=name, linestyle=style.next())
            lines.append(l)
        plt.ylim(0, max(self.dfs[self.dfs.keys()[0]]["cumsum"]) + 1)
        plt.legend(handles=lines)
        fname = fname + ".png"
        logger.info("Saving the figure to {}".format(fname))
        plt.savefig(fname)


    def __get_ideal_cumsum(self):
        """ Draw a straight `y = x` line that represents an ideal selection
        criterion: all the top prioritized inputs are fault-revealing. """
        df = self.dfs[self.dfs.keys()[0]]
        mx, l = df["cumsum"].max(), len(df["cumsum"])
        cumsum = range(1, l+1)
        for i, s in enumerate(cumsum):
            if s > mx:
                cumsum[i] = mx
        return pd.DataFrame(cumsum, columns=["cumsum"], index=range(1, l+1))



class Evaluator(object):

    def __init__(self, model, dataset, db_name, config):
        self.db = tinydb.TinyDB(db_name)
        self.model = model
        self.dataset = dataset
        self.config = config
        self.ds = []


    def demo(self, ):
        buggy_vector = [
                1, 1, 0, 1, 0, 0, 1, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        poor_vector = [
                0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
        df = pd.DataFrame(buggy_vector, columns=["buggy"],
                index=range(1, len(buggy_vector)+1))
        df["cumsum"] = df["buggy"].cumsum()
        self.dfs["technique1"] = df
        df = pd.DataFrame(poor_vector, columns=["buggy"],
                index=range(1, len(poor_vector)+1))
        df["cumsum"] = df["buggy"].cumsum()
        self.dfs["poor"] = df


    def display_selected(self, selected):
        x, _ = self.dataset.gets_query(selected)
        y = self.model.predict_and_process(x)
        self.dataset.display_selected(selected, y, (7,7))
        self.dataset.display_selected(selected, y, (7,7), reverse=True)


    def save_good_bad(self, query):
        inputs, _ = self.dataset.gets_query(query)
        predictions = self.model.predict_and_process(inputs)
        yq = query.copy(variable="y")
        labels, _ = self.dataset.gets_query(yq)
        corrects = self.dataset.is_correct(query, predictions)
        cnt = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        print("BAD")
        for i, c in enumerate(corrects):
            if c == False:
                cnt += 1
                print(i, predictions[i], labels[i])
                cv2.imwrite('demo/bad_{}.jpg'.format(i), inputs[i])
            if cnt >= 200:
                break

        print("GOOD")
        cnt = 0
        for i, c in reversed(list(enumerate(corrects))):
            if c == True:
                cnt += 1
                print(i, predictions[i], labels[i])
                cv2.imwrite('demo/good_{}.jpg'.format(i), inputs[i])
            if cnt >= 200:
                break


    def evaluate_and_save(self, ptech, query, extra_data={}):
        """ Evaluate a prioritized suite
        :param ptech: name of the prioritization technique
        :param query: prioritized test suite as `dataset.Query` object
        """
        #print(query)
        inputs, indices = self.dataset.gets_query(query)
        # prediction
        ys = self.model.predict_and_process(inputs)
        corrects = self.dataset.is_correct(query, ys)
        #errors = self.dataset.get_error(query, ys)
        # True if buggy
        buggy_int = [0 if c == True else 1 for c in corrects]
        if False:
            # visualize the fault-reavling of the prioritized inputs
            w = ''.join(map(lambda d: '_' if d == 0 else '#', buggy_int))
            logger.debug("buggy: " + w)
        #logger.debug("errors: " + str(list(errors)))
        d = {
                "id": run_id,
                "datetime": {
                    "year": now.year,
                    "month": now.month,
                    "day": now.day,
                    "hour": now.hour,
                    "minute": now.minute,
                    },
                "config": self.config,
                "ptech": ptech,
                # Save only the top prioritized test indices to save space
                "indices": list(indices[:SAVE_LIMIT]),
                "buggy": buggy_int,
                "cnt": len(buggy_int),
#                "query": query.to_dict(),
#                "error": list(errors),
                }
        d.update(extra_data)
        self.db.insert(d)

    def save(self):
        pass


