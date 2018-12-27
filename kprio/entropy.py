# 8/13/18
# Taejoon Byun
from __future__ import print_function

import numpy as np
import IPython, sys, logging, copy
import pandas as pd
import tqdm, joblib


class KerasEntropy():
    """
    - KerasEntropy loads a CoverageMatrix.
      - Each entry can have an integer value >= 0
      - Calc frequency of each value for all entries.
    """

    __logger = logging.getLogger('kerascov')

    def __init__(self):
        # a set of observed values in all cells
        self.values = None  # discrete coverage values. i.e. {0, 1}
        self.obligations = None
        self.length = 0        # number of observed inputs
        self.count = {}     # count
        self.p = {}         # probability


    def __init_attributes(self, df):
        self.values = set()
        for col in df:
            for item in df[col]:
                if item not in self.values:
                    self.values.add(item)
        self.values = map(int, self.values)
        self.__logger.debug("{} values: {}".format(len(self.values), self.values))
        self.obligations = df.columns
        self.p = pd.DataFrame(columns=df.columns, index=self.values).fillna(0.0)
        self.count = pd.DataFrame(columns=df.columns, index=self.values).fillna(0)


    def measure(self, cov_df):
        """ get the "entropy matrix" where the rows are for input, cols are for
        neurons.
        :param cov_df: coverage DataFrame from KerasCoverage
        """
        KerasEntropy.__logger.info('Measuring entropy of {} runs'.format(len(cov_df)))
        if not self.values:
            self.__init_attributes(cov_df)

        self.__logger.debug("KerasEntropy: counting ...")
        # obligation * value -> count
        for column in cov_df:
            # for each column in the coverage metric, count the occurrences of
            # unique values (has to be integers). Doing this rather than going
            # through each item saves a ton of time spent in random RAM access.
            cnt_per_col = dict(cov_df[column].value_counts())
            for val, cnt in cnt_per_col.iteritems():
                self.count.at[val, column] = cnt
        self.length += len(cov_df)
        self.count = self.count.fillna(0)
        self.p = self.count / float(self.length)
        self.p = self.p.fillna(0.0)


    def merge(self, other):
        #assert(self.values == other.values)
        new = copy.copy(self)
        new.length = self.length + other.length
        new.count += other.count
        new.count = new.count.fillna(0)
        new.p = new.count / float(new.length)
        new.p = new.p.fillna(0)
        return new


    def load_data(self, filename, length):
        self.length = length
        KerasEntropy.__logger.debug("KerasEntropy.load_data: reading csv")
        self.p = pd.read_csv(filename, index_col=0)
        self.values = list(self.p.index.values)
        self.obligations = self.p.columns
        KerasEntropy.__logger.debug("KerasEntropy.load_data: setting")
        self.count = (self.p * self.length).astype(int)


    def load_from_dir(self, path, length):
        files = glob.glob("{}/*".format(path))
        # some/dir/path/entropy_100_120.csv
        get_key = lambda f: int(f.split('/')[-1].split('_')[1])
        #TODO (8/15/18)
        entropy = KerasEntropy()
        for f in sorted(files, key=get_key):
            KerasEntropy.__logger.debug("parsing entropy from {}".format(f))
            e = KerasEntropy()
            e.load_data(f, length)
            entropy = entropy.merge(e)


    def save_data(self, filename):
        self.p.to_csv(filename)


    def get_entropies(self, cov_df):
        entropies = joblib.Parallel(n_jobs=-1) \
                (joblib.delayed(get_entropy)(self.p, self.length, cov_df.iloc[i]) \
                for i in tqdm.tqdm(range(len(cov_df))))
        return entropies
        #return [self.get_entropy(self.p, self.length, row[1]) for row in df.iterrows()]

# TODO (8/15/18): self.p should be organized in the item/col order. By doing
# so, 60k something columns can be multiplied directly with act_pattern for
# each input.

def get_entropy(freq, length, act_pattern):
    """ Given an activation pattern of hidden layers (in Pandas.Series
    format), calculate the sum of the entropies.
    """
    entropy = 0.0
    #for col in freq:
    for col in freq.columns.tolist()[:-64]:
        item = act_pattern[col]
        p = freq[col][item]
        # chance of seeing `item` in the training dataset
        if p == 0.0:
            # special treatment when p == 0 is encountered: to avoid
            # `inf`, treat this case as if it was seen at least once by
            # 10% chance.
            p = 0.1 / length 
        #elif p == 1.0: p = 1 - 0.1 / length
        #e = -np.log2(p)
        e = -np.log2(p)
        #e = np.log10(-np.log2(p))
        #if e <= 0.0: e = 0.0     # relu-like behavior
        entropy += e
    return entropy


"""
        entropies = joblib.Parallel(backend="threading", n_jobs=24) \
                (joblib.delayed(calc)(self.p[col], act_pattern[i]) \
                for i, col in tqdm.tqdm(enumerate(self.p)))
"""
