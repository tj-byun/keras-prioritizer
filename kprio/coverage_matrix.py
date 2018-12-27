#!/usr/bin/env python
#
# @author: Taejoon Byun <taejoon@umn.edu> #
# first : Dec 13 2017
# last  : May 6 2018

"""
A coverage-satisfaction matrix conceptually looks like the following, where
the column corresponds to the mutant (sorted by name, or instruction address,
in ascending order) and the row corresponds to the test (sorted by id, in an
ascending order).

    |        | mutant 1   | mutant 2   | mutant 3 |
    |--------|------------|------------|----------|
    | test 1 | 0          | 1 (killed) | 1 (k)    |
    | test 2 | 1 (killed) | 0          | 1 (k)    |

The cvs file stores only the contents of the matrix without the header nor
index. The file for the same matrix looks as follows:

    0,1,1
    1,0,1

This Pyton script parses the matrix in the csv file and stores it as a list of
bit-vectors encoded as integers, which allows a fast bit-wise operation across
tests. For instance, the two obligation-satisfaction vectors `011` for test 1
and `101` for test 2 can be or-ed to see if adding test 2 to test 1 (or vice
versa) improves the coverage (of killed mutant, in this case).
"""

import random, sys, re
import pandas as pd
import copy


class CoverageMatrix():

    def __init__(self, columns=[], n_test=0):
        self.matrix = []
        self.dropped_columns = {}   # a set of dropped columns
        self.df = pd.DataFrame(index=range(n_test), columns=columns)
        self.bvs = {}

    def __getitem__(self, key):
        """ Access the matrix by row (test case ID: int) """
        if self.df is None:
            raise Exception('matrix is not set')
        return self.get_row_bv(key)

### Load, save, set

    def set(self, mname, tid, truth):
        self.df[mname][tid] = 1 if bool(truth) else 0

    def parse(self, matrix_fname):
        self.df = pd.read_csv(matrix_fname)

    def save(self, fname):
        self.df.to_csv(fname, index=False)

    def visualize(self):
        # header
        sys.stdout.write('   ')
        for i in range(len(self.df.columns)):
            if i % 10 == 0:
                d = i/10 if i < 100 else (i/10) % 10
                sys.stdout.write('|' + str(d))
            else:
                sys.stdout.write(' ')
        sys.stdout.write('\n')
        # rows
        i = 0
        for _, row in self.df.iterrows():
            if i % 10 == 0:
                sys.stdout.write('{: >3}'.format(i))
            else:
                sys.stdout.write('   ')
            for j, item in enumerate(row.tolist()):
                if j % 10 == 0:
                    sys.stdout.write('|')
                sys.stdout.write(' ' if item == 0 else '#')
            sys.stdout.write('$\n')
            i += 1
        sys.stdout.flush()

    def get_obligation_cnt(self):
        return len(self.df.columns)

    def listize(self, bitvector, len_=0):
        """ Given a bit-vector (int), return a list of Booleans """
        len_ = len(self.df.columns) if len_ == 0 else len_
        strred = str(bin(bitvector))[2:]
        # pad with leading '0's when its length is shorter than `len_`
        strred = '0' * (len_ - len(strred)) + strred
        assert len(strred) == len_
        return [int(bit) for bit in strred]

    def iterbvs(self):
        """ Iterate the rows as bitvectors """
        for i in self.df.index.tolist():
            yield self.get_row_bv(i)

    def get_row_bv(self, i):
        """ Get the i-th row of the matrix as a bit vector """
        if i not in self.bvs:
            self.bvs[i] = int(''.join(map(str, self.df.loc[i].tolist())), 2)
        return self.bvs[i]

    def get_col_bv(self, j):
        """ Get the i-th column of the matrix as a bit vector """
        return int(''.join(map(str, self.df[self.df.columns[j]].tolist())), 2)

    def get_col_bv_by_name(self, colname):
        """ Get the i-th column of the matrix as a bit vector """
        return int(''.join(map(str, self.df[colname].tolist())), 2)

### Measurement, reduction ###

    def get_sat_tests_per_obligation(self, obligation_id):
        """ Returns the list of test case IDs that satisfied the coverage
        obligation given as obligation index.
        :param obligation_id: The index of the obligation to check. The index
            starts from 0, where 0 indicate the first coverage obligation, or
            self.header[0].
        """
        assert 0 <= obligation_id < len(self.df.columns)
        # create a bitmask of which only the n-th bit (from the left) is 1.
        tids = []
        for tid in self.df.index.tolist():
            if df.loc[tid][obligation_id] == 1:
                tids.append(tid)
        return tids

    def get_sat_obligations(self, tests):
        """ Get satisfied obligations after running a list of tests """
        cov = self.get_coverage_bv(tests)
        sat_vector = self.listize(cov)
        assert len(sat_vector) == len(self.df.columns)
        sats = []
        for i, oblg in enumerate(self.df.columns):
            if sat_vector[i] == 1:
                sats.append(oblg)
        return sats

    def get_unsat_obligations(self, tests):
        """ Get unsatisfied obligations after running a list of tests """
        return sorted(set(self.df.columns) - set(get_sat_obligations(tests)))

    def get_coverage_in_percentage(self, suites=None):
        """ Calculate coverage of test suites against a given coverage matrix.
        :param matrix_file: matrix file
        :param suites: list of list of int
        """
        if not suites:
            return float(self.count_sat_obligations()) / len(self.df.columns) * 100.0
        else:
            return [float(self.count_sat_obligations(suite))\
                    / len(self.df.columns) * 100.0 for suite in suites]

    def count_sat_obligations(self, tests=None):
        return str(bin(self.get_coverage_bv(tests)))[2:].count('1')

    def get_coverage_bv(self, tests=None):
        """ Returns the coverage of a test suite in bit-vector format. """
        cov = 0
        if not tests:
            tests = range(len(self.df))
        for tid in tests:
            cov |= self.get_row_bv(tid)
        return cov

    def get_max_coverage_bv(self):
        """ Returns the maximum coverage of the master suite in bit-vector
        format. """
        maxc = 0
        for cov in self.iterbvs():
            maxc |= cov     # bitwise or
        assert maxc != 0
        return maxc

    def get_maximum_coverage_vector(self):
        """ Get maximum coverage vector in Boolean list format """
        return self.listize(self.get_max_coverage_bv(), len(self.df.columns))

### Column manip ###

    def drop_columns(self, columns):
        self.df = self.df.drop(columns, axis=1)

    def drop_duplicate(self):
        dup_list_list = self.__get_duplicate_oblgs()
        self.dropped_columns['duplicate'] = []
        for dup_list in dup_list_list:
            # keep only the 1st column in the "duplicate" set and drop the rest
            to_drop = sorted(dup_list)[1:]
            self.drop_columns(to_drop)
            self.dropped_columns['duplicate'] += to_drop

    def drop_trivial(self):
        trivials = self.__get_trivial_oblgs()
        self.dropped_columns['trivial'] = trivials
        self.drop_columns(trivials)

    def drop_live(self):
        uncovered = self.__get_uncovered_oblgs()
        self.dropped_columns['uncovered'] = uncovered
        self.drop_columns(uncovered)

    def drop_subsumed(self, k=1):
        subsuming = self.__get_subsuming_oblgs_sanjai(k)
        subsumed = set(self.df.columns) - set(subsuming)
        self.df = self.df[subsuming]    # retain the subsuming ones only
        self.dropped_columns['subsumed'] = list(subsumed)

    def sanitize_obligations(self, k=1):
        """ Sanitize the trivial / equivalent / duplicate obligations. Drop the
        columns in the matrix that correspond to those "useless" obligations.
        """
        self.drop_live()
        self.drop_trivial()
        #self.drop_duplicate() ##TJ 4/11
        #self.drop_subsumed(k)
        return self.dropped_columns

    def __get_trivial_oblgs(self):
        """ Returns a list of trivial obligations: the obligations that are
        trivially satisfied by every test case. """
        oblgs = []
        for column in self.df.columns:
            all_sat = True
            for item in self.df[column]:
                if item == 0:
                    # found a test case that doesn't cover this obligation.
                    all_sat = False
                    break
            if all_sat:
                # if every test case SATs this obligation, keep it.
                oblgs.append(column)
        return oblgs

    def __get_uncovered_oblgs(self):
        """ Returns a set of obligations that were not covered by any test case
        """
        uncoverables = []
        for column in self.df.columns:
            if reduce(lambda x, y: x + y, self.df[column].tolist()) == 0:
                # empty column: no test case covers this obligation.
                uncoverables.append(column)
        return uncoverables

    def __get_duplicate_oblgs(self):
        """ Returns a list of duplicate obligations sets: A set of obligation
        is duplicate to each other if they are covered by the exact same set of
        test cases. Exclude the obligations that were never covered.
        """
        cov_per_column = [int(''.join(map(str, self.df[col].tolist())), 2)
                for col in self.df.columns]
        dupsets = []
        for i, cov1 in enumerate(cov_per_column):
            dupset = {self.df.columns[i]}
            for j, cov2 in enumerate(cov_per_column):
                if cov1 == cov2:
                    column = self.df.columns[j]
                    # don't add if this column was already found
                    is_new_dupset = True
                    for ds in dupsets:
                        if column in ds:
                            # already found
                            is_new_dupset = False
                    if is_new_dupset:
                        dupset.add(column)
            if len(dupset) >= 2:
                dupsets.append(dupset)
        return dupsets

    def __get_subsuming_oblgs(self):
        """ Returns a list of (approximately) subsuming (dominant) obligations.

        Note:

        This method implements the greedy algorithm for subsuming mutant
        identification, introduced in the following paper:

        > Papadakis et. al, "Threats to the Validity of Mutation-based Test
          Assessment", ISSTA 2016. (https://dl.acm.org/citation.cfm?id=2931040)
        """
        df = self.df
        max_subsuming_oblgs = set()
        prev_len = len(df.columns)
        while len(df.columns) > 0:
            max_subsumed = 0
            max_subsuming_oblg = None
            subsumed_set = set()
            for i, m1 in enumerate(df.columns):
                # pick a mutant that subsumes the largest number of mutants
                subsumed_by_m1 = set()
                for j, m2 in enumerate(df.columns):
                    v1 = self.get_col_bv(i)
                    v2 = self.get_col_bv(j)
                    if i != j and v1 & v2 == v1:
                        # m1 subsumes m2
                        subsumed_by_m1.add(m2)
                if len(subsumed_by_m1) > max_subsumed:
                    #print 'subsumed_by %s: %d' % (m1, len(subsumed_by_m1))
                    max_subsumed = len(subsumed_by_m1)
                    subsumed_set = subsumed_by_m1 # XXX ??? union?
                    max_subsuming_oblg = m1
            if max_subsuming_oblg:
                max_subsuming_oblgs.add(max_subsuming_oblg)
                df = df.drop(columns=[max_subsuming_oblg])
            # remove subsumed obligations
            df = df.drop(columns=list(subsumed_set))
            if prev_len == len(df.columns):
                max_subsuming_oblgs |= set(df.columns)
                break
            else:
                prev_len = len(df.columns)
        return sorted(max_subsuming_oblgs)

    def __get_subsuming_oblgs_sanjai(self, k=1):
        """ Returns a list of (approximately) subsuming (dominant) obligations.
        """
        def get_subsumption_k(m1, m2):
            # m1 subsumes m2 (m1 -> m2) if bitvector(m1) -> bitvector(m2)
            v1 = self.get_col_bv_by_name(m1)
            v2 = self.get_col_bv_by_name(m2)
            if m1 != m2 and v1 & v2 == v1:
                # return k: the number of test cases that kills both m1 & m2
                return bin(v1 & v2)[2:].count('1')
            else:
                # does not subsume.
                return 0
        df = self.df
        D = set()    # dominant set
        # sort the columns of df
        col_weight_pairs = [(col, df[col].sum()) for col in df.columns]
        sorted_columns = map(lambda pair: pair[0], sorted(col_weight_pairs,
            key=lambda pair: pair[1]))
        drops = set()
        for m1 in sorted_columns:
            subsumed = False
            for m0 in D:
                if get_subsumption_k(m0, m1) >= k:
                    subsumed = True
                    break
            if not subsumed:
                D.add(m1)
                #TODO (4/10)
        return sorted(D)

### Row manip ###

    def drop_rows_except(self, tids):
        todrop = sorted(set(self.df.index.tolist()) - set(tids))
        self.df = self.df.drop(todrop)




class SubsuiteGenerator():
    """ Generate a subsuite """

    def create_random_suites(self, m, repeat_cnt):
        suites = self.create_reduced_suites(m, repeat_cnt)
        return [self.__get_random_sequence(m)[:len(suite)] for suite in suites]

    def __get_random_sequence(self, m):
        """ generate a random sequence of test IDs """
        seq = map(int, m.df.index.tolist())
        random.shuffle(seq)
        return seq

    def select_minimal(self, m):
        """ Select minimal subset of test cases that preserves the maximal
        coverage.
        :return: list of test case IDs (integers)
        """
        tests = self.__get_random_sequence(m)
        prev_cov, current_cov = 0, 0
        max_cov = m.get_max_coverage_bv()
        minimal = []
        for test_id in tests:
            # from a randomized sequence of test ids, add the test only if it
            # increases the coverage of the `minimal` set.
            current_cov = prev_cov | m.get_row_bv(test_id)
            if current_cov > prev_cov:
                minimal.append(test_id)
                prev_cov = current_cov
                if current_cov == max_cov:
                    return sorted(minimal)
            else:
                current_cov = prev_cov
        assert False    # Shouldn't happen: max_cov shall always be reachable

    def select_increased_by(self, m, n):
        """ Select minimal subset of test cases that preserves the maximal
        coverage.
        :return: list of test case IDs (integers)
        """
        tests = self.__get_random_sequence(m)
        prev_cov, current_cov = 0, 0
        minimal = []
        for test_id in tests:
            # from a randomized sequence of test ids, add the test only if it
            # increases the coverage of the `minimal` set by `n`.
            current_cov = prev_cov | m.get_row_bv(test_id)
            if current_cov > prev_cov:
                if str(current_cov - prev_cov).count('1') < n:
                    continue
                minimal.append(test_id)
                prev_cov = current_cov
            else:
                current_cov = prev_cov      # don't select; roll back
        return sorted(minimal)



def get_sats(m):
    suites = []
    with open('/home/taejoon/git/cba/coverage/exp1000-out/microwave_auto-gcc/suites/reduced/oofc_O0.csv') as f:
        f.readline()
        suites = [map(int, line.strip().split(',')) for line in f]
    sats = set()
    for suite in suites[:5]:
        sats |= set(m.get_sat_obligations(suite))
    return sats


def stringify_tid(tid):
    return "%04d" % tid

def parse_reduced_suites_file(suite_file):
    """ Parse the CSV file where the list of "reduced suites" is stored.
    :returns: 2 by 2 matrix of test case IDs
    """
    with open(suite_file) as f:
        return [[int(tid) for tid in line.split(',')] for line in f]

def print_suites(suites, out_file):
    with open(out_file, 'w') as f:
        for suite in suites:
            f.write(','.join(map(lambda i: stringify_tid(i), suite)) + '\n')


## public functions

def print_reduced_suites(matrix_file, repeat_cnt, out_file):
    """ Generate `repeat_cnt` numbers of minimally reduced suites that
    preserve the maximal coverage of the given coverage matrix.
    :param matrix_file: coverage matrix file
    :param repeat_cnt: number of minimal suites to generate
    """
    matrix = CoverageMatrix()
    matrix.parse(matrix_file)
    suites = SubsuiteGenerator().create_reduced_suites(matrix, repeat_cnt)
    print_suites(suites, out_file)

def print_random_suites(matrix_file, repeat_cnt, out_file):
    """ Generate `repeat_cnt` numbers of randomly reduced suites that
    have the same length of reduced suites given a matrix file.
    When an MC/DC matrix is given, for example, generate random suites with the
    same length of reduced MC/DC suites.
    :param matrix_file: coverage matrix file
    :param repeat_cnt: number of minimal suites to generate
    """
    matrix = CoverageMatrix()
    matrix.parse(matrix_file)
    suites = SubsuiteGenerator().create_random_suites(matrix, repeat_cnt)
    print_suites(suites, out_file)

def is_reduced_suite_valid(matrix_file, suite_file):
    matrix = CoverageMatrix()
    matrix.parse(matrix_file)
    suite_list = parse_reduced_suites_file(suite_file)
    max_cov = matrix.get_max_coverage_bv()
    retval = True
    for suite in suite_list:
        cov = 0
        for tid in suite:
            cov |= matrix[tid]
        assert cov <= max_cov
        if cov < max_cov:
            # a reduced suite did not achieve the maximum coverage
            str_cov = str(bin(cov))
            str_max_cov = str(bin(max_cov))
            cnt = 0
            for i in range(len(str_cov)):
                if str_cov[i] == '0' and str_max_cov[i] == '1':
                    cnt += 1
            print 'unsat obligations: ' + str(cnt)
            retval = False
    return retval

def repl():
    if len(sys.argv) != 2:
        print 'Invalid #args'
        return

    N = 5
    ms = []     # original matrices
    suites = [] # test suites
    k1s = []   # sanitized matrices
    k1os = []
    k2s = []   # sanitized matrices
    k2os = []

    m = CoverageMatrix()
    m.parse(sys.argv[1])
    ms.append(m)
    for i in range(1, N):
        ms.append(copy.copy(m))

    suites = SubsuiteGenerator().create_reduced_suites(m, N-1)
    for i in range(0, N):
        print i
        k1s.append(copy.copy(ms[i]))
        k2s.append(copy.copy(ms[i]))
        if i >= 1:
            k1s[i].drop_rows_except(suites[i-1])
            k2s[i].drop_rows_except(suites[i-1])
        k1s[i].sanitize_obligations(k=1)
        k2s[i].sanitize_obligations(k=2)
        k1os.append(set(k1s[i].df.columns))
        k2os.append(set(k2s[i].df.columns))

    import IPython
    IPython.embed()

    return

if __name__ == '__main__':
    repl()


"""
    def get_coverage_of_suite_file(self, suite_file):
        tests = set()
        with open(suite_file, 'r') as f:
            for line in f:
                tests |= set(map(int, line.strip().split(',')))
        return self.get_coverage_bv(tests)
"""
