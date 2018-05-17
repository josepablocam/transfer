# Use boruca functions to create extract features
# and try to improve upon a baseline model
from argparse import ArgumentParser

import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sklearn.feature_selection
import tqdm

from transfer.build_db import *
from transfer.utils import sort_by_values


class ColumnRelationshipsCache(object):
    def __init__(self, db):
        self.db = db
        self._rels = {}

    def defines(self, f):
        return set(self.rels(f).get(RelationshipTypes.DEFINES, []))

    def uses(self, f):
        return set(self.rels(f).get(RelationshipTypes.USES, []))

    def rels(self, f):
        rels_dict = self._rels.get(f, None)
        if rels_dict is None:
            rels_dict = {}
            for r, v in get_extracted_function_relationships(f):
                if not r in rels_dict:
                    rels_dict[r] = []
                rels_dict[r].append(v)
            self._rels[f] = rels_dict
        return rels_dict


def get_new_column_transforms(db):
    funs = db.extracted_functions()
    eligible_funs = []
    criteria = {'num_args': 0, 'return_var': 0}
    for node in funs:
        f = db.get_function_from_node(node)
        # only consider functions with a single input argument
        if len(f.formal_arg_vars) != 1:
            criteria['num_args'] += 1
            continue
        # # only consider functions that return the same input value (modified)
        # if f.formal_arg_vars[0] != f.return_var:
        #     criteria['return_var'] += 1
        #     continue
        eligible_funs.append(f)
    # sort these by LOC
    locs = [len(f.source.split('\n')) for f in eligible_funs]
    return sort_by_values(eligible_funs, locs), criteria

def get_column_value(X, col):
    try:
        return X[col]
    except KeyError:
        if col == 'index':
            return X.index
        else:
            return None

def add_column_from_function(f, X, rels_cache):
    X_copy = X.copy()
    applied = {}
    info = None
    try:
        X_copy = f(X_copy)
        # make sure X's number of rows are maintained
        # since just care about adding new columns, not general transforms
        if X_copy.shape[0] != X.shape[0]:
            raise Exception('Changed dimensions')

        cols_used = rels_cache.uses(f)
        cols_defined = rels_cache.defines(f)

        info = {}
        info['function'] = f
        # only care about columns used that were not defined by transform itself!
        info['cols_used'] = cols_used.intersection(X.columns.tolist() + ['index'])
        info['cols_defined'] = cols_defined.intersection(X_copy.columns.tolist() + ['index'])
        info['old_values'] = {}
        info['new_values'] = {}

        for col in info['cols_defined']:
            info['new_values'][col] = get_column_value(X_copy, col)

        for col in info['cols_used']:
            info['old_values'][col] = get_column_value(X_copy, col)
    except:
        info = None
        pass

    return info

def is_numpy_numeric_type(_type):
    return np.issubdtype(_type, np.number)

def is_just_renaming(new_series, collection_old_series):
    for old_series in collection_old_series:
        if len(old_series) != len(new_series):
            continue
        if np.all(old_series.values == new_series.values):
            return True
    return False

def mutual_info_wrapper(mi_op, x, y):
    x = np.copy(x)

    if x.ndim == 1:
        # mutual info wants column vector
        x = x.reshape(-1, 1)

    try:
        if np.isnan(x).any():
            # does not work with nans, so fill these with zero
            x[np.isnan(x)] = 0
    except TypeError:
        # some numpy types cant be checked for nan
        pass

    if x.dtype == np.dtype('object'):
        # mutual info functions don't like object and fail to convert
        # event when discrete_features=True
        x = x.astype(str)

    discrete_x = not is_numpy_numeric_type(x.dtype)
    return mi_op(x, y, discrete_features=[discrete_x])[0]

def run(db, data, target_column, n_iters):
    results = []

    feature_funs, criteria = get_new_column_transforms(db)
    print('{} eligible feature functions'.format(len(feature_funs)))
    print('Removals based on criteria {}'.format(criteria))

    rels_cache = ColumnRelationshipsCache(db)

    # this is not a very useful metric....since only works for categorical info
    if is_numpy_numeric_type(data[target_column].dtype):
        mutual_info = sklearn.feature_selection.mutual_info_regression
    else:
        mutual_info = sklearn.feature_selection.mutual_info_classif

    for i in tqdm.tqdm(list(range(n_iters))):
        X = data.sample(1000)
        y = X[target_column]

        for f in tqdm.tqdm(feature_funs):
            info = add_column_from_function(f, X, rels_cache)
            if info is None:
                continue

            info['iter'] = i
            info['cols_defined_info'] = {}
            info['cols_used_info'] = {}

            for c in info['cols_defined']:
                new_vec = info['new_values'][c]
                just_renaming = is_just_renaming(new_vec, info['old_values'].values())
                _type = new_vec.dtype
                ct_unique_vals = len(new_vec.unique())
                mi = mutual_info_wrapper(mutual_info, new_vec.values, y.values)
                col_info = dict(
                    name=c,
                    just_renaming=False,
                    type=_type,
                    ct_unique_vals=ct_unique_vals,
                    mi=mi,
                )
                info['cols_defined_info'][c] = col_info

            for c in info['cols_used']:
                old_vec = info['old_values'][c]
                _type = old_vec.dtype
                ct_unique_values = len(old_vec.unique())
                mi = mutual_info_wrapper(mutual_info, old_vec.values, y.values)
                col_info = dict(
                    name=c,
                    just_renaming=False,
                    type=_type,
                    ct_unique_vals=ct_unique_vals,
                    mi=mi,
                )
                info['cols_used_info'][c] = col_info

            info.pop('new_values')
            info.pop('old_values')
            if info['cols_used'] or info['cols_defined']:
                results.append(info)

    return results


def cleanup_for_pickle(res):
    for entry in res:
        entry['function']._obj = None

loan_data_config = dict(
    target_column = 'loan_status',
    n_iters=1,
)


def main(args):
    with open(args.database_file, 'rb') as f:
        db = pickle.load(f)
    db.startup()

    data = pd.read_csv(args.dataset_path)

    if args.dataset_config == 'loan_data':
        config = loan_data_config
    else:
        raise Exception('Invalid configuration selection')

    results = run(db, data, config['target_column'], config['n_iters'])
    cleanup_for_pickle(results)

    with open(args.output_path, 'wb') as f:
        pickle.dump(results, f)

    db.shutdown()


if __name__ == '__main__':
    parser = ArgumentParser(description='Run feature addition evaluation')
    parser.add_argument('database_file', type=str, help='Path to pickled db interface')
    parser.add_argument('dataset_path', type=str, help='Path to data used by programs')
    parser.add_argument('dataset_config', type=str, help='Config to run one of [loan_data]')
    parser.add_argument('output_path', type=str, help='Path to store pickled results')
    args = parser.parse_args()

    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
