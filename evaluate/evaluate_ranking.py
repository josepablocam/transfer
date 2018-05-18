from argparse import ArgumentParser
from collections import defaultdict
import glob
import os
import pickle

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model
import sklearn.tree
import tqdm

from transfer.build_db import *
from transfer.rank_functions import *
from transfer.utils import sort_by_values

def get_test_file_info_from_db(db):
    info = defaultdict(lambda: [])
    nodes = db.extracted_functions()
    for n in nodes:
        f = db.get_function_from_node(n)
        graph_file_name = f.graph.graph['filename']
        info[graph_file_name].append(n)
    return info

def precompute_distances(db, distance_type):
    distance_computer = get_distance_computer_by_type(distance_type)
    funs = db.extracted_functions()
    print('Pre-computing distances')
    # compute and caches underneath covers
    for i, f1_node in tqdm.tqdm(list(enumerate(funs))):
        f1 = db.get_function_from_node(f1_node)
        for f2_node in tqdm.tqdm(funs[(i + 1):]):
            f2 = db.get_function_from_node(f2_node)
            distance_computer.distance(f1, f2)
    return distance_computer

def load_distance_computer(db, distance_computer, distance_type):
    if distance_computer is None:
        obj = precompute_distances(db, distance_type)
        return obj
    elif isinstance(distance_computer, FunctionDistanceComputer):
        return distance_computer
    elif isinstance(distance_computer, str):
        if os.path.exists(distance_computer):
            print('Attempting read pre-computed distances from: {}'.format(distance_computer))
            with open(distance_computer, 'rb') as f:
                obj = pickle.load(f)
            return obj
        else:
            obj = load_distance_computer(db, None, distance_type)
            with open(distance_computer, 'wb') as f:
                pickle.dump(obj, f)
            return obj
    else:
        raise ValueError('Invalid distance computer')

def distances_from_nodes(ref_fun, db, nodes, distance_computer):
    distances = []
    for n in nodes:
        f = db.get_function_from_node(n)
        dist = distance_computer.distance(ref_fun, f)
        distances.append(dist)
    return distances

def compute_test_data(db, test_nodes, distance_computer):
    print('Compute test node data')
    data = {}
    for test_node in tqdm.tqdm(test_nodes):
        rels_for_query = db.get_extracted_function_relationships_from_node(test_node)
        query_results = db.query_by_relationships(rels_for_query)
        # remove self
        query_results.remove(test_node)
        # remove anything from the test file
        query_results = list(query_results.difference(test_nodes))
        test_fun = db.get_function_from_node(test_node)
        # optimally sorted query results
        distances = distances_from_nodes(test_fun, db, query_results, distance_computer)
        # use function names to break ties for deterministic sorting...
        ranking_metric = list(zip(distances, [n['name'] for n in query_results]))
        query_results = sort_by_values(query_results, distances)

        data[test_node] = {}
        data[test_node]['rels_for_query'] = rels_for_query
        data[test_node]['query_results'] = query_results
        data[test_node]['fun'] = test_fun
    return data

def spearman_correlation(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    res = scipy.stats.spearmanr(l1, l2)
    return res.correlation, res.pvalue


class BaselineRandomRegressor(object):
    def __init__(self, seed=None):
        self.random_state = np.random.RandomState(seed=seed)

    def fit(self, X, y):
        pass

    def predict(self, X):
        return self.random_state.uniform(size=X.shape[0])


def run(db, n_iters, model_constuctors, distance_computer=None, distance_type=None):
    if distance_type is None:
        distance_type = 'string'
    distance_computer = load_distance_computer(db, distance_computer, distance_type)
    data = []
    test_file_info = get_test_file_info_from_db(db)

    prep = BinaryRelationshipDataPreparer(db, distance_computer)
    X, y, ref_query_nodes, ref_result_nodes = prep.prepare()
    ixs = np.arange(X.shape[0])

    # LOOCV style evaluation
    for test_file_name, test_nodes in tqdm.tqdm(test_file_info.items()):
        test_data = compute_test_data(db, test_nodes, distance_computer)

        is_test_query = np.isin(ref_query_nodes, test_nodes)
        is_test_result = np.isin(ref_result_nodes, test_nodes)
        is_training = ~is_test_query & ~is_test_result
        X_train = X[is_training, :]
        y_train = y[is_training]

        for i in tqdm.tqdm(list(range(n_iters))):
            # some models may have random initializations, so they
            # should be constructed within the iteration loop
            for model_constructor in tqdm.tqdm(model_constuctors):
                model = model_constructor()
                model.fit(X_train, y_train)

                for test_node in tqdm.tqdm(test_nodes):
                    info = dict(
                        test_file=test_file_name,
                        iter=i,
                        model=model.__class__.__qualname__,
                        test_node_name=test_node['name'],
                        spearman_corr=np.nan,
                        spearman_pval=np.nan,
                        distances_for_preds=[],
                        distances_for_optimal=[],
                        query_results=[],
                    )
                    rels_for_query = test_data[test_node]['rels_for_query']
                    optimally_ranked_results = test_data[test_node]['query_results']

                    if len(optimally_ranked_results) > 0:
                        # retrieve indices for test data and make sure
                        # their order matches the optimally_ranked_results
                        # so that prediction results are correctly aligned to sort
                        test_ix = ixs[(ref_query_nodes == test_node) & ~is_test_result]
                        orig_ordered_result_nodes = ref_result_nodes[test_ix]
                        test_ix_dict = dict(zip(orig_ordered_result_nodes, test_ix))
                        test_ix = [test_ix_dict[r] for r in optimally_ranked_results]

                        X_test = X[test_ix, :]
                        assert X_test.shape[0] == len(optimally_ranked_results)

                        # break ranking ties with function name
                        ordered_result_nodes = ref_result_nodes[test_ix]
                        result_nodes_names = [n['name'] for n in ordered_result_nodes]
                        pred_ranking_metric = list(zip(model.predict(X_test), result_nodes_names))
                        pred_ranked_results = sort_by_values(optimally_ranked_results, pred_ranking_metric)
                        assert len(pred_ranked_results) == len(optimally_ranked_results)

                        # evaluate ranking relative to ideal
                        optimally_ranked_results_names = [n['name'] for n in optimally_ranked_results]
                        pred_ranked_results_names = [n['name'] for n in pred_ranked_results]
                        info['spearman_corr'], info['spearman_pval'] = spearman_correlation(
                            optimally_ranked_results_names,
                            pred_ranked_results_names
                            )

                        # evaluate observed distances in order of ranked results
                        test_fun = db.get_function_from_node(test_node)
                        info['distances_for_preds'] = distances_from_nodes(test_fun, db, pred_ranked_results, distance_computer)
                        info['distances_for_optimal'] = distances_from_nodes(test_fun, db, optimally_ranked_results, distance_computer)
                        info['query_results_names'] = optimally_ranked_results_names
                    data.append(info)

    return pd.DataFrame(data)

def main(args):
    # pre-set for now...
    model_constructors = [
        lambda: sklearn.linear_model.LinearRegression(),
        lambda: sklearn.linear_model.Ridge(),
        lambda: sklearn.linear_model.Lasso(),
        lambda: sklearn.linear_model.ElasticNet(),
        lambda: sklearn.tree.DecisionTreeRegressor(),
        lambda: BaselineRandomRegressor(),
    ]

    database_file = args.database_file
    with open(database_file, 'rb') as f:
        db = pickle.load(f)
    db.startup()
    n_iters = args.n_iters
    distance_computer = args.distance_computer
    distance_type=args.distance_type
    result_df = run(db, n_iters, model_constructors, distance_computer=distance_computer, distance_type=distance_type)
    result_df.to_pickle(args.output_path)
    db.shutdown()

if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate database query result ranking by LOOCV')
    parser.add_argument('database_file', type=str, help='Pickled database interface file')
    parser.add_argument('n_iters', type=int, help='Number of iterations for each test_node')
    parser.add_argument('output_path', type=str, help='Path to store pickled dataframe with evaluation raw data')
    parser.add_argument('-t', '--distance_type', type=str, help='Type of distance computation', default='string')
    parser.add_argument('-d', '--distance_computer', type=str, help='Path to existing (or desired location for storage) for distance computer')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
