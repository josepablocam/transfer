from argparse import ArgumentParser
from collections import defaultdict
import glob
import os
import pickle

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model
import tqdm

from transfer.build_db import *
from transfer.rank_functions import *

def get_test_file_info_from_db(db):
    info = defaultdict(lambda: [])
    nodes = db.extracted_functions()
    for n in nodes:
        f = db.get_function_from_node(n)
        graph_file_name = f.graph.graph['filename']
        info[graph_file_name].append(n)
    return info

def precompute_distances(db):
    distance_computer = SourceCodeStringDistance('levenshtein')
    funs = db.extracted_functions()
    print('Pre-computing distances')
    # compute and caches underneath covers
    for i, f1 in tqdm.tqdm(list(enumerate(funs))):
        for f2 in tqdm.tqdm(funs[(i + 1):]):
            distance_computer.distance(f1, f2)
    return distance_computer

def load_distance_computer(db, distance_computer):
    if distance_computer is None:
        obj = precompute_distances(db)
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
            obj = load_distance_computer(db, None)
            with open(distance_computer, 'wb') as f:
                f.write(obj)
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
        test_fun = db.get_function_from_node(test_node)
        # optimally sorted query results
        distances = distances_from_nodes(test_fun, db, query_results, distance_computer)
        query_results = sorted(zip(query_results, distances), key = lambda x: x[1])
        query_results = [r for r, _  in query_results]

        data[test_node] = {}
        data[test_node]['rels_for_query'] = rels_for_query
        data[test_node]['query_results'] = query_results
        data[test_node]['fun'] = test_fun
    return data

def spearman_correlation(l1, l2):
    res = scipy.stats.spearmanr(l1, l2)
    return res.correlation, res.pvalue


class BaselineRandomRegressor(object):
    def __init__(self, seed=None):
        self.random_state = np.random.RandomState(seed=seed)

    def predict(self, X):
        return self.random_state.uniform()


def run(db, n_iters, model_constuctors, distance_computer=None):
    distance_computer = load_distance_computer(db, distance_computer)
    data = []
    test_file_info = get_test_file_info_from_db(db)

    # LOOCV style evaluation
    for test_file_name, test_nodes in tqdm.tqdm(test_file_info.items()):
        test_data = compute_test_data(db, test_nodes, distance_computer)

        prep = BinaryRelationshipDataPreparer(db, distance_computer)
        X, y = prep.prepare(remove_nodes=test_nodes)

        for i in tqdm.tqdm(list(range(n_iters))):
            # some models may have random initializations, so they
            # should be constructed within the iteration loop
            for model_constructor in tqdm.tqdm(model_constuctors):
                model = model_constructor()
                model.fit(X, y)

                for test_node in tqdm.tqdm(test_nodes):
                    rels_for_query = test_data[test_node]['rels_for_query']
                    optimally_ranked_results = test_data[test_node]['query_results']
                    # evaluate ranking relative to ideal
                    pred_ranked_results = rank_query_results(prep, model, rels_for_query, optimally_ranked_results)
                    corr, pval = spearman_correlation(optimally_ranked_results, pred_ranked_results)
                    # evaluate observed distances in order of ranked results
                    test_fun = db.get_function_from_node(test_node)
                    distances = distances_from_nodes(test_fun, db, pred_ranked_results, distance_computer)

                    info = dict(
                        test_file=test_file_name,
                        iter=i,
                        model=model.__qualname__,
                        spearman_corr=corr,
                        sprearman_pval=pval,
                        predicted_dists = distances,
                    )
                    data.append(info)

    return pd.DataFrame(data)

def main(args):
    # pre-set for now...
    model_constructors = [
        lambda: sklearn.linear_model.LinearRegression(),
        lambda: BaselineRandomRegressor(),
    ]

    database_path = args.database_path
    with open(database_path, 'rb') as f:
        db = pickle.load(f)
    n_iters = args.n_iters
    distance_computer = args.distance_computer
    run(db, n_iters, model_constructors, distance_computer=distance_computer)

if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate database query result ranking by LOOCV')
    parser.add_argument('database_file', type=str, help='Pickled database interface file')
    parser.add_argument('n_iters', type=int, help='Number of iterations for each test_node')
    parser.add_argument('output_path', type=str, help='Path to store pickled dataframe with evaluation raw data')
    parser.add_argument('-d', '--distance_computer', type=str, help='Path to existing (or desired location for storage) for distance computer')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
