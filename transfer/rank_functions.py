from abc import ABC, abstractmethod
import ast

import editdistance
import networkx as nx
import numpy as np
import py2neo
import sklearn.feature_extraction
import tqdm
import zss

from .lift_donations import DonatedFunction
from .utils import sort_by_values

class FunctionDistanceComputer(ABC):
    """
    Pair-wise distance between two functions
    """
    def __init__(self):
        self.cache = {}

    def distance(self, function_1, function_2):
        if (function_1, function_2) in self.cache:
            return self.cache[(function_1, function_2)]
        if (function_2, function_1) in self.cache:
            return self.cache[(function_2, function_1)]
        dist = self._distance(function_1, function_2)
        self.cache[(function_1, function_2)] = dist
        return dist

    @abstractmethod
    def _distance(self, function_1, function_2):
        pass


# TODO we should probably handle commutativity ourselves (currently ignores that)
# TODO: this is uber slow...consider PQ-GRAM instead, if we want AST-based options
class ZhangShashaTreeDistance(FunctionDistanceComputer):
    @staticmethod
    def get_node_label(node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        else:
            return type(node).__name__

    @staticmethod
    def get_children(node):
        return list(ast.iter_child_nodes(node))

    @staticmethod
    def label_dist(label_1, label_2):
        # assume single edits, not string distance
        return 1 if label_1 != label_2 else 0

    @staticmethod
    def get_tree(elem):
        if isinstance(elem, DonatedFunction):
            return elem.ast
        elif isinstance(elem, str):
            return ast.parse(elem)
        elif isinstance(elem, ast.AST):
            return elem
        else:
            raise ValueError('Unhandled function type {} for tree distance'.format(type(elem)))

    def _distance(self, function_1, function_2):
        ast_1 = self.get_tree(function_1)
        ast_2 = self.get_tree(function_2)

        return zss.simple_distance(
            ast_1, ast_2,
            self.get_children,
            self.get_node_label,
            self.label_dist
            )

# TODO: WRITE THIS AT SOME POINT....
# class DependencyGraphDistance(FunctionDistanceComputer):
#     @staticmethod
#     def get_graph(elem):
#         if isinstance(elem, DonatedFunction):
#             return elem.graph
#         elif isinstance(elem, nx.DiGraph):
#             return elem
#         else:
#             raise ValueError('Unhandled function type {} for graph distance'.format(type(elem)))
#
#     def _build_dag(self, graph_1, graph_2, costs):
#         # construct DAG by pairing nodes such that
#         # sum of distances for paired nodes + sum of delete nodes is
#         # minimized
#         # graph_1 and graph_2 are DAGS by definition
#         # so we only add edges from graph_1 to graph_2, so this is still a DAG
#         # and we want no cross-graph edges to cross (indicating different data flow)
#         # objective:
#         # min sum_{n_1 in graph_1} sum_{n_2 in graph_2} cost[(n_1, n_2)] * chosen((n_1, n_2)) +
#         #     sum_{n_1 in graph_1} cost[n_1] * (1 - chosen(n_1)) +
#         #     sum_{n_2 in graph_2} cost[n_2] * (1 - chosen(n_2))
#         # subject to
#         # each n_1, n_2 can be chosen at most once
#         # # no crossing edges
#         # for all n_1 in graph_1:
#         #   for all pred in predecessors(n_1):
#         #       edge_from(p).to <= edge_from(n_1).to
#         return cost
#
#     def _distance(self, function_1, function_2):
#         raise NotImplementedError('jose is still working on this')
#
#         graph_1 = self.get_graph(function_1)
#         graph_2 = self.get_graph(function_2)
#
#         # cost for pairing up each possible combo
#         node_distance_computer = ZhangShashaTreeDistance()
#         costs = {}
#         for node_id_1, node_data_1 in graph_1.nodes(data=True):
#             for node_id_2, node_data_2 in graph_2.nodes(data=True):
#                 dist = node_distance_computer.distance(node_data_1['line'], node_data_2['line'])
#                 costs[(node_id_1, node_id_2)] = dist
#
#         # cost for just removing a node altogether
#         for node_id, node_data in graph_1.nodes(data=True) + graph_2.nodes(data=True):
#             costs[node_id] = node_distance_computer(node_data['line'], '')
#
#         return self._cost_build_dag(graph_1, graph_2, costs)


class SourceCodeStringDistance(FunctionDistanceComputer):
    def __init__(self, str_dist_fun=None):
        super().__init__()
        if str_dist_fun is None:
            str_dist_fun = editdistance.eval
        if isinstance(str_dist_fun, str):
            if str_dist_fun == 'levenshtein':
                str_dist_fun = editdistance.eval
            else:
                raise ValueError('Unknown string distance name')
        self.str_dist_fun = str_dist_fun

    @staticmethod
    def get_string(elem):
        if isinstance(elem, DonatedFunction):
            return elem.source
        elif isinstance(elem, str):
            return elem
        elif isinstance(elem, ast.AST):
            return ast.dump(elem)
        else:
            raise ValueError('Unhandled function type {} for string distance'.format(type(elem)))

    def _distance(self, function_1, function_2):
        src_1 = self.get_string(function_1)
        src_2 = self.get_string(function_2)
        return self.str_dist_fun(src_1, src_2)

def get_distance_computer_by_type(distance_type):
    if distance_type == 'string':
        return SourceCodeStringDistance('levenshtein')
    elif distance_type == 'tree':
        return ZhangShashaTreeDistance()
    else:
        raise Exception('invalid source code distance type (must be one of string or tree)')

class LearningDataPreparer(ABC):
    def prepare(self, db):
        # takes database
        # returns a matrix and target vector to learn
        pass


class BinaryRelationshipDataPreparer(LearningDataPreparer):
    def __init__(self, db, distance_computer=None):
        self.db = db
        if distance_computer is None:
            # choose fastets option for now...
            distance_computer = SourceCodeStringDistance('levenshtein')
        self.distance_computer = distance_computer
        self.relationships_and_features_cache = {}
        self.vectorizer = sklearn.feature_extraction.DictVectorizer()

    def _get_relationships_and_features(self, node, prefix=None):
        result = self.relationships_and_features_cache.get(node, None)
        if result is None:
            # query database and get relations from that...
            # no need to go through this again
            rels = self.db.get_extracted_function_relationships_from_node(node)
            feats = self._relationships_to_dict(rels, prefix)
            result = (rels, feats)
            self.relationships_and_features_cache[node] = result
        rels, feats = result
        return rels, dict(feats)

    def _relationships_to_dict(self, relationships, prefix=None):
        assert prefix is not None
        if prefix is None:
            prefix = ''
        else:
            prefix = prefix + '+'
        _dict = {}
        for relationship_type, val in relationships:
            _dict[prefix + relationship_type.name + '+' + val] = True
        return _dict

    # TODO: this function is an obscenity....jesus. fix this
    def prepare(self, query_elems=None, query_results=None, remove_nodes=None):
        self.db.startup()
        X_data = []
        y_data = []
        ref_query_elems = []
        ref_result_elems = []

        if query_elems is None:
            query_elems = self.db.extracted_functions()

        if isinstance(query_elems[0], tuple):
            query_elems = [query_elems]

        for query_elem in tqdm.tqdm(query_elems):

            if isinstance(query_elem, py2neo.Node):
                query_rels, query_feats = self._get_relationships_and_features(query_elem, prefix='query-node')
                query_fun = self.db.get_function_from_node(query_elem)
            else:
                query_rels = query_elem
                query_feats = self._relationships_to_dict(query_rels, prefix='query-node')
                query_fun = None

            if query_results is None:
                result_nodes = self.db.query_by_relationships(query_rels)
            else:
                result_nodes = set(query_results)

            # remove self from results
            result_nodes.remove(query_elem)
            # drop other nodes (may be useful for train/test setups)
            if remove_nodes:
                result_nodes = result_nodes.difference(remove_nodes)

            for result_node in tqdm.tqdm(result_nodes):
                result_rels, result_feats = self._get_relationships_and_features(result_node, prefix='result-node')
                result_fun = self.db.get_function_from_node(result_node)

                row = {}
                row.update(query_feats)
                row.update(result_feats)
                X_data.append(row)
                ref_query_elems.append(query_elem)
                ref_result_elems.append(result_node)

                if query_fun is not None:
                    dist = self.distance_computer.distance(query_fun, result_fun)
                    y_data.append(dist)

        if not hasattr(self.vectorizer, 'vocabulary_'):
            print('Fitting transformer')
            self.vectorizer.fit(X_data)

        X_matrix = self.vectorizer.transform(X_data)
        y_vec = np.array(y_data)
        ref_query_elems_vec = np.array(ref_query_elems)
        ref_result_elems_vec = np.array(ref_result_elems)
        return X_matrix, y_vec, ref_query_elems_vec, ref_result_elems_vec


def rank_query_results(data_preparer, model, rels_for_query, query_results):
    X, _ = data_preparer.prepare(rels_for_query, query_results)
    predicted_distances = model.predict(X)
    # use to break ties deterministically
    result_names = [n['name'] for n in query_results]
    rank_metric = list(zip(predicted_distances, result_names))
    return sort_by_values(query_results, rank_metric)
