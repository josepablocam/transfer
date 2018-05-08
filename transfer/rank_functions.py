from abc import ABC, abstractmethod
import ast
import editdistance
import networkx as nx
import zss

from .lift_donations import DonatedFunction

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


class DependencyGraphDistance(FunctionDistanceComputer):
    @staticmethod
    def get_graph(elem):
        if isinstance(elem, DonatedFunction):
            return elem.graph
        elif isinstance(elem, nx.DiGraph):
            return elem
        else:
            raise ValueError('Unhandled function type {} for graph distance'.format(type(elem)))

    def _build_dag(self, graph_1, graph_2, costs):
        # construct DAG by pairing nodes such that
        # sum of distances for paired nodes + sum of delete nodes is
        # minimized
        # graph_1 and graph_2 are DAGS by definition
        # so we only add edges from graph_1 to graph_2, so this is still a DAG
        # and we want no cross-graph edges to cross (indicating different data flow)
        # objective:
        # min sum_{n_1 in graph_1} sum_{n_2 in graph_2} cost[(n_1, n_2)] * chosen((n_1, n_2)) +
        #     sum_{n_1 in graph_1} cost[n_1] * (1 - chosen(n_1)) +
        #     sum_{n_2 in graph_2} cost[n_2] * (1 - chosen(n_2))
        # subject to
        # each n_1, n_2 can be chosen at most once
        # # no crossing edges
        # for all n_1 in graph_1:
        #   for all pred in predecessors(n_1):
        #       edge_from(p).to <= edge_from(n_1).to
        return cost

    def _distance(self, function_1, function_2):
        raise NotImplementedError('jose is still working on this')

        graph_1 = self.get_graph(function_1)
        graph_2 = self.get_graph(function_2)

        # cost for pairing up each possible combo
        node_distance_computer = ZhangShashaTreeDistance()
        costs = {}
        for node_id_1, node_data_1 in graph_1.nodes(data=True):
            for node_id_2, node_data_2 in graph_2.nodes(data=True):
                dist = node_distance_computer.distance(node_data_1['line'], node_data_2['line'])
                costs[(node_id_1, node_id_2)] = dist

        # cost for just removing a node altogether
        for node_id, node_data in graph_1.nodes(data=True) + graph_2.nodes(data=True):
            costs[node_id] = node_distance_computer(node_data['line'], '')

        return self._cost_build_dag(graph_1, graph_2, costs)


class LevenshteinStringDistance(FunctionDistanceComputer):
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
        return editdistance.eval(src_1, src_2)


# class FunctionAbstractor(ABC):
#     """
#     Abstract source code of a function
#     """
#     @abstractmethod
#     def abstract(self, _function):
#         pass

#
# class FunctionRanker(ABC):
#     def __init__(self, graph, abstractor, distance_computer):
#         pass
#
#     @abstractmethod
#     def train(self):
#         pass
#
#     @abstractmethod
#     def predict(self, function_1, relationships):
#         pass
#
#     def rank(self, functions, query_relationships):
#         return sorted(functions, lambda x: self.predict(x, query_relationships))
#
# class LinearRegressionRanker(FunctionRanker):
#     def __init__(self, graph, distance_function):
#         self.graph = graph
#         self.X = None
#         self.y = None
#         self.model = None
#
#     def _prepare(self):
#         # for each node inthe graph
#         # compute features, and store
#         pass
#
#     def train(self):
#         self.model = sklearn.linear_models.LinearRegression()
#         self.model.fit(self.X, self.y)
#
#     def predict(self, function_1, query_relationships):
#         X = self._prepare(function_1, query_relationships)
#         return self.model.predict(X)
#
#
# class DecisionTreeRegressionRanker(FunctionRanker):
#     # pass
#
# class NeuralRegressionRanker(FunctionRanker):
#     # this one should be stochastic
#     pass
