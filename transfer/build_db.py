# Construct graph database for extracted functions and their relationships
# to columns and third party libraries

from argparse import ArgumentParser
from collections import Counter
from enum import Enum
import pickle

import networkx as nx
import numpy as np
import py2neo
from plpy.analyze.dynamic_trace_events import ExecLine

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering

from .lift_donations import (
    DonatedFunction,
    lower_lifted_rewrites,
    rename_lowered_rewrites,
    remove_table_noops,
    collect_key_tokens,
)


class NodeTypes(Enum):
    EXTRACTED_FUNCTION = 1
    FUNCTION = 2
    COLUMN = 3


class RelationshipTypes(Enum):
    DEFINES = 1
    USES = 2
    CALLS = 3
    WRANGLES_FOR = 4


def get_function_full_name(elem):
    if isinstance(elem, str):
        return elem
    elif isinstance(elem, dict):
        call_details = elem
        modulename = call_details['module']
        qualname = call_details['qualname']
        co_name = call_details['co_name']
    else:
        try:
            fun = elem
            modulename = getattr(fun, '__module__', None)
            qualname = getattr(fun, '__qualname__', None)
            code = getattr(fun, '__code__', None)
            co_name = '' if code is None else code.co_name
        except AttributeError:
            raise ValueError(
                'Unable to compute full_name from {}, provide string of name'.
                format(elem)
            )
    full_name = []
    full_name.append('' if modulename is None else modulename + '.')
    full_name.append(co_name if qualname is None else qualname)
    full_name = ''.join(full_name)
    return None if full_name == '' else full_name


def get_column_name(elem):
    if isinstance(elem, str):
        return elem
    elif isinstance(elem, py2neo.Node):
        assert elem.has_label(NodeTypes.COLUMN.name)
        return elem['name']
    else:
        raise ValueError(
            'Unable to compute column_name from {}, provide string of name'.
            format(elem)
        )


def get_wrangles_for(fun, program_graph):
    # use node id for the value associated with the return
    _return_node_id = max(fun.graph.nodes)
    fwd_nodes = nx.dfs_tree(program_graph, _return_node_id)
    fwd_slice = program_graph.subgraph(fwd_nodes)
    full_names = set([])
    for _, node_data in fwd_slice.nodes(data=True):
        if isinstance(node_data['event'], ExecLine):
            # this may or may not be annotated, so be safe...
            if node_data.get('treat_as_comment', False):
                continue
            if node_data['calls'] is None:
                continue
            for call in node_data['calls']:
                full_name = get_function_full_name(call.details)
                if full_name is not None:
                    full_names.add(full_name)
    return full_names


def get_extracted_function_relationships(fun, program_graph=None):
    assert isinstance(fun, DonatedFunction), 'Can only add extracted functions'
    relationship_tuples = set([])

    for _, node_data in fun.graph.nodes(data=True):
        if node_data.get('treat_as_comment', False):
            continue
        for col in node_data['columns_defined']:
            relationship_tuples.add((RelationshipTypes.DEFINES, col))
        for col in node_data['columns_used']:
            relationship_tuples.add((RelationshipTypes.USES, col))
        if node_data['calls'] is not None:
            for call in node_data['calls']:
                full_name = get_function_full_name(call.details)
                if full_name is not None:
                    relationship_tuples.add(
                        (RelationshipTypes.CALLS, full_name)
                    )

    if not program_graph is None:
        for full_name in get_wrangles_for(fun, program_graph):
            relationship_tuples.add(
                (RelationshipTypes.WRANGLES_FOR, full_name)
            )

    return relationship_tuples


class FunctionDatabase(object):
    def __init__(self):
        self.id_to_fun = {}
        # tuple of (node_type, X)
        self.node_info_to_id_cache = {}
        self.graph_db = None
        self.selectors = None
        # maintain a global count of functions extracted and added
        self.fun_counter = 0
        self._has_started_up = False
        # accumulate function bodies to avoid duplicates
        self.function_acc = set([])

    def startup(self):
        self.graph_db = py2neo.Graph()
        self.selectors = {}
        # setup selectors
        for node_type in NodeTypes:
            self.selectors[node_type] = py2neo.NodeSelector(
                self.graph_db
            ).select(node_type.name)
        self._has_started_up = True

    def shutdown(self):
        self.graph_db = None
        self.selectors = {}
        self._has_started_up = False

    def _get_or_create_node(self, node_type, **kwargs):
        name = kwargs['name']
        node_id = self.node_info_to_id_cache.get((node_type, name), None)
        if node_id is None:
            node = py2neo.Node(node_type.name, **kwargs)
            self.graph_db.create(node)
            node_id = self._get_node_id(node)
            self.node_info_to_id_cache[(node_type, name)] = node_id
            return node
        return self.graph_db.node(node_id)

    def _convert_to_target_node(self, relationship_type, target_node):
        if isinstance(target_node, py2neo.Node):
            return target_node
        elif isinstance(target_node, str):
            if relationship_type in set([RelationshipTypes.DEFINES,
                                         RelationshipTypes.USES]):
                target_node_type = NodeTypes.COLUMN
            elif relationship_type in set([RelationshipTypes.CALLS,
                                           RelationshipTypes.WRANGLES_FOR]):
                target_node_type = NodeTypes.FUNCTION
            else:
                raise Exception(
                    "Need to know target_node_type for relationship type: {}".
                    format(relationship_type)
                )
            return self._get_or_create_node(target_node_type, name=target_node)
        else:
            raise Exception(
                "Don't know how to convert type {} to py2neo.Node".format(
                    type(target_node)
                )
            )

    def _create_relationship(
        self, src_node, relationship_type, target_node, **kwargs
    ):
        assert isinstance(src_node, py2neo.Node)
        # must create target_node if doesn't exist
        target_node = self._convert_to_target_node(
            relationship_type, target_node
        )
        relation = py2neo.Relationship(
            src_node, relationship_type.name, target_node, **kwargs
        )
        self.graph_db.create(relation)
        return relation

    def _get_node_id(self, node):
        return py2neo.remote(node)._id

    def _allocate_global_fun_name(self, name):
        ct = self.fun_counter
        self.fun_counter += 1
        return '{}_{}'.format(name, ct)

    def _get_cleaning_code_signature(self, fun, normalize_names=True):
        # remove comments
        cleaning_code = [l for l in fun.cleaning_code if not l.startswith("#")]
        cleaning_code_str = "\n".join(cleaning_code)
        # get rid of intermediate rewrite vars
        lowered = lower_lifted_rewrites(cleaning_code_str)
        # remove "no-ops"
        lowered = remove_table_noops(lowered)
        # remove imports... not critical to function "signature"
        lowered_lines = lowered.split("\n")
        lowered_lines = [
            l for l in lowered_lines if not l.startswith(("import", "from"))
        ]
        # normalize variable names
        lowered = "\n".join(lowered_lines)
        if normalize_names:
            lowered = rename_lowered_rewrites(lowered)
        # extract key tokens as signature
        key_tokens = collect_key_tokens(lowered)
        return frozenset(key_tokens)

    def add_extracted_function(self, fun, program_graph):
        assert isinstance(
            fun, DonatedFunction
        ), 'Can only add extracted functions'
        global_fun_name = self._allocate_global_fun_name(fun.name)
        lines_of_code = len(fun.source.split("\n"))
        extracted_node = self._get_or_create_node(
            NodeTypes.EXTRACTED_FUNCTION,
            name=global_fun_name,
            lines_of_code=lines_of_code,
        )
        _id = self._get_node_id(extracted_node)
        self.id_to_fun[_id] = fun

        # relationships to place in database
        relationship_tuples = get_extracted_function_relationships(
            fun, program_graph
        )
        for relationship_type, end_node in relationship_tuples:
            self._create_relationship(
                extracted_node, relationship_type, end_node
            )

        return extracted_node

    def populate(self, funs, program_graph=None):
        if not self._has_started_up:
            raise Exception('Must start database by running self.startup()')
        for f in funs:
            # some functions can be duplicated across programs
            # TODO: should we highlight these somehow? Seems they are useful
            # if multiple programs have them
            fun_signature = self._get_cleaning_code_signature(
                f,
                normalize_names=True,
            )
            # exact match
            already_collected = False
            for other_signature in self.function_acc:
                # exact match
                if fun_signature == other_signature:
                    already_collected = True
                    break
                # only difference is import statements
                diff = fun_signature.symmetric_difference(other_signature)
                is_import = [l.startswith(("import", "from")) for l in diff]
                if all(is_import):
                    already_collected = True
                    break

            if not already_collected:
                # we can add some more info if we have the complete program graph
                # though we can still do useful things without it
                self.add_extracted_function(f, program_graph=program_graph)
                self.function_acc.add(fun_signature)

    def _run_relationship_query(
        self,
        start_node,
        rel_type,
        end_node,
        _lambda=None,
        query_for_ranking=None,
        **rank_kwargs
    ):
        if not self._has_started_up:
            raise Exception('Must start database by running self.startup()')
        if _lambda is None:
            _lambda = lambda x: x
        results = []
        for rel in self.graph_db.match(start_node=start_node,
                                       rel_type=rel_type.name,
                                       end_node=end_node):
            results.append(_lambda(rel))
        if query_for_ranking is not None:
            self.rank_query_results(results, query_for_ranking, **rank_kwargs)
        else:
            # sort by shortest
            results = sorted(results, key=lambda x: x["lines_of_code"])

        return results

    def rank_query_results(
        self,
        results,
        query,
        alpha=0.5,
        n_clusters=5,
        per_cluster=3,
        random_state=None,
    ):
        """
        Rank query results using:
            - a combination (based on alpha) of lines of code and
            token overlap with the query.
            We do some "special" preprocessing to get only relevant tokens
            We remove any result that has zero overlap with the query
        If there are many results (i.e. more than n_clusters * per_cluster)
        results, then we cluster results based on:
            - spectral clustering of the correlation matrix computed
            from the code fragments tf-idf encoding
        We then take the top K results per cluster, and sort them based
        on the score (explained above), and return that set
        """
        assert 0 <= alpha <= 1.0
        assert n_clusters > 1
        assert per_cluster >= 1

        scored_results = []

        # normalizing constants
        min_loc = np.inf
        max_overlap = -np.inf

        for node in results:
            # lines of code
            src = self.get_code(node)
            loc = len(src.split("\n"))
            func = self.get_function_from_node(node)
            # token overlap (do not normalize names since otherwise
            # lose meaningful tokens and replaces with ___id_{#}___ )
            sig = self._get_cleaning_code_signature(
                func, normalize_names=False
            )
            token_overlap = len(sig.intersection(query))
            if token_overlap > 0:
                # we ignore results that have zero token overlap
                min_loc = min(min_loc, loc)
                max_overlap = max(max_overlap, token_overlap)
                scored_results.append((node, (loc, token_overlap)))

        if len(scored_results) <= 1:
            return [n for n, _ in scored_results]

        # normalize and combine loc/token scores using alpha
        scored_results = [
            (n, (min_loc / l) * alpha + (t / max_overlap) * (1 - alpha))
            for n, (l, t) in scored_results
        ]
        scored_results = sorted(
            scored_results, key=lambda x: x[-1], reverse=True
        )

        if len(scored_results) <= (n_clusters * per_cluster):
            return [n for n, _, in scored_results]
        # take code as string and count
        code = [self.get_code(n) for n, _ in scored_results]
        # encode by accounting for token frequency
        encoded_code = TfidfVectorizer().fit_transform(code)
        # correlation matrix
        corr_mat = np.corrcoef(encoded_code.todense())
        # clamped at zero (need to take sqrt in spectral clustering)
        corr_mat[corr_mat < 0.0] = 0.0
        # spectral clusterign
        clusters = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=random_state,
        ).fit_predict(corr_mat)
        #  take first K based on score per cluster
        cluster_cts = Counter()
        pruned_results = []
        for c, (elem, _) in zip(clusters, scored_results):
            if cluster_cts[c] < per_cluster:
                pruned_results.append(elem)
                cluster_cts[c] += 1

        return pruned_results

    def _get_selector(self, node_type):
        if not self._has_started_up:
            raise Exception('Must start database by running self.startup()')
        return self.selectors[node_type]

    # Querying API
    def defines(
        self,
        column_name,
        start_node=None,
        _lambda=None,
        rank=False,
        alpha=0.5,
        n_clusters=5,
        per_cluster=3
    ):
        if _lambda is None:
            _lambda = lambda x: x.start_node()
        column_name = get_column_name(column_name)
        end_node = self._get_selector(NodeTypes.COLUMN
                                      ).where(name=column_name).first()
        if end_node is None and column_name is not None:
            return []
        return self._run_relationship_query(
            start_node,
            RelationshipTypes.DEFINES,
            end_node,
            _lambda,
            query_for_ranking=[column_name] if rank else None,
            alpha=alpha,
            n_clusters=n_clusters,
            per_cluster=per_cluster,
        )

    def uses(
        self,
        column_name,
        start_node=None,
        _lambda=None,
        rank=False,
        alpha=0.5,
        n_clusters=5,
        per_cluster=3
    ):
        if _lambda is None:
            _lambda = lambda x: x.start_node()
        column_name = get_column_name(column_name)
        end_node = self._get_selector(NodeTypes.COLUMN
                                      ).where(name=column_name).first()
        if end_node is None and column_name is not None:
            return []
        return self._run_relationship_query(
            start_node,
            RelationshipTypes.USES,
            end_node,
            _lambda,
            query_for_ranking=[column_name] if rank else None,
            alpha=alpha,
            n_clusters=n_clusters,
            per_cluster=per_cluster,
        )

    def calls(
        self,
        fun,
        start_node=None,
        _lambda=None,
        rank=False,
        alpha=0.5,
        n_clusters=5,
        per_cluster=3
    ):
        if _lambda is None:
            _lambda = lambda x: x.start_node()
        full_name = get_function_full_name(fun) if fun is not None else None
        end_node = self._get_selector(NodeTypes.FUNCTION
                                      ).where(name=full_name).first()
        if end_node is None and fun is not None:
            return []
        return self._run_relationship_query(
            start_node,
            RelationshipTypes.CALLS,
            end_node,
            _lambda,
            query_for_ranking=full_name.split(".") if rank else None,
            alpha=alpha,
            n_clusters=n_clusters,
            per_cluster=per_cluster,
        )

    def wrangles_for(
        self,
        fun,
        start_node=None,
        _lambda=None,
        rank=False,
        alpha=0.5,
        n_clusters=5,
        per_cluster=3
    ):
        if _lambda is None:
            _lambda = lambda x: x.start_node()
        full_name = get_function_full_name(fun) if fun is not None else None
        end_node = self._get_selector(NodeTypes.FUNCTION
                                      ).where(name=full_name).first()
        if end_node is None and fun is not None:
            return []
        return self._run_relationship_query(
            start_node,
            RelationshipTypes.WRANGLES_FOR,
            end_node,
            _lambda,
            query_for_ranking=full_name.split(".") if rank else None,
            alpha=alpha,
            n_clusters=n_clusters,
            per_cluster=per_cluster,
        )

    def query(
        self,
        terms,
        alpha=0.5,
        n_clusters=5,
        per_cluster=3,
        random_state=None,
    ):
        # defines/calls
        # uses/wrangles_for
        results = []
        processed_terms = []
        for t in terms:
            if isinstance(t, str):
                results.extend(self.defines(t, rank=False))
                results.extend(self.uses(t, rank=False))
                processed_terms.append(t)
            else:
                # assume it is a function
                results.extend(self.calls(t, rank=False))
                results.extend(self.wrangles_for(t, rank=False))
                full_func_name = get_function_full_name(t)
                # add in function name tokenized by path
                processed_terms.extend(full_func_name.split("."))
        query = set(processed_terms)
        results = set(results)
        return self.rank_query_results(
            results,
            query,
            alpha=alpha,
            n_clusters=n_clusters,
            per_cluster=per_cluster,
            random_state=random_state,
        )

    def query_by_relationships(self, relationship_tuples):
        if isinstance(relationship_tuples, tuple):
            relationship_tuples = [relationship_tuples]
        criteria = set(relationship_tuples)
        query_results = set([])
        query_handler = {
            RelationshipTypes.CALLS: self.calls,
            RelationshipTypes.DEFINES: self.defines,
            RelationshipTypes.USES: self.uses,
            RelationshipTypes.WRANGLES_FOR: self.wrangles_for,
        }

        while criteria:
            relationship_type, end_node_value = criteria.pop()
            query_fun = query_handler.get(relationship_type, None)
            if query_fun is None:
                raise ValueError(
                    'Invalid relation type: {}'.format(relationship_type)
                )
            query_results.update(query_fun(end_node_value))

        return query_results

    def get_extracted_function_relationships_from_node(self, node):
        assert node.has_label(NodeTypes.EXTRACTED_FUNCTION.name)
        rels = set([])
        results = self.graph_db.match(
            start_node=node, rel_type=None, end_node=None
        )
        for res in results:
            rel_type = RelationshipTypes[res.type()]
            end_name = res.end_node()['name']
            rels.add((rel_type, end_name))
        return rels

    def get_function_from_node(self, node):
        if not node.has_label(NodeTypes.EXTRACTED_FUNCTION.name):
            raise ValueError('Can only retrieve code for extracted functions')
        _id = self._get_node_id(node)
        return self.id_to_fun[_id]

    def get_code(self, node):
        if not node.has_label(NodeTypes.EXTRACTED_FUNCTION.name):
            raise ValueError('Can only retrieve code for extracted functions')
        src = self.get_function_from_node(node).source
        return lower_lifted_rewrites(src)

    def get_executable(self, node):
        if not node.has_label(NodeTypes.EXTRACTED_FUNCTION.name):
            raise ValueError('Can only retrieve code for extracted functions')
        return self.get_function_from_node(node).obj

    def functions(self):
        return list(self._get_selector(NodeTypes.FUNCTION))

    def extracted_functions(self):
        return list(self._get_selector(NodeTypes.EXTRACTED_FUNCTION))

    def columns(self):
        return list(self._get_selector(NodeTypes.COLUMN))

    # TODO: way to combine query results
    # and then return (rather than have to figure out)
    # basically should be able to compose these queries


def main(args):
    functions_files = args.function_files
    graph_files = args.graph_files

    db = FunctionDatabase()
    db.startup()

    # make sure we align files
    name_to_functions = {
        f.replace("_functions.pkl", ""): f
        for f in functions_files if f.endswith("functions.pkl")
    }

    name_to_graphs = {
        f.replace("_graph.pkl", ""): f
        for f in graph_files if f.endswith("graph.pkl")
    }

    for kernel_name, funs_file in name_to_functions.items():
        if kernel_name not in name_to_graphs:
            continue
        graph_file = name_to_graphs[kernel_name]
        print('Populating database with functions from {}'.format(funs_file))
        print('Using graph file:{}'.format(graph_file))
        # functions we extracted
        with open(funs_file, 'rb') as f:
            funs = pickle.load(f)
        # corresponding program dynamic dep graph
        with open(graph_file, 'rb') as f:
            program_graph = pickle.load(f)

        db.populate(funs, program_graph)
    db.shutdown()

    with open(args.output_file, 'wb') as f:
        pickle.dump(db, f)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Populate graph database with donated functions'
    )
    parser.add_argument(
        '--function_files',
        type=str,
        nargs='+',
        help='Pickled donated function files'
    )
    parser.add_argument(
        '--graph_files',
        type=str,
        nargs='+',
        help='Pickled program dependency graph files'
    )
    parser.add_argument(
        '-o',
        '--output_file',
        type=str,
        help='Path to store database',
        default='boruca.pkl'
    )
    args = parser.parse_args()

    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
