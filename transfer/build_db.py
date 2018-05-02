# Construct graph database for extracted functions and their relationships
# to columns and third party libraries

from argparse import ArgumentParser
from enum import Enum
import pickle
import inspect

import networkx as nx
import py2neo
from plpy.analyze.dynamic_trace_events import ExecLine

from .lift_donations import DonatedFunction

class NodeTypes(Enum):
    EXTRACTED_FUNCTION = 1
    FUNCTION = 2
    COLUMN = 3

class RelationshipTypes(Enum):
    DEFINES = 1
    USES = 2
    CALLS = 3
    WRANGLES_FOR = 4

class FunctionDatabase(object):
    def __init__(self):
        self.id_to_fun = {}
        self.column_to_id = {}
        self.qualname_to_id = {}
        self.graph_db = None
        self.selectors = None
        # maintain a global count of functions extracted and added
        self.fun_counter = 0

    def startup(self):
        self.graph_db = py2neo.Graph()
        self.selectors = {}
        # setup selectors
        for node_type in NodeTypes:
            self.selectors[node_type] = py2neo.NodeSelector(self.graph_db).select(node_type.name)

    def shutdown(self):
        self.graph_db = None
        self.selectors = {}

    def _create_node(self, node_type, **kwargs):
        node = py2neo.Node(node_type.name, **kwargs)
        self.graph_db.create(node)
        return node

    def _create_relationship(self, src_node, relationship_type, target_node, **kwargs):
        relation = py2neo.Relationship(src_node, relationship_type.name, target_node, **kwargs)
        self.graph_db.create(relation)
        return relation

    def _get_node_id(self, node):
        return py2neo.remote(node)._id

    def _allocate_global_fun_name(self, name):
        ct = self.fun_counter
        self.fun_counter += 1
        return '{}_{}'.format(name, ct)

    def add_extracted_function(self, fun, program_graph):
        assert isinstance(fun, DonatedFunction), 'Can only add extracted functions'
        global_fun_name = self._allocate_global_fun_name(fun.name)
        extracted_node = self._create_node(NodeTypes.EXTRACTED_FUNCTION, name=global_fun_name)
        _id = self._get_node_id(extracted_node)
        self.id_to_fun[_id] = fun

        columns_defined = set([])
        columns_used = set([])
        functions_called = set([])
        wrangles_for_calls = set([])

        for _, node_data in fun.graph.nodes(data=True):
            columns_defined.update(node_data['columns_defined'])
            columns_used.update(node_data['columns_used'])
            if node_data['calls'] is not None:
                for call in node_data['calls']:
                    functions_called.add(call.details['qualname'])

        wrangles_for_calls.update(self._get_wrangles_for(fun, program_graph))

        for col in columns_defined:
            col_node = self.add_column(col)
            self._create_relationship(extracted_node, RelationshipTypes.DEFINES, col_node)

        for col in columns_used:
            col_node = self.add_column(col)
            self._create_relationship(extracted_node, RelationshipTypes.USES, col_node)

        for fun in functions_called:
            fun_node = self.add_function(fun)
            self._create_relationship(extracted_node, RelationshipTypes.CALLS, fun_node)

        for fun in wrangles_for_calls:
            fun_node = self.add_function(fun)
            self._create_relationship(extracted_node, RelationshipTypes.WRANGLES_FOR, fun_node)

        return extracted_node

    def _get_wrangles_for(self, fun, program_graph):
        # use node id for the value associated with the return
        _return_node_id = max(fun.graph.nodes)
        fwd_nodes = nx.dfs_tree(program_graph, _return_node_id)
        fwd_slice = program_graph.subgraph(fwd_nodes)
        qualnames = set([])
        for _, node_data in fwd_slice.nodes(data=True):
            if isinstance(node_data['event'], ExecLine):
                if node_data['calls'] is None:
                    continue
                for call in node_data['calls']:
                    qualname = call.details['qualname']
                    if qualname is not None:
                        qualnames.add(qualname)
        return qualnames

    def add_column(self, _str):
        node_id = self.column_to_id.get(_str, None)
        assert len(_str) > 1
        if node_id is None:
            node =  self._create_node(NodeTypes.COLUMN, name=_str)
            node_id = self._get_node_id(node)
            self.column_to_id[_str] = node_id
            return node
        else:
            return self.graph_db.node(node_id)

    def add_function(self, qualname):
        node_id = self.qualname_to_id.get(qualname, None)
        if node_id is None:
            node = self._create_node(NodeTypes.FUNCTION, name=qualname)
            node_id = self._get_node_id(node)
            self.qualname_to_id[qualname] = node_id
            return node
        else:
            return self.graph_db.node(node_id)

    def populate(self, funs, program_graph=None):
        for f in funs:
            # we can add some more info if we have the complete program graph
            # though we can still do useful things without it
            self.add_extracted_function(f, program_graph=program_graph)

    @staticmethod
    def _get_qualname(fun):
        if isinstance(fun, str):
            return fun
        else:
            return fun.__qualname__

    def _get_relationships(self, start_node, rel_type, end_node, _lambda=None):
        if _lambda is None:
            _lambda = lambda x: x
        results = []
        for rel in self.graph_db.match(start_node=start_node, rel_type=rel_type.name, end_node=end_node):
            results.append(_lambda(rel))
        return results

    # Querying API
    def calls(self, fun, start_node=None, _lambda=None):
        if _lambda is None:
            _lambda = lambda x: x.start_node()
        qualname = self._get_qualname(fun) if fun is not None else None
        end_node = self.selectors[NodeTypes.FUNCTION].where(name=qualname).first()
        if end_node is None and fun is not None:
            return []
        return self._get_relationships(start_node, RelationshipTypes.CALLS, end_node, _lambda)

    def defines(self, column_name, start_node=None, _lambda=None):
        if _lambda is None:
            _lambda = lambda x: x.start_node()
        end_node = self.selectors[NodeTypes.COLUMN].where(name=column_name).first()
        if end_node is None and column_name is not None:
            return []
        return self._get_relationships(start_node, RelationshipTypes.DEFINES, end_node, _lambda)

    def uses(self, column_name, start_node=None, _lambda=None):
        if _lambda is None:
            _lambda = lambda x: x.start_node()
        end_node = self.selectors[NodeTypes.COLUMN].where(name=column_name).first()
        if end_node is None and column_name is not None:
            return []
        return self._get_relationships(start_node, RelationshipTypes.USES, end_node, _lambda)

    def wrangles_for(self, fun, start_node=None, _lambda=None):
        if _lambda is None:
            _lambda = lambda x: x.start_node()
        qualname = self._get_qualname(fun) if fun is not None else None
        end_node = self.selectors[NodeTypes.FUNCTION].where(name=qualname).first()
        if end_node is None and fun is not None:
            return []
        return self._get_relationships(start_node, RelationshipTypes.WRANGLES_FOR, end_node, _lambda)

    def get_function_from_node(self, node):
        if not node.has_label(NodeTypes.EXTRACTED_FUNCTION.name):
            raise ValueError('Can only retrieve code for extracted functions')
        _id = self._get_node_id(node)
        return self.id_to_fun[_id]

    def functions(self):
        return list(self.selectors[NodeTypes.FUNCTION])

    def extracted_functions(self):
        return list(self.selectors[NodeTypes.EXTRACTED_FUNCTION])

    def columns(self):
        return list(self.selectors[NodeTypes.COLUMN])


def main(args):
    functions_files = args.functions_files_list.split(',')
    graph_files = args.graph_files_list.split(',')

    if len(functions_files) != len(graph_files):
        raise Exception('Must provide same number of input and graph files')

    db = FunctionDatabase()
    db.startup()

    for funs_file, graph_file in zip(functions_files, graph_files):
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
    parser = ArgumentParser(description='Populate graph database with donated functions')
    parser.add_argument('functions_files_list', type=str, help='CSV list of paths to pickled donated function files')
    parser.add_argument('graph_files_list', type=str, help='CSV list of paths to pickled program dependency graph files')
    parser.add_argument('-o', '--output_file', type=str, help='Path to store database', default='boruca.pkl')
    args = parser.parse_args()

    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()