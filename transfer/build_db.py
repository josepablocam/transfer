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
            raise ValueError('Unable to compute full_name from {}, provide string of name'.format(elem))
    full_name = []
    full_name.append('' if modulename is None else modulename + '.')
    full_name.append(co_name if qualname is None else qualname)
    full_name =  ''.join(full_name)
    return None if full_name == '' else full_name

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
                    relationship_tuples.add((RelationshipTypes.CALLS, full_name))

    if not program_graph is None:
        for full_name in get_wrangles_for(fun, program_graph):
            relationship_tuples.add((RelationshipTypes.WRANGLES_FOR, full_name))

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

    def startup(self):
        self.graph_db = py2neo.Graph()
        self.selectors = {}
        # setup selectors
        for node_type in NodeTypes:
            self.selectors[node_type] = py2neo.NodeSelector(self.graph_db).select(node_type.name)
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
            if relationship_type in set([RelationshipTypes.DEFINES, RelationshipTypes.USES]):
                target_node_type = NodeTypes.COLUMN
            elif relationship_type in set([RelationshipTypes.CALLS, RelationshipTypes.WRANGLES_FOR]):
                target_node_type = NodeTypes.FUNCTION
            else:
                raise Exception("Need to know target_node_type for relationship type: {}".format(relationship_type))
            return self._get_or_create_node(target_node_type, name=target_node)
        else:
            raise Exception("Don't know how to convert type {} to py2neo.Node".format(type(target_node)))

    def _create_relationship(self, src_node, relationship_type, target_node, **kwargs):
        assert isinstance(src_node, py2neo.Node)
        # must create target_node if doesn't exist
        target_node = self._convert_to_target_node(relationship_type, target_node)
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
        extracted_node = self._get_or_create_node(NodeTypes.EXTRACTED_FUNCTION, name=global_fun_name)
        _id = self._get_node_id(extracted_node)
        self.id_to_fun[_id] = fun

        # relationships to place in database
        relationship_tuples = get_extracted_function_relationships(fun, program_graph)
        for relationship_type, end_node in relationship_tuples:
            self._create_relationship(extracted_node, relationship_type, end_node)

        return extracted_node

    def populate(self, funs, program_graph=None):
        if not self._has_started_up:
            raise Exception('Must start database by running self.startup()')
        for f in funs:
            # we can add some more info if we have the complete program graph
            # though we can still do useful things without it
            self.add_extracted_function(f, program_graph=program_graph)

    def _run_relationship_query(self, start_node, rel_type, end_node, _lambda=None):
        if not self._has_started_up:
            raise Exception('Must start database by running self.startup()')
        if _lambda is None:
            _lambda = lambda x: x
        results = []
        for rel in self.graph_db.match(start_node=start_node, rel_type=rel_type.name, end_node=end_node):
            results.append(_lambda(rel))
        return results

    def _get_selector(self, node_type):
        if not self._has_started_up:
            raise Exception('Must start database by running self.startup()')
        return self.selectors[node_type]

    # Querying API
    def defines(self, column_name, start_node=None, _lambda=None):
        if _lambda is None:
            _lambda = lambda x: x.start_node()
        end_node = self._get_selector(NodeTypes.COLUMN).where(name=column_name).first()
        if end_node is None and column_name is not None:
            return []
        return self._run_relationship_query(start_node, RelationshipTypes.DEFINES, end_node, _lambda)

    def uses(self, column_name, start_node=None, _lambda=None):
        if _lambda is None:
            _lambda = lambda x: x.start_node()
        end_node = self._get_selector(NodeTypes.COLUMN).where(name=column_name).first()
        if end_node is None and column_name is not None:
            return []
        return self._run_relationship_query(start_node, RelationshipTypes.USES, end_node, _lambda)

    def calls(self, fun, start_node=None, _lambda=None):
        if _lambda is None:
            _lambda = lambda x: x.start_node()
        full_name = get_function_full_name(fun) if fun is not None else None
        end_node = self._get_selector(NodeTypes.FUNCTION).where(name=full_name).first()
        if end_node is None and fun is not None:
            return []
        return self._run_relationship_query(start_node, RelationshipTypes.CALLS, end_node, _lambda)

    def wrangles_for(self, fun, start_node=None, _lambda=None):
        if _lambda is None:
            _lambda = lambda x: x.start_node()
        full_name = get_function_full_name(fun) if fun is not None else None
        end_node = self._get_selector(NodeTypes.FUNCTION).where(name=full_name).first()
        if end_node is None and fun is not None:
            return []
        return self._run_relationship_query(start_node, RelationshipTypes.WRANGLES_FOR, end_node, _lambda)

    def query_by_relationships(self, relationship_tuples):
        if isinstance(relationship_tuples, tuple):
            relationship_tuples = [relationship_tuples]
        criteria = set(relationship_tuples)
        query_results = set([])
        query_handler = {
            RelationshipTypes.CALLS:        self.calls,
            RelationshipTypes.DEFINES:      self.defines,
            RelationshipTypes.USES:         self.uses,
            RelationshipTypes.WRANGLES_FOR: self.wrangles_for,
        }

        while criteria:
            relationship_type, end_node_value = criteria.pop()
            query_fun = query_handler.get(relationship_type, None)
            if query_fun is None:
                raise ValueError('Invalid relation type: {}'.format(relationship_type))
            query_results.update(query_fun(end_node_value))

        return query_results

    def get_function_from_node(self, node):
        if not node.has_label(NodeTypes.EXTRACTED_FUNCTION.name):
            raise ValueError('Can only retrieve code for extracted functions')
        _id = self._get_node_id(node)
        return self.id_to_fun[_id]

    def functions(self):
        return list(self._get_selector(NodeTypes.FUNCTION))

    def extracted_functions(self):
        return list(self._get_selector(NodeTypes.EXTRACTED_FUNCTION))

    def columns(self):
        return list(self._get_selector(NodeTypes.COLUMN))


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
