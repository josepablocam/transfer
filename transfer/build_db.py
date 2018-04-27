# Construct graph database for extracted functions and their relationships
# to columns and third party libraries

from argparse import ArgumentParser
from enum import Enum
import pickle

import py2neo

from .lift_donations import DonatedFunction

class NodeTypes(Enum):
    EXTRACTED_FUNCTION = 1
    FUNCTION = 2
    COLUMN = 3

class RelationshipTypes(Enum):
    DEFINES = 1
    USES = 2
    CALLS = 3

class FunctionDatabase(object):
    def __init__(self):
        self.id_to_fun = {}
        self.column_to_id = {}
        self.qualname_to_id = {}
        self.graph_db = None

    def startup(self):
        self.graph_db = py2neo.Graph()

    def shutdown(self):
        self.graph_db = None

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

    def add_extracted_function(self, fun):
        assert isinstance(fun, DonatedFunction), 'Can only add extracted functions'
        extracted_node = self._create_node(NodeTypes.EXTRACTED_FUNCTION, name=fun.name)
        _id = self._get_node_id(extracted_node)
        self.id_to_fun[_id] = fun

        columns_defined = set([])
        columns_used = set([])
        functions_called = set([])

        for _, node_data in fun.graph.nodes(data=True):
            columns_defined.update(node_data['columns_defined'])
            columns_used.update(node_data['columns_used'])
            if node_data['calls'] is not None:
                for call in node_data['calls']:
                    functions_called.add(call.details['qualname'])

        for col in columns_defined:
            col_node = self.add_column(col)
            self._create_relationship(extracted_node, RelationshipTypes.DEFINES, col_node)

        for col in columns_used:
            col_node = self.add_column(col)
            self._create_relationship(extracted_node, RelationshipTypes.USES, col_node)

        for fun in functions_called:
            fun_node = self.add_function(fun)
            self._create_relationship(extracted_node, RelationshipTypes.CALLS, fun_node)

        return extracted_node

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

    def populate(self, funs):
        for f in funs:
            self.add_extracted_function(f)

    def calls():
        # write query here
        raise NotImplementedError

    def defines():
        # write query here
        raise NotImplementedError

    def uses():
        # write query
        raise NotImplementedError

    def map(self, f):
        # write query
        raise NotImplementedError

    def filter(self, f):
        # write query
        raise NotImplementedError


def main(arg):
    db = FunctionDatabase()
    db.startup()
    for _file in arg.input_files:
        with open(_file, 'rb') as f:
            funs = pickle.load(f)
        db.populate(funs)
    db.shutdown()

    with open(arg.output_file, 'wb') as f:
        pickle.dump(db, f)


if __name__ == '__main__':
    parser = ArgumentParser(description='Populate graph database with donated functions')
    parser.add_argument('input_files', nargs='+', type=str, help='Path to pickled donated function files')
    parser.add_argument('-o', '--output_file', type=str, help='Path to store database', default='pythia.pkl')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
