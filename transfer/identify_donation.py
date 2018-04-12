from argparse import ArgumentParser
import ast
import networkx as nx
import pickle

from plpy.analyze.graph_builder import draw
from plpy.analyze.dynamic_tracer import to_ast_node
from plpy.analyze.dynamic_trace_events import ExecLine, MemoryUpdate

class DonationSlices(object):
    def __init__(self, graph, seed_columns, slices):
        self.graph = graph
        self.seed_columns = seed_columns
        self.slices = slices

    def __iter__(self):
        return iter(self.slices)

# will produce false positives for attributes
class PossibleColumnCollector(ast.NodeVisitor):
    def __init__(self):
        self.acc = []

    def visit_Str(self, node):
        self.acc.append(node.s)

    def visit_Attribute(self, node):
        self.acc.append(node.attr)

    def run(self, tree):
        self.visit(tree)
        return set(self.acc)

# TODO: would it be better to compute the columns assigned/used based on the memory locations?
def columns_assigned_to(src_line):
    try:
        ast_node = to_ast_node(src_line)
    except SyntaxError:
        # not standalone assignment/expressions
        # FIXME: this is ignoring the fact that we can parse portions of this
        return set([])

    if not isinstance(ast_node, (ast.Assign, ast.AugAssign)):
        return set([])

    lhs = []
    if isinstance(ast_node, ast.Assign):
        lhs.extend(ast_node.targets)
    elif isinstance(ast_node, ast.AugAssign):
        lhs.append(ast_node.target)
    else:
        lhs.extend(ast_node)

    cols_assigned = set([])
    for node in lhs:
        cols_assigned.update(PossibleColumnCollector().run(node))

    return cols_assigned

def columns_used(src_line):
    try:
        ast_node = to_ast_node(src_line)
    except SyntaxError:
        return set([])

    # FIXME: we should consider the lhs of augmente assigne as used as well
    if isinstance(ast_node, (ast.Assign, ast.AugAssign)):
        ast_node = ast_node.value
    return PossibleColumnCollector().run(ast_node)

def remove_subgraphs(graphs):
    # graphs from least to most number of nodes
    sorted_graphs = sorted(graphs, key=lambda x: len(x.nodes))
    # take set of nodes as key for each graph
    keys = [frozenset(g.nodes) for g in sorted_graphs]
    clean = []
    for i, graph in enumerate(sorted_graphs):
        larger_keys = keys[(i + 1):]
        key = keys[i]
        # ignore any graph that is a subset of a larger graph
        # later on in our sorted list of graphs
        if any(key.issubset(k) for k in larger_keys):
            continue
        clean.append(graph)
    return clean

def show_lines(graph, annotate=True):
    if annotate:
        nodes = [(node_id, dict(data)) for node_id, data in graph.nodes(data=True)]
        for node_id, data in nodes:
            if isinstance(data['event'], ExecLine):
                data['uses'] = data['event'].uses_mem_locs
    else:
        nodes = graph.nodes(data=True)
    return sorted(nodes, key=lambda x: x[1]['lineno'])

def to_code_block(nodes):
    if isinstance(nodes, nx.Graph):
        nodes = show_lines(nodes)
    return '\n'.join(map(lambda x: x[1]['src'], nodes))


class ColumnAssignmentExtractor(object):
    """
    Extracts slices by:
        - take as seeds any stmt that assigns to a target column
        - slice backward from these seeds
    """
    def __init__(self, graph):
        self.graph = graph

    def get_donation_slices(self, target_columns):
        seeds = []
        if isinstance(target_columns, str):
            target_columns = [target_columns]
        target_columns = set(target_columns)

        for node_id, node_data in self.graph.nodes.items():
            # contains target_columns
            assigns_to = columns_assigned_to(node_data['src'])
            if any(col in assigns_to for col in target_columns):
                seeds.append(node_id)

        reversed_graph = self.graph.reverse(copy=False)
        # frozenset of nodes as key to avoid extracting slices
        # that are strict subset of existing slice
        slices = []

        for node_id in seeds:
            slice_nodes = nx.dfs_tree(reversed_graph, node_id)
            _slice = self.graph.subgraph(slice_nodes)
            slices.append(_slice)

        # keep only the largest slice when there are subslices
        slices = remove_subgraphs(slices)
        return DonationSlices(self.graph, target_columns, slices)


class ColumnUseExtractor(object):
    """
    Extracts slices by:
        - take as seeds any stmt that uses a target column
        - slice forward on each seed
        - identify columns that are assigned to in these slices
        - uses these columns as seeds to slice backward using the column assignment extractor
    """
    def __init__(self, graph):
        self.graph = graph

    def get_donation_slices(self, target_columns):
        seeds = []
        if isinstance(target_columns, str):
            target_columns = [target_columns]
        target_columns = set(target_columns)

        for node_id, node_data in self.graph.nodes.items():
            # contains target_columns
            assigns_to = columns_used(node_data['src'])
            if any(col in assigns_to for col in target_columns):
                seeds.append(node_id)

        # slice forward on the seeds extract columns assigned to
        cols = set([])
        for node_id in seeds:
            # TODO: these slices could presumably grow to the end
            # of each script
            # we should likely consider: topological sort and then
            # remove nodes that are "too far away"
            slice_nodes = nx.dfs_tree(self.graph, node_id)
            for child_id in slice_nodes:
                child_src = self.graph.nodes[child_id]['src']
                cols.update(columns_assigned_to(child_src))

        helper = ColumnAssignmentExtractor(self.graph)
        return helper.get_donation_slices(cols)



def main(args):
    graph_file = args.graph_file
    slices_file = args.output_file
    target_columns = args.targets

    with open(graph_file, 'rb') as f:
        graph = pickle.load(f)
    extractor = ColumnAssignmentExtractor(graph)
    donations = extractor.get_donation_slices(target_columns)

    with open(slices_file, 'wb') as f:
        pickle.dump(donations, f)


if __name__ == '__main__':
    parser = ArgumentParser(description='Extract candidate donation slices')
    parser.add_argument('graph_file', type=str, help='File path to pickled networkx graph')
    parser.add_argument('output_file', type=str, help='File path to pickle output slices')
    args = parser.parse_args()
    try:
        main(args)
    except:
        import pdb
        pdb.post_mortem()






