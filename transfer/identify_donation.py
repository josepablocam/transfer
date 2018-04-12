from argparse import ArgumentParser
import ast
from collections import defaultdict
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

    def __getitem__(self, ix):
        return self.slices[ix]

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
def columns_assigned_to_line(src_line):
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

def columns_used_line(src_line):
    try:
        ast_node = to_ast_node(src_line)
    except SyntaxError:
        return set([])

    # FIXME: we should consider the lhs of augmente assigne as used as well
    if isinstance(ast_node, (ast.Assign, ast.AugAssign)):
        ast_node = ast_node.value
    return PossibleColumnCollector().run(ast_node)

def annotate_graph(graph):
    for node_id, data in graph.nodes(data=True):
        if data.get('annotator') == __file__:
            return graph
        defs = columns_assigned_to_line(data['src'])
        used = columns_used_line(data['src'])
        data['columns_defined'] = defs
        data['columns_used'] = used
        data['annotator'] = __file__
    return graph

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

def merge_common_prefixes(graphs):
    # TODO merge graphs that share a common prefix
    pass

def remove_noop_assignments(graph):
    pass

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


class RepairSliceImplicitUses(object):
    """
    Code can make implicit use of a column. By implicit we mean that its 'memory address' (as indicated by id) is not
    directly accessed by the statement in user code. Instead, it is accessed by library code that we do not trace
    (on purpose, since that is not particularly useful). To repair a slice, we can then:
        - find uses of columns by scanning for strings/attribute accesses
        - look at statements earlier in the program (i.e. earlier nodes in our graph) and identify assignments
        - if these two sets intersect, we need to extract relevant assignments (and their slices)
        - this process is repeatedly recursively to repair any new slices included
    """
    def __init__(self, graph):
        self.graph = annotate_graph(graph)

    def run(self, _slices):
        if isinstance(_slices, nx.Graph):
            return self.repair_slice(_slice)
        else:
            repaired = []
            for _slice in _slices:
                repaired.append(self.repair_slice(_slice))
            return repaired

    def repair_slice(self, _slice):
        # note that we explicitly make this while loop to avoid recursion errors (stack depth bound by Python)
        curr_slice = _slice
        final_nodes = set(_slice.nodes)
        while True:
            # columns used in the slices
            cols_used = set([col for _, data in curr_slice.nodes(data=True) for col in data['columns_used']])
            if not cols_used:
                break

            # we need to inspect any node before our current node (i.e. stmts executed before)
            max_slice_node_id = max(curr_slice.nodes)
            earlier_nodes = [node_id for node_id in self.graph if node_id < max_slice_node_id]
            earlier_graph = self.graph.subgraph(earlier_nodes)
            if not earlier_graph.nodes:
                break

            # columns assigned during earlier stmts that are not part of our slice
            earlier_defs = set([])
            for node_id, data in earlier_graph.nodes(data=True):
                if not node_id in curr_slice.nodes:
                    earlier_defs.update(data['columns_defined'])
            if cols_used.isdisjoint(earlier_defs):
                break

            defs_needed = cols_used.intersection(earlier_defs)
            repair_slices = ColumnDefExtractor(earlier_graph)._get_raw_donation_slices(defs_needed).slices
            if not repair_slices:
                break

            repair_nodes = set([])
            for _s in repair_slices:
                repair_nodes.update(_s.nodes)
            # add these repair nodes to the final set of nodes
            final_nodes.update(repair_nodes)
            # set this repair subgraph as the current slice
            # to solve any implicit uses here
            curr_slice = self.graph.subgraph(repair_nodes)

        return self.graph.subgraph(final_nodes)



class ColumnDefExtractor(object):
    """
    Extracts slices by:
        - take as seeds any stmt that assigns to a target column
        - slice backward from these seeds
    """
    def __init__(self, graph, col_to_node_ids=None):
        self.graph = annotate_graph(graph)

        if col_to_node_ids is None:
            col_to_node_ids = defaultdict(lambda: [])
            for node_id, data in self.graph.nodes(data=True):
                for col in data['columns_defined']:
                    col_to_node_ids[col].append(node_id)
        self.col_to_node_ids = col_to_node_ids

    def _get_raw_donation_slices(self, target_columns):
        if isinstance(target_columns, str):
            target_columns = [target_columns]
        target_columns = set(target_columns)

        seeds = []
        for col in target_columns:
            seeds.extend(self.col_to_node_ids[col])

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

    def run(self, target_columns):
        _slices = self._get_raw_donation_slices(target_columns)
        return DonationSlices(self.graph, target_columns, _slices)



class ColumnUseExtractor(object):
    """
    Extracts slices by:
        - take as seeds any stmt that uses a target column
        - slice forward on each seed
        - identify columns that are assigned to in these slices
        - uses these columns as seeds to slice backward using the column assignment extractor
    """
    def __init__(self, graph, col_to_node_ids=None):
        self.graph = annotate_graph(graph)

        if col_to_node_ids is None:
            col_to_node_ids = defaultdict(lambda: [])
            for node_id, data in self.graph.nodes(data=True):
                for col in data['columns_used']:
                    col_to_node_ids[col].append(node_id)
        self.col_to_node_ids = col_to_node_ids

    def _get_raw_donation_slices(self, target_columns):
        if isinstance(target_columns, str):
            target_columns = [target_columns]
        target_columns = set(target_columns)

        seeds = []
        for col in target_columns:
            seeds.extend(self.col_to_node_ids[col])

        # slice forward on the seeds to find columns that are assigned to
        cols_assigned_to = set([])
        for node_id in seeds:
            # TODO: these slices could presumably grow to the end
            # of each script
            # we should likely consider: topological sort and then
            # remove nodes that are "too far away"
            slice_nodes = nx.dfs_tree(self.graph, node_id)
            for child_id in slice_nodes:
                child_defs = self.graph.nodes[child_id]['columns_defined']
                cols_assigned_to.update(child_defs)

        backward_helper = ColumnDefExtractor(self.graph)
        return backward_helper._get_raw_donation_slices(cols_assigned_to)

    def run(self, target_columns):
        _slices = self._get_raw_donation_slices(target_columns)
        _slices =  RepairSliceImplicitUses(self.graph).run(_slices)
        return DonationSlices(self.graph, target_columns, _slices)


# Main concerns (in order of priority):
#      - library code that mutates dataframe without info so no dependency established
        # * possible solution: given that we're focusing on pandas methods, we could
        #     look for the call argument 'inplace=True' and if so
        #     this updates memory location for everything involved here
        # This should be exposed through graph_builder
#       - need to decide what to do about control-flow
#      - common prefix traces
#      - no-op assignments that are indistinguishable given memory location info


def main(args):
    raise Exception('nyi')
    graph_file = args.graph_file
    slices_file = args.output_file
    target_columns = args.targets

    with open(graph_file, 'rb') as f:
        graph = pickle.load(f)
    # extractor = ColumnDefExtractor(graph)
    # donations = extractor.get_donation_slices(target_columns)

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






