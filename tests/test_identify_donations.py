import ast
import astunparse
import pytest
import textwrap

import networkx as nx
from plpy.rewrite import expr_lifter
from plpy.analyze import dynamic_tracer, graph_builder

from transfer import identify_donations as _id


def construct_dynamic_graph(src):
    lifted = astunparse.unparse(expr_lifter.lift_expressions(src))
    tracer = dynamic_tracer.DynamicDataTracer(loop_bound=None)
    tracer.run(lifted)
    grapher = graph_builder.DynamicTraceToGraph(ignore_unknown=True, memory_refinement=graph_builder.MemoryRefinementStrategy.IGNORE_BASE)
    graph = grapher.run(tracer)
    return graph

def construct_graph_manual(src_lines, edges=None):
    g = nx.DiGraph()
    for i, line in enumerate(src_lines):
        g.add_node(i)
        g.nodes[i]['src'] = line
        if i > 0 and edges is None:
            g.add_edge(i - 1, i)
    if not edges is None:
        g.add_edges_from(edges)
    return g

possible_column_collector_cases = [
    ("df['a'] = 'b'", ['a', 'b']),
    ("df.a = 'b'", ['a', 'b']),
    ("df.a.b = 2", ['b']),
    ("df = 2", []),
]

@pytest.mark.parametrize('src,expected', possible_column_collector_cases)
def test_possible_column_collector(src, expected):
    tree = ast.parse(src)
    result = _id.PossibleColumnCollector().run(tree)
    assert result == set(expected)
    pass


columns_assigned_to_cases = [
    ("df['a'] = 2 + df['b']", ['a']),
    ("df['a'], df['b'] = (1, 2)", ['a', 'b']),
    ("df['a'].b = 2", ['b']),
    ("x = df['a']", [])
]

@pytest.mark.parametrize('src,expected', columns_assigned_to_cases)
def test_columns_assigned_to(src, expected):
    fake_node_data = {'src': src}
    result = _id.columns_assigned_to(fake_node_data)
    assert result == set(expected)


columns_used_cases = [
    ("df['a'] = 2 + df['b']", ['b']),
    ("df['a'], df['b'] = (1, 2)", []),
    ("df['a'].b = 2 * df['a'].b", ['b']),
    ("x = df['a']", ['a']),
    ("df['a']", ['a'])
]

@pytest.mark.parametrize('src,expected', columns_used_cases)
def test_columns_used(src, expected):
    fake_node_data = {'src': src}
    result = _id.columns_used(fake_node_data)
    assert result == set(expected)


annotate_graph_cases = [
    dict(
        src_lines=["df['a'] = 2 + df['b']", "df['a'], df['b'] = (1, 2)"],
        defined=[['a'], ['a', 'b']],
        used=[['b'], []]
    ),
    dict(
        src_lines=["df['a'].b = 2 * df['a'].b", "x = df['a']"],
        defined=[['b'], []],
        used=[['b'], ['a']]
    ),
]

@pytest.mark.parametrize('info', annotate_graph_cases)
def test_annotate_graph(info):
    src_lines = info['src_lines']

    graph = construct_graph_manual(src_lines)
    _id.annotate_graph(graph)

    for ix in range(len(src_lines)):
        node_data = graph.nodes[ix]
        assert node_data['columns_defined'] == set(info['defined'][ix])
        assert node_data['columns_used'] == set(info['used'][ix])
        assert node_data['annotator'] == _id.__file__


subgraphs_cases = [
    ["df['a'] = 2 + df['b']", "df['a'], df['b'] = (1, 2)"],
    ["df['a'].b = 2 * df['a'].b", "x = df['a']"],
]

@pytest.mark.parametrize('lines', subgraphs_cases)
def test_remove_subgraphs(lines):
    overall_graph = construct_graph_manual(lines)

    for ix in range(len(lines)):
        line_graph = construct_graph_manual([lines[ix]])
        prefix_graph = construct_graph_manual(lines[:ix])
        suffix_graph = construct_graph_manual(lines[ix:])
        subgraphs = [line_graph, prefix_graph, suffix_graph]
        for g in subgraphs:
            assert _id.remove_subgraphs([g, overall_graph]) == [overall_graph]

        assert _id.remove_subgraphs(subgraphs + [overall_graph]) == [overall_graph]



duplicate_graphs_cases = [
    ([[(1, 2), (2, 3)], [(1, 2), (2, 3)], [(1, 3), (2, 3)]], [0, 2]),
    ([[(2, 3), (2, 4), (2, 5)], [(3, 2), (4, 2), (5, 2)]], [0, 1]),
    ([[], [], [(1, 1)]], [0, 2]),
]
@pytest.mark.parametrize('edge_lists,expected_ix', duplicate_graphs_cases)
def test_remove_duplicate_graphs(edge_lists, expected_ix):
    graphs = []
    for _list in edge_lists:
        g = nx.DiGraph()
        g.add_edges_from(_list)
        graphs.append(g)

    result = _id.remove_duplicate_graphs(graphs)
    result_adjacency = [sorted(tuple(g.adjacency())) for g in result]
    expected_adjacency = [sorted(tuple(graphs[ix].adjacency())) for ix in expected_ix]
    assert result_adjacency == expected_adjacency



repair_cases = [
    dict(
        src=
        """
        irrelevant = 0
        df['a'] = 2 * 3
        df['b'] = df['a'] + 3
        x = df['b'] * 4
        """,
        edges=[(0, 1), (1, 2), (2, 3)],
        init_slice_nodes=[1, 2, 3],
        expected_slice_nodes=[1, 2, 3],
    ),
    dict(
        src=
        """
        df['c'] = 2
        _var0 = df.groupby
        _var1 = _var0['c']
        df['b'] = _var1.max()
        """,
        edges=[(0, 1), (1, 2), (2, 3)],
        init_slice_nodes=[1, 2, 3],
        # add in missing definition of df['c']
        expected_slice_nodes=[0, 1, 2, 3],
    ),
    dict(
        src=
        """
        irrelevant = 10
        df['a'] = 100
        df['c'] = df['a'] + 4
        _var0 = df.groupby
        _var1 = _var0['c']
        df['a'] = dummy
        df['b'] = _var1.max() + df['a']
        """,
        edges=[(0, 0), (1, 2), (2, 3), (3, 4), (4, 6), (5, 6)],
        init_slice_nodes=[3, 4, 5, 6],
        # add in missing definition of df['c'] which depends on a previous df['a'] def
        expected_slice_nodes=[1, 2, 3, 4, 5, 6],
    )
]

@pytest.mark.parametrize('info', repair_cases)
def test_repair_slice_implicit_uses(info):
    src_lines = textwrap.dedent(info["src"]).strip().split('\n')
    edges = info['edges']

    init_slice_nodes = info["init_slice_nodes"]
    expected_slice_nodes = info["expected_slice_nodes"]

    graph = construct_graph_manual(src_lines, edges=edges)
    _slice = graph.subgraph(init_slice_nodes)
    repaired_slice = _id.RepairSliceImplicitUses(graph).run(_slice)
    expected_slice = graph.subgraph(expected_slice_nodes)

    assert sorted(list(repaired_slice.edges)) == sorted(list(expected_slice.edges))

# src, ((target_column, expected count))for def, and then for use
column_extractor_cases = [
    (
        """
        import pandas as pd
        df = pd.DataFrame([(1, 2), (2, 3)], columns=['c1', 'c2'])
        df['a'] = df['c1'].max()
        """,
        [('a', 1), ('b', 0)],
        [('c1', 1), ('a', 0)],
    ),
    (
        """
        import pandas as pd
        df = pd.DataFrame([(1, 2), (2, 3)], columns=['c1', 'c2'])
        df['c1'] = df['c2'] * 2
        df['other'] = df['c1'] + 3
        df['c1'] = 3
        """,
        # first and second c1 defs are independent (under graph_builder.MemoryRefinementStrategy.IGNORE_BASE)
        [('c1', 2)],
        # we actually have two for c1 as it slices forward and finds two defines for other and c1
        # same for c2
        [('c1', 2), ('c2', 2)],
    ),

    (
        """
        import pandas as pd
        df = pd.DataFrame([(1, 2), (2, 3)], columns=['c1', 'c2'])
        df['c1'] = df['c2'] * 2
        df['other'] = df['c1'] + 3
        df['c1'] = df['c1'] * 2
        """,
        # the second def for c1 is superset of first
        [('c1', 1)],
        # slicing forward from c1 leads to other and c1
        [('c1', 2)],
    ),
]

@pytest.mark.parametrize('src, expected_def, expected_use', column_extractor_cases)
def test_column_def_extractor(src, expected_def, expected_use):
    src = textwrap.dedent(src)
    graph = construct_dynamic_graph(src)
    # extractors
    from_defs = _id.ColumnDefExtractor(graph)
    from_uses = _id.ColumnUseExtractor(graph)

    for target_column, ct in expected_def:
        _slices = from_defs.run(target_column)
        assert len(_slices) == ct

    for target_column, ct in expected_use:
        _slices = from_uses.run(target_column)
        assert len(_slices) == ct
