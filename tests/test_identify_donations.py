import ast
import astunparse
import pytest
import textwrap

import networkx as nx
import pandas as pd
from plpy.rewrite import expr_lifter
from plpy.analyze import dynamic_tracer, graph_builder
from plpy.analyze.dynamic_trace_events import Variable

from transfer import identify_donations as _id


def construct_dynamic_graph(src):
    lifted = astunparse.unparse(expr_lifter.lift_expressions(src))
    tracer = dynamic_tracer.DynamicDataTracer(loop_bound=None)
    tracer.run(lifted)
    grapher = graph_builder.DynamicTraceToGraph(ignore_unknown=True, memory_refinement=graph_builder.MemoryRefinementStrategy.IGNORE_BASE)
    graph = grapher.run(tracer)
    return graph

def construct_graph_manual(src_lines, defs=None, uses=None, edges=None):
    g = nx.DiGraph()
    for i, line in enumerate(src_lines):
        g.add_node(i)
        g.nodes[i]['src'] = line
        if defs:
            g.nodes[i]['defs'] = defs[i]
        if uses:
            g.nodes[i]['uses'] = uses[i]
        if i > 0 and edges is None:
            g.add_edge(i - 1, i)
    if not edges is None:
        g.add_edges_from(edges)
    return g

possible_column_collector_cases = [
    ("df['a'] = 'b'", ['a']),
    ("df.a = 'b'", ['a']),
    ("df.a.b = 2", ['a', 'b']),
    ("df = 2", []),
]

@pytest.mark.parametrize('src,expected', possible_column_collector_cases)
def test_possible_column_collector(src, expected):
    tree = ast.parse(src)
    result = _id.PossibleColumnCollector().run(tree)
    assert result == set(expected)
    pass


columns_assigned_to_cases = [
    ("df['a'] = 2 + df['b']", ['a'], [Variable("df['a']", 0, pd.Series)], [Variable("df['b']", 0, pd.Series)]),
    ("df['a'], df['b'] = (1, 2)", ['a', 'b'], [Variable("df['a']", 0, pd.Series), Variable("df['b']", 0, pd.Series)], []),
    # we consider 'b' relevant as an assignment involving a series/dataframe takes place, regardless of 'b's type
    ("df['a'].b = 2", ['a', 'b'], [Variable("df['a'].b", 0, int)], [Variable("df['a']", 0, pd.Series)]),
    ("x = df['a']", [], [], [Variable("df['a']", 0, pd.Series)]),
]

@pytest.mark.parametrize('src,expected,defs,uses', columns_assigned_to_cases)
def test_columns_assigned_to(src, expected, defs, uses):
    fake_node_data = {'src': src, 'defs': defs, 'uses': uses}
    result = _id.columns_assigned_to(fake_node_data)
    assert result == set(expected)


columns_used_cases = [
    ("df['a'] = 2 + df['b']", ['b'], [Variable("df['a']", 0, pd.Series)], [Variable("df['b']", 0, pd.Series)]),
    ("df['a'], df['b'] = (1, 2)", [], [Variable("df['a']", 0, pd.Series), Variable("df['b']", 0, pd.Series)], []),
    # we consider 'b' relevant as an assignment involving a series/dataframe takes place, regardless of 'b's type
    ("df['a'].b = 2", [], [Variable("df['a'].b", 0, int)], [Variable("df['a']", 0, pd.Series)]),
    ("x = df['a']", ['a'], [], [Variable("df['a']", 0, pd.Series)]),
    ("df['a']", ['a'], [], [Variable("df['a']", 0, pd.Series)]),
]

@pytest.mark.parametrize('src,expected,defs,uses', columns_used_cases)
def test_columns_used(src, expected, defs, uses):
    fake_node_data = {'src': src, 'defs': defs, 'uses': uses}
    result = _id.columns_used(fake_node_data)
    assert result == set(expected)


annotate_graph_cases = [
    dict(
        src_lines=["df['a'] = 2 + df['b']", "df['a'], df['b'] = (1, 2)"],
        defs = [
            [Variable("df['a']", 0, pd.Series)],
            [Variable("df[a']", 0, pd.Series), Variable("df['b']", 1, pd.Series)],
        ],
        uses = [
            [Variable("df['b']", 0, pd.Series)],
            [],
        ],
        columns_defined=[['a'], ['a', 'b']],
        columns_used=[['b'], []]
    ),
    dict(
        src_lines=["df['a'].b = 2 * df['a'].b", "x = df['a']"],
        defs = [
            [Variable("df['a']", 0, pd.Series), Variable("df['a'].b", 0, int)],
            [],
        ],
        uses = [
            [Variable("df['a']", 0, pd.Series), Variable("df['a'].b", 1, int)],
            [Variable("df['a']", 0, pd.Series)],
        ],
        columns_defined=[['a', 'b'], []],
        columns_used=[['a', 'b'], ['a']]
    ),
]

@pytest.mark.parametrize('info', annotate_graph_cases)
def test_annotate_graph(info):
    src_lines = info['src_lines']
    defs = info['defs']
    uses = info['uses']
    graph = construct_graph_manual(src_lines, defs=defs, uses=uses)

    _id.annotate_graph(graph)

    for ix in range(len(src_lines)):
        node_data = graph.nodes[ix]
        assert node_data['columns_defined'] == set(info['columns_defined'][ix])
        assert node_data['columns_used'] == set(info['columns_used'][ix])
        assert node_data['annotator'] == _id.__file__


def are_equal_graphs(graph1, graph2):
    if sorted(graph1.nodes) != sorted(graph2.nodes):
        return False
    if sorted(graph1.edges) != sorted(graph2.edges):
        return False
    for i in graph1.nodes:
        data_graph1 = graph1.nodes[i]
        data_graph2 = graph2.nodes[i]
        if data_graph1 != data_graph2:
            return False
    return True


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
        expected = [overall_graph]
        for g in subgraphs:
            result = _id.remove_subgraphs([g, overall_graph])
            assert len(result) == 1 and len(expected) == 1
            assert are_equal_graphs(result[0], expected[0])

        result = _id.remove_subgraphs(subgraphs + [overall_graph])
        assert len(result) == 1 and len(expected) == 1
        assert are_equal_graphs(result[0], expected[0])



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
        defs = [
            [],
            [Variable("df['a']", 0, pd.Series)],
            [Variable("df['b']", 1, pd.Series)],
            [],
        ],
        uses = [
            [],
            [],
            [Variable("df['a']", 0, pd.Series)],
            [Variable("df['b']", 1, pd.Series)],
        ],
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
        defs = [
            [Variable("df['c']", 0, pd.Series)],
            [Variable("_var0", 0, pd.core.groupby.GroupBy)],
            [Variable("_var1", 0, pd.Series)],
            [Variable("df['b']", 1, pd.Series)],
        ],
        uses = [
            [],
            [Variable("df", 0, pd.DataFrame)],
            [Variable("_var0['c']", 1, pd.Series)],
            [Variable("_var1", 0, pd.Series)],
        ],
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
        defs = [
            [],
            [Variable('df["a"]', 0, pd.Series)],
            [Variable('df["c"]', 1, pd.Series)],
            [Variable("_var0", 2, pd.DataFrame)],
            [Variable("_var1", 1, pd.Series)],
            [Variable("df['a']", 3, pd.Series)],
            [Variable("df['b']", 4, pd.Series)],
        ],
        uses = [
            [],
            [],
            [Variable("df['a']", 0, pd.Series)],
            [Variable("df", 1, pd.DataFrame)],
            [Variable("_var0['c']", 1, pd.Series)],
            [],
            [Variable("df['a']", 0, pd.Series)],
        ],
        init_slice_nodes=[3, 4, 5, 6],
        # add in missing definition of df['c'] which depends on a previous df['a'] def
        expected_slice_nodes=[1, 2, 3, 4, 5, 6],
    )
]

@pytest.mark.parametrize('info', repair_cases)
def test_repair_slice_implicit_uses(info):
    src_lines = textwrap.dedent(info["src"]).strip().split('\n')
    edges = info['edges']
    defs = info['defs']
    uses = info['uses']

    init_slice_nodes = info["init_slice_nodes"]
    expected_slice_nodes = info["expected_slice_nodes"]

    graph = construct_graph_manual(src_lines, defs=defs, uses=uses, edges=edges)
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
        # c2 slices forward and finds df['c1'] = ... and df['other'] = ...
        # df['c1'] use slices forward and finds df['other'] = ...
        [('c1', 1), ('c2', 2)],
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
def test_column_extractor(src, expected_def, expected_use):
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
