import ast
import pytest

import networkx as nx
from plpy.rewrite import expr_lifter
from plpy.analyze import dynamic_tracer, graph_builder

from transfer import identify_donations as _id


def construct_dynamic_graph(src):
    lifted = expr_lifter.lift_expressions(src)
    tracer = dynamic_tracer.DynamicDataTracer(loop_bound=None)
    tracer.run(lifted)
    grapher = graph_builder.DynamicTraceToGraph(ignore_unknown=True, memory_refinement=graph_builder.MemoryRefinementStrategy.INCLUDE_ALL)
    graph = grapher.run(tracer)
    return graph

def construct_fake_graph(src_lines):
    g = nx.DiGraph()
    for i, line in enumerate(src_lines):
        g.add_node(i)
        g.nodes[i]['src'] = line
        if i > 0:
            g.add_edge(i - 1, i)
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

    graph = construct_fake_graph(src_lines)
    _id.annotate_graph(graph)

    for ix in range(len(src_lines)):
        node_data = graph.nodes[ix]
        assert node_data['columns_defined'] == set(info['defined'][ix])
        assert node_data['columns_used'] == set(info['used'][ix])
        assert node_data['annotator'] == _id.__file__


def test_remove_subgraphs():
    pass

def test_remove_duplicate_graphs():
    pass

def test_repair_slice_implicit_uses():
    pass

def test_column_def_extractor():
    pass

def test_column_use_extractor():
    pass

def test_get_all_donations():
    pass


