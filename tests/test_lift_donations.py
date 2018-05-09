import ast

import networkx as nx
import pandas as pd
from plpy.analyze.dynamic_trace_events import Variable
import pytest
import textwrap

from transfer.lift_donations import DonatedFunction



def test_args_and_return_vars_donated_functions():
    g = nx.DiGraph()
    name = 'f'
    cleaning_code = []
    context_code = []

    bad_formals = [Variable('x', 1, pd.Series), Variable('x[1]', 2, int)]
    bad_return  = Variable('df.b', 0, pd.Series)
    ok_return = Variable('df', 1, pd.DataFrame)

    try:
        DonatedFunction(g, name, bad_formals, ok_return, [], [])
        pytest.fail('Should fail with non-ast.Name formal args')
    except ValueError:
        pass

    try:
        DonatedFunction(g, name, [], bad_return, [], [])
        pytest.fail('Should fail with non-ast.Name return var')
    except ValueError:
        pass

def to_ast_dump(elem):
    if isinstance(elem, ast.AST):
        return ast.dump(elem)
    elif isinstance(elem, str):
        return ast.dump(ast.parse(elem))
    else:
        raise ValueError('unhandled type for ast dump')


source_donated_functions_cases = [
    dict(
        cleaning_code=['x', 'x + 2'],
        context_code=None,
        formal_args = [],
        return_var = Variable('x', 0, int),
        expected = "def f(): x; x + 2; return x",
    ),
    dict(
        cleaning_code=['x', 'x + 2'],
        context_code=None,
        formal_args = [Variable('a', 0, int), Variable('b', 1, int)],
        return_var = Variable('x', 0, int),
        expected = "def f(a, b): x; x + 2; return x",
    ),
    dict(
        cleaning_code=['x', 'x + 2'],
        context_code=[],
        formal_args = [Variable('a', 0, int), Variable('b', 1, int)],
        return_var = Variable('x', 0, int),
        expected = "def f(a, b): x; x + 2; return x",
    ),
    dict(
        cleaning_code=['x', 'x + 2'],
        context_code=['def a(): return 10'],
        formal_args = [Variable('z', 0, int)],
        return_var = Variable('x', 0, int),
        expected = """
        def f(z):
            def a(): return 10
            x
            x + 2
            return x
        """
    ),
]

@pytest.mark.parametrize('info', source_donated_functions_cases)
def test_source_donated_functions(info):
    g = nx.DiGraph()
    name = 'f'
    formal_args = info['formal_args']
    return_var = info['return_var']
    cleaning_code = info['cleaning_code']
    context_code = info['context_code']
    expected = textwrap.dedent(info['expected'])

    fun = DonatedFunction(g, name, formal_args, return_var, cleaning_code, context_code)
    expected_src = "def f(): x; x + 2; return x"
    assert to_ast_dump(fun.source) == to_ast_dump(expected)
    assert to_ast_dump(fun.ast) == to_ast_dump(expected)
