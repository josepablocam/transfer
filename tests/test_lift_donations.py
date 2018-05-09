import ast
import argparse
import os
import shutil
import sys
import tempfile
import textwrap

import dill
import networkx as nx
import pandas as pd
from plpy.analyze.dynamic_trace_events import Variable
import pytest

from transfer.lift_donations import DonatedFunction
from transfer.utils import build_script_paths
sys.path.append('../')
from runner import run_pipeline


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


def from_source_to_functions(src, tempdir):
    script_path = os.path.join(tempdir, 'script.py')

    with open(script_path, 'w') as f:
        f.write(src)

    args = argparse.Namespace(
        timeout='2h',
        script_path=script_path,
        output_dir=tempdir,
        loop_bound=2,
        memory_refinement=1,
        log=None,
    )
    run_pipeline.main(args)

    functions_path = build_script_paths(script_path)['functions_path']
    with open(functions_path, 'rb') as f:
        functions = dill.load(f)

    return functions


lift_functions_cases = [
    (
        """
        import pandas as pd
        df = pd.read_csv('data.csv')
        df['c1'] = 2
        """,
        """
        def cleaning_func_0(df):
            import pandas as pd
            df['c1'] = 2
            return df
        """
    ),
    #
    # (
    #     """
    #     import pandas as pd
    #     df = pd.read_csv('dummy.csv')
    #     x = 'c1'
    #     df['c1'] = df[x]
    #     """,
    #     """
    #     def f(df):
    #         import pandas as pd
    #         x = 'c1'
    #         _var0 = df[x]
    #         df['c1'] = _var0
    #     """
    # )
]


@pytest.mark.parametrize('src,expected_src', lift_functions_cases)
def test_lift_functions(src, expected_src):
    src = textwrap.dedent(src)
    expected_src = textwrap.dedent(expected_src)

    tempdir = tempfile.mkdtemp()
    # prepare data for examples to run
    data_path = os.path.join(tempdir, 'data.csv')
    dummy_data = pd.DataFrame([(1, 2), (3, 4)], columns=['c1', 'c2'])
    dummy_data.to_csv(data_path, index=False)

    from transfer.lift_donations import DonatedFunction
    functions = from_source_to_functions(src, tempdir)
    assert len(functions) == 1
    function = functions[0]
    assert to_ast_dump(function.source) == to_ast_dump(expected_src)

    shutil.rmtree(tempdir)
