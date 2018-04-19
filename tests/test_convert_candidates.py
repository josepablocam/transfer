import ast
import astunparse
import os
import tempfile

import nbformat
import pytest
import shutil

from transfer import convert_candidates as cc

def create_dir_with_files(_dir, file_names):
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    file_paths = [os.path.join(_dir, n) for n in file_names]
    for path in file_paths:
        with open(path, 'w') as f:
            f.write('')

def write_src_to_notebook(src, ipynb_file_name, temp_dir):
    # wrap source in a notebook and write out
    nb = nbformat.v4.new_notebook()
    nb['cells'].append(nbformat.v4.new_code_cell(src))
    ipynb_file_path = os.path.join(temp_dir, ipynb_file_name)
    nbformat.write(nb, open(ipynb_file_path, 'w'))
    return ipynb_file_path




get_files_cases = [
    ('/tmp/_test1', ['f1.ipynb', 'f1.xypnb', 'f2.R', 'f3.py'], ['/tmp/_test1/f1.ipynb'], ['/tmp/_test1/f3.py']),
    ('/tmp/_test2', ['f1.xypnb', 'f2.R', 'f3.py'], [], ['/tmp/_test2/f3.py']),
    ('/tmp/_test3', ['f1.xypnb', 'f2.R'], [], []),
]

@pytest.mark.parametrize('_dir,file_names,expected_ipynb,expected_py', get_files_cases)
def test_get_files(_dir, file_names, expected_ipynb, expected_py):
    create_dir_with_files(_dir, file_names)
    assert sorted(cc.get_ipython_notebooks(_dir)) == sorted(expected_ipynb)
    assert sorted(cc.get_py_scripts(_dir)) == sorted(expected_py)
    shutil.rmtree(_dir)



check_can_parse_cases = [
    ('def f(): return x', True),
    ('blah ble', False)
]

@pytest.mark.parametrize('src,expected', check_can_parse_cases)
def test_check_can_parse(src, expected):
    with tempfile.NamedTemporaryFile(mode='w', delete=True) as f:
        f.write(src)
        f.flush()
        assert cc.check_can_parse(f.name) == expected



convert_2_to_3_cases = [
    ('print 2', 'print(2)'),
    ('xrange(10)', 'range(10)'),
    ('d = {}; d.iteritems()', 'd = {}; iter(d.items())'),
]

@pytest.mark.parametrize('src,expected', convert_2_to_3_cases)
def test_convert_2_to_3(src, expected):
    temp_dir = tempfile.mkdtemp()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as f:
        f.write(src)
        f.flush()
        converted_path = cc.convert_2_to_3(f.name, temp_dir)
        with open(converted_path, 'r') as converted_file:
            result = converted_file.read().strip()
        assert result == expected

    shutil.rmtree(temp_dir)



convert_notebook_to_script_cases = [
    'print(2)',
    '1 + 2',
    'def f(x): return x',
    'class A(object): pass'
]

@pytest.mark.parametrize('src', convert_notebook_to_script_cases)
def test_convert_notebook_to_script(src):
    temp_dir = tempfile.mkdtemp()
    ipynb_file_path = write_src_to_notebook(src, 'file.ipynb', temp_dir)
    py_file_path = os.path.join(temp_dir, 'file.py')

    cc.convert_notebook_to_script(ipynb_file_path, temp_dir)
    print(py_file_path)
    print(ipynb_file_path)

    # parse and unparse to remove ipython notebook comments etc
    with open(py_file_path, 'r') as f:
        result = f.read()
        result_tree = ast.parse(result)
        result = astunparse.unparse(result_tree)

    src_tree = ast.parse(src)
    src = astunparse.unparse(src_tree)
    assert result == src

    shutil.rmtree(temp_dir)


def test_filter_candidates():
    files_and_contents = [
        # when things just work
        ('plain_python.py', '1'),
        ('plain_notebook.ipynb', '1'),
        # when can't parse
        #('bad_python.py', 'blah bleh'),
        #('bad_notebook.ipynb', 'blah bleh'),
        # when can convert from python2
        ('convertible_2to3_python.py', 'xrange(1)'),
        ('convertible_2to3_notebook.ipynb', 'xrange(1)'),
        # ignored files
        ('ignore.R', '1')
    ]

    # construct directory and files
    temp_dir = tempfile.mkdtemp()

    for file_name, src in files_and_contents:
        file_path = os.path.join(temp_dir, file_name)
        ext = file_name.split('.')[-1]
        if ext == 'ipynb':
            write_src_to_notebook(src, file_name, temp_dir)
        else:
            with open(file_path, 'w') as f:
                f.write(src)

    parsed_dir = os.path.join(temp_dir, 'parsed_dir')
    converted_dir = os.path.join(temp_dir, 'converted_dir')

    cc.filter_candidates(temp_dir, parsed_dir, converted_dir)

    # note that bad_notebook.ipynb is still converted, just not parseable later on
    # expected_converted_dir = sorted(['convertible_2to3_notebook.py', 'bad_notebook.py', 'plain_notebook.py'])
    expected_converted_dir = sorted(['convertible_2to3_notebook.py',  'plain_notebook.py'])
    expected_parsed_dir = sorted(['plain_python.py', 'convertible_2to3_python.py', 'convertible_2to3_notebook.py', 'plain_notebook.py'])
    assert sorted(os.listdir(converted_dir)) == expected_converted_dir
    assert sorted(os.listdir(parsed_dir)) == expected_parsed_dir

    filename_to_expected_src = {
        'plain_python.py'             : '1',
        'plain_notebook.py'           : '1',
        'convertible_2to3_python.py'  : 'range(1)',
        'convertible_2to3_notebook.py': 'range(1)',
    }
    for file_name in expected_parsed_dir:
        with open(os.path.join(temp_dir, 'parsed_dir', file_name), 'r') as f:
            result = f.read()

        result = astunparse.unparse(ast.parse(result))
        expected = filename_to_expected_src[file_name]
        expected = astunparse.unparse(ast.parse(expected))
        assert result == expected, 'Failed on %s' % file_name

    shutil.rmtree(temp_dir)
