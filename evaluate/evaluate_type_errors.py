import transfer.identify_donations as _id
from transfer.build_db import *
from transfer import utils
import os
import networkx as nx
import pickle
import textwrap


def get_lifted_source_lines(entry, src_dir):
    func = entry['function']
    basename = os.path.basename(func.graph.graph['filename'])
    basename = '_'.join(basename.split('_')[:-1])
    basepath = os.path.join(src_dir, basename + '.py')
    paths = utils.build_script_paths(basepath)
    lifted_path = paths['lifted_path']
    with open(lifted_path, 'r') as f:
        src_lines = f.readlines()
    return src_lines


def get_py_whitespace(line):
    n = len(line) - len(textwrap.dedent(line))
    if n > 0:
        char = line[0]
        return char * n
    return ''


def wrap_whitespace(reference_line, content):
    return '{}{}'.format(get_py_whitespace(reference_line), content)


def comment_out_assignments(entry, orig_lines):
    func = entry['function']
    cols_defined = entry['cols_defined']
    comment_lines = set([])

    for node_id, node_data in func.graph.nodes(data=True):
        node_defines = node_data.get('columns_defined', set([]))
        if not node_defines.isdisjoint(cols_defined):
            comment_lines.add(node_data['src'].strip())

    rewritten = []
    for i, line in enumerate(orig_lines):
        if line.strip() in comment_lines:
            rewritten.append('# commented out for type error testing\n')
            line = '# {}'.format(line)
        rewritten.append(line)

    return ''.join(rewritten)


def add_nullifying_assignments(entry, orig_lines):
    func = entry['function']
    max_node_id = max(func.graph)
    end_of_func_line = func.graph.nodes[max_node_id]['src'].strip()

    col_used = list(entry['cols_used'])[0]
    cols_defined = entry['cols_defined']
    df_name = func.return_var.name
    modified = False

    rewritten = []
    for i, line in enumerate(orig_lines):
        if line.strip() == end_of_func_line and not modified:
            rewritten.append(
                '# adding nullying assignments for type error testing\n'
            )
            for col in cols_defined:
                # assign old value to new value column to remove effect of derivation
                assignment = '{df}["{col_defined}"] = {df}["{col_used}"]\n'.format(
                    df=df_name, col_defined=col, col_used=col_used
                )
                new_line = wrap_whitespace(line, assignment)
                rewritten.append(new_line)
            modified = True
        else:
            rewritten.append(line)

    return ''.join(rewritten)


def execute_rewritten(new_src):
    try:
        exec(new_src)
        return (True, None)
    except:
        exc_info = sys.exc_info
        return (False, exc_info)


# simple: single input
single_input = [entry for entry in results if len(entry['cols_used']) == 1]
redefinitions = [
    entry for entry in single_input
    if entry['cols_used'] == entry['cols_defined']
]
derivations = [
    entry for entry in single_input
    if entry['cols_used'] != entry['cols_defined']
]


def get_type_testing_entries(results):
    results = pickle.load(
        open('evaluation_results/new_columns_results.pkl', 'rb')
    )
    entries = []
    for entry in results:
        if len(entry['cols_used']) == 1:
            label = 'redefinition' if entry['cols_used'] == entry[
                'cols_defined'] else 'derivation'
            entries.append((label, entry))
    return entries


def main(data):
    # exec_dir = os.path.abspath(exec_dir)
    entries = get_type_testing_entries(data)
    src_dir = 'program_data/loan_data/scripts'
    results = {}
    for i, (entry_label, entry) in enumerate(entries):
        lifted_lines = get_lifted_source_lines(entry, src_dir)
        rewrite = comment_out_assignments if entry_label == 'redefinition' else add_nullifying_assignments
        new_src = rewrite(entry, lifted_lines)
        results[i] = (entry_label, entry, ''.join(lifted_lines), new_src)
        # curr_dir = os.path.abspath(os.getcwd())
        # os.chdir(exec_dir)
        # outcome = execute_rewritten(new_src)
        # os.chdir(curr_dir)
        # results[func] = outcome

    return results
