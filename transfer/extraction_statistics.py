
from argparse import ArgumentParser
import glob
import inspect
import os
import pickle

import matplotlib.pyplot as plt
plt.ion()
import networkx as nx
import numpy as np
import pandas as pd
import tabulate
from plpy.analyze.dynamic_tracer import DynamicDataTracer

from .identify_donations import ColumnUse, ColumnDef, remove_duplicate_graphs
from .lift_donations import DonatedFunction
from .utils import build_script_paths

def print_df(df):
    print(tabulate.tabulate(df, headers='keys', tablefmt='grid', showindex=False))

def summarize_lifted(lifted_path):
    if not os.path.exists(lifted_path):
        return dict(lifted=False)
    else:
        return dict(lifted=True)

def summarize_trace(trace_path):
    info = dict(has_trace=False, trace_len=0)
    if not os.path.exists(trace_path):
        return info
    info['has_trace'] = True
    with open(trace_path, 'rb') as f:
        tracer = pickle.load(f)
    info['trace_len'] = len(tracer.trace_events)
    return info

def summarize_graph(graph_path):
    info = dict(has_graph=False, num_graph_nodes=0, num_graph_edges=0)
    if not os.path.exists(graph_path):
        return info
    info['has_graph'] = True
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    info['num_graph_nodes'] = graph.number_of_nodes()
    info['num_graph_edges'] = graph.number_of_edges()
    return info

def summarize_donations(donations_path):
    info = dict(has_donations=False, num_donations=0)
    if not os.path.exists(donations_path):
        return info
    info['has_donations'] = True
    with open(donations_path, 'rb') as f:
        donations = pickle.load(f)
    info['num_donations'] = len(donations)
    return info

def summarize_functions(functions_path):
    info = dict(has_functions=False, num_functions=0)
    if not os.path.exists(functions_path):
        return info
    info['has_functions'] = True
    with open(functions_path, 'rb') as f:
        functions = pickle.load(f)
    info['num_functions'] = len(functions)
    info['avg_function_len'] = np.mean([f.graph.number_of_nodes() for f in functions])
    # FIXME: this line fails because we currently lift some functions incorrectly
    # func_objs = [f._get_func_obj() for f in functions]
    func_objs = []
    info['fraction_more_than_one_arg'] = np.mean([len(inspect.getfullargspec(f).args) > 1 for f in func_objs])
    return info

def functions_length_distribution(functions):
    s = pd.Series([f.graph.number_of_nodes() for f in functions])
    return s, s.plot(kind='hist')

def functions_args_distribution(functions):
    func_objs = [f._get_func_obj() for f in functions]
    s = pd.Series([len(inspect.getfullargspec(f).args) for f in func_objs])
    return s, s.plot(kind='hist')

def summarize(scripts_dir, results_dir, detailed=False):
    scripts_paths = glob.glob(os.path.join(scripts_dir, '*[0-9].py'))
    functions_paths = []

    results = []
    for s in scripts_paths:
        paths = build_script_paths(s, output_dir=results_dir)
        info = dict(script_path=paths['script_path'])
        info['name'] = os.path.basename(info['script_path'])
        info.update(summarize_lifted(paths['lifted_path']))
        info.update(summarize_trace(paths['trace_path']))
        info.update(summarize_graph(paths['graph_path']))
        info.update(summarize_donations(paths['donations_path']))
        info.update(summarize_functions(paths['functions_path']))
        results.append(info)
        if info['has_functions']:
            functions_paths.append(paths['functions_path'])

    summary_df = pd.DataFrame(results)
    if not detailed:
        return summary_df

    functions = []
    for f_path in functions_paths:
        with open(f_path, 'rb') as f:
            functions.extend(pickle.load(f))
    length_dist_results = functions_length_distribution(functions)
    arg_dist_results = functions_args_distribution(functions)
    return summary_df, length_dist_results, arg_dist_results

def print_report(summary_df):
    total_ct = summary_df.shape[0]
    ct_fields = ['has_trace', 'has_graph', 'has_donations', 'has_functions']
    mean_fields = ['avg_function_len', 'fraction_more_than_one_arg']
    sum_fields = ['num_donations', 'num_functions']
    print('General Summary')
    print('---------------------')
    for f in ct_fields:
        ct = summary_df[f].sum()
        print('Files {}: {}/{} ({})'.format(f, ct, total_ct, round(ct / total_ct, 2)))
    for f in sum_fields:
        print('Total {}: {}'.format(f, summary_df[f].sum()))
    for f in mean_fields:
        print('Mean {}: {}'.format(f, round(np.mean(summary_df[f]), 2)))
    print('======================')
    print('Detailed Report (only entries with a trace)')
    detailed_fields = ['name', 'trace_len', 'num_donations', 'num_functions', 'avg_function_len']
    reduced_df = summary_df.loc[summary_df['has_trace']][detailed_fields]
    print_df(reduced_df)

def print_failed_trace(summary_df):
    mask = ~summary_df['has_trace']
    if any(mask):
        failed = summary_df[mask]
        print('Failed to collect a trace: {} / {}'.format(failed.shape[0], summary_df.shape[0]))
        print_df(failed[['script_path']])
    else:
        print('No trace collection failures')

def print_failed_graph(summary_df):
    has_trace = summary_df['has_trace']
    missing_graph = ~summary_df['has_graph']
    mask = has_trace & missing_graph
    if any(mask):
        failed = summary_df[mask]
        print('Failed to build a graph: {} / {}'.format(failed.shape[0], summary_df.shape[0]))
        print_df(failed[['script_path']])
    else:
        print('No graph building failures')

def main(args):
    summary_df = summarize(args.scripts_dir, args.results_dir)

    if not args.silent_report:
        print_report(summary_df)

    if args.failed_trace:
        print_failed_trace(summary_df)

    if args.failed_graph:
        print_failed_graph(summary_df)

    if args.output_path:
        summary_df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    parser = ArgumentParser(description='Summarize extraction statistics')
    parser.add_argument('scripts_dir', type=str, help='Directory for scripts')
    parser.add_argument('results_dir', type=str, help='Directory to results (trace, graph, etc)')
    parser.add_argument('-o', '--output_path', type=str, help='Path to store csv of summary')
    parser.add_argument('-t', '--failed_trace', action='store_true', help='Print info for scripts that failed to trace')
    parser.add_argument('-g', '--failed_graph', action='store_true', help='Print info for scripts that failed to graph')
    parser.add_argument('-s', '--silent_report', action='store_true', help='Do not print out main report')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
