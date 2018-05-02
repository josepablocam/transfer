from argparse import ArgumentParser
import os
import subprocess

def rewrite(script_path, lifted_path):
    print('Rewriting {} to {}'.format(script_path, lifted_path))
    cmd = ['python', '-m', 'plpy.rewrite.expr_lifter']
    cmd += [script_path, lifted_path]
    return subprocess.call(cmd)

def trace(timeout, execution_dir, lifted_name, trace_path, loop_bound):
    current_dir = os.getcwd()
    os.chdir(execution_dir)
    trace_path = os.path.abspath(trace_path)
    print('Executing from {}'.format(execution_dir))
    print('Tracing {} to {} with loop bound={} (timeout={})'.format(lifted_name, trace_path, loop_bound, timeout))
    cmd = ['timeout', timeout]
    cmd += ['ipython3', '-m', 'plpy.analyze.dynamic_tracer', '--']
    cmd += [lifted_name, trace_path, '--loop_bound', str(loop_bound)]
    return_code = subprocess.call(cmd)
    os.chdir(current_dir)
    return return_code

def graph(trace_path, graph_path, memory_refinement):
    print('Graphing {} to {} with refinement={}'.format(trace_path, graph_path, memory_refinement))
    cmd = ['python', '-m', 'plpy.analyze.graph_builder']
    cmd += [trace_path, graph_path, '--ignore_unknown', '--memory_refinement', memory_refinement]
    return subprocess.call(cmd)

def identify_donations(graph_path, donations_path):
    print('Identifying donations from {} to {}'.format(graph_path, donations_path))
    cmd = ['python', '-m', 'transfer.identify_donations']
    cmd += [graph_path, donations_path]
    return subprocess.call(cmd)

def lift_donations(donations_path, script_path, functions_path):
    print('Lifting from {} to {} (script={})'.format(donations_path, functions_path, script_path))
    cmd = ['python', '-m', 'transfer.lift_donations']
    cmd += [donations_path, script_path, functions_path]
    return subprocess.call(cmd)

def main(args):
    script_path = args.script_path
    script_dir = os.path.dirname(script_path)
    basename = '.'.join(os.path.basename(script_path).split('.')[:-1])

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        print('Creating output directory {}'.format(output_dir))
        os.makedirs(output_dir)

    lifted_path = os.path.join(script_dir, basename + '_lifted.py')
    lifted_name = os.path.basename(lifted_path)
    trace_path = os.path.join(output_dir, basename + '_tracer.pkl')
    graph_path = os.path.join(output_dir, basename + '_graph.pkl')
    donations_path = os.path.join(output_dir, basename + '_donations.pkl')
    functions_path = os.path.join(output_dir, basename + '_functions.pkl')

    timeout = str(args.timeout)
    loop_bound = str(args.loop_bound)
    memory_refinement = str(args.memory_refinement)

    rewrite(script_path, lifted_path)
    # need to execute from same directory as script
    trace(timeout, script_dir, lifted_name, trace_path, loop_bound)
    graph(trace_path, graph_path, memory_refinement)
    identify_donations(graph_path, donations_path)
    lift_donations(donations_path, script_path, functions_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='Run extraction pipeline for a given Kaggle script')
    parser.add_argument('timeout', type=str, help='Timeout string for tracing portion')
    parser.add_argument('script_path', type=str, help='Path to Kaggle script')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    parser.add_argument('-l', '--loop_bound', type=int, help='Loop bound for tracing', default=2)
    parser.add_argument('-m', '--memory_refinement', type=int, help='Memory refinement strategy for graph builder', default=1)
    args = parser.parse_args()
    print(args)
    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
