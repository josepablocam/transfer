from argparse import ArgumentParser
import os
import subprocess
import sys
import time

from transfer.utils import build_script_paths


def exit_with_error_if_file_missing(file_path):
    # TODO: we check for the file because the subprocess.calls
    # will actually always return 0, because we have some pdb code
    # catching exceptions at the bottom of each of our library scripts...
    if not os.path.exists(file_path):
        sys.exit(1)


def run_plain(timeout, script_path, confirmation_dir):
    # useful when debugging to figure out if script is just plain wrong
    # get absolute paths so things work when we change directories
    script_path = os.path.abspath(script_path)
    confirmation_dir = os.path.abspath(confirmation_dir)
    script_name = '.'.join(os.path.basename(script_path).split('.')[:-1])

    current_dir = os.getcwd()
    os.chdir(os.path.dirname(script_path))

    cmd = ['timeout', timeout, 'ipython3', script_path]
    confirmation_path = os.path.join(confirmation_dir, script_name + '.plain')
    with open(confirmation_path, 'w') as stdout:
        proc = subprocess.Popen(cmd, stdout=stdout)
    proc.communicate()
    return_status = proc.returncode

    with open(confirmation_path, 'a') as f:
        f.write('Program return value: {}'.format(return_status))
    # rename file to be clear what happened
    confirmation_ending = '.success' if return_status == 0 else '.failure'
    new_confirmation_path = confirmation_path + confirmation_ending
    os.rename(confirmation_path, new_confirmation_path)

    os.chdir(current_dir)
    return return_status


def rewrite(script_path, lifted_path):
    print('Rewriting {} to {}'.format(script_path, lifted_path))
    cmd = ['python', '-m', 'plpy.rewrite.expr_lifter']
    cmd += [script_path, lifted_path]
    return subprocess.call(cmd)


def filter_file(script_path):
    print(
        'Removing lines that cause failures in IPython3 for {}'.
        format(script_path)
    )
    cmd = ['python', '-m', 'transfer.filter_file']
    cmd += [script_path]
    return subprocess.call(cmd)


def trace(timeout, execution_dir, lifted_name, trace_path, loop_bound, log):
    trace_path = os.path.abspath(trace_path)
    current_dir = os.getcwd()
    os.chdir(execution_dir)
    print('Executing from {}'.format(execution_dir))
    print(
        'Tracing {} to {} with loop bound={} (timeout={})'.format(
            lifted_name, trace_path, loop_bound, timeout
        )
    )
    cmd = ['timeout', timeout]
    cmd += ['ipython3', '-m', 'plpy.analyze.dynamic_tracer', '--']
    cmd += [lifted_name, trace_path, '--loop_bound', str(loop_bound)]
    if log is not None:
        cmd += ['--log', log]
    return_code = subprocess.call(cmd)
    os.chdir(current_dir)
    return return_code


def graph(trace_path, graph_path, memory_refinement):
    print(
        'Graphing {} to {} with refinement={}'.format(
            trace_path, graph_path, memory_refinement
        )
    )
    cmd = ['python', '-m', 'plpy.analyze.graph_builder']
    cmd += [
        trace_path, graph_path, '--ignore_unknown', '--memory_refinement',
        memory_refinement
    ]
    return subprocess.call(cmd)


def identify_donations(graph_path, donations_path):
    print(
        'Identifying donations from {} to {}'.format(
            graph_path, donations_path
        )
    )
    cmd = ['python', '-m', 'transfer.identify_donations']
    cmd += [graph_path, donations_path]
    return subprocess.call(cmd)


def lift_donations(donations_path, script_path, functions_path):
    print(
        'Lifting from {} to {} (script={})'.format(
            donations_path, functions_path, script_path
        )
    )
    cmd = ['python', '-m', 'transfer.lift_donations']
    cmd += [donations_path, script_path, functions_path]
    return subprocess.call(cmd)


def record_time(script_path, start_time, output_dir):
    total_time = time.time() - start_time

    script_name = script_path.replace("/", "_")
    script_name = os.path.splitext(script_name)[0]
    time_file = os.path.join(output_dir, script_name, "_time.txt")
    print('Recording time to', time_file)
    with open(time_file, "w") as fout:
        fout.write("{}:{}".format(script_path, total_time))


def cleanup(script_path):
    script_dir = os.path.dirname(script_path)
    instr_script = os.path.join(script_dir, "_instrumented.py")
    if os.path.exists(instr_script):
        os.remove(instr_script)
    script_name = os.path.splitext(os.path.split(script_path)[-1])[0]
    lifted_script = os.path.join(script_dir, script_name + "_lifted.py")
    if os.path.exists(lifted_script):
        os.remove(lifted_script)


def main(args):
    script_path = args.script_path
    script_dir = os.path.dirname(script_path)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        print('Creating output directory {}'.format(output_dir))
        os.makedirs(output_dir)

    paths = build_script_paths(script_path, output_dir=output_dir)

    lifted_path = paths['lifted_path']
    lifted_name = os.path.basename(lifted_path)
    trace_path = paths['trace_path']
    graph_path = paths['graph_path']
    donations_path = paths['donations_path']
    functions_path = paths['functions_path']

    timeout = str(args.timeout)
    loop_bound = str(args.loop_bound)
    memory_refinement = str(args.memory_refinement)
    tracer_log = args.log

    start_time = time.time()

    filter_file(script_path)

    # only execute plain after the filtering...otherwise guaranteed to fail
    # for a lot of scripts...
    if args.plain:
        return_plain = run_plain(timeout, script_path, output_dir)
        if args.time:
            record_time(script_path, start_time, output_dir)
        sys.exit(return_plain)

    rewrite(script_path, lifted_path)
    exit_with_error_if_file_missing(lifted_path)

    # need to execute from same directory as script
    timeout_return = trace(
        timeout, script_dir, lifted_name, trace_path, loop_bound, tracer_log
    )
    # we can actually still check the timeout
    if timeout_return != 0:
        sys.exit(timeout_return)
    exit_with_error_if_file_missing(trace_path)

    graph(trace_path, graph_path, memory_refinement)
    exit_with_error_if_file_missing(graph_path)

    identify_donations(graph_path, donations_path)
    exit_with_error_if_file_missing(donations_path)

    lift_donations(donations_path, script_path, functions_path)
    exit_with_error_if_file_missing(functions_path)

    if args.time:
        record_time(script_path, start_time, output_dir)

    cleanup(script_path)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Run extraction pipeline for a given script'
    )
    parser.add_argument(
        'timeout', type=str, help='Timeout string for tracing portion'
    )
    parser.add_argument('script_path', type=str, help='Path to script')
    parser.add_argument(
        'output_dir', type=str, help='Path to output directory'
    )
    parser.add_argument(
        '-b',
        '--loop_bound',
        type=int,
        help='Loop bound for tracing',
        default=2
    )
    parser.add_argument(
        '-m',
        '--memory_refinement',
        type=int,
        help='Memory refinement strategy for graph builder',
        default=1
    )
    parser.add_argument(
        '-l', '--log', type=str, help='Turn on logging for the tracer'
    )
    parser.add_argument(
        '-p',
        '--plain',
        action='store_true',
        help='Run program as plain ipython (with timeout) and store stdout'
    )
    parser.add_argument(
        "-t",
        "--time",
        action="store_true",
        help="Save execution time of pipeline",
    )
    args = parser.parse_args()
    print(args)
    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
