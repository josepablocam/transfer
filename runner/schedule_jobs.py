from argparse import ArgumentParser
import os
import subprocess

def build_docker_command(script_path, vm_output_dir, docker_output_dir, mem_limit, timeout):
    docker_command = [
    'docker',
    '--name', script_path,
    'run',
    '-v', '{}:{}'.format(vm_output_dir, docker_output_dir),
    '--mem', mem_limit,
    script_path, docker_output_dir, timeout
    ]
    return docker_command

def schedule(command):
    at_command = ['at', '-q', 'b', '-m', 'now']
    full_command = at_command + command
    return subprocess.call(full_command)

def setup_at_config(load_avg):
    load_avg_cmd = ['atd', '-l', str(load_avg)]
    return subprocess.call(load_avg_cmd)

def schedule_jobs(scripts_dir, vm_output_dir, docker_output_dir, load_avg, mem_limit, timeout):
    setup_at_config(load_avg)
    scripts = os.listdir(scripts_dir)
    for script_name in scripts:
        script_path = os.path.join(scripts_dir, script_name)
        cmd = build_docker_command(script_path, vm_output_dir, docker_output_dir, mem_limit, timeout)
        schedule(cmd)

def main(args):
    schedule_jobs(
        args.scripts_dir,
        args.vm_output_dir,
        args.docker_output_dir,
        args.load_average,
        args.mem_limit,
        args.timeout,
    )

if __name__ == '__main__':
    parser = ArgumentParser(description='Schedule batch of Kaggle scripts')
    parser.add_argument('scripts_dir', type=str, help='Path to directory with scripts')
    parser.add_argument('docker_output_dir', type=str, help='Path in docker container to save any outputs generated')
    parser.add_argument('vm_output_dir', type=str, help='Path in VM that maps to docker output path')
    parser.add_argument('-m', '--mem_limit', type=str, help='Maximum memory per docker container', default='20GB')
    parser.add_argument('-t', '--timeout', type=str, help='Timeout per tracing portion', default='2h')
    parser.add_argument('-l', '--load_average', type=float, help='Load average for atd command', default=5.0)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
