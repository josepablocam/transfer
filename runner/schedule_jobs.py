from argparse import ArgumentParser
import os
import re
import subprocess
import tempfile

def build_docker_command(docker_image, timeout, script_path, vm_output_dir, docker_output_dir, mem_limit):
    # to be able to mount volumes, must use absolute path
    vm_output_dir = os.path.abspath(vm_output_dir)
    # TODO: we should probably fix this
    docker_output_dir = os.path.join('/', docker_output_dir)
    container_name = script_path.replace('/', '_')
    docker_command = [
        'docker',
        'run',
        '--rm',
        '--name', container_name,
        '-v', '{}:{}'.format(vm_output_dir, docker_output_dir),
        '--memory', mem_limit,
        docker_image,
        timeout, script_path, docker_output_dir
    ]
    return docker_command

def schedule(command):
    at_command = ['at', '-q', 'b', '-m', 'now', '-f']
    # need to write out command to a temporary file
    # before feeding to at
    script_file =  tempfile.NamedTemporaryFile(suffix='.sh', delete=False)
    if not isinstance(command, str):
        command = ' '.join(command)
    print('Scheduling: {}'.format(command))
    script_file.write('#!/bin/bash\n')
    script_file.write('{}\n'.format(command))
    script_file.flush()
    script_file.close()
    full_command = at_command + [script_file.name]
    return_code =  subprocess.call(full_command)
    os.remove(script_file.name)
    return return_code

def setup_at_config(load_avg):
    load_avg_cmd = ['atd', '-l', str(load_avg)]
    return subprocess.call(load_avg_cmd)

def schedule_jobs(docker_image, scripts_dir, vm_output_dir, docker_output_dir, timeout, load_avg, mem_limit, regex_pattern):
    setup_at_config(load_avg)
    scripts = [s for s in os.listdir(scripts_dir) if s.split('.')[-1] == 'py']
    for script_name in scripts:
        script_path = os.path.join(scripts_dir, script_name)
        if regex_pattern is not None and re.match(regex_pattern, script_path) is None:
            continue
        cmd = build_docker_command(docker_image, timeout, script_path, vm_output_dir, docker_output_dir, mem_limit)
        schedule(cmd)

def main(args):
    schedule_jobs(
        args.docker_image,
        args.scripts_dir,
        args.vm_output_dir,
        args.docker_output_dir,
        args.timeout,
        args.load_average,
        args.mem_limit,
        args.regex
    )

if __name__ == '__main__':
    parser = ArgumentParser(description='Schedule batch of Kaggle scripts')
    parser.add_argument('docker_image', type=str, help='Name for docker image to use to execute scripts')
    parser.add_argument('scripts_dir', type=str, help='Path to directory with scripts')
    parser.add_argument('vm_output_dir', type=str, help='Path in VM that maps to docker output path')
    parser.add_argument('docker_output_dir', type=str, help='Path in docker container to save any outputs generated')
    parser.add_argument('-m', '--mem_limit', type=str, help='Maximum memory per docker container', default='20GB')
    parser.add_argument('-t', '--timeout', type=str, help='Timeout per tracing portion', default='4h')
    parser.add_argument('-l', '--load_average', type=float, help='Load average for atd command', default=5.0)
    parser.add_argument('-r', '--regex', type=str, help='Only schedule scripts with a path that matches the regex', default=None)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
