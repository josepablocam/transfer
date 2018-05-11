from argparse import ArgumentParser
import os
import re
import subprocess
import tempfile

def build_docker_command(
    docker_image,
    timeout,
    script_path,
    vm_output_dir,
    docker_output_dir,
    mem_limit,
    keep,
    plain):
    # to be able to mount volumes, must use absolute path
    vm_output_dir = os.path.abspath(vm_output_dir)
    # TODO: we should probably fix this
    docker_output_dir = os.path.join('/', docker_output_dir)
    container_name = script_path.replace('/', '_')
    docker_command = [
        'docker',
        'run',
        '--name', container_name,
        '-v', '{}:{}'.format(vm_output_dir, docker_output_dir),
        '--memory', mem_limit
        ]
    if not keep:
        docker_command += ['--rm']
    docker_command += [
        docker_image,
        timeout, script_path, docker_output_dir
    ]
    if plain:
        docker_command += ['--plain']
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

def schedule_jobs(docker_image, scripts_dir, vm_output_dir, docker_output_dir, timeout, mem_limit, keep, regex_pattern, plain):
    scripts = [s for s in os.listdir(scripts_dir) if s.split('.')[-1] == 'py']
    for script_name in scripts:
        script_path = os.path.join(scripts_dir, script_name)
        if regex_pattern is not None and re.match(regex_pattern, script_path) is None:
            continue
        cmd = build_docker_command(docker_image, timeout, script_path, vm_output_dir, docker_output_dir, mem_limit, keep, plain)
        schedule(cmd)

def main(args):
    schedule_jobs(
        args.docker_image,
        args.scripts_dir,
        args.vm_output_dir,
        args.docker_output_dir,
        args.timeout,
        args.mem_limit,
        args.keep,
        args.regex,
        args.plain
    )

if __name__ == '__main__':
    parser = ArgumentParser(description='Schedule batch of Kaggle scripts')
    parser.add_argument('docker_image', type=str, help='Name for docker image to use to execute scripts')
    parser.add_argument('scripts_dir', type=str, help='Path to directory with scripts')
    parser.add_argument('vm_output_dir', type=str, help='Path in VM that maps to docker output path')
    parser.add_argument('docker_output_dir', type=str, help='Path in docker container to save any outputs generated')
    parser.add_argument('-m', '--mem_limit', type=str, help='Maximum memory per docker container', default='20GB')
    parser.add_argument('-t', '--timeout', type=str, help='Timeout per tracing portion', default='4h')
    parser.add_argument('-k', '--keep', action='store_true', help='Keep container fs (i.e. run w/o --rm)')
    parser.add_argument('-r', '--regex', type=str, help='Only schedule scripts with a path that matches the regex', default=None)
    parser.add_argument('-p', '--plain', action='store_true', help='Run program as plain ipython (with timeout) and store stdout')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
