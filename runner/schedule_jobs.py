from argparse import ArgumentParser
import os
import re
import subprocess
import tempfile


def build_docker_command(docker_image, timeout, script_path, host_output_dir,
                         docker_output_dir, mem_limit, keep, plain):
    # to be able to mount volumes, must use absolute path
    host_output_dir = os.path.abspath(host_output_dir)
    # TODO: we should probably fix this
    docker_output_dir = os.path.join('/', docker_output_dir)
    container_name = script_path.replace('/', '_')
    docker_command = [
        'docker', 'run', '--name', container_name, '-v', '{}:{}'.format(
            host_output_dir, docker_output_dir), '--memory', mem_limit
    ]
    if not keep:
        docker_command += ['--rm']
    docker_command += [docker_image, timeout, script_path, docker_output_dir]
    if plain:
        docker_command += ['--plain']
    return docker_command


def set_max_jobs(n):
    assert n > 0
    tsp_command = "tsp -S {}".format(n)
    return_code = subprocess.call(tsp_command, shell=True)
    return return_code


def schedule(command):
    tsp_command = "tsp {}".format(command)
    return_code = subprocess.call(tsp_command, shell=True)
    return return_code


def create_job_commands(
        docker_image,
        scripts,
        host_output_dir,
        docker_output_dir,
        timeout,
        mem_limit,
        keep,
        plain,
):
    # and remove any scripts we already lifted etc
    scripts = [s for s in scripts if not s.endswith("lifted.py")]
    commands = []
    for script_path in scripts:
        cmd = build_docker_command(
            docker_image,
            timeout,
            script_path,
            host_output_dir,
            docker_output_dir,
            mem_limit,
            keep,
            plain,
        )
        cmd = " ".join(cmd)
        commands.append(cmd)
    return commands


def main(args):
    commands = create_job_commands(
        args.docker_image,
        args.scripts,
        args.host_output_dir,
        args.docker_output_dir,
        args.timeout,
        args.mem_limit,
        args.keep,
        args.plain,
    )
    set_max_jobs(args.max_jobs)
    for c in commands:
        schedule(c)


if __name__ == '__main__':
    parser = ArgumentParser(description='Schedule batch of Kaggle scripts')
    parser.add_argument(
        '--docker_image',
        type=str,
        help='Name for docker image to use to execute scripts')
    parser.add_argument(
        '--scripts', type=str, nargs="+", help='Scripts to run')
    parser.add_argument(
        '--host_output_dir',
        type=str,
        help='Path in host machine that maps to docker output path')
    parser.add_argument(
        '--docker_output_dir',
        type=str,
        help='Path in docker container to save any outputs generated')
    parser.add_argument(
        '-m',
        '--mem_limit',
        type=str,
        help='Maximum memory per docker container',
        default='20GB')
    parser.add_argument(
        '-t',
        '--timeout',
        type=str,
        help='Timeout per tracing portion',
        default='4h')
    parser.add_argument(
        '-k',
        '--keep',
        action='store_true',
        help='Keep container fs (i.e. run w/o --rm)')
    parser.add_argument(
        '-p',
        '--plain',
        action='store_true',
        help='Run program as plain ipython (with timeout) and store stdout')
    parser.add_argument(
        "--max_jobs",
        type=int,
        help="Max number of concurrent jobs in task-spooler",
        default=10,
    )
    args = parser.parse_args()
    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
