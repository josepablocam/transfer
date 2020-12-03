# Setup to run Kaggle scripts on Rhino group server (rhino.csail.mit.edu)

We provide some utilities to build a vagrant-based VM (for isolation)
and Docker container (for convenience). We suggest using
the VM, but note that scripts execute more slowly there due to
expected overhead.

Instructions assume our group's machine, but they should
be easily adaptable to your own.


## Setting up a VM

1. Install Vagrant
    ```
    sudo apt-get install vagrant
    ```
2. Install vagrant disk-size plugin
    ```
    sudo vagrant plugin install vagrant-disksize
    ```
3. Clone the main repository to `/raid/` on Rhino (make sure nothing uses AFS)

4. Make sure that any data you want to run on is stored in
   `transfer/runner/program_data`.
   The directory structure assumed is:
   ```
program_data
└── <project_name>
    ├── input
    │   └── <data>.csv
    └── scripts
        ├── <script>.py
        ├── ...
```

This directory is copied on to the VM and then the Docker container within that.

You can prepare this directory structure by running

```
bash prepare_kaggle.sh
```

for the datasets we provide. This will also generate a `requirements.txt`
for the docker build (note we already include such a file,
but you can overwrite
it for your own set of scripts.)

5. Build the vagrant VM, used to sandbox the Kaggle scripts
    ```
    cd transfer/runner; make build_vagrant; vagrant halt
    ```
6. Start up the VM with `longjob` to remain logged in for a day (change this for longer)
    ```
    longjob --renew 1d vagrant up
    ```
    This step is specific to our group's machine, so feel free to skip.

7. Connect to the VM
    ```
    vagrant ssh
    ```

Note that you only need to build the VM if you want to true isolation,
otherwise you can just build the docker container.  

## Building Docker container
You can run this from within your VM, if you built one, or on
the original host machine

1. Run

`bash prepare_kaggle.sh`


This prepares the expected folder structure, validates which kernels we
stand a reasonable chance of running, and produces a requirements.txt
file with any missing packages that the docker build should try
to install to run kernels.

2. Build docker
    ```
    cd runner; make build_docker
    ```

Occasionally `make build_docker`may result in the following error:

```
Get https://registry-1.docker.io/v2/: net/http: TLS handshake timeout
```

If so, I recommend just running `make build_docker` again, and that
tends to solve the timeout.


You can now schedule jobs to running by using the command below
and modifying script locations etc as desired.
The actual scheduling is done with
`task-spooler` (which you will need to install if not available
on your system already). Timeout is
implemented with the `timeout` command.
The memory limit is handed directly by the
docker container that executes each job.

```
python schedule_jobs.py \
    --docker_image cleaning \
    --scripts program_data/loan_data/scripts/*.py \
    --host_output_dir program_data/loan_data/results \
    --docker_output_dir program_data/loan_data/results/ \
    --mem_limit 20GB \
    --timeout 2h \
    --max_jobs 2
```

Note that if you have already built the VM with all the data needed, and docker has been built accordingly, then
you can just skip to the last step directly.

You can set the number of jobs you'd like to run concurrently by using

```
tsp -S <num>
```

or by passing in the `--max_jobs` flag to `schedule_jobs.py` as done above.


### Reproduce our datasets
After you build the VM/docker as desired, you can run our kernels/datasets by executing

```
bash run_kaggle.sh
```

from the `runner/` folder.


After you have executed your desired scripts, the corresponding outputs
will be in `program_data/<dataset>/results`. You may want to
`sudo chown -R $(USER) program_data/<dataset>/results` as they will be written
using root user. You can then zip these up and download/move around at your convenience.

# Known Issues

* It seems that on occasion, `vagrant` can fail when building and not actually include docker. If this happens, I suggest removing the box (`vagrant destroy`), cleaning up, and calling `make build_vagrant` again. That seems to solve the issue in all cases I've encountered.

* If anything hangs for a long time, I suggest deleting the `.vagrant*`
folders created in the `transfer-cleaning/runner` folder. You may also want
to delete the `/raid/jcamsan/virtualbox_vms` folder as well. Also, kill any
`vboxmanage` or `vagrant` processes and then try again. I realize this may be
overkill (no pun intended) but not sure how to fix otherwise.
