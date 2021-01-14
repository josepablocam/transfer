#!/usr/bin/env bash
source ../setup.sh


MAX_JOBS=10
MEM_LIMIT=20GB
TIMEOUT=2h
DATASETS="house_sales loan_data_v2 university_rankings"


for d in ${DATASETS}
do
python schedule_jobs.py \
    --docker_image ${DOCKER_RUNNER_IMAGE} \
    --scripts program_data/${d}/scripts/kernel_*.py \
    --host_output_dir program_data/${d}/results \
    --docker_output_dir program_data/${d}/results/ \
    --mem_limit ${MEM_LIMIT} \
    --timeout ${TIMEOUT} \
    --max_jobs ${MAX_JOBS} \
    --time
done
