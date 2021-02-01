#!/usr/bin/env bash
export DOWNLOAD_KERNELS_FOLDER="downloaded_kernels/"
export KERNELS_RUNNER_FOLDER="runner/"
export S3_BUCKET="https://wranglesearch.s3.us-east-2.amazonaws.com"
export CONDA_WRANGLESEARCH_ENV="wranglesearch-env"
export DOCKER_RUNNER_IMAGE="wranglesearch"

function check_wranglesearch_conda_env {
  echo $(conda env list | grep ${CONDA_WRANGLESEARCH_ENV} | wc -l)
}

function activate_wranglesearch_conda_env {
    if command -v conda
    then
       config_path=$(realpath "$(dirname $(which conda))/../etc/profile.d")
       source "${config_path}/conda.sh"
       conda activate ${CONDA_WRANGLESEARCH_ENV}
    fi
}

function create_wranglesearch_conda_env {
  if command -v conda
  then
    if [ $(check_wranglesearch_conda_env) -eq 0 ]
    then
      # install if missing
      echo "Building conda environment: wranglesearch-env"
      yes | conda create -n ${CONDA_WRANGLESEARCH_ENV} python=3.7
    fi
  fi
}
