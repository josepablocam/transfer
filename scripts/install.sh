#!/usr/bin/env bash
set -e

source scripts/setup.sh
create_wranglesearch_conda_env
activate_wranglesearch_conda_env

# Setup wranglesearch
pip install -e .

# Run quick tests
bash test.sh

# Download pre-collected trace files
wget "${S3_BUCKET}/program_data.zip"
yes | unzip program_data.zip
rm -rf program_data.zip

# Download dataset used by programs in demo trace files
mkdir -p demo-data/
pushd demo-data
wget "${S3_BUCKET}/loan_data.zip"
yes | unzip loan_data.zip
rm -rf loan_data.zip
popd
