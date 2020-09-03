#!/usr/bin/env bash

# Setup transfer
pip install -e .

# Run quick tests
bash test.sh

# Download pre-collected trace files
wget https://transfer-snippets.s3.us-east-2.amazonaws.com/program_data.zip
unzip program_data.zip
rm -rf program_data.zip

# Download dataset used by programs in demo trace files
mkdir -p demo-data/
pushd demo-data
wget https://transfer-snippets.s3.us-east-2.amazonaws.com/loan.zip
unzip loan.zip
rm -rf loan.zip
popd
