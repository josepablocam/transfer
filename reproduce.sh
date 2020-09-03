#!/usr/bin/env bash

# Setup transfer
pip install -e .

# Run quick tests
bash test.sh

# Download pre-collected trace files
wget https://transfer-snippets.s3.us-east-2.amazonaws.com/program_data.zip
unzip program_data.zip

# Start up the neo4j database
# used to store snippets/relationships
neo4j start

# neo4j takes a bit of time to startup
sleep 5s

# Build database
python -m transfer.build_db \
  --function_files program_data/loan_data/results/*functions* \
  --graph_files program_data/loan_data/results/*graph* \
  --output sample_db_info.pkl
