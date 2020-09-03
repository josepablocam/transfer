#!/usr/bin/env bash

# Start up the neo4j database
# used to store snippets/relationships
neo4j start || exit 1

# neo4j takes a bit of time to startup
echo "Wait for neo4j to start up"
sleep 10s

# Build database
python -m transfer.build_db \
  --function_files program_data/loan_data/results/*functions* \
  --graph_files program_data/loan_data/results/*graph* \
  --output sample_db.pkl
