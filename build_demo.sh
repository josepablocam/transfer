#!/usr/bin/env bash
set -e

# Prompt to delete current neo4j database
read -p "This demo deletes your current neo4j database, are you sure you want to continue? [y/N] " yn


if [ ${yn} != "y" ]
then
  echo "Abort"
  exit 0
fi

# Clean up any existing neo4j db
database_path=$(neo4j start | grep data: | awk {'print $2}')
neo4j stop
echo "Deleting neo4j database..."
rm -rf ${database_path}
neo4j start
sleep 10s


# Build database
python -m transfer.build_db \
  --function_files program_data/loan_data/results/*functions* \
  --graph_files program_data/loan_data/results/*graph* \
  --output sample_db.pkl
