#!/usr/bin/env bash
source setup.sh
activate_wranglesearch_conda_env

set -e

DATASET="loan_data"

function help {
  echo "Usage: bash build_demo.sh [--dataset <name>]"
  exit 0
}

while [[ "$#" -gt 0 ]]
do
    case $1 in
        --dataset) shift; DATASET=${1}; shift;;
        -h|--help) shift; help;;
        *) echo "Unknown parameter: $1";exit 1;;
    esac
done

echo "Building demo using functions extracted for dataset=${DATASET}"

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
python -m wranglesearch.build_db \
  --function_files program_data/${DATASET}/results/*functions* \
  --graph_files program_data/${DATASET}/results/*graph* \
  --output sample_db.pkl
