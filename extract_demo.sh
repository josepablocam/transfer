#!/usr/bin/env bash

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

# remove in case ran anytime before..
rm -rf demo-data/program-data/

# NOTE: you can replace the path for scripts, output folder etc
INPUT_SCRIPTS=$(ls demo-data/scripts/example*.py)
OUTPUT_DIR=demo-data/program-data/
OUTPUT_DB=sample_db.pkl
TIMEOUT=2m

mkdir -p ${OUTPUT_DIR}

# to point to your scripts
# collect info from each script in our demo scripts
for f in ${INPUT_SCRIPTS}
do
  echo "Extracting info from ${f}"
  python runner/run_pipeline.py \
    ${TIMEOUT} \
    ${f} \
    ${OUTPUT_DIR}
done

# Build database with newly extracted functions
python -m transfer.build_db \
  --function_files ${OUTPUT_DIR}/*functions* \
  --graph_files ${OUTPUT_DIR}/*graph* \
  --output ${OUTPUT_DB}
