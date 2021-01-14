#!/usr/bin/env bash
source setup.sh

# Survey results analysis
python -m analysis.survey_analysis \
  --input prolific.csv \
  --output_dir survey-results \
  --prolific_meta prolific_info.csv

# Pipeline time analysis
for d in ${DATASETS}
do
  echo "Time analysis: ${d}"
  python -m analysis.pipeline_time_analysis \
  --plain runner/program_data/${d}/results-plain/*time.txt \
  --full runner/program_data/${d}/results/*time.txt \
  --output_dir time-result/${d}/
dine
