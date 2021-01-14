#!/usr/bin/env bash
source setup.sh

ANALYSIS_DIR="analysis-results/"
mkdir -p ${ANALYSIS_DIR}


# Survey results analysis
python -m analysis.survey_analysis \
  --input prolific.csv \
  --output_dir "${ANALYSIS_DIR}/survey-results" \
  --prolific_meta prolific_info.csv

# Pipeline time analysis
DATASETS="house_sales university_rankings"
for d in ${DATASETS}
do
  echo "Time analysis: ${d}"
  python -m analysis.pipeline_time_analysis \
  --plain program_data/${d}/results-plain/*time.txt \
  --full program_data/${d}/results/*time.txt \
  --output_dir "${ANALYSIS_DIR}/time-result/${d}/" \
  --max_plot 5.0
done
