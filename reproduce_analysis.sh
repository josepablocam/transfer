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
DATASETS="house_sales loan_data_v2 university_rankings"

# Some instrumented scripts timed out at 2 hours
timedout="program_data_house_sales_scripts_kernel_57"
timedout+=" program_data_loan_data_v2_scripts_kernel_130.py"
timedout+=" program_data_loan_data_v2_scripts_kernel_142.py"
timedout+=" program_data_loan_data_v2_scripts_kernel_25.py"
timedout+=" program_data_loan_data_v2_scripts_kernel_34.py"
timeout_seconds=$((2 * 60 * 60))

for d in ${DATASETS}
do
  echo "Time analysis: ${d}"
  python -m analysis.pipeline_time_analysis \
  --plain program_data/${d}/results-plain/*time.txt \
  --full program_data/${d}/results/*time.txt \
  --output_dir "${ANALYSIS_DIR}/time-result/${d}/" \
  --max_plot 5.0 \
  --timedout ${timedout} \
  --timeout ${timeout_seconds}
done
