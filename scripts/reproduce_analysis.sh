#!/usr/bin/env bash
set -ex

source scripts/setup.sh

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

echo "Time analysis"
python -m analysis.pipeline_time_analysis \
--plain program_data/*/results-plain/*time.txt \
--full program_data/*/results/*time.txt \
--output_dir "${ANALYSIS_DIR}/time-results/" \
--max_plot 5.0 \
--timedout ${timedout} \
--timeout ${timeout_seconds} \
--rename "loan_data_v2:loan_data_2021"


# Function executability
DATASETS="house_sales loan_data loan_data_v2 university_rankings"
for d in ${DATASETS}
do
  echo "Function executability: ${d}"
  yes | head -n1 | bash build_demo.sh --dataset ${d}
  sleep 5s
  python -m analysis.function_executability \
    --data program_data/${d}/input/*.csv \
    --downsample_n 100 \
    --seed 42 \
    --output_dir "${ANALYSIS_DIR}/executability-results/${d}/"
done
