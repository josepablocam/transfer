#!/usr/bin/env bash
source setup.sh
activate_transfer_conda_env

FORCE=0
while [[ "$#" -gt 0 ]]
do
    case $1 in
        --force) FORCE=1; shift;;
        *) echo "Unknown parameter: $1";exit 1;;
    esac
done

DATASETS="house_sales loan_data university_rankings"


# for d in ${DATASETS}
# do
#   echo "Preparing dataset: ${d}"
#
#   kernels_dir="${DOWNLOAD_KERNELS_FOLDER}/${d}"
#   converted_dir="${kernels_dir}/converted_notebooks"
#   parsed_dir="${kernels_dir}/parsed_kernels"
#
#   if [ -d ${parsed_dir} ] && [ ${FORCE} -eq 0 ]
#   then
#     echo "${parsed_dir} exists and FORCE=0, so skipping"
#     echo "use --force if you want to execute regardless"
#     continue
#   fi
#
#   mkdir -p ${converted_dir}
#   mkdir -p ${parsed_dir}
#
#   python -m transfer.convert_candidates \
#       ${kernels_dir} \
#       --converted_dir ${converted_dir} \
#       --parsed_dir ${parsed_dir}
# done
#
#
# # copy over files to to runner
# for d in ${DATASETS}
# do
#     runner_dir="${KERNELS_RUNNER_FOLDER}/program_data/${d}"
#     scripts_dir="${runner_dir}/scripts/"
#     data_dir="${runner_dir}/input/"
#
#     mkdir -p ${scripts_dir}
#     mkdir -p ${data_dir}
#
#     # copy source code over
#     cp -r ${DOWNLOAD_KERNELS_FOLDER}/${d}/parsed_kernels/*.py ${scripts_dir}
#     # download data set (need to put these up)
#     pushd ${data_dir}
#     wget "${S3_BUCKET}/${d}.zip"
#     yes | unzip "${d}.zip"
#     rm "${d}.zip"
#     popd
# done


# create a requirements.txt file with necessary installs
# for docker
all_scripts=""
for d in ${DATASETS}
do
    runner_dir="${KERNELS_RUNNER_FOLDER}/program_data/${d}"
    scripts_dir="${runner_dir}/scripts/"
    all_scripts+=" "
    all_scripts+=$(ls ${scripts_dir}/*.py)
done

python -m transfer.collect_requirements \
  --input ${all_scripts} \
  --filter > "${KERNELS_RUNNER_FOLDER}/requirements.txt"
