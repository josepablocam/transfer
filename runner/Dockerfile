# Kaggle docker container config
FROM gcr.io/kaggle-images/python:v10

# RUN apt-get install -y --no-install-recommends gnupg2 curl ca-certificates apt-transport-https
RUN apt-get update
RUN apt-get install zip -y
RUN apt-get install vim -y
# set up basic .vimrc file
RUN echo -e ":syntax on\n:set expandtab\n:set number" > ~/.vimrc

# RUN pip uninstall -y tensorflow
# Downgrade tensorflow to avoid core dump in Ubuntu VM
# Distributor ID: Ubuntu
# Description: Ubuntu 16.04.4 LTS
# Release: 16.04
# Codename: xenial
# root@daa5dbbc0f6b:/# python3 -c 'import tensorflow'
# 2018-05-04 18:09:06.153333: F tensorflow/core/platform/cpu_feature_guard.cc:36] The TensorFlow library was compiled to use AVX2 instructions, but these aren't available on your machine.
# Aborted (core dumped)
# Known issue: https://github.com/tensorflow/tensorflow/issues/17411
# RUN pip install tensorflow==1.15

# Python analysis tools
RUN git clone https://github.com/josepablocam/python-pl
RUN pip install -e python-pl

# Transfer analysis tools
ADD transfer.zip transfer.zip
RUN unzip transfer.zip -d transfer
RUN pip install -e transfer

# # install ipython to succesfully run
# RUN pip install ipython
#
# # copy in data and source for script to execute
# # top-level directory with [projectx/[scripts, input]]
# ADD program_data/ program_data/
#
# # copy in the driver
# ADD run_pipeline.py run_pipeline.py
#
# ENTRYPOINT ["python", "run_pipeline.py"]
