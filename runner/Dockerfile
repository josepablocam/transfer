FROM ubuntu:18.04

# Basic prereqs
RUN apt update
RUN apt-get update
RUN apt-get install -y openjdk-8-jdk
RUN apt-get install -y wget
RUN apt-get install -y gnupg
RUN apt-get install -y zip
RUN apt-get install -y vim

# install/setup python
RUN apt-get install -y python3.7
RUN apt-get install -y python3-pip
RUN apt-get install 2to3
RUN ln -s $(which python3.7) /usr/bin/python
RUN ln -s $(which python3.7) /usr/bin/python3 --force
RUN ln -s $(which pip3) /usr/bin/pip
RUN pip install -U pip
RUN apt-get install -y git
# needed for ipython to work properly
RUN apt-get install python3-tk

# packages needed for kaggle scripts
ADD requirements.txt requirements.txt
# add true as xargs may result in 123 exit code
RUN (cat requirements.txt | grep -v "#" | xargs -I {} pip install {}) || true

# PL analysis tools
RUN git clone https://github.com/josepablocam/python-pl
RUN pip install -e python-pl

# Transfer analysis tools
ADD transfer.zip transfer.zip
RUN unzip transfer.zip -d transfer
RUN pip install -e transfer

# make sure can run matplotlib "headless"
# i.e. no $DISPLAY set
RUN echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc

# copy in data and source for script to execute
# top-level directory with [projectx/[scripts, input]]
ADD program_data/ program_data/

# copy in the driver
ADD run_pipeline.py run_pipeline.py

ENTRYPOINT ["python", "run_pipeline.py"]
