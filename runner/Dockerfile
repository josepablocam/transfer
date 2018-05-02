# Kaggle docker container config
FROM kaggle/python

RUN apt-get update
RUN apt-get install zip -y
RUN apt-get install vim -y

# Python analysis tools
RUN git clone https://github.com/josepablocam/python-pl
RUN pip install -e python-pl

# Transfer analysis tools
ADD transfer-cleaning.zip transfer-cleaning.zip
RUN unzip transfer-cleaning.zip -d transfer-cleaning
RUN pip install -e transfer-cleaning

# install ipython to succesfully run
RUN pip install ipython

# copy in data and source for script to execute
# top-level directory with [projectx/[scripts, input]]
ADD program_data/ program_data/

# copy in the driver
ADD run_kaggle_program.py run_kaggle_program.py

ENTRYPOINT ["python", "run_kaggle_program.py"]
