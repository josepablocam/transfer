# Kaggle docker container config
FROM kaggle/python

RUN apt-get update
RUN apt-get install zip -y
RUN apt-get install vim -y

# Python analysis tools
RUN pip install git+git@github.com:josepablocam/python-pl.git

# Transfer analysis tools
RUN pip install -e ../

# install ipython to succesfully run
RUN pip install ipython

# copy in data and source for script to execute
# top-level directory with [projectx/[scripts, input]]
ADD program_data/ program_data/

# copy in the driver
ADD run_kaggle_program.py run_kaggle_program.py

ENTRYPOINT ["python", "run_kaggle_program.py"]
