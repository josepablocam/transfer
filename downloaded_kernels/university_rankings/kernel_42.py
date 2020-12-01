######### INSTRUCTIONS #########
#
# Fork this script and change the university name to see what rank it gets:
#
my_university_name = ["University of Toronto"]
#
# Look at the log for a full list of universities you can choose from.
#
# If your university is listed under multiple names, you can combine as many names as you want like this:
# my_university_name = ["The Johns Hopkins University", "Johns Hopkins University"]
#
################################
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.path as path

import matplotlib.patches as patches

from scipy import stats
import statsmodels.api as sm

import pylab as pl
# this allows plots to appear directly in the notebook

from sklearn.gaussian_process import GaussianProcessRegressor


# Import Data


shanghaiTable = pd.read_csv("../input/shanghaiData.csv")
shanghaiData = shanghaiTable.loc[shanghaiTable['year'] == 2015]
shanghaiData = shanghaiData.head(n=100)
feature = ['alumni','award','hici','ns','pub','pcp']
x = shanghaiData[feature]
y=shanghaiData['total_score']

model = GaussianProcessRegressor()
model.fit(x,y)
print(model.score(x,y))
