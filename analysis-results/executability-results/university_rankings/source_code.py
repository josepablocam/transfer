# Function 0
def cleaning_func_0(timesData):
	# core cleaning code
	my_university_name = ['University of Illinois at Urbana-Champaign']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['source'] = 'Times'
	return times_plot_data
#=============

# Function 1
def cleaning_func_2(shanghaiData):
	# core cleaning code
	my_university_name = ['University of Illinois at Urbana-Champaign']
	import pandas as pd
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data['source'] = 'Shanghai'
	return shanghai_plot_data
#=============

# Function 2
def cleaning_func_4(cwurData):
	# core cleaning code
	my_university_name = ['University of Illinois at Urbana-Champaign']
	import pandas as pd
	# cwurData = pd.read_csv('../input/cwurData.csv')
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data['source'] = 'CWUR'
	return cwur_plot_data
#=============

# Function 3
def cleaning_func_6(timesData):
	# core cleaning code
	my_university_name = ['University of Illinois at Urbana-Champaign']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 4
def cleaning_func_7(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Illinois at Urbana-Champaign']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 5
def cleaning_func_9(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Illinois at Urbana-Champaign']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return shanghaiData[shanghaiData.university_name.isin(my_university_name)]
#=============

# Function 6
def cleaning_func_11(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Illinois at Urbana-Champaign']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 7
def cleaning_func_12(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Illinois at Urbana-Champaign']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return cwurData[cwurData.institution.isin(my_university_name)]
#=============

# Function 8
def cleaning_func_16(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Illinois at Urbana-Champaign']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	plot_data = times_plot_data.append(shanghai_plot_data).append(cwur_plot_data)
	plot_data['world_rank'] = plot_data['world_rank'].astype(int)
	return plot_data
#=============

# Function 9
def cleaning_func_1(timesData):
	# core cleaning code
	my_university_name = ['University of Illinois at Urbana-Champaign']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	return timesData[timesData.university_name.isin(my_university_name)]
#=============

# Function 10
def cleaning_func_17(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Illinois at Urbana-Champaign']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	return cwur_plot_data
#=============

# Function 11
def cleaning_func_0(df):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# df = pd.read_csv('../input/timesData.csv')
	df['income'] = df['income'].replace('-', np.NaN)
	return df
#=============

# Function 12
def cleaning_func_0(timesData):
	# core cleaning code
	my_university_name = ['Duke University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	return timesData[timesData.university_name.isin(my_university_name)]
#=============

# Function 13
def cleaning_func_1(timesData):
	# core cleaning code
	my_university_name = ['Duke University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['source'] = 'Times'
	return times_plot_data
#=============

# Function 14
def cleaning_func_2(shanghaiData):
	# core cleaning code
	my_university_name = ['Duke University']
	import pandas as pd
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data['source'] = 'Shanghai'
	return shanghai_plot_data
#=============

# Function 15
def cleaning_func_5(cwurData):
	# core cleaning code
	my_university_name = ['Duke University']
	import pandas as pd
	# cwurData = pd.read_csv('../input/cwurData.csv')
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data['source'] = 'CWUR'
	return cwur_plot_data
#=============

# Function 16
def cleaning_func_6(timesData):
	# core cleaning code
	my_university_name = ['Duke University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 17
def cleaning_func_7(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Duke University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	return shanghai_plot_data
#=============

# Function 18
def cleaning_func_9(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Duke University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return shanghaiData[shanghaiData.university_name.isin(my_university_name)]
#=============

# Function 19
def cleaning_func_11(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Duke University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data.append(shanghai_plot_data)
#=============

# Function 20
def cleaning_func_12(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Duke University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return cwurData[cwurData.institution.isin(my_university_name)]
#=============

# Function 21
def cleaning_func_13(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Duke University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	return cwur_plot_data
#=============

# Function 22
def cleaning_func_15(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Duke University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	plot_data = times_plot_data.append(shanghai_plot_data).append(cwur_plot_data)
	plot_data['world_rank'] = plot_data['world_rank'].astype(int)
	return plot_data
#=============

# Function 23
def cleaning_func_0(timesData):
	# core cleaning code
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	times_plot_data['source'] = 'Times'
	return times_plot_data
#=============

# Function 24
def cleaning_func_1(timesData):
	# core cleaning code
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	return timesData[timesData.university_name.isin(['University of Texas at Dallas'])]
#=============

# Function 25
def cleaning_func_2(shanghaiData):
	# core cleaning code
	import pandas as pd
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	shanghai_plot_data['source'] = 'Shanghai'
	return shanghai_plot_data
#=============

# Function 26
def cleaning_func_5(cwurData):
	# core cleaning code
	import pandas as pd
	# cwurData = pd.read_csv('../input/cwurData.csv')
	cwur_plot_data = cwurData[cwurData.institution.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	cwur_plot_data['source'] = 'CWUR'
	return cwur_plot_data
#=============

# Function 27
def cleaning_func_6(timesData):
	# core cleaning code
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 28
def cleaning_func_8(shanghaiData,timesData):
	# core cleaning code
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 29
def cleaning_func_10(shanghaiData,timesData):
	# core cleaning code
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	return shanghaiData[shanghaiData.university_name.isin(['University of Texas at Dallas'])]
#=============

# Function 30
def cleaning_func_12(cwurData,shanghaiData,timesData):
	# core cleaning code
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data.append(shanghai_plot_data)
#=============

# Function 31
def cleaning_func_13(cwurData,shanghaiData,timesData):
	# core cleaning code
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	return cwur_plot_data
#=============

# Function 32
def cleaning_func_16(cwurData,shanghaiData,timesData):
	# core cleaning code
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	return cwurData[cwurData.institution.isin(['University of Texas at Dallas'])]
#=============

# Function 33
def cleaning_func_18(cwurData,shanghaiData,timesData):
	# core cleaning code
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(['University of Texas at Dallas'])][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	plot_data = times_plot_data.append(shanghai_plot_data).append(cwur_plot_data)
	plot_data['world_rank'] = plot_data['world_rank'].astype(int)
	return plot_data
#=============

# Function 34
def cleaning_func_0(timesData):
	# core cleaning code
	my_university_name = ['City University of New York City College']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	return timesData[timesData.university_name.isin(my_university_name)]
#=============

# Function 35
def cleaning_func_1(timesData):
	# core cleaning code
	my_university_name = ['City University of New York City College']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['source'] = 'Times'
	return times_plot_data
#=============

# Function 36
def cleaning_func_3(shanghaiData):
	# core cleaning code
	my_university_name = ['City University of New York City College']
	import pandas as pd
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data['source'] = 'Shanghai'
	return shanghai_plot_data
#=============

# Function 37
def cleaning_func_5(cwurData):
	# core cleaning code
	my_university_name = ['City University of New York City College']
	import pandas as pd
	# cwurData = pd.read_csv('../input/cwurData.csv')
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data['source'] = 'CWUR'
	return cwur_plot_data
#=============

# Function 38
def cleaning_func_6(timesData):
	# core cleaning code
	my_university_name = ['City University of New York City College']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 39
def cleaning_func_7(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['City University of New York City College']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	return shanghai_plot_data
#=============

# Function 40
def cleaning_func_8(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['City University of New York City College']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return shanghaiData[shanghaiData.university_name.isin(my_university_name)]
#=============

# Function 41
def cleaning_func_11(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['City University of New York City College']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	return shanghai_plot_data
#=============

# Function 42
def cleaning_func_12(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['City University of New York City College']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return cwurData[cwurData.institution.isin(my_university_name)]
#=============

# Function 43
def cleaning_func_15(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['City University of New York City College']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	return cwur_plot_data
#=============

# Function 44
def cleaning_func_16(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['City University of New York City College']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	plot_data = times_plot_data.append(shanghai_plot_data).append(cwur_plot_data)
	plot_data['world_rank'] = plot_data['world_rank'].astype(int)
	return plot_data
#=============

# Function 45
def cleaning_func_0(shanghai):
	# core cleaning code
	import pandas as pd
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	shanghai.world_rank = [(int(x.split('-')[0]) if (type(x) == str) else x) for x in shanghai.world_rank]
	return shanghai
#=============

# Function 46
def cleaning_func_0(timesData):
	# core cleaning code
	my_university_name = ['University of Toronto']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	return timesData[timesData.university_name.isin(my_university_name)]
#=============

# Function 47
def cleaning_func_1(timesData):
	# core cleaning code
	my_university_name = ['University of Toronto']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['source'] = 'Times'
	return times_plot_data
#=============

# Function 48
def cleaning_func_3(shanghaiData):
	# core cleaning code
	my_university_name = ['University of Toronto']
	import pandas as pd
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data['source'] = 'Shanghai'
	return shanghai_plot_data
#=============

# Function 49
def cleaning_func_5(cwurData):
	# core cleaning code
	my_university_name = ['University of Toronto']
	import pandas as pd
	# cwurData = pd.read_csv('../input/cwurData.csv')
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data['source'] = 'CWUR'
	return cwur_plot_data
#=============

# Function 50
def cleaning_func_6(timesData):
	# core cleaning code
	my_university_name = ['University of Toronto']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 51
def cleaning_func_8(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Toronto']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 52
def cleaning_func_9(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Toronto']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return shanghaiData[shanghaiData.university_name.isin(my_university_name)]
#=============

# Function 53
def cleaning_func_11(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Toronto']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	return cwur_plot_data
#=============

# Function 54
def cleaning_func_13(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Toronto']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	return shanghai_plot_data
#=============

# Function 55
def cleaning_func_14(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Toronto']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	plot_data = times_plot_data.append(shanghai_plot_data).append(cwur_plot_data)
	plot_data['world_rank'] = plot_data['world_rank'].astype(int)
	return plot_data
#=============

# Function 56
def cleaning_func_16(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Toronto']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return cwurData[cwurData.institution.isin(my_university_name)]
#=============

# Function 57
def cleaning_func_0(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/timesData.csv')
	odf_cp5 = df.copy()
	f = (lambda x: (int(((int(x.split('-')[0]) + int(x.split('-')[1])) / 2)) if (len(str(x).strip()) > 3) else x))
	odf_cp5['world_rank'] = odf_cp5['world_rank'].str.replace('=', '').map(f).astype('float')
	return odf_cp5
#=============

# Function 58
def cleaning_func_1(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/timesData.csv')
	odf_cp2 = df.copy()
	odf_cp2['num_students'] = odf_cp2['num_students'].str.replace(',', '')
	odf_cp2['num_students'] = odf_cp2['num_students'].astype(np.float)
	return odf_cp2
#=============

# Function 59
def cleaning_func_2(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/timesData.csv')
	odf_cp1 = df.copy()
	odf_cp1['international_students'] = odf_cp1['international_students'].str.replace('%', '')
	odf_cp1['international_students'] = odf_cp1['international_students'].astype(np.float)
	return odf_cp1
#=============

# Function 60
def cleaning_func_3(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/timesData.csv')
	odf_cp4 = df.copy()
	return odf_cp4
#=============

# Function 61
def cleaning_func_4(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/timesData.csv')
	odf_cp4 = df.copy()
	odf_cp4 = odf_cp4[(odf_cp4['total_score'].str.len() > 1)]
	odf_cp4['total_score'] = odf_cp4['total_score'].astype(np.float)
	return odf_cp4
#=============

# Function 62
def cleaning_func_5(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/timesData.csv')
	odf_cp3 = df.copy()
	odf_cp3 = odf_cp3[(odf_cp3['female_male_ratio'].str.len() > 0)]
	odf_cp3['female_male_ratio'] = odf_cp3['female_male_ratio'].str.replace('-', '0')
	odf_cp3['female_male_ratio'] = odf_cp3['female_male_ratio'].str.split(':', expand=True)
	odf_cp3['female_male_ratio'] = odf_cp3['female_male_ratio'].str[slice(0, 2, None)]
	odf_cp3['female_male_ratio'] = odf_cp3['female_male_ratio'].astype(np.float)
	return odf_cp3
#=============

# Function 63
def cleaning_func_7(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/timesData.csv')
	odf_cp5 = df.copy()
	f = (lambda x: (int(((int(x.split('-')[0]) + int(x.split('-')[1])) / 2)) if (len(str(x).strip()) > 3) else x))
	odf_cp5['world_rank'] = odf_cp5['world_rank'].str.replace('=', '').map(f).astype('float')
	odf_cp4 = df.copy()
	return odf_cp4
#=============

# Function 64
def cleaning_func_9(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/timesData.csv')
	odf_cp5 = df.copy()
	f = (lambda x: (int(((int(x.split('-')[0]) + int(x.split('-')[1])) / 2)) if (len(str(x).strip()) > 3) else x))
	odf_cp5['world_rank'] = odf_cp5['world_rank'].str.replace('=', '').map(f).astype('float')
	odf_cp4 = df.copy()
	odf_cp4 = odf_cp4[(odf_cp4['total_score'].str.len() > 1)]
	odf_cp4['total_score'] = odf_cp4['total_score'].astype(np.float)
	f = (lambda x: (int(((int(x.split('-')[0]) + int(x.split('-')[1])) / 2)) if (len(str(x).strip()) > 3) else x))
	odf_cp4['world_rank'] = odf_cp4['world_rank'].str.replace('=', '').map(f).astype('float')
	return odf_cp4
#=============

# Function 65
def cleaning_func_0(shanghaiData):
	# core cleaning code
	my_university_name = ['University of Cambridge']
	import pandas as pd
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	return shanghaiData[shanghaiData.university_name.isin(my_university_name)]
#=============

# Function 66
def cleaning_func_1(shanghaiData):
	# core cleaning code
	my_university_name = ['University of Cambridge']
	import pandas as pd
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data['source'] = 'Shanghai'
	return shanghai_plot_data
#=============

# Function 67
def cleaning_func_2(cwurData):
	# core cleaning code
	my_university_name = ['University of Cambridge']
	import pandas as pd
	# cwurData = pd.read_csv('../input/cwurData.csv')
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data['source'] = 'CWUR'
	return cwur_plot_data
#=============

# Function 68
def cleaning_func_5(timesData):
	# core cleaning code
	my_university_name = ['University of Cambridge']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['source'] = 'Times'
	return times_plot_data
#=============

# Function 69
def cleaning_func_6(timesData):
	# core cleaning code
	my_university_name = ['University of Cambridge']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 70
def cleaning_func_7(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Cambridge']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return shanghaiData[shanghaiData.university_name.isin(my_university_name)]
#=============

# Function 71
def cleaning_func_8(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Cambridge']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	return shanghai_plot_data
#=============

# Function 72
def cleaning_func_12(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Cambridge']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	return shanghai_plot_data
#=============

# Function 73
def cleaning_func_14(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Cambridge']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	plot_data = times_plot_data.append(shanghai_plot_data).append(cwur_plot_data)
	plot_data['world_rank'] = plot_data['world_rank'].astype(int)
	return plot_data
#=============

# Function 74
def cleaning_func_16(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Cambridge']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	return cwur_plot_data
#=============

# Function 75
def cleaning_func_18(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Cambridge']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return cwurData[cwurData.institution.isin(my_university_name)]
#=============

# Function 76
def cleaning_func_0(timesData):
	# core cleaning code
	my_university_name = ['Korea University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	return timesData[timesData.university_name.isin(my_university_name)]
#=============

# Function 77
def cleaning_func_1(timesData):
	# core cleaning code
	my_university_name = ['Korea University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['source'] = 'Times'
	return times_plot_data
#=============

# Function 78
def cleaning_func_2(shanghaiData):
	# core cleaning code
	my_university_name = ['Korea University']
	import pandas as pd
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data['source'] = 'Shanghai'
	return shanghai_plot_data
#=============

# Function 79
def cleaning_func_4(cwurData):
	# core cleaning code
	my_university_name = ['Korea University']
	import pandas as pd
	# cwurData = pd.read_csv('../input/cwurData.csv')
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data['source'] = 'CWUR'
	return cwur_plot_data
#=============

# Function 80
def cleaning_func_6(timesData):
	# core cleaning code
	my_university_name = ['Korea University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 81
def cleaning_func_8(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Korea University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return shanghaiData[shanghaiData.university_name.isin(my_university_name)]
#=============

# Function 82
def cleaning_func_9(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Korea University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	return shanghai_plot_data
#=============

# Function 83
def cleaning_func_12(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Korea University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return cwurData[cwurData.institution.isin(my_university_name)]
#=============

# Function 84
def cleaning_func_13(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Korea University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	plot_data = times_plot_data.append(shanghai_plot_data).append(cwur_plot_data)
	plot_data['world_rank'] = plot_data['world_rank'].astype(int)
	return plot_data
#=============

# Function 85
def cleaning_func_14(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Korea University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 86
def cleaning_func_18(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Korea University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	return cwur_plot_data
#=============

# Function 87
def cleaning_func_0(timesData):
	# core cleaning code
	my_university_name = ['University of Waterloo']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['source'] = 'Times'
	return times_plot_data
#=============

# Function 88
def cleaning_func_1(timesData):
	# core cleaning code
	my_university_name = ['University of Waterloo']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	return timesData[timesData.university_name.isin(my_university_name)]
#=============

# Function 89
def cleaning_func_3(shanghaiData):
	# core cleaning code
	my_university_name = ['University of Waterloo']
	import pandas as pd
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data['source'] = 'Shanghai'
	return shanghai_plot_data
#=============

# Function 90
def cleaning_func_4(cwurData):
	# core cleaning code
	my_university_name = ['University of Waterloo']
	import pandas as pd
	# cwurData = pd.read_csv('../input/cwurData.csv')
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data['source'] = 'CWUR'
	return cwur_plot_data
#=============

# Function 91
def cleaning_func_6(timesData):
	# core cleaning code
	my_university_name = ['University of Waterloo']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 92
def cleaning_func_7(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Waterloo']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 93
def cleaning_func_9(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Waterloo']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return shanghaiData[shanghaiData.university_name.isin(my_university_name)]
#=============

# Function 94
def cleaning_func_11(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Waterloo']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data.append(shanghai_plot_data)
#=============

# Function 95
def cleaning_func_12(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Waterloo']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	plot_data = times_plot_data.append(shanghai_plot_data).append(cwur_plot_data)
	plot_data['world_rank'] = plot_data['world_rank'].astype(int)
	return plot_data
#=============

# Function 96
def cleaning_func_15(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Waterloo']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return cwurData[cwurData.institution.isin(my_university_name)]
#=============

# Function 97
def cleaning_func_18(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Waterloo']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	return cwur_plot_data
#=============

# Function 98
def cleaning_func_0(timesData):
	# core cleaning code
	my_university_name = ['Indiana University Bloomington']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	return timesData[timesData.university_name.isin(my_university_name)]
#=============

# Function 99
def cleaning_func_1(timesData):
	# core cleaning code
	my_university_name = ['Indiana University Bloomington']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['source'] = 'Times'
	return times_plot_data
#=============

# Function 100
def cleaning_func_3(shanghaiData):
	# core cleaning code
	my_university_name = ['Indiana University Bloomington']
	import pandas as pd
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data['source'] = 'Shanghai'
	return shanghai_plot_data
#=============

# Function 101
def cleaning_func_4(cwurData):
	# core cleaning code
	my_university_name = ['Indiana University Bloomington']
	import pandas as pd
	# cwurData = pd.read_csv('../input/cwurData.csv')
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data['source'] = 'CWUR'
	return cwur_plot_data
#=============

# Function 102
def cleaning_func_6(timesData):
	# core cleaning code
	my_university_name = ['Indiana University Bloomington']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 103
def cleaning_func_7(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Indiana University Bloomington']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return shanghaiData[shanghaiData.university_name.isin(my_university_name)]
#=============

# Function 104
def cleaning_func_9(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Indiana University Bloomington']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	return shanghai_plot_data
#=============

# Function 105
def cleaning_func_12(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Indiana University Bloomington']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data.append(shanghai_plot_data)
#=============

# Function 106
def cleaning_func_13(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Indiana University Bloomington']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	plot_data = times_plot_data.append(shanghai_plot_data).append(cwur_plot_data)
	plot_data['world_rank'] = plot_data['world_rank'].astype(int)
	return plot_data
#=============

# Function 107
def cleaning_func_14(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Indiana University Bloomington']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	return cwur_plot_data
#=============

# Function 108
def cleaning_func_16(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Indiana University Bloomington']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return cwurData[cwurData.institution.isin(my_university_name)]
#=============

# Function 109
def cleaning_func_0(timesData):
	# core cleaning code
	my_university_name = ['University of California, Berkeley', 'University of California-Berkeley']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['source'] = 'Times'
	return times_plot_data
#=============

# Function 110
def cleaning_func_1(timesData):
	# core cleaning code
	my_university_name = ['University of California, Berkeley', 'University of California-Berkeley']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	return timesData[timesData.university_name.isin(my_university_name)]
#=============

# Function 111
def cleaning_func_2(shanghaiData):
	# core cleaning code
	my_university_name = ['University of California, Berkeley', 'University of California-Berkeley']
	import pandas as pd
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data['source'] = 'Shanghai'
	return shanghai_plot_data
#=============

# Function 112
def cleaning_func_5(cwurData):
	# core cleaning code
	my_university_name = ['University of California, Berkeley', 'University of California-Berkeley']
	import pandas as pd
	# cwurData = pd.read_csv('../input/cwurData.csv')
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data['source'] = 'CWUR'
	return cwur_plot_data
#=============

# Function 113
def cleaning_func_6(timesData):
	# core cleaning code
	my_university_name = ['University of California, Berkeley', 'University of California-Berkeley']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 114
def cleaning_func_7(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of California, Berkeley', 'University of California-Berkeley']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 115
def cleaning_func_8(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of California, Berkeley', 'University of California-Berkeley']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return shanghaiData[shanghaiData.university_name.isin(my_university_name)]
#=============

# Function 116
def cleaning_func_11(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of California, Berkeley', 'University of California-Berkeley']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 117
def cleaning_func_12(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of California, Berkeley', 'University of California-Berkeley']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return cwurData[cwurData.institution.isin(my_university_name)]
#=============

# Function 118
def cleaning_func_14(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of California, Berkeley', 'University of California-Berkeley']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	return cwur_plot_data
#=============

# Function 119
def cleaning_func_16(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of California, Berkeley', 'University of California-Berkeley']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	plot_data = times_plot_data.append(shanghai_plot_data).append(cwur_plot_data)
	plot_data['world_rank'] = plot_data['world_rank'].astype(int)
	return plot_data
#=============

# Function 120
def cleaning_func_0(df_sac):
	# core cleaning code
	import pandas as pd
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	df_sac.columns = ['university_name', 'country']
	return df_sac
#=============

# Function 121
def cleaning_func_1(df):
	# core cleaning code
	import pandas as pd
	# df_times = pd.read_csv('../input/timesData.csv')
	a = []
	df.columns = a
	return df
#=============

# Function 122
def cleaning_func_2(df,df_cwur,df_sac):
	# core cleaning code
	import pandas as pd
	# df_cwur = pd.read_csv('../input/cwurData.csv')
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	df_cwur = df_cwur.rename(columns={'institution': 'university_name'})
	return df_cwur
#=============

# Function 123
def cleaning_func_3(df,df_cwur,df_sac):
	# core cleaning code
	import pandas as pd
	# df_cwur = pd.read_csv('../input/cwurData.csv')
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	df_cwur = df_cwur.rename(columns={'institution': 'university_name'})
	df_cwur = df_cwur.merge(df_sac, how='left', on='university_name')
	return df_cwur
#=============

# Function 124
def cleaning_func_4(df,df_cwur,df_sac):
	# core cleaning code
	import pandas as pd
	# df_cwur = pd.read_csv('../input/cwurData.csv')
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	df_cwur = df_cwur.rename(columns={'institution': 'university_name'})
	df_cwur = df_cwur.merge(df_sac, how='left', on='university_name')
	a = []
	df.columns = a
	return df
#=============

# Function 125
def cleaning_func_5(df_cwur,df_sac,df_shanghai,df_times):
	# core cleaning code
	import pandas as pd
	# df_cwur = pd.read_csv('../input/cwurData.csv')
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	# df_shanghai = pd.read_csv('../input/shanghaiData.csv')
	# df_times = pd.read_csv('../input/timesData.csv')
	df_shanghai = df_shanghai.merge(df_sac, how='left', on='university_name')
	return df_shanghai
#=============

# Function 126
def cleaning_func_6(df_cwur,df_sac,df_shanghai,df_times):
	# core cleaning code
	import pandas as pd
	# df_cwur = pd.read_csv('../input/cwurData.csv')
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	# df_shanghai = pd.read_csv('../input/shanghaiData.csv')
	# df_times = pd.read_csv('../input/timesData.csv')
	df_shanghai = df_shanghai.merge(df_sac, how='left', on='university_name')
	df_cwur = df_cwur.rename(columns={'institution': 'university_name'})
	df_cwur = df_cwur.merge(df_sac, how='left', on='university_name')
	df_full = df_times.merge(df_cwur, how='outer', on=['university_name', 'year'])
	df_full = df_full.merge(df_shanghai, how='outer', on=['university_name', 'year'])
	df_ranks = df_full[['university_name', 't_country', 'year', 't_world_rank', 'c_world_rank', 's_world_rank']].copy()
	return df_ranks[(df_ranks.year == 2015)]
#=============

# Function 127
def cleaning_func_7(df_cwur,df_sac,df_shanghai,df_times):
	# additional context code from user definitions

	def f(x):
	    a = []
	    for i in ['t_world_rank', 's_world_rank', 'c_world_rank']:
	        try:
	            if (x[i] == float(x[i])):
	                a.append(x[i])
	        except:
	            pass
	    return min(a)

	# core cleaning code
	import pandas as pd
	# df_cwur = pd.read_csv('../input/cwurData.csv')
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	# df_shanghai = pd.read_csv('../input/shanghaiData.csv')
	# df_times = pd.read_csv('../input/timesData.csv')
	df_shanghai = df_shanghai.merge(df_sac, how='left', on='university_name')
	df_cwur = df_cwur.rename(columns={'institution': 'university_name'})
	df_cwur = df_cwur.merge(df_sac, how='left', on='university_name')
	df_full = df_times.merge(df_cwur, how='outer', on=['university_name', 'year'])
	df_full = df_full.merge(df_shanghai, how='outer', on=['university_name', 'year'])
	df_ranks = df_full[['university_name', 't_country', 'year', 't_world_rank', 'c_world_rank', 's_world_rank']].copy()
	df_ranks2015 = df_ranks[(df_ranks.year == 2015)].copy()
	df_ranks2015['min_rank'] = df_ranks2015.apply(f, axis=1)
	return df_ranks2015
#=============

# Function 128
def cleaning_func_8(df_cwur,df_sac,df_shanghai,df_times):
	# core cleaning code
	import pandas as pd
	# df_cwur = pd.read_csv('../input/cwurData.csv')
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	# df_shanghai = pd.read_csv('../input/shanghaiData.csv')
	# df_times = pd.read_csv('../input/timesData.csv')
	df_shanghai = df_shanghai.merge(df_sac, how='left', on='university_name')
	df_cwur = df_cwur.rename(columns={'institution': 'university_name'})
	return df_cwur
#=============

# Function 129
def cleaning_func_10(df_cwur,df_sac,df_shanghai,df_times):
	# core cleaning code
	import pandas as pd
	# df_cwur = pd.read_csv('../input/cwurData.csv')
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	# df_shanghai = pd.read_csv('../input/shanghaiData.csv')
	# df_times = pd.read_csv('../input/timesData.csv')
	df_shanghai = df_shanghai.merge(df_sac, how='left', on='university_name')
	df_cwur = df_cwur.rename(columns={'institution': 'university_name'})
	df_cwur = df_cwur.merge(df_sac, how='left', on='university_name')
	df_full = df_times.merge(df_cwur, how='outer', on=['university_name', 'year'])
	df_full = df_full.merge(df_shanghai, how='outer', on=['university_name', 'year'])
	return df_full[['university_name', 't_country', 'year', 't_world_rank', 'c_world_rank', 's_world_rank']]
#=============

# Function 130
def cleaning_func_14(df_cwur,df_sac,df_shanghai,df_times):
	# additional context code from user definitions

	def f(x):
	    a = []
	    for i in ['t_world_rank', 's_world_rank', 'c_world_rank']:
	        try:
	            if (x[i] == float(x[i])):
	                a.append(x[i])
	        except:
	            pass
	    return min(a)

	# core cleaning code
	import pandas as pd
	# df_cwur = pd.read_csv('../input/cwurData.csv')
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	# df_shanghai = pd.read_csv('../input/shanghaiData.csv')
	# df_times = pd.read_csv('../input/timesData.csv')
	df_shanghai = df_shanghai.merge(df_sac, how='left', on='university_name')
	df_cwur = df_cwur.rename(columns={'institution': 'university_name'})
	df_cwur = df_cwur.merge(df_sac, how='left', on='university_name')
	df_full = df_times.merge(df_cwur, how='outer', on=['university_name', 'year'])
	df_full = df_full.merge(df_shanghai, how='outer', on=['university_name', 'year'])
	df_ranks = df_full[['university_name', 't_country', 'year', 't_world_rank', 'c_world_rank', 's_world_rank']].copy()
	f = (lambda x: (int(((int(x.split('-')[0]) + int(x.split('-')[1])) / 2)) if (len(str(x).strip()) > 3) else x))
	df_ranks['s_world_rank'] = df_ranks['s_world_rank'].str.replace('=', '').map(f).astype('float')
	return df_ranks
#=============

# Function 131
def cleaning_func_17(df_cwur,df_sac,df_shanghai,df_times):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# df_cwur = pd.read_csv('../input/cwurData.csv')
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	# df_shanghai = pd.read_csv('../input/shanghaiData.csv')
	# df_times = pd.read_csv('../input/timesData.csv')
	df_shanghai = df_shanghai.merge(df_sac, how='left', on='university_name')
	df_cwur = df_cwur.rename(columns={'institution': 'university_name'})
	df_cwur = df_cwur.merge(df_sac, how='left', on='university_name')
	df_full = df_times.merge(df_cwur, how='outer', on=['university_name', 'year'])
	df_full = df_full.merge(df_shanghai, how='outer', on=['university_name', 'year'])
	df_ranks = df_full[['university_name', 't_country', 'year', 't_world_rank', 'c_world_rank', 's_world_rank']].copy()
	df_ranks2015 = df_ranks[(df_ranks.year == 2015)].copy()
	df_ranks2015['std_dev'] = df_ranks2015.apply((lambda x: np.std([x['s_world_rank'], x['t_world_rank'], x['c_world_rank']])), axis=1)
	return df_ranks2015
#=============

# Function 132
def cleaning_func_25(df_cwur,df_sac,df_shanghai,df_times):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# df_cwur = pd.read_csv('../input/cwurData.csv')
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	# df_shanghai = pd.read_csv('../input/shanghaiData.csv')
	# df_times = pd.read_csv('../input/timesData.csv')
	df_shanghai = df_shanghai.merge(df_sac, how='left', on='university_name')
	df_cwur = df_cwur.rename(columns={'institution': 'university_name'})
	df_cwur = df_cwur.merge(df_sac, how='left', on='university_name')
	df_full = df_times.merge(df_cwur, how='outer', on=['university_name', 'year'])
	df_full = df_full.merge(df_shanghai, how='outer', on=['university_name', 'year'])
	df_ranks = df_full[['university_name', 't_country', 'year', 't_world_rank', 'c_world_rank', 's_world_rank']].copy()
	df_ranks2015 = df_ranks[(df_ranks.year == 2015)].copy()
	df_ranks2015['mean_rank'] = df_ranks2015.apply((lambda x: np.mean([x['s_world_rank'], x['t_world_rank'], x['c_world_rank']]).round()), axis=1)
	return df_ranks2015
#=============

# Function 133
def cleaning_func_26(df_cwur,df_sac,df_shanghai,df_times):
	# core cleaning code
	import pandas as pd
	# df_cwur = pd.read_csv('../input/cwurData.csv')
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	# df_shanghai = pd.read_csv('../input/shanghaiData.csv')
	# df_times = pd.read_csv('../input/timesData.csv')
	df_shanghai = df_shanghai.merge(df_sac, how='left', on='university_name')
	df_cwur = df_cwur.rename(columns={'institution': 'university_name'})
	df_cwur = df_cwur.merge(df_sac, how='left', on='university_name')
	df_full = df_times.merge(df_cwur, how='outer', on=['university_name', 'year'])
	df_full = df_full.merge(df_shanghai, how='outer', on=['university_name', 'year'])
	df_ranks = df_full[['university_name', 't_country', 'year', 't_world_rank', 'c_world_rank', 's_world_rank']].copy()
	df_rankstime = df_ranks[((df_ranks['year'] <= 2015) & (df_ranks['year'] >= 2012))]
	return df_rankstime.pivot('university_name', 'year', 't_world_rank')
#=============

# Function 134
def cleaning_func_27(df_cwur,df_sac,df_shanghai,df_times):
	# core cleaning code
	import pandas as pd
	# df_cwur = pd.read_csv('../input/cwurData.csv')
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	# df_shanghai = pd.read_csv('../input/shanghaiData.csv')
	# df_times = pd.read_csv('../input/timesData.csv')
	df_shanghai = df_shanghai.merge(df_sac, how='left', on='university_name')
	df_cwur = df_cwur.rename(columns={'institution': 'university_name'})
	df_cwur = df_cwur.merge(df_sac, how='left', on='university_name')
	df_full = df_times.merge(df_cwur, how='outer', on=['university_name', 'year'])
	df_full = df_full.merge(df_shanghai, how='outer', on=['university_name', 'year'])
	df_ranks = df_full[['university_name', 't_country', 'year', 't_world_rank', 'c_world_rank', 's_world_rank']].copy()
	df_rankstime = df_ranks[((df_ranks['year'] <= 2015) & (df_ranks['year'] >= 2012))]
	df_tranks = df_rankstime.pivot('university_name', 'year', 't_world_rank').reset_index()
	df_tranks.columns = ['university_name', '2012', '2013', '2014', '2015']
	return df_tranks
#=============

# Function 135
def cleaning_func_34(df_cwur,df_sac,df_shanghai,df_times):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# df_cwur = pd.read_csv('../input/cwurData.csv')
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	# df_shanghai = pd.read_csv('../input/shanghaiData.csv')
	# df_times = pd.read_csv('../input/timesData.csv')
	df_shanghai = df_shanghai.merge(df_sac, how='left', on='university_name')
	df_cwur = df_cwur.rename(columns={'institution': 'university_name'})
	df_cwur = df_cwur.merge(df_sac, how='left', on='university_name')
	df_full = df_times.merge(df_cwur, how='outer', on=['university_name', 'year'])
	df_full = df_full.merge(df_shanghai, how='outer', on=['university_name', 'year'])
	df_ranks = df_full[['university_name', 't_country', 'year', 't_world_rank', 'c_world_rank', 's_world_rank']].copy()
	df_rankstime = df_ranks[((df_ranks['year'] <= 2015) & (df_ranks['year'] >= 2012))]
	df_tranks = df_rankstime.pivot('university_name', 'year', 't_world_rank').reset_index()
	df_tranks['std_dev'] = df_tranks.apply((lambda x: np.std([x['2012'], x['2013'], x['2014'], x['2015']])), axis=1)
	return df_tranks
#=============

# Function 136
def cleaning_func_40(df_cwur,df_sac,df_shanghai,df_times):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# df_cwur = pd.read_csv('../input/cwurData.csv')
	# df_sac = pd.read_csv('../input/school_and_country_table.csv')
	# df_shanghai = pd.read_csv('../input/shanghaiData.csv')
	# df_times = pd.read_csv('../input/timesData.csv')
	df_shanghai = df_shanghai.merge(df_sac, how='left', on='university_name')
	df_cwur = df_cwur.rename(columns={'institution': 'university_name'})
	df_cwur = df_cwur.merge(df_sac, how='left', on='university_name')
	df_full = df_times.merge(df_cwur, how='outer', on=['university_name', 'year'])
	df_full = df_full.merge(df_shanghai, how='outer', on=['university_name', 'year'])
	df_ranks = df_full[['university_name', 't_country', 'year', 't_world_rank', 'c_world_rank', 's_world_rank']].copy()
	df_rankstime = df_ranks[((df_ranks['year'] <= 2015) & (df_ranks['year'] >= 2012))]
	df_tranks = df_rankstime.pivot('university_name', 'year', 't_world_rank').reset_index()
	df_tranks['mean'] = df_tranks.apply((lambda x: np.mean([x['2012'], x['2013'], x['2014'], x['2015']])), axis=1)
	return df_tranks
#=============

# Function 137
def cleaning_func_0(timesData):
	# core cleaning code
	my_university_name = ['University of Groningen']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	return timesData[timesData.university_name.isin(my_university_name)]
#=============

# Function 138
def cleaning_func_1(timesData):
	# core cleaning code
	my_university_name = ['University of Groningen']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['source'] = 'Times'
	return times_plot_data
#=============

# Function 139
def cleaning_func_3(shanghaiData):
	# core cleaning code
	my_university_name = ['University of Groningen']
	import pandas as pd
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data['source'] = 'Shanghai'
	return shanghai_plot_data
#=============

# Function 140
def cleaning_func_4(cwurData):
	# core cleaning code
	my_university_name = ['University of Groningen']
	import pandas as pd
	# cwurData = pd.read_csv('../input/cwurData.csv')
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data['source'] = 'CWUR'
	return cwur_plot_data
#=============

# Function 141
def cleaning_func_6(timesData):
	# core cleaning code
	my_university_name = ['University of Groningen']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 142
def cleaning_func_8(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Groningen']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 143
def cleaning_func_10(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Groningen']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return shanghaiData[shanghaiData.university_name.isin(my_university_name)]
#=============

# Function 144
def cleaning_func_13(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Groningen']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	return shanghai_plot_data
#=============

# Function 145
def cleaning_func_14(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Groningen']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	return cwur_plot_data
#=============

# Function 146
def cleaning_func_15(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Groningen']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return cwurData[cwurData.institution.isin(my_university_name)]
#=============

# Function 147
def cleaning_func_18(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['University of Groningen']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	plot_data = times_plot_data.append(shanghai_plot_data).append(cwur_plot_data)
	plot_data['world_rank'] = plot_data['world_rank'].astype(int)
	return plot_data
#=============

# Function 148
def cleaning_func_0(timesData):
	# core cleaning code
	my_university_name = ['Australian National University', 'The Australian National University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['source'] = 'Times'
	return times_plot_data
#=============

# Function 149
def cleaning_func_1(timesData):
	# core cleaning code
	my_university_name = ['Australian National University', 'The Australian National University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	return timesData[timesData.university_name.isin(my_university_name)]
#=============

# Function 150
def cleaning_func_2(shanghaiData):
	# core cleaning code
	my_university_name = ['Australian National University', 'The Australian National University']
	import pandas as pd
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data['source'] = 'Shanghai'
	return shanghai_plot_data
#=============

# Function 151
def cleaning_func_4(cwurData):
	# core cleaning code
	my_university_name = ['Australian National University', 'The Australian National University']
	import pandas as pd
	# cwurData = pd.read_csv('../input/cwurData.csv')
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data['source'] = 'CWUR'
	return cwur_plot_data
#=============

# Function 152
def cleaning_func_6(timesData):
	# core cleaning code
	my_university_name = ['Australian National University', 'The Australian National University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 153
def cleaning_func_7(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Australian National University', 'The Australian National University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	return shanghai_plot_data
#=============

# Function 154
def cleaning_func_10(shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Australian National University', 'The Australian National University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return shanghaiData[shanghaiData.university_name.isin(my_university_name)]
#=============

# Function 155
def cleaning_func_11(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Australian National University', 'The Australian National University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	return cwurData[cwurData.institution.isin(my_university_name)]
#=============

# Function 156
def cleaning_func_12(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Australian National University', 'The Australian National University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	return times_plot_data
#=============

# Function 157
def cleaning_func_16(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Australian National University', 'The Australian National University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	return cwur_plot_data
#=============

# Function 158
def cleaning_func_18(cwurData,shanghaiData,timesData):
	# core cleaning code
	my_university_name = ['Australian National University', 'The Australian National University']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	# shanghaiData = pd.read_csv('../input/shanghaiData.csv')
	# cwurData = pd.read_csv('../input/cwurData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank', 'year']]
	cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
	shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]
	plot_data = times_plot_data.append(shanghai_plot_data).append(cwur_plot_data)
	plot_data['world_rank'] = plot_data['world_rank'].astype(int)
	return plot_data
#=============

# Function 159
def cleaning_func_0(cwur):
	# core cleaning code
	import pandas as pd
	# cwur = pd.read_csv('../input/cwurData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	cwur = cwur[['university_name', 'score']]
	cwur.score = pd.to_numeric(cwur.score, errors='coerce')
	return cwur
#=============

# Function 160
def cleaning_func_2(cwur):
	# core cleaning code
	import pandas as pd
	import re
	# cwur = pd.read_csv('../input/cwurData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	cwur = cwur[['university_name', 'score']]
	return cwur.groupby('university_name').mean()
#=============

# Function 161
def cleaning_func_4(cwur):
	# core cleaning code
	import pandas as pd
	import re
	# cwur = pd.read_csv('../input/cwurData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	cwur = cwur[['university_name', 'score']]
	cwur = cwur.groupby('university_name').mean().reset_index()
	cwur.university_name = [re.sub('[^a-zA-Z\\s0-9]+', '', string) for string in cwur.university_name]
	return cwur
#=============

# Function 162
def cleaning_func_7(cwur):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# cwur = pd.read_csv('../input/cwurData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	cwur = cwur[['university_name', 'score']]
	cwur = cwur.groupby('university_name').mean().reset_index()
	data = cwur
	return data
#=============

# Function 163
def cleaning_func_9(cwur):
	# additional context code from user definitions

	def calcScore(series):
	    scores = [x for x in series.values[1:] if (not np.isnan(x))]
	    return np.mean(scores)

	# core cleaning code
	import numpy as np
	import pandas as pd
	# cwur = pd.read_csv('../input/cwurData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	cwur = cwur[['university_name', 'score']]
	cwur = cwur.groupby('university_name').mean().reset_index()
	data = cwur
	data = data.rename(columns={'score': 'score_cwur'})
	data['mean_score'] = data.apply(calcScore, axis=1)
	return data
#=============

# Function 164
def cleaning_func_10(cwur):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# cwur = pd.read_csv('../input/cwurData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	cwur = cwur[['university_name', 'score']]
	cwur = cwur.groupby('university_name').mean().reset_index()
	return cwur
#=============

# Function 165
def cleaning_func_12(cwur,shanghai):
	# core cleaning code
	import pandas as pd
	import re
	# cwur = pd.read_csv('../input/cwurData.csv')
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	shanghai = shanghai.rename(columns={'total_score': 'score'})
	cwur = cwur[['university_name', 'score']]
	shanghai = shanghai[['university_name', 'score']]
	cwur = cwur.groupby('university_name').mean().reset_index()
	shanghai = shanghai.groupby('university_name').mean().reset_index()
	cwur.university_name = [re.sub('[^a-zA-Z\\s0-9]+', '', string) for string in cwur.university_name]
	shanghai.university_name = [re.sub('[^a-zA-Z\\s0-9]+', '', string) for string in shanghai.university_name]
	return shanghai
#=============

# Function 166
def cleaning_func_15(cwur,shanghai):
	# core cleaning code
	import pandas as pd
	import re
	# cwur = pd.read_csv('../input/cwurData.csv')
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	shanghai = shanghai.rename(columns={'total_score': 'score'})
	cwur = cwur[['university_name', 'score']]
	shanghai = shanghai[['university_name', 'score']]
	cwur = cwur.groupby('university_name').mean().reset_index()
	return shanghai.groupby('university_name').mean()
#=============

# Function 167
def cleaning_func_16(cwur,shanghai):
	# core cleaning code
	import pandas as pd
	import re
	# cwur = pd.read_csv('../input/cwurData.csv')
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	shanghai = shanghai.rename(columns={'total_score': 'score'})
	cwur = cwur[['university_name', 'score']]
	return cwur
#=============

# Function 168
def cleaning_func_26(cwur,shanghai):
	# core cleaning code
	import pandas as pd
	# cwur = pd.read_csv('../input/cwurData.csv')
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	shanghai = shanghai.rename(columns={'total_score': 'score'})
	cwur = cwur[['university_name', 'score']]
	shanghai = shanghai[['university_name', 'score']]
	cwur = cwur.groupby('university_name').mean().reset_index()
	shanghai = shanghai.groupby('university_name').mean().reset_index()
	data = cwur
	return data
#=============

# Function 169
def cleaning_func_27(cwur,shanghai):
	# additional context code from user definitions

	def check_for_uni_shanghai(series):
	    university = series['university_name']
	    for uni in shanghai['university_name'].values:
	        if is_fuzzy_match(university, uni):
	            return shanghai[(shanghai['university_name'] == uni)]['score'].values[0]


	def is_fuzzy_match(string1, string2, threshold=0.9):
	    similarity = SM(None, str(string1), str(string2)).ratio()
	    if (similarity > threshold):
	        return True
	    else:
	        return False

	# core cleaning code
	import pandas as pd
	# cwur = pd.read_csv('../input/cwurData.csv')
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	shanghai = shanghai.rename(columns={'total_score': 'score'})
	cwur = cwur[['university_name', 'score']]
	shanghai = shanghai[['university_name', 'score']]
	cwur = cwur.groupby('university_name').mean().reset_index()
	shanghai = shanghai.groupby('university_name').mean().reset_index()
	data = cwur
	data = data.rename(columns={'score': 'score_cwur'})
	data['score_shanghai'] = data.apply(check_for_uni_shanghai, axis=1)
	return data
#=============

# Function 170
def cleaning_func_32(cwur,shanghai,times):
	# core cleaning code
	import pandas as pd
	# cwur = pd.read_csv('../input/cwurData.csv')
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	# times = pd.read_csv('../input/timesData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	shanghai = shanghai.rename(columns={'total_score': 'score'})
	times = times.rename(columns={'total_score': 'score'})
	cwur = cwur[['university_name', 'score']]
	shanghai = shanghai[['university_name', 'score']]
	times = times[['university_name', 'score']]
	times = times[(~ (times['score'] == '-'))]
	cwur.score = pd.to_numeric(cwur.score, errors='coerce')
	shanghai.score = pd.to_numeric(shanghai.score, errors='coerce')
	times.score = pd.to_numeric(shanghai.score, errors='coerce')
	return times
#=============

# Function 171
def cleaning_func_33(cwur,shanghai,times):
	# core cleaning code
	import pandas as pd
	# cwur = pd.read_csv('../input/cwurData.csv')
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	# times = pd.read_csv('../input/timesData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	shanghai = shanghai.rename(columns={'total_score': 'score'})
	times = times.rename(columns={'total_score': 'score'})
	cwur = cwur[['university_name', 'score']]
	shanghai = shanghai[['university_name', 'score']]
	times = times[['university_name', 'score']]
	return times
#=============

# Function 172
def cleaning_func_37(cwur,shanghai,times):
	# core cleaning code
	import pandas as pd
	# cwur = pd.read_csv('../input/cwurData.csv')
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	# times = pd.read_csv('../input/timesData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	shanghai = shanghai.rename(columns={'total_score': 'score'})
	times = times.rename(columns={'total_score': 'score'})
	cwur = cwur[['university_name', 'score']]
	shanghai = shanghai[['university_name', 'score']]
	times = times[['university_name', 'score']]
	times = times[(~ (times['score'] == '-'))]
	cwur.score = pd.to_numeric(cwur.score, errors='coerce')
	shanghai.score = pd.to_numeric(shanghai.score, errors='coerce')
	times.score = pd.to_numeric(shanghai.score, errors='coerce')
	cwur = cwur.groupby('university_name').mean().reset_index()
	cwur.score = ((cwur.score - cwur.score.mean()) / cwur.score.std())
	return cwur
#=============

# Function 173
def cleaning_func_41(cwur,shanghai,times):
	# core cleaning code
	import pandas as pd
	# cwur = pd.read_csv('../input/cwurData.csv')
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	# times = pd.read_csv('../input/timesData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	shanghai = shanghai.rename(columns={'total_score': 'score'})
	times = times.rename(columns={'total_score': 'score'})
	cwur = cwur[['university_name', 'score']]
	shanghai = shanghai[['university_name', 'score']]
	times = times[['university_name', 'score']]
	times = times[(~ (times['score'] == '-'))]
	cwur.score = pd.to_numeric(cwur.score, errors='coerce')
	shanghai.score = pd.to_numeric(shanghai.score, errors='coerce')
	times.score = pd.to_numeric(shanghai.score, errors='coerce')
	cwur = cwur.groupby('university_name').mean().reset_index()
	return shanghai.groupby('university_name').mean()
#=============

# Function 174
def cleaning_func_46(cwur,shanghai,times):
	# core cleaning code
	import pandas as pd
	import re
	# cwur = pd.read_csv('../input/cwurData.csv')
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	# times = pd.read_csv('../input/timesData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	shanghai = shanghai.rename(columns={'total_score': 'score'})
	times = times.rename(columns={'total_score': 'score'})
	cwur = cwur[['university_name', 'score']]
	shanghai = shanghai[['university_name', 'score']]
	times = times[['university_name', 'score']]
	times = times[(~ (times['score'] == '-'))]
	cwur.score = pd.to_numeric(cwur.score, errors='coerce')
	shanghai.score = pd.to_numeric(shanghai.score, errors='coerce')
	times.score = pd.to_numeric(shanghai.score, errors='coerce')
	cwur = cwur.groupby('university_name').mean().reset_index()
	shanghai = shanghai.groupby('university_name').mean().reset_index()
	times = times.groupby('university_name').mean().reset_index()
	cwur.score = ((cwur.score - cwur.score.mean()) / cwur.score.std())
	shanghai.score = ((shanghai.score - shanghai.score.mean()) / shanghai.score.std())
	times.score = ((times.score - times.score.mean()) / times.score.std())
	cwur.university_name = [re.sub('[^a-zA-Z\\s0-9]+', '', string) for string in cwur.university_name]
	shanghai.university_name = [re.sub('[^a-zA-Z\\s0-9]+', '', string) for string in shanghai.university_name]
	return shanghai
#=============

# Function 175
def cleaning_func_59(cwur,shanghai,times):
	# core cleaning code
	import pandas as pd
	# cwur = pd.read_csv('../input/cwurData.csv')
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	# times = pd.read_csv('../input/timesData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	shanghai = shanghai.rename(columns={'total_score': 'score'})
	times = times.rename(columns={'total_score': 'score'})
	cwur = cwur[['university_name', 'score']]
	shanghai = shanghai[['university_name', 'score']]
	times = times[['university_name', 'score']]
	times = times[(~ (times['score'] == '-'))]
	cwur.score = pd.to_numeric(cwur.score, errors='coerce')
	shanghai.score = pd.to_numeric(shanghai.score, errors='coerce')
	times.score = pd.to_numeric(shanghai.score, errors='coerce')
	cwur = cwur.groupby('university_name').mean().reset_index()
	shanghai = shanghai.groupby('university_name').mean().reset_index()
	times = times.groupby('university_name').mean().reset_index()
	cwur.score = ((cwur.score - cwur.score.mean()) / cwur.score.std())
	shanghai.score = ((shanghai.score - shanghai.score.mean()) / shanghai.score.std())
	times.score = ((times.score - times.score.mean()) / times.score.std())
	data = cwur
	return data
#=============

# Function 176
def cleaning_func_60(cwur,shanghai,times):
	# additional context code from user definitions

	def check_for_uni_times(series):
	    university = series['university_name']
	    for uni in times['university_name'].values:
	        if is_fuzzy_match(university, uni):
	            return times[(times['university_name'] == uni)]['score'].values[0]


	def is_fuzzy_match(string1, string2, threshold=0.9):
	    similarity = SM(None, str(string1), str(string2)).ratio()
	    if (similarity > threshold):
	        return True
	    else:
	        return False

	# core cleaning code
	import pandas as pd
	# cwur = pd.read_csv('../input/cwurData.csv')
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	# times = pd.read_csv('../input/timesData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	shanghai = shanghai.rename(columns={'total_score': 'score'})
	times = times.rename(columns={'total_score': 'score'})
	cwur = cwur[['university_name', 'score']]
	shanghai = shanghai[['university_name', 'score']]
	times = times[['university_name', 'score']]
	times = times[(~ (times['score'] == '-'))]
	cwur.score = pd.to_numeric(cwur.score, errors='coerce')
	shanghai.score = pd.to_numeric(shanghai.score, errors='coerce')
	times.score = pd.to_numeric(shanghai.score, errors='coerce')
	cwur = cwur.groupby('university_name').mean().reset_index()
	shanghai = shanghai.groupby('university_name').mean().reset_index()
	times = times.groupby('university_name').mean().reset_index()
	cwur.score = ((cwur.score - cwur.score.mean()) / cwur.score.std())
	shanghai.score = ((shanghai.score - shanghai.score.mean()) / shanghai.score.std())
	times.score = ((times.score - times.score.mean()) / times.score.std())
	data = cwur
	data = data.rename(columns={'score': 'score_cwur'})
	data['score_times'] = data.apply(check_for_uni_times, axis=1)
	return data
#=============

# Function 177
def cleaning_func_61(cwur,shanghai,times):
	# core cleaning code
	import pandas as pd
	import re
	# cwur = pd.read_csv('../input/cwurData.csv')
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	# times = pd.read_csv('../input/timesData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	shanghai = shanghai.rename(columns={'total_score': 'score'})
	times = times.rename(columns={'total_score': 'score'})
	cwur = cwur[['university_name', 'score']]
	shanghai = shanghai[['university_name', 'score']]
	times = times[['university_name', 'score']]
	times = times[(~ (times['score'] == '-'))]
	cwur.score = pd.to_numeric(cwur.score, errors='coerce')
	shanghai.score = pd.to_numeric(shanghai.score, errors='coerce')
	times.score = pd.to_numeric(shanghai.score, errors='coerce')
	cwur = cwur.groupby('university_name').mean().reset_index()
	shanghai = shanghai.groupby('university_name').mean().reset_index()
	times = times.groupby('university_name').mean().reset_index()
	cwur.score = ((cwur.score - cwur.score.mean()) / cwur.score.std())
	shanghai.score = ((shanghai.score - shanghai.score.mean()) / shanghai.score.std())
	times.score = ((times.score - times.score.mean()) / times.score.std())
	cwur.university_name = [re.sub('[^a-zA-Z\\s0-9]+', '', string) for string in cwur.university_name]
	shanghai.university_name = [re.sub('[^a-zA-Z\\s0-9]+', '', string) for string in shanghai.university_name]
	times.university_name = [re.sub('[^a-zA-Z\\s0-9]+', '', string) for string in times.university_name]
	data = cwur
	return data
#=============

# Function 178
def cleaning_func_62(cwur,shanghai,times):
	# core cleaning code
	import pandas as pd
	import re
	# cwur = pd.read_csv('../input/cwurData.csv')
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	# times = pd.read_csv('../input/timesData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	shanghai = shanghai.rename(columns={'total_score': 'score'})
	times = times.rename(columns={'total_score': 'score'})
	cwur = cwur[['university_name', 'score']]
	shanghai = shanghai[['university_name', 'score']]
	times = times[['university_name', 'score']]
	times = times[(~ (times['score'] == '-'))]
	cwur.score = pd.to_numeric(cwur.score, errors='coerce')
	shanghai.score = pd.to_numeric(shanghai.score, errors='coerce')
	times.score = pd.to_numeric(shanghai.score, errors='coerce')
	cwur = cwur.groupby('university_name').mean().reset_index()
	shanghai = shanghai.groupby('university_name').mean().reset_index()
	times = times.groupby('university_name').mean().reset_index()
	cwur.score = ((cwur.score - cwur.score.mean()) / cwur.score.std())
	shanghai.score = ((shanghai.score - shanghai.score.mean()) / shanghai.score.std())
	times.score = ((times.score - times.score.mean()) / times.score.std())
	cwur.university_name = [re.sub('[^a-zA-Z\\s0-9]+', '', string) for string in cwur.university_name]
	shanghai.university_name = [re.sub('[^a-zA-Z\\s0-9]+', '', string) for string in shanghai.university_name]
	times.university_name = [re.sub('[^a-zA-Z\\s0-9]+', '', string) for string in times.university_name]
	data = cwur
	data = data.rename(columns={'score': 'score_cwur'})
	return data
#=============

# Function 179
def cleaning_func_63(cwur,shanghai,times):
	# additional context code from user definitions

	def check_for_master(university):
	    for program in programs:
	        similarity = is_fuzzy_match(university, program)
	        if similarity:
	            return True


	def is_fuzzy_match(string1, string2, threshold=0.9):
	    similarity = SM(None, str(string1), str(string2)).ratio()
	    if (similarity > threshold):
	        return True
	    else:
	        return False

	# core cleaning code
	import pandas as pd
	import re
	# cwur = pd.read_csv('../input/cwurData.csv')
	# shanghai = pd.read_csv('../input/shanghaiData.csv')
	# times = pd.read_csv('../input/timesData.csv')
	cwur = cwur.rename(columns={'institution': 'university_name'})
	shanghai = shanghai.rename(columns={'total_score': 'score'})
	times = times.rename(columns={'total_score': 'score'})
	cwur = cwur[['university_name', 'score']]
	shanghai = shanghai[['university_name', 'score']]
	times = times[['university_name', 'score']]
	times = times[(~ (times['score'] == '-'))]
	cwur.score = pd.to_numeric(cwur.score, errors='coerce')
	shanghai.score = pd.to_numeric(shanghai.score, errors='coerce')
	times.score = pd.to_numeric(shanghai.score, errors='coerce')
	cwur = cwur.groupby('university_name').mean().reset_index()
	shanghai = shanghai.groupby('university_name').mean().reset_index()
	times = times.groupby('university_name').mean().reset_index()
	cwur.score = ((cwur.score - cwur.score.mean()) / cwur.score.std())
	shanghai.score = ((shanghai.score - shanghai.score.mean()) / shanghai.score.std())
	times.score = ((times.score - times.score.mean()) / times.score.std())
	cwur.university_name = [re.sub('[^a-zA-Z\\s0-9]+', '', string) for string in cwur.university_name]
	shanghai.university_name = [re.sub('[^a-zA-Z\\s0-9]+', '', string) for string in shanghai.university_name]
	times.university_name = [re.sub('[^a-zA-Z\\s0-9]+', '', string) for string in times.university_name]
	data = cwur
	data = data.rename(columns={'score': 'score_cwur'})
	data = data.sort_values('mean_score', ascending=False)
	programs = ['Auburn University', 'The University of Alabama', 'Arkansas Tech University', 'University of Arkansas', 'University of Arkansas at Little Rock', 'Arizona State University', 'University of Arizona', 'California Polytechnic State University', 'California State UniversityEast Bay', 'California State UniversityFullerton', 'California State UniversityLong Beach', 'California State UniversitySan Bernardino', 'Chapman University', 'Claremont Graduate University', 'Galvanize U', 'National University', 'San Jose State University', 'Santa Clara University', 'Stanford University', 'University of California Hastings College of Law', 'University of CaliforniaDavis', 'University of CaliforniaIrvine', 'University of CaliforniaSan Diego', 'University of CaliforniaBerkeley', 'University of San Francisco', 'University Of San Francisco', 'University of Southern California', 'University of the Pacific', 'American Sentinel University', 'Aspen University', 'Colorado State UniversityFort Collins', 'Colorado State UniversityGlobal Campus', 'Colorado Technical University', 'Regis University', 'University of Colorado Boulder', 'University of Colorado Denver', 'University of Denver', 'Central Connecticut State University', 'Quinnipiac University', 'University of Connecticut', 'University of New Haven', 'American University', 'George Washington University', 'Georgetown University', 'The George Washington University', 'New College of Florida', 'Florida International University', 'Florida Polytechnic University', 'Full Sail University', 'Nova Southeastern University', 'PascoHernando State College', 'University of Central Florida', 'University of Florida', 'University of Miami', 'University of South Florida SarasotaManatee', 'University of South FloridaMain Campus', 'Georgia Southern University', 'Georgia State University', 'Georgia Tech', 'Kennesaw State University', 'Mercer University', 'University of Georgia', 'Loras College', 'Northwestern College', 'Luther College', 'The University of Iowa', 'Aurora University', 'Benedictine University', 'DePaul University', 'Elmhurst College', 'Illinois Institute of Technology', 'Lewis University', 'Loyola University Chicago', 'Northwestern University', 'University of Chicago', 'University of Illinois at Chicago', 'University of Illinois at Springfield', 'University of Illinois at UrbanaChampaign', 'University of St Francis', 'Indiana University Bloomington', 'Indiana UniversityPurdue UniversityIndianapolis', 'Purdue UniversityMain Campus', 'Saint Marys College', 'University of Notre Dame', 'University of Evansville', 'University of Kansas', 'Northern Kentucky University', 'University of Louisville', 'Louisiana State University', 'Babson College', 'Becker College', 'Bentley University', 'Brandeis University', 'Harvard University', 'Northeastern University', 'University of Massachusetts Amherst', 'Worcester Polytechnic Institute', 'Smith College', 'Johns Hopkins University', 'Notre Dame of Maryland University', 'University of MarylandBaltimore County', 'University of MarylandCollege Park', 'University of MarylandUniversity College', 'Baker College', 'Central Michigan University', 'Davenport University', 'Eastern Michigan University', 'Grand Valley State University', 'Michigan State University', 'Michigan Technological University', 'Oakland University', 'University of MichiganAnn Arbor', 'University of MichiganDearborn', 'Capella University', 'The College of Saint Scholastica', 'University of Minnesota', 'University of MinnesotaDuluth', 'University of St Thomas', 'Winona State University', 'Grantham University', 'Missouri University of Science and Technology', 'Rockhurst University', 'Saint Louis University', 'Saint Louis UniversityMain Campus', 'University of MissouriSt Louis', 'Jackson State University', 'University of Montana', 'Elon University', 'North Carolina State University at Raleigh', 'University of North Carolina at Chapel Hill', 'University of North Carolina at Charlotte', 'University of North Carolina at Greensboro', 'Wake forest University', 'Bellevue University', 'Creighton University', 'Nebraska College of Technical Agriculture', 'University of Nebraska at Omaha', 'Southern New Hampshire University', 'New Jersey Institute of Technology', 'Rutgers University', 'Saint Peters University', 'Stevens Institute of Technology', 'Thomas Edison State College', 'University of NevadaReno', 'Columbia University in the City of New York', 'Cornell University', 'CUNY Bernard M Baruch College', 'CUNY Graduate School and University Center', 'CUNY Queens College', 'Fordham University', 'Keller Graduate School of Management', 'Marist College', 'New York University', 'Pace UniversityNew York', 'Rensselaer Polytechnic Institute', 'St Johns UniversityNew York', 'Syracuse University', 'The New School', 'Trocaire College', 'Union Graduate College', 'University at Buffalo', 'University of Rochester', 'Bowling Green State University', 'Case Western Reserve University', 'Cleveland State University', 'Miami University of Ohio', 'Notre Dame College', 'Ohio University', 'The Ohio State University', 'University of CincinnatiMain Campus', 'Oklahoma State University Center for Health Sciences', 'Southwestern Oklahoma State University', 'University of Oklahoma Norman Campus', 'Oregon State University', 'Albright College', 'Carnegie Mellon University', 'Drexel University', 'Harrisburg University of Science and Technology', 'La Salle University', 'Misericordia University', 'Pennsylvania State University', 'Philadelphia University', 'Saint Josephs University', 'Temple University', 'University of PittsburghBradford', 'University of PittsburghPittsburgh Campus', 'Villanova University', 'Brown University', 'College of Charleston', 'Medical University of South Carolina', 'University of South CarolinaColumbia', 'Dakota State University', 'South Dakota State University', 'Austin Peay State University', 'Middle Tennessee State University', 'Tennessee Technological University', 'The University of Tennessee', 'The University of Tennessee at Chattanooga', 'University of Memphis', 'Southern Methodist University', 'St Marys University', 'Tarleton State University', 'Texas A  M UniversityCollege Station', 'The University of Texas at Austin', 'The University of Texas at Dallas', 'The University of Texas at San Antonio', 'University of Dallas', 'University of North Texas', 'University of Utah', 'George Mason University', 'Radford University', 'University of Virginia', 'Virginia Commonwealth University', 'Virginia Polytechnic Institute and State University', 'Statisticscom', 'Bellevue College', 'City University of Seattle', 'Seattle University', 'University of WashingtonSeattle Campus', 'University of WashingtonTacoma Campus', 'University of WisconsinMadison', 'University of Wisconsin Colleges', 'University of WisconsinMilwaukee', 'West Virginia University', 'Ukrainian Catholic Univeristy', 'Sabanc University', 'National University of Singapore', 'Dalarna University', 'Blekinge Institute of Technology', 'Kth Royal Institute Of Technology', 'Linkping University', 'Universidade Nova de Lisboa', 'University of Otago', 'Massey University', 'Erasmus University', 'Maastricht University', 'Radboud Universiteit Nijmegen', 'Eindhoven University of TechnologyTUe', 'Utrecht University', 'Vrije Universiteit Amsterdam', 'Autonomous Technological Institute of Mexico', 'Mykolas Romeris University', 'Sangmyung University', 'BAICR', 'Polytechnic University Of Turin', 'University Of MilanBicocca', 'University Of Pisa', 'BenGurion University Of The Negev', 'Dublin City University', 'Dublin Institute Of Technology', 'Institute Of Technology Blanchardstown', 'Irish Management Institute', 'National College Of Ireland', 'National University Of Ireland Galway', 'University College Cork', 'University College Dublin', 'Chinese University of Hong Kong', 'Hong Kong University of Science  Technology', 'Lancaster University', 'Aston University', 'Birmingham City University', 'Bournemouth University', 'Brunel University London', 'City University London', 'Coventry University', 'De Montfort University', 'Goldsmiths University of London', 'Imperial College London', 'Leeds Met', 'Newcastle University', 'Robert Gordon University', 'Royal Holloway University Of London', 'Sheffield Hallam University', 'The University Of Edinburgh', 'The University Of Manchester', 'University College London', 'University Of Bristol', 'University of Derby', 'University of Dundee', 'University Of East Anglia', 'University Of East London', 'University Of Essex', 'University Of Greenwich', 'University of Kent', 'University Of Leeds', 'University of Leicester', 'University Of Liverpool', 'University of Manchester', 'University of Nottingham', 'University of Southampton', 'University Of St Andrews', 'University of Strathclyde', 'University of Surrey', 'University of Warwick', 'University Of Warwick', 'University Of Westminster', 'Data ScienceTech Institute', 'EISTI', 'ENSAE Paris Tech', 'Telecom Paris Tech', 'Telecom Sudparis', 'Universit Pierre Et Marie Curie', 'Aalto University', 'University Of Helsinki', 'Universit De Nantes', 'Barcelona School of Management', 'Instituto de Empresa', 'Universidad Rey Juan Carlos', 'Universitat Pompeu Fabra', 'Universities Of Alicante', 'University of Barcelona', 'University of Oviedo', 'Aalborg University', 'Aarhus University', 'Technical University of Denmark', 'Otto Von Guericke University Magdeburg', 'TU Dortmund', 'Universitt Konstanz', 'Queens University', 'Simon Fraser University', 'University Of Alberta', 'University of the Fraser Valley', 'York University', 'Mackenzie Presbyterian Institute', 'Deakin University', 'Macquarie University', 'University of South Australia', 'University of Technology Sydney', 'Vienna University of Economics and Business', 'University of Vienna']
	programs = [re.sub('[^a-zA-Z\\s0-9]+', '', string) for string in programs]
	data['ds_master'] = data['university_name'].apply(check_for_master)
	return data
#=============

