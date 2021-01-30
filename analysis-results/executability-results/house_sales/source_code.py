# Function 0
def cleaning_func_0(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/kc_house_data.csv')
	dataset['condition'] = dataset['condition'].astype('category', ordered=True)
	return dataset
=============

# Function 1
def cleaning_func_1(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/kc_house_data.csv')
	dataset['grade'] = dataset['grade'].astype('category', ordered=False)
	return dataset
=============

# Function 2
def cleaning_func_2(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/kc_house_data.csv')
	dataset['view'] = dataset['view'].astype('category', ordered=True)
	return dataset
=============

# Function 3
def cleaning_func_3(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/kc_house_data.csv')
	dataset['waterfront'] = dataset['waterfront'].astype('category', ordered=True)
	return dataset
=============

# Function 4
def cleaning_func_4(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/kc_house_data.csv')
	dataset = dataset.drop(['id', 'date'], axis=1)
	dataset['is_renovated'] = dataset['yr_renovated'].apply((lambda x: (1 if (x > 0) else 0)))
	dataset['is_renovated'] = dataset['is_renovated'].astype('category', ordered=False)
	return dataset
=============

# Function 5
def cleaning_func_5(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/kc_house_data.csv')
	dataset = dataset.drop(['id', 'date'], axis=1)
	dataset['basement_is_present'] = dataset['sqft_basement'].apply((lambda x: (1 if (x > 0) else 0)))
	dataset['basement_is_present'] = dataset['basement_is_present'].astype('category', ordered=False)
	return dataset
=============

# Function 6
def cleaning_func_0(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn import tree
	# data = pd.read_csv('../input/kc_house_data.csv', encoding='ISO-8859-1')
	Y = data['price']
	X = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long']]
	return X
=============

# Function 7
def cleaning_func_1(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn import tree
	# data = pd.read_csv('../input/kc_house_data.csv', encoding='ISO-8859-1')
	Y = data['price']
	X = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long']]
	colnames = X.columns
	ranks = {}
	y = Y
	y = Y
	(X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2, random_state=10)
	return X_train
=============

# Function 8
def cleaning_func_0(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data['date'] = pd.to_datetime(data['date'])
	return data
=============

# Function 9
def cleaning_func_1(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data['date'] = pd.to_datetime(data['date'])
	data['month'] = data['date'].dt.month
	return data
=============

# Function 10
def cleaning_func_2(data):
	# core cleaning code
	import pandas as pd
	from datetime import datetime, date, time
	# data = pd.read_csv('../input/kc_house_data.csv')
	current_year = datetime.now().year
	data['house_age'] = (current_year - data['yr_built'])
	return data
=============

# Function 11
def cleaning_func_3(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data['date'] = pd.to_datetime(data['date'])
	data['year'] = data['date'].dt.year
	return data
=============

# Function 12
def cleaning_func_0(dataFrame):
	# core cleaning code
	import pandas as pd
	# dataFrame = pd.read_csv('../input/kc_house_data.csv', nrows=1000)
	Cols = ['price', 'sqft_living']
	dataFrame = dataFrame[Cols]
	return dataFrame[['price']]
=============

# Function 13
def cleaning_func_1(mydata):
	# core cleaning code
	import pandas as pd
	# mydata = pd.read_csv('../input/kc_house_data.csv', parse_dates=['date'])
	mydata['zipcode'] = mydata['zipcode'].astype(str)
	return mydata
=============

# Function 14
def cleaning_func_0(df_data):
	# core cleaning code
	import pandas as pd
	# df_data = pd.read_csv('../input/kc_house_data.csv')
	df_data['price'] = (df_data['price'] / (10 ** 6))
	return df_data
=============

# Function 15
def cleaning_func_1(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data = data.set_index('id')
	data.floors = data.floors.astype(int)
	return data
=============

# Function 16
def cleaning_func_2(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data = data.set_index('id')
	data.bathrooms = data.bathrooms.astype(int)
	return data
=============

# Function 17
def cleaning_func_3(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data = data.set_index('id')
	data.price = data.price.astype(int)
	return data
=============

# Function 18
def cleaning_func_4(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data = data.set_index('id')
	data['renovated'] = data['yr_renovated'].apply((lambda yr: (0 if (yr == 0) else 1)))
	return data
=============

# Function 19
def cleaning_func_5(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data = data.set_index('id')
	return data
=============

# Function 20
def cleaning_func_6(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data = data.set_index('id')
	data = data.drop('date', axis=1)
	data = data.drop('yr_renovated', axis=1)
	return data
=============

# Function 21
def cleaning_func_7(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data = data.set_index('id')
	data = data.drop('date', axis=1)
	return data
=============

# Function 22
def cleaning_func_8(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data = data.set_index('id')
	data = data.drop('date', axis=1)
	data = data.drop('yr_renovated', axis=1)
	data = data.drop('yr_built', axis=1)
	data['sqft_living'] = np.log(data['sqft_living'])
	return data
=============

# Function 23
def cleaning_func_9(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data['date'] = pd.to_datetime(data['date'])
	data = data.set_index('id')
	data['house_age'] = (data['date'].dt.year - data['yr_built'])
	return data
=============

# Function 24
def cleaning_func_10(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data = data.set_index('id')
	data.price = data.price.astype(int)
	data['renovated'] = data['yr_renovated'].apply((lambda yr: (0 if (yr == 0) else 1)))
	return data
=============

# Function 25
def cleaning_func_11(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data = data.set_index('id')
	data.price = data.price.astype(int)
	data['renovated'] = data['yr_renovated'].apply((lambda yr: (0 if (yr == 0) else 1)))
	data = data.drop('date', axis=1)
	data = data.drop('yr_renovated', axis=1)
	return data
=============

# Function 26
def cleaning_func_13(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data = data.set_index('id')
	data.price = data.price.astype(int)
	data['renovated'] = data['yr_renovated'].apply((lambda yr: (0 if (yr == 0) else 1)))
	data = data.drop('date', axis=1)
	data = data.drop('yr_renovated', axis=1)
	data = data.drop('yr_built', axis=1)
	data['price'] = np.log(data['price'])
	return data
=============

# Function 27
def cleaning_func_0(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df = pd.get_dummies(df, 'zipcode')
	df['yr_renovated'] = df.loc[((df.yr_renovated > 2007), 'yr_renovated')] = 1
	df['yr_renovated'] = df.loc[((df.yr_renovated <= 2007), 'yr_renovated')] = 0
	return df
=============

# Function 28
def cleaning_func_0(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data['date'] = pd.to_datetime(data['date'])
	data['date'] = data['date'].dt.dayofyear
	return data
=============

# Function 29
def cleaning_func_1(data):
	# core cleaning code
	import pandas as pd
	from datetime import datetime, date, time
	# data = pd.read_csv('../input/kc_house_data.csv')
	currYear = datetime.now().year
	data['yr_built'] = (currYear - data['yr_built'])
	return data
=============

# Function 30
def cleaning_func_2(data):
	# core cleaning code
	import pandas as pd
	from datetime import datetime, date, time
	# data = pd.read_csv('../input/kc_house_data.csv')
	currYear = datetime.now().year
	data = data.rename(columns={'yr_built': 'house_age'})
	data = data.drop('id', axis=1)
	data = data.drop('zipcode', axis=1)
	data['yr_renovated'] = (currYear - data['yr_renovated'])
	data['yr_renovated'] = data['yr_renovated'].where((data['yr_renovated'] != currYear), 0)
	return data
=============

# Function 31
def cleaning_func_3(data):
	# core cleaning code
	import pandas as pd
	from datetime import datetime, date, time
	# data = pd.read_csv('../input/kc_house_data.csv')
	currYear = datetime.now().year
	data = data.rename(columns={'yr_built': 'house_age'})
	data = data.drop('id', axis=1)
	return data
=============

# Function 32
def cleaning_func_4(data):
	# core cleaning code
	import pandas as pd
	from datetime import datetime, date, time
	# data = pd.read_csv('../input/kc_house_data.csv')
	currYear = datetime.now().year
	data = data.rename(columns={'yr_built': 'house_age'})
	return data
=============

# Function 33
def cleaning_func_0(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df = df.drop('id', axis=1)
	df['date'] = pd.to_datetime(df['date'])
	return df
=============

# Function 34
def cleaning_func_1(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df = df.drop('id', axis=1)
	df['zipcode_str'] = df['zipcode'].astype(str).map((lambda x: ('zip_' + x)))
	return df
=============

# Function 35
def cleaning_func_2(df):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df = df.drop('id', axis=1)
	df['yr_renovated_bin'] = (np.array((df['yr_renovated'] != 0)) * 1)
	return df
=============

# Function 36
def cleaning_func_3(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df = df.drop('id', axis=1)
	df['date'] = pd.to_datetime(df['date'])
	df['dow'] = pd.to_datetime(df.date).map((lambda x: ('dow' + str(x.weekday()))))
	return df
=============

# Function 37
def cleaning_func_4(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df = df.drop('id', axis=1)
	df['date'] = pd.to_datetime(df['date'])
	df['month'] = pd.to_datetime(df.date).map((lambda x: ('month' + str(x.month))))
	return df
=============

# Function 38
def cleaning_func_0(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df['zipcode_str'] = df['zipcode'].astype(str).map((lambda x: ('zip_' + x)))
	return df
=============

# Function 39
def cleaning_func_1(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df['dow'] = pd.to_datetime(df.date).map((lambda x: ('dow' + str(x.weekday()))))
	return df
=============

# Function 40
def cleaning_func_3(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df['month'] = pd.to_datetime(df.date).map((lambda x: ('month' + str(x.month))))
	return df
=============

# Function 41
def cleaning_func_4(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/kc_house_data.csv')
	df['yr_renovated_bin'] = (np.array((df['yr_renovated'] != 0)) * 1)
	return df
=============

# Function 42
def cleaning_func_0(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	X = df.drop('price', axis=1)
	X['sold_year'] = X['date'].apply((lambda x: int(x[slice(None, 4, None)])))
	return X
=============

# Function 43
def cleaning_func_1(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	X = df.drop('price', axis=1)
	X['sold_year'] = X['date'].apply((lambda x: int(x[slice(None, 4, None)])))
	X['yrs_since_renovated'] = (X['sold_year'] - X['yr_renovated'][(X['yr_renovated'] != 0)]).fillna(0)
	return X
=============

# Function 44
def cleaning_func_1(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	(df['Year'], df['Month']) = (df['date'].str[slice(None, 4, None)], df['date'].str[slice(4, 6, None)])
	return df
=============

# Function 45
def cleaning_func_0(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df['year_sell'] = [int(i[slice(None, 4, None)]) for i in df.date]
	return df
=============

# Function 46
def cleaning_func_1(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df['age_of_renov'] = 100
	return df
=============

# Function 47
def cleaning_func_2(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df['price_per_sqft'] = (df['price'] / df['sqft_living'])
	return df
=============

# Function 48
def cleaning_func_3(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df['area_floor'] = (df['sqft_above'] + df['sqft_living'])
	return df
=============

# Function 49
def cleaning_func_4(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df['basement_present'] = df['sqft_basement'].apply((lambda x: (1 if (x > 0) else 0)))
	return df
=============

# Function 50
def cleaning_func_5(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df.loc[((df['yr_renovated'] != 0), 'age_of_renov')] = (2015 - df.loc[((df['yr_renovated'] != 0), 'yr_renovated')])
	return df
=============

# Function 51
def cleaning_func_6(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df.loc[((df['yr_renovated'] != 0), 'age_of_renov')] = (2015 - df.loc[((df['yr_renovated'] != 0), 'yr_renovated')])
	df['renovated'] = df['yr_renovated'].apply((lambda x: (1 if (x > 0) else 0)))
	return df
=============

# Function 52
def cleaning_func_0(house):
	# core cleaning code
	import numpy as np
	import pandas as pd
	from sklearn.ensemble import RandomForestRegressor
	# house = pd.read_csv('../input/kc_house_data.csv')
	X = house[house.columns[slice(1, 19, None)]]
	return X
=============

# Function 53
def cleaning_func_0(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv', parse_dates=['date'])
	df['month'] = df['date'].dt.month
	return df
=============

# Function 54
def cleaning_func_1(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv', parse_dates=['date'])
	df['year'] = df['date'].dt.year
	return df
=============

# Function 55
def cleaning_func_0(housing_data):
	# core cleaning code
	import pandas as pd
	# housing_data = pd.read_csv('../input/kc_house_data.csv')
	import datetime
	current_year = datetime.datetime.now().year
	housing_data['age_of_house'] = (current_year - pd.to_datetime(housing_data['date']).dt.year)
	return housing_data
=============

# Function 56
def cleaning_func_0(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df['family'] = (df.view + df.condition)
	return df
=============

# Function 57
def cleaning_func_0(df):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df['date'] = pd.to_datetime(df['date'])
	df['date'] = ((df['date'] - df['date'].min()) / np.timedelta64(1, 'D'))
	return df
=============

# Function 58
def cleaning_func_1(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/kc_house_data.csv')
	data['zipcode'] = data['zipcode'].astype('category', ordered=False)
	return data
=============

# Function 59
def cleaning_func_0(df):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# df = pd.read_csv('../input/kc_house_data.csv')
	df['sqft_living'] = np.log1p(df['sqft_living'])
	return df
=============

