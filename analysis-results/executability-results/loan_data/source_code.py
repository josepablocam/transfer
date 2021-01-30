# Function 0
def cleaning_func_0(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', low_memory=False)
	data.earliest_cr_line = pd.to_datetime(data.earliest_cr_line)
	return data
=============

# Function 1
def cleaning_func_1(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', low_memory=False)
	data['term'] = data['term'].apply((lambda x: x.lstrip()))
	return data
=============

# Function 2
def cleaning_func_2(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', low_memory=False)
	data.earliest_cr_line = pd.to_datetime(data.earliest_cr_line)
	data['earliest_cr_line_year'] = data['earliest_cr_line'].dt.year
	return data
=============

# Function 3
def cleaning_func_3(data):
	# additional context code from user definitions

	def impute_missing_algo(df, target, cats, cols, algo):
	    y = pd.DataFrame(df[target])
	    X = df[cols].copy()
	    X.drop(cats, axis=1, inplace=True)
	    cats = pd.get_dummies(df[cats])
	    X = pd.concat([X, cats], axis=1)
	    y['null'] = y[target].isnull()
	    y['null'] = y.loc[:, target].isnull()
	    X['null'] = y[target].isnull()
	    y_missing = y[(y['null'] == True)]
	    y_notmissing = y[(y['null'] == False)]
	    X_missing = X[(X['null'] == True)]
	    X_notmissing = X[(X['null'] == False)]
	    y_missing.loc[:, target] = ''
	    dfs = [y_missing, y_notmissing, X_missing, X_notmissing]
	    for df in dfs:
	        df.drop('null', inplace=True, axis=1)
	    y_missing = y_missing.values.ravel(order='C')
	    y_notmissing = y_notmissing.values.ravel(order='C')
	    X_missing = X_missing.as_matrix()
	    X_notmissing = X_notmissing.as_matrix()
	    algo.fit(X_notmissing, y_notmissing)
	    y_missing = algo.predict(X_missing)
	    y.loc[((y['null'] == True), target)] = y_missing
	    y.loc[((y['null'] == False), target)] = y_notmissing
	    return y[target]

	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', low_memory=False)
	from sklearn.ensemble import RandomForestClassifier
	rf = RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1)
	catiables = ['term', 'purpose', 'grade']
	columns = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'grade', 'purpose', 'term']
	data['earliest_cr_line_year'] = impute_missing_algo(data, 'earliest_cr_line_year', catiables, columns, rf)
	return data
=============

# Function 4
def cleaning_func_4(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', low_memory=False)
	data['emp_length'] = data['emp_length'].astype(int)
	return data
=============

# Function 5
def cleaning_func_5(data):
	# additional context code from user definitions

	def impute_missing_algo(df, target, cats, cols, algo):
	    y = pd.DataFrame(df[target])
	    X = df[cols].copy()
	    X.drop(cats, axis=1, inplace=True)
	    cats = pd.get_dummies(df[cats])
	    X = pd.concat([X, cats], axis=1)
	    y['null'] = y[target].isnull()
	    y['null'] = y.loc[:, target].isnull()
	    X['null'] = y[target].isnull()
	    y_missing = y[(y['null'] == True)]
	    y_notmissing = y[(y['null'] == False)]
	    X_missing = X[(X['null'] == True)]
	    X_notmissing = X[(X['null'] == False)]
	    y_missing.loc[:, target] = ''
	    dfs = [y_missing, y_notmissing, X_missing, X_notmissing]
	    for df in dfs:
	        df.drop('null', inplace=True, axis=1)
	    y_missing = y_missing.values.ravel(order='C')
	    y_notmissing = y_notmissing.values.ravel(order='C')
	    X_missing = X_missing.as_matrix()
	    X_notmissing = X_notmissing.as_matrix()
	    algo.fit(X_notmissing, y_notmissing)
	    y_missing = algo.predict(X_missing)
	    y.loc[((y['null'] == True), target)] = y_missing
	    y.loc[((y['null'] == False), target)] = y_notmissing
	    return y[target]

	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', low_memory=False)
	from sklearn.ensemble import RandomForestClassifier
	rf = RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1)
	catiables = ['term', 'purpose', 'grade']
	columns = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'grade', 'purpose', 'term']
	data['emp_length'] = impute_missing_algo(data, 'emp_length', catiables, columns, rf)
	return data
=============

# Function 6
def cleaning_func_6(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', low_memory=False)
	data.issue_d = pd.Series(data.issue_d).str.replace('-2015', '')
	return data
=============

# Function 7
def cleaning_func_8(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', low_memory=False)
	data.earliest_cr_line = pd.to_datetime(data.earliest_cr_line)
	s = pd.value_counts(data['earliest_cr_line']).to_frame().reset_index()
	s.columns = ['date', 'count']
	return s
=============

# Function 8
def cleaning_func_9(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', low_memory=False)
	data.earliest_cr_line = pd.to_datetime(data.earliest_cr_line)
	s = pd.value_counts(data['earliest_cr_line']).to_frame().reset_index()
	s['year'] = s['date'].dt.year
	return s
=============

# Function 9
def cleaning_func_11(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', low_memory=False)
	data['emp_length'] = data['emp_length'].astype(int)
	s = pd.value_counts(data['emp_length']).to_frame().reset_index()
	s.columns = ['type', 'count']
	return s
=============

# Function 10
def cleaning_func_12(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', low_memory=False)
	data.earliest_cr_line = pd.to_datetime(data.earliest_cr_line)
	s = pd.value_counts(data['earliest_cr_line']).to_frame().reset_index()
	s['month'] = s['date'].dt.month
	return s
=============

# Function 11
def cleaning_func_0(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['term'] = dataset['term'].astype('category').cat.codes
	return dataset
=============

# Function 12
def cleaning_func_1(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['verification_status'] = dataset['verification_status'].astype('category').cat.codes
	return dataset
=============

# Function 13
def cleaning_func_2(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['purpose'] = dataset['purpose'].astype('category').cat.codes
	return dataset
=============

# Function 14
def cleaning_func_3(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['application_type'] = dataset['application_type'].astype('category').cat.codes
	return dataset
=============

# Function 15
def cleaning_func_4(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['addr_state'] = dataset['addr_state'].astype('category').cat.codes
	return dataset
=============

# Function 16
def cleaning_func_5(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['sub_grade'] = dataset['sub_grade'].astype('category').cat.codes
	return dataset
=============

# Function 17
def cleaning_func_6(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['loan_status'] = dataset['loan_status'].astype('category').cat.codes
	return dataset
=============

# Function 18
def cleaning_func_7(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['initial_list_status'] = dataset['initial_list_status'].astype('category').cat.codes
	return dataset
=============

# Function 19
def cleaning_func_8(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['emp_length'] = dataset['emp_length'].astype('category').cat.codes
	return dataset
=============

# Function 20
def cleaning_func_9(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['verification_status_joint'] = dataset['verification_status_joint'].astype('category').cat.codes
	return dataset
=============

# Function 21
def cleaning_func_10(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['home_ownership'] = dataset['home_ownership'].astype('category').cat.codes
	return dataset
=============

# Function 22
def cleaning_func_11(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['pymnt_plan'] = dataset['pymnt_plan'].astype('category').cat.codes
	return dataset
=============

# Function 23
def cleaning_func_12(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['grade'] = dataset['grade'].astype('category').cat.codes
	return dataset
=============

# Function 24
def cleaning_func_13(dataset):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['earliest_cr_line'] = pd.to_datetime(dataset['earliest_cr_line'])
	dataset['earliest_cr_line'] = ((dataset['earliest_cr_line'] - dataset['earliest_cr_line'].min()) / np.timedelta64(1, 'D'))
	return dataset
=============

# Function 25
def cleaning_func_14(dataset):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['last_pymnt_d'] = pd.to_datetime(dataset['last_pymnt_d'])
	dataset['last_pymnt_d'] = ((dataset['last_pymnt_d'] - dataset['last_pymnt_d'].min()) / np.timedelta64(1, 'D'))
	return dataset
=============

# Function 26
def cleaning_func_15(dataset):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['last_credit_pull_d'] = pd.to_datetime(dataset['last_credit_pull_d'])
	dataset['last_credit_pull_d'] = ((dataset['last_credit_pull_d'] - dataset['last_credit_pull_d'].min()) / np.timedelta64(1, 'D'))
	return dataset
=============

# Function 27
def cleaning_func_16(dataset):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['issue_d'] = pd.to_datetime(dataset['issue_d'])
	dataset['issue_d'] = ((dataset['issue_d'] - dataset['issue_d'].min()) / np.timedelta64(1, 'D'))
	return dataset
=============

# Function 28
def cleaning_func_17(dataset):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['next_pymnt_d'] = pd.to_datetime(dataset['next_pymnt_d'])
	dataset['next_pymnt_d'] = ((dataset['next_pymnt_d'] - dataset['next_pymnt_d'].min()) / np.timedelta64(1, 'D'))
	return dataset
=============

# Function 29
def cleaning_func_18(dataset):
	# additional context code from user definitions

	def LoanResult(status):
	    if ((status == 5) or (status == 1) or (status == 7)):
	        return 1
	    else:
	        return 0

	# core cleaning code
	import numpy as np
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['loan_status'] = dataset['loan_status'].astype('category').cat.codes
	non_numerics = [x for x in dataset.columns if (not ((dataset[x].dtype == np.float64) or (dataset[x].dtype == np.int8) or (dataset[x].dtype == np.int64)))]
	df = dataset
	df = df.drop(non_numerics, 1)
	df['loan_status'] = df['loan_status'].apply(LoanResult)
	return df
=============

# Function 30
def cleaning_func_19(dataset):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['loan_status'] = dataset['loan_status'].astype('category').cat.codes
	non_numerics = [x for x in dataset.columns if (not ((dataset[x].dtype == np.float64) or (dataset[x].dtype == np.int8) or (dataset[x].dtype == np.int64)))]
	df = dataset
	return df
=============

# Function 31
def cleaning_func_0(df):
	# additional context code from user definitions

	def status_class(text):
	    if (text == 'Fully Paid'):
	        return 'Fully Paid'
	    elif ((text == 'Charged Off') or (text == 'Default')):
	        return 'Default'
	    elif ((text == 'Current') or (text == 'Issued')):
	        return 'Current'
	    else:
	        return 'Late'

	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df['status_class'] = df['loan_status'].apply(status_class)
	return df
=============

# Function 32
def cleaning_func_1(df):
	# additional context code from user definitions

	def emp_length_class(text):
	    if ((text == '< 1 year') or (text == '1 year') or (text == '2 years') or (text == '3 years')):
	        return '<=3 years'
	    elif ((text == '4 years') or (text == '5 years') or (text == '6 years')):
	        return '4-6 years'
	    elif ((text == '7 years') or (text == '8 years') or (text == '9 years')):
	        return '7-9 years'
	    elif (text == '10+ years'):
	        return '>=10 years'
	    else:
	        return None

	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df['emp_length_class'] = df['emp_length'].apply(emp_length_class)
	return df
=============

# Function 33
def cleaning_func_2(df):
	# additional context code from user definitions

	def inc_class(num):
	    if (num <= 50000):
	        return '<=50000'
	    elif (num <= 75000):
	        return '50000-75000'
	    elif (num <= 100000):
	        return '75000-100000'
	    elif (num <= 125000):
	        return '100000-125000'
	    elif (num <= 150000):
	        return '125000-150000'
	    else:
	        return '>150000'

	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df['inc_class'] = df['annual_inc'].apply(inc_class)
	return df
=============

# Function 34
def cleaning_func_3(df):
	# additional context code from user definitions

	def loan_class(num):
	    if (num <= 10000):
	        return '<=10000'
	    elif (num <= 20000):
	        return '10000-20000'
	    elif (num <= 30000):
	        return '20000-30000'
	    else:
	        return '>30000'

	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df['loan_class'] = df['loan_amnt'].apply(loan_class)
	return df
=============

# Function 35
def cleaning_func_4(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	df = df.dropna(thresh=(len(df) / 2), axis=1)
	return df
=============

# Function 36
def cleaning_func_5(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	return df
=============

# Function 37
def cleaning_func_6(df):
	# additional context code from user definitions

	def status_binary(text):
	    if (text == 'Fully Paid'):
	        return 0
	    elif ((text == 'Current') or (text == 'Issued')):
	        return (- 1)
	    else:
	        return 1

	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	df = df.dropna(thresh=(len(df) / 2), axis=1)
	df = df.dropna()
	df['loan_status'] = df['loan_status'].apply(status_binary)
	return df
=============

# Function 38
def cleaning_func_7(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	new_df = df[(df['addr_state'] == x)]
	new_df['weighted'] = ((new_df['int_rate'] / 100) * new_df['funded_amnt'])
	return new_df
=============

# Function 39
def cleaning_func_9(df):
	# additional context code from user definitions

	def purpose_class(text):
	    if ((text == 'debt_consolidation') or (text == 'credit_card')):
	        return 'refinance'
	    elif ((text == 'house') or (text == 'home_improvement') or (text == 'renewable_energy') or (text == 'moving')):
	        return 'home'
	    elif ((text == 'car') or (text == 'major_purchase')):
	        return 'major_purchase'
	    else:
	        return 'other'


	def status_binary(text):
	    if (text == 'Fully Paid'):
	        return 0
	    elif ((text == 'Current') or (text == 'Issued')):
	        return (- 1)
	    else:
	        return 1

	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	df = df.dropna(thresh=(len(df) / 2), axis=1)
	df = df.dropna()
	df['loan_status'] = df['loan_status'].apply(status_binary)
	df = df[(df['loan_status'] != (- 1))]
	df['purpose'] = df['purpose'].apply(purpose_class)
	return df
=============

# Function 40
def cleaning_func_12(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv')
	State_List = []
	Loan_Amount = []
	Average_Balance = []
	Default_Rate = []
	Weighted_Rate = []
	Average_Income = []
	Average_Employment_Length = []
	Average_Inq_12 = []
	Average_Inq_6 = []
	from collections import OrderedDict
	combine_data = OrderedDict([('Loan_Funding', Loan_Amount), ('Average_Balance', Average_Balance), ('Default_Rate', Default_Rate), ('Weighted_Rate', Weighted_Rate), ('Average_Income', Average_Income), ('Average_Employment_Length', Average_Employment_Length), ('Average_DTI', DTI_Average), ('12m_Inquiries', Average_Inq_12), ('6m_Inquiries', Average_Inq_6), ('code', State_List)])
	df_plot = pd.DataFrame.from_dict(combine_data)
	df_plot = df_plot.round(decimals=2)
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col]
	df_plot[col].astype = df_plot[col].astype
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col].astype(str)
	return df_plot
=============

# Function 41
def cleaning_func_13(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv')
	State_List = []
	Loan_Amount = []
	Average_Balance = []
	Default_Rate = []
	Weighted_Rate = []
	Average_Income = []
	Average_Employment_Length = []
	Average_Inq_12 = []
	Average_Inq_6 = []
	from collections import OrderedDict
	combine_data = OrderedDict([('Loan_Funding', Loan_Amount), ('Average_Balance', Average_Balance), ('Default_Rate', Default_Rate), ('Weighted_Rate', Weighted_Rate), ('Average_Income', Average_Income), ('Average_Employment_Length', Average_Employment_Length), ('Average_DTI', DTI_Average), ('12m_Inquiries', Average_Inq_12), ('6m_Inquiries', Average_Inq_6), ('code', State_List)])
	df_plot = pd.DataFrame.from_dict(combine_data)
	return df_plot
=============

# Function 42
def cleaning_func_14(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv')
	State_List = []
	Loan_Amount = []
	Average_Balance = []
	Default_Rate = []
	Weighted_Rate = []
	Average_Income = []
	Average_Employment_Length = []
	Average_Inq_12 = []
	Average_Inq_6 = []
	from collections import OrderedDict
	combine_data = OrderedDict([('Loan_Funding', Loan_Amount), ('Average_Balance', Average_Balance), ('Default_Rate', Default_Rate), ('Weighted_Rate', Weighted_Rate), ('Average_Income', Average_Income), ('Average_Employment_Length', Average_Employment_Length), ('Average_DTI', DTI_Average), ('12m_Inquiries', Average_Inq_12), ('6m_Inquiries', Average_Inq_6), ('code', State_List)])
	df_plot = pd.DataFrame.from_dict(combine_data)
	df_plot = df_plot.round(decimals=2)
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col]
	df_plot[col].astype = df_plot[col].astype
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col]
	df_plot[col].astype = df_plot[col].astype
	state_average_int_rate = df.groupby('addr_state').agg({'int_rate': np.average, 'id': np.count_nonzero, 'annual_inc': np.average})
	state_average_int_rate['interest'] = state_average_int_rate['int_rate']
	return state_average_int_rate
=============

# Function 43
def cleaning_func_15(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv')
	State_List = []
	Loan_Amount = []
	Average_Balance = []
	Default_Rate = []
	Weighted_Rate = []
	Average_Income = []
	Average_Employment_Length = []
	Average_Inq_12 = []
	Average_Inq_6 = []
	from collections import OrderedDict
	combine_data = OrderedDict([('Loan_Funding', Loan_Amount), ('Average_Balance', Average_Balance), ('Default_Rate', Default_Rate), ('Weighted_Rate', Weighted_Rate), ('Average_Income', Average_Income), ('Average_Employment_Length', Average_Employment_Length), ('Average_DTI', DTI_Average), ('12m_Inquiries', Average_Inq_12), ('6m_Inquiries', Average_Inq_6), ('code', State_List)])
	df_plot = pd.DataFrame.from_dict(combine_data)
	df_plot = df_plot.round(decimals=2)
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col]
	df_plot[col].astype = df_plot[col].astype
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col]
	df_plot[col].astype = df_plot[col].astype
	state_average_int_rate = df.groupby('addr_state').agg({'int_rate': np.average, 'id': np.count_nonzero, 'annual_inc': np.average})
	state_average_int_rate['id'] = state_average_int_rate['id'].astype(str)
	return state_average_int_rate
=============

# Function 44
def cleaning_func_16(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv')
	State_List = []
	Loan_Amount = []
	Average_Balance = []
	Default_Rate = []
	Weighted_Rate = []
	Average_Income = []
	Average_Employment_Length = []
	Average_Inq_12 = []
	Average_Inq_6 = []
	from collections import OrderedDict
	combine_data = OrderedDict([('Loan_Funding', Loan_Amount), ('Average_Balance', Average_Balance), ('Default_Rate', Default_Rate), ('Weighted_Rate', Weighted_Rate), ('Average_Income', Average_Income), ('Average_Employment_Length', Average_Employment_Length), ('Average_DTI', DTI_Average), ('12m_Inquiries', Average_Inq_12), ('6m_Inquiries', Average_Inq_6), ('code', State_List)])
	df_plot = pd.DataFrame.from_dict(combine_data)
	df_plot = df_plot.round(decimals=2)
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col]
	df_plot[col].astype = df_plot[col].astype
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col]
	df_plot[col].astype = df_plot[col].astype
	state_average_int_rate = df.groupby('addr_state').agg({'int_rate': np.average, 'id': np.count_nonzero, 'annual_inc': np.average})
	state_average_int_rate['int_rate'] = (('Average Interest Rate: ' + state_average_int_rate['int_rate'].apply((lambda x: str(round(x, 2))))) + '%')
	return state_average_int_rate
=============

# Function 45
def cleaning_func_17(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv')
	State_List = []
	Loan_Amount = []
	Average_Balance = []
	Default_Rate = []
	Weighted_Rate = []
	Average_Income = []
	Average_Employment_Length = []
	Average_Inq_12 = []
	Average_Inq_6 = []
	from collections import OrderedDict
	combine_data = OrderedDict([('Loan_Funding', Loan_Amount), ('Average_Balance', Average_Balance), ('Default_Rate', Default_Rate), ('Weighted_Rate', Weighted_Rate), ('Average_Income', Average_Income), ('Average_Employment_Length', Average_Employment_Length), ('Average_DTI', DTI_Average), ('12m_Inquiries', Average_Inq_12), ('6m_Inquiries', Average_Inq_6), ('code', State_List)])
	df_plot = pd.DataFrame.from_dict(combine_data)
	df_plot = df_plot.round(decimals=2)
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col]
	df_plot[col].astype = df_plot[col].astype
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col]
	df_plot[col].astype = df_plot[col].astype
	state_average_int_rate = df.groupby('addr_state').agg({'int_rate': np.average, 'id': np.count_nonzero, 'annual_inc': np.average})
	state_average_int_rate['annual_inc'] = (state_average_int_rate['annual_inc'] / 1000.0)
	state_average_int_rate['annual_inc'] = state_average_int_rate['annual_inc'].apply((lambda x: str(round(x, 2))))
	return state_average_int_rate
=============

# Function 46
def cleaning_func_19(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv')
	State_List = []
	Loan_Amount = []
	Average_Balance = []
	Default_Rate = []
	Weighted_Rate = []
	Average_Income = []
	Average_Employment_Length = []
	Average_Inq_12 = []
	Average_Inq_6 = []
	from collections import OrderedDict
	combine_data = OrderedDict([('Loan_Funding', Loan_Amount), ('Average_Balance', Average_Balance), ('Default_Rate', Default_Rate), ('Weighted_Rate', Weighted_Rate), ('Average_Income', Average_Income), ('Average_Employment_Length', Average_Employment_Length), ('Average_DTI', DTI_Average), ('12m_Inquiries', Average_Inq_12), ('6m_Inquiries', Average_Inq_6), ('code', State_List)])
	df_plot = pd.DataFrame.from_dict(combine_data)
	df_plot = df_plot.round(decimals=2)
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col]
	df_plot[col].astype = df_plot[col].astype
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col].astype(str)
	df_plot[col] = df_plot[col]
	df_plot[col].astype = df_plot[col].astype
	state_average_int_rate = df.groupby('addr_state').agg({'int_rate': np.average, 'id': np.count_nonzero, 'annual_inc': np.average})
	state_average_int_rate['id'] = state_average_int_rate['id'].astype(str)
	state_average_int_rate['annual_inc'] = (state_average_int_rate['annual_inc'] / 1000.0)
	state_average_int_rate['annual_inc'] = state_average_int_rate['annual_inc'].apply((lambda x: str(round(x, 2))))
	state_average_int_rate['text'] = ((((('Number of Applicants: ' + state_average_int_rate['id']) + '<br>') + 'Average Annual Inc: $') + state_average_int_rate['annual_inc']) + 'k')
	return state_average_int_rate
=============

# Function 47
def cleaning_func_20(df):
	# additional context code from user definitions

	def status_binary(text):
	    if (text == 'Fully Paid'):
	        return 0
	    elif ((text == 'Current') or (text == 'Issued')):
	        return (- 1)
	    else:
	        return 1

	# core cleaning code
	import pandas as pd
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	df = df.dropna(thresh=(len(df) / 2), axis=1)
	df = df.dropna()
	df['loan_status'] = df['loan_status'].apply(status_binary)
	df = df[(df['loan_status'] != (- 1))]
	dummy_df = pd.get_dummies(df[['home_ownership', 'verification_status', 'purpose', 'term']])
	df = df.drop(['home_ownership', 'verification_status', 'purpose', 'term'], axis=1)
	df = pd.concat([df, dummy_df], axis=1)
	mapping_dict = {'emp_length': {'10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6, '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1, '< 1 year': 0, 'n/a': 0}, 'grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}}
	df = df.replace(mapping_dict)
	cols = list(df)
	df = df.ix[(slice(None, None, None), cols)]
	from sklearn.model_selection import train_test_split
	(train, test) = train_test_split(df, test_size=0.3)
	x_train = train.iloc[(slice(0, None, None), slice(1, 34, None))]
	y_train = train[['loan_status']]
	return y_train
=============

# Function 48
def cleaning_func_22(df):
	# additional context code from user definitions

	def status_binary(text):
	    if (text == 'Fully Paid'):
	        return 0
	    elif ((text == 'Current') or (text == 'Issued')):
	        return (- 1)
	    else:
	        return 1

	# core cleaning code
	import pandas as pd
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	df = df.dropna(thresh=(len(df) / 2), axis=1)
	df = df.dropna()
	df['loan_status'] = df['loan_status'].apply(status_binary)
	df = df[(df['loan_status'] != (- 1))]
	dummy_df = pd.get_dummies(df[['home_ownership', 'verification_status', 'purpose', 'term']])
	df = df.drop(['home_ownership', 'verification_status', 'purpose', 'term'], axis=1)
	df = pd.concat([df, dummy_df], axis=1)
	mapping_dict = {'emp_length': {'10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6, '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1, '< 1 year': 0, 'n/a': 0}, 'grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}}
	df = df.replace(mapping_dict)
	cols = list(df)
	df = df.ix[(slice(None, None, None), cols)]
	return df
=============

# Function 49
def cleaning_func_23(df):
	# additional context code from user definitions

	def status_binary(text):
	    if (text == 'Fully Paid'):
	        return 0
	    elif ((text == 'Current') or (text == 'Issued')):
	        return (- 1)
	    else:
	        return 1

	# core cleaning code
	import pandas as pd
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	df = df.dropna(thresh=(len(df) / 2), axis=1)
	df = df.dropna()
	df['loan_status'] = df['loan_status'].apply(status_binary)
	df = df[(df['loan_status'] != (- 1))]
	dummy_df = pd.get_dummies(df[['home_ownership', 'verification_status', 'purpose', 'term']])
	df = df.drop(['home_ownership', 'verification_status', 'purpose', 'term'], axis=1)
	df = pd.concat([df, dummy_df], axis=1)
	mapping_dict = {'emp_length': {'10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6, '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1, '< 1 year': 0, 'n/a': 0}, 'grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}}
	df = df.replace(mapping_dict)
	cols = list(df)
	df = df.ix[(slice(None, None, None), cols)]
	from sklearn.model_selection import train_test_split
	(train, test) = train_test_split(df, test_size=0.3)
	x_train = train.iloc[(slice(0, None, None), slice(1, 34, None))]
	return x_train
=============

# Function 50
def cleaning_func_25(df):
	# additional context code from user definitions

	def status_binary(text):
	    if (text == 'Fully Paid'):
	        return 0
	    elif ((text == 'Current') or (text == 'Issued')):
	        return (- 1)
	    else:
	        return 1

	# core cleaning code
	import pandas as pd
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	df = df.dropna(thresh=(len(df) / 2), axis=1)
	df = df.dropna()
	df['loan_status'] = df['loan_status'].apply(status_binary)
	df = df[(df['loan_status'] != (- 1))]
	dummy_df = pd.get_dummies(df[['home_ownership', 'verification_status', 'purpose', 'term']])
	df = df.drop(['home_ownership', 'verification_status', 'purpose', 'term'], axis=1)
	df = pd.concat([df, dummy_df], axis=1)
	mapping_dict = {'emp_length': {'10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6, '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1, '< 1 year': 0, 'n/a': 0}, 'grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}}
	df = df.replace(mapping_dict)
	cols = list(df)
	df = df.ix[(slice(None, None, None), cols)]
	from sklearn.model_selection import train_test_split
	(train, test) = train_test_split(df, test_size=0.3)
	x_train = train.iloc[(slice(0, None, None), slice(1, 34, None))]
	y_train = train[['loan_status']]
	x_test = test.iloc[(slice(0, None, None), slice(1, 34, None))]
	y_test = test[['loan_status']]
	return y_test
=============

# Function 51
def cleaning_func_26(df):
	# additional context code from user definitions

	def status_binary(text):
	    if (text == 'Fully Paid'):
	        return 0
	    elif ((text == 'Current') or (text == 'Issued')):
	        return (- 1)
	    else:
	        return 1

	# core cleaning code
	import pandas as pd
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	df = df.dropna(thresh=(len(df) / 2), axis=1)
	df = df.dropna()
	df['loan_status'] = df['loan_status'].apply(status_binary)
	df = df[(df['loan_status'] != (- 1))]
	dummy_df = pd.get_dummies(df[['home_ownership', 'verification_status', 'purpose', 'term']])
	df = df.drop(['home_ownership', 'verification_status', 'purpose', 'term'], axis=1)
	df = pd.concat([df, dummy_df], axis=1)
	mapping_dict = {'emp_length': {'10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6, '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1, '< 1 year': 0, 'n/a': 0}, 'grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}}
	df = df.replace(mapping_dict)
	cols = list(df)
	df = df.ix[(slice(None, None, None), cols)]
	from sklearn.model_selection import train_test_split
	(train, test) = train_test_split(df, test_size=0.3)
	x_train = train.iloc[(slice(0, None, None), slice(1, 34, None))]
	y_train = train[['loan_status']]
	x_test = test.iloc[(slice(0, None, None), slice(1, 34, None))]
	y_test = test[['loan_status']]
	method = ['Decision Tree', 'Random Forests', 'Logistic Regression']
	false_paid = pd.DataFrame([[0, 0, 0], [0, 0, 0]], columns=method, index=['train', 'test'])
	default_identified = pd.DataFrame([[0, 0, 0], [0, 0, 0]], columns=method, index=['train', 'test'])
	from sklearn.tree import DecisionTreeClassifier
	from sklearn import tree
	model = tree.DecisionTreeClassifier(max_depth=5, criterion='entropy', class_weight={0: 0.15, 1: 0.85})
	from sklearn.metrics import confusion_matrix
	import numpy as np
	p_train = model.predict(x_train)
	p_test = model.predict(x_test)
	(fully_paid, default) = confusion_matrix(p_train, np.array(y_train))
	false_paid.loc[('train', 'Decision Tree')] = ((100 * fully_paid[1]) / (fully_paid[0] + fully_paid[1]))
	default_identified.loc[('train', 'Decision Tree')] = ((100 * default[1]) / (default[1] + fully_paid[1]))
	(fully_paid, default) = confusion_matrix(p_test, np.array(y_test))
	false_paid.loc[('test', 'Decision Tree')] = ((100 * fully_paid[1]) / (fully_paid[0] + fully_paid[1]))
	return false_paid
=============

# Function 52
def cleaning_func_27(df):
	# additional context code from user definitions

	def status_binary(text):
	    if (text == 'Fully Paid'):
	        return 0
	    elif ((text == 'Current') or (text == 'Issued')):
	        return (- 1)
	    else:
	        return 1

	# core cleaning code
	import pandas as pd
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	df = df.dropna(thresh=(len(df) / 2), axis=1)
	df = df.dropna()
	df['loan_status'] = df['loan_status'].apply(status_binary)
	df = df[(df['loan_status'] != (- 1))]
	dummy_df = pd.get_dummies(df[['home_ownership', 'verification_status', 'purpose', 'term']])
	df = df.drop(['home_ownership', 'verification_status', 'purpose', 'term'], axis=1)
	df = pd.concat([df, dummy_df], axis=1)
	mapping_dict = {'emp_length': {'10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6, '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1, '< 1 year': 0, 'n/a': 0}, 'grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}}
	df = df.replace(mapping_dict)
	cols = list(df)
	df = df.ix[(slice(None, None, None), cols)]
	from sklearn.model_selection import train_test_split
	(train, test) = train_test_split(df, test_size=0.3)
	return train
=============

# Function 53
def cleaning_func_28(df):
	# additional context code from user definitions

	def status_binary(text):
	    if (text == 'Fully Paid'):
	        return 0
	    elif ((text == 'Current') or (text == 'Issued')):
	        return (- 1)
	    else:
	        return 1

	# core cleaning code
	import pandas as pd
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	df = df.dropna(thresh=(len(df) / 2), axis=1)
	df = df.dropna()
	df['loan_status'] = df['loan_status'].apply(status_binary)
	df = df[(df['loan_status'] != (- 1))]
	dummy_df = pd.get_dummies(df[['home_ownership', 'verification_status', 'purpose', 'term']])
	df = df.drop(['home_ownership', 'verification_status', 'purpose', 'term'], axis=1)
	df = pd.concat([df, dummy_df], axis=1)
	mapping_dict = {'emp_length': {'10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6, '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1, '< 1 year': 0, 'n/a': 0}, 'grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}}
	df = df.replace(mapping_dict)
	return df
=============

# Function 54
def cleaning_func_30(df):
	# additional context code from user definitions

	def status_binary(text):
	    if (text == 'Fully Paid'):
	        return 0
	    elif ((text == 'Current') or (text == 'Issued')):
	        return (- 1)
	    else:
	        return 1

	# core cleaning code
	import pandas as pd
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	df = df.dropna(thresh=(len(df) / 2), axis=1)
	df = df.dropna()
	df['loan_status'] = df['loan_status'].apply(status_binary)
	df = df[(df['loan_status'] != (- 1))]
	dummy_df = pd.get_dummies(df[['home_ownership', 'verification_status', 'purpose', 'term']])
	return dummy_df
=============

# Function 55
def cleaning_func_35(df):
	# additional context code from user definitions

	def status_binary(text):
	    if (text == 'Fully Paid'):
	        return 0
	    elif ((text == 'Current') or (text == 'Issued')):
	        return (- 1)
	    else:
	        return 1

	# core cleaning code
	import pandas as pd
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	df = df.dropna(thresh=(len(df) / 2), axis=1)
	df = df.dropna()
	df['loan_status'] = df['loan_status'].apply(status_binary)
	df = df[(df['loan_status'] != (- 1))]
	dummy_df = pd.get_dummies(df[['home_ownership', 'verification_status', 'purpose', 'term']])
	df = df.drop(['home_ownership', 'verification_status', 'purpose', 'term'], axis=1)
	df = pd.concat([df, dummy_df], axis=1)
	return df
=============

# Function 56
def cleaning_func_36(df):
	# additional context code from user definitions

	def status_binary(text):
	    if (text == 'Fully Paid'):
	        return 0
	    elif ((text == 'Current') or (text == 'Issued')):
	        return (- 1)
	    else:
	        return 1

	# core cleaning code
	import pandas as pd
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	df = df.dropna(thresh=(len(df) / 2), axis=1)
	df = df.dropna()
	df['loan_status'] = df['loan_status'].apply(status_binary)
	df = df[(df['loan_status'] != (- 1))]
	dummy_df = pd.get_dummies(df[['home_ownership', 'verification_status', 'purpose', 'term']])
	df = df.drop(['home_ownership', 'verification_status', 'purpose', 'term'], axis=1)
	df = pd.concat([df, dummy_df], axis=1)
	mapping_dict = {'emp_length': {'10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6, '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1, '< 1 year': 0, 'n/a': 0}, 'grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}}
	df = df.replace(mapping_dict)
	cols = list(df)
	df = df.ix[(slice(None, None, None), cols)]
	from sklearn.model_selection import train_test_split
	(train, test) = train_test_split(df, test_size=0.3)
	x_train = train.iloc[(slice(0, None, None), slice(1, 34, None))]
	y_train = train[['loan_status']]
	x_test = test.iloc[(slice(0, None, None), slice(1, 34, None))]
	return x_test
=============

# Function 57
def cleaning_func_37(df):
	# additional context code from user definitions

	def status_binary(text):
	    if (text == 'Fully Paid'):
	        return 0
	    elif ((text == 'Current') or (text == 'Issued')):
	        return (- 1)
	    else:
	        return 1

	# core cleaning code
	import pandas as pd
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	df = df.dropna(thresh=(len(df) / 2), axis=1)
	df = df.dropna()
	df['loan_status'] = df['loan_status'].apply(status_binary)
	df = df[(df['loan_status'] != (- 1))]
	dummy_df = pd.get_dummies(df[['home_ownership', 'verification_status', 'purpose', 'term']])
	df = df.drop(['home_ownership', 'verification_status', 'purpose', 'term'], axis=1)
	df = pd.concat([df, dummy_df], axis=1)
	mapping_dict = {'emp_length': {'10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6, '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1, '< 1 year': 0, 'n/a': 0}, 'grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}}
	df = df.replace(mapping_dict)
	cols = list(df)
	df = df.ix[(slice(None, None, None), cols)]
	from sklearn.model_selection import train_test_split
	(train, test) = train_test_split(df, test_size=0.3)
	x_train = train.iloc[(slice(0, None, None), slice(1, 34, None))]
	y_train = train[['loan_status']]
	x_test = test.iloc[(slice(0, None, None), slice(1, 34, None))]
	y_test = test[['loan_status']]
	method = ['Decision Tree', 'Random Forests', 'Logistic Regression']
	false_paid = pd.DataFrame([[0, 0, 0], [0, 0, 0]], columns=method, index=['train', 'test'])
	default_identified = pd.DataFrame([[0, 0, 0], [0, 0, 0]], columns=method, index=['train', 'test'])
	from sklearn.tree import DecisionTreeClassifier
	from sklearn import tree
	model = tree.DecisionTreeClassifier(max_depth=5, criterion='entropy', class_weight={0: 0.15, 1: 0.85})
	from sklearn.metrics import confusion_matrix
	import numpy as np
	p_train = model.predict(x_train)
	p_test = model.predict(x_test)
	(fully_paid, default) = confusion_matrix(p_train, np.array(y_train))
	false_paid.loc[('train', 'Decision Tree')] = ((100 * fully_paid[1]) / (fully_paid[0] + fully_paid[1]))
	default_identified.loc[('train', 'Decision Tree')] = ((100 * default[1]) / (default[1] + fully_paid[1]))
	(fully_paid, default) = confusion_matrix(p_test, np.array(y_test))
	false_paid.loc[('test', 'Decision Tree')] = ((100 * fully_paid[1]) / (fully_paid[0] + fully_paid[1]))
	default_identified.loc[('test', 'Decision Tree')] = ((100 * default[1]) / (default[1] + fully_paid[1]))
	from sklearn.ensemble import RandomForestClassifier
	model = RandomForestClassifier(max_depth=6, n_estimators=10, class_weight={0: 0.15, 1: 0.85})
	from sklearn.metrics import confusion_matrix
	p_train = model.predict(x_train)
	p_test = model.predict(x_test)
	(fully_paid, default) = confusion_matrix(p_train, np.array(y_train))
	false_paid.loc[('train', 'Random Forests')] = ((100 * fully_paid[1]) / (fully_paid[0] + fully_paid[1]))
	default_identified.loc[('train', 'Random Forests')] = ((100 * default[1]) / (default[1] + fully_paid[1]))
	(fully_paid, default) = confusion_matrix(p_test, np.array(y_test))
	false_paid.loc[('test', 'Random Forests')] = ((100 * fully_paid[1]) / (fully_paid[0] + fully_paid[1]))
	return false_paid
=============

# Function 58
def cleaning_func_39(df):
	# additional context code from user definitions

	def status_binary(text):
	    if (text == 'Fully Paid'):
	        return 0
	    elif ((text == 'Current') or (text == 'Issued')):
	        return (- 1)
	    else:
	        return 1

	# core cleaning code
	import pandas as pd
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title', 'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type', 'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'], axis=1)
	df = df.dropna(thresh=(len(df) / 2), axis=1)
	df = df.dropna()
	df['loan_status'] = df['loan_status'].apply(status_binary)
	df = df[(df['loan_status'] != (- 1))]
	dummy_df = pd.get_dummies(df[['home_ownership', 'verification_status', 'purpose', 'term']])
	df = df.drop(['home_ownership', 'verification_status', 'purpose', 'term'], axis=1)
	df = pd.concat([df, dummy_df], axis=1)
	mapping_dict = {'emp_length': {'10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6, '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1, '< 1 year': 0, 'n/a': 0}, 'grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}}
	df = df.replace(mapping_dict)
	cols = list(df)
	df = df.ix[(slice(None, None, None), cols)]
	from sklearn.model_selection import train_test_split
	(train, test) = train_test_split(df, test_size=0.3)
	x_train = train.iloc[(slice(0, None, None), slice(1, 34, None))]
	y_train = train[['loan_status']]
	x_test = test.iloc[(slice(0, None, None), slice(1, 34, None))]
	y_test = test[['loan_status']]
	method = ['Decision Tree', 'Random Forests', 'Logistic Regression']
	false_paid = pd.DataFrame([[0, 0, 0], [0, 0, 0]], columns=method, index=['train', 'test'])
	default_identified = pd.DataFrame([[0, 0, 0], [0, 0, 0]], columns=method, index=['train', 'test'])
	from sklearn.tree import DecisionTreeClassifier
	from sklearn import tree
	model = tree.DecisionTreeClassifier(max_depth=5, criterion='entropy', class_weight={0: 0.15, 1: 0.85})
	from sklearn.metrics import confusion_matrix
	import numpy as np
	p_train = model.predict(x_train)
	p_test = model.predict(x_test)
	(fully_paid, default) = confusion_matrix(p_train, np.array(y_train))
	false_paid.loc[('train', 'Decision Tree')] = ((100 * fully_paid[1]) / (fully_paid[0] + fully_paid[1]))
	default_identified.loc[('train', 'Decision Tree')] = ((100 * default[1]) / (default[1] + fully_paid[1]))
	(fully_paid, default) = confusion_matrix(p_test, np.array(y_test))
	false_paid.loc[('test', 'Decision Tree')] = ((100 * fully_paid[1]) / (fully_paid[0] + fully_paid[1]))
	default_identified.loc[('test', 'Decision Tree')] = ((100 * default[1]) / (default[1] + fully_paid[1]))
	from sklearn.ensemble import RandomForestClassifier
	model = RandomForestClassifier(max_depth=6, n_estimators=10, class_weight={0: 0.15, 1: 0.85})
	from sklearn.metrics import confusion_matrix
	p_train = model.predict(x_train)
	p_test = model.predict(x_test)
	(fully_paid, default) = confusion_matrix(p_train, np.array(y_train))
	false_paid.loc[('train', 'Random Forests')] = ((100 * fully_paid[1]) / (fully_paid[0] + fully_paid[1]))
	default_identified.loc[('train', 'Random Forests')] = ((100 * default[1]) / (default[1] + fully_paid[1]))
	(fully_paid, default) = confusion_matrix(p_test, np.array(y_test))
	false_paid.loc[('test', 'Random Forests')] = ((100 * fully_paid[1]) / (fully_paid[0] + fully_paid[1]))
	default_identified.loc[('test', 'Random Forests')] = ((100 * default[1]) / (default[1] + fully_paid[1]))
	from sklearn.linear_model import LogisticRegression
	import numpy as np
	model = LogisticRegression(class_weight={0: 0.15, 1: 0.85})
	from sklearn.metrics import confusion_matrix
	p_train = model.predict(x_train)
	(fully_paid, default) = confusion_matrix(p_train, np.array(y_train))
	false_paid.loc[('train', 'Logistic Regression')] = ((100 * fully_paid[1]) / (fully_paid[0] + fully_paid[1]))
	return false_paid
=============

# Function 59
def cleaning_func_0(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['90day_worse_rating'] = np.where(loan['mths_since_last_major_derog'].isnull(), 0, 1)
	return loan
=============

# Function 60
def cleaning_func_1(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['tot_coll_amt'] = loan['tot_coll_amt'].fillna(loan['tot_coll_amt'].median())
	return loan
=============

# Function 61
def cleaning_func_2(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['revol_util'] = loan['revol_util'].fillna(loan['revol_util'].median())
	return loan
=============

# Function 62
def cleaning_func_3(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['total_acc'] = np.where(loan['total_acc'].isnull(), 0, loan['total_acc'])
	return loan
=============

# Function 63
def cleaning_func_4(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['tot_cur_bal'] = loan['tot_cur_bal'].fillna(loan['tot_cur_bal'].median())
	return loan
=============

# Function 64
def cleaning_func_5(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['open_acc'] = np.where(loan['open_acc'].isnull(), 0, loan['open_acc'])
	return loan
=============

# Function 65
def cleaning_func_6(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['collections_12_mths_ex_med'] = np.where(loan['collections_12_mths_ex_med'].isnull(), 0, loan['collections_12_mths_ex_med'])
	return loan
=============

# Function 66
def cleaning_func_7(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['total_rev_hi_lim'] = loan['total_rev_hi_lim'].fillna(loan['total_rev_hi_lim'].median())
	return loan
=============

# Function 67
def cleaning_func_8(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['title'] = np.where(loan['title'].isnull(), 0, loan['title'])
	return loan
=============

# Function 68
def cleaning_func_9(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['annual_inc'] = loan['annual_inc'].fillna(loan['annual_inc'].median())
	return loan
=============

# Function 69
def cleaning_func_10(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['delinq_2yrs'] = np.where(loan['delinq_2yrs'].isnull(), 0, loan['delinq_2yrs'])
	return loan
=============

# Function 70
def cleaning_func_11(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['acc_now_delinq'] = np.where(loan['acc_now_delinq'].isnull(), 0, loan['acc_now_delinq'])
	return loan
=============

# Function 71
def cleaning_func_12(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['inq_last_6mths'] = np.where(loan['inq_last_6mths'].isnull(), 0, loan['inq_last_6mths'])
	return loan
=============

# Function 72
def cleaning_func_13(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['pub_rec'] = np.where(loan['pub_rec'].isnull(), 0, loan['pub_rec'])
	return loan
=============

# Function 73
def cleaning_func_14(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['emp_title'] = np.where(loan['emp_title'].isnull(), 'Job title not given', loan['emp_title'])
	return loan
=============

# Function 74
def cleaning_func_15(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['mths_since_last_delinq'] = np.where(loan['mths_since_last_delinq'].isnull(), 188, loan['mths_since_last_delinq'])
	return loan
=============

# Function 75
def cleaning_func_0(ld):
	# core cleaning code
	import pandas as pd
	# ld = pd.read_csv('../input/loan.csv', low_memory=False, parse_dates=True)
	pct_full = (ld.count() / len(ld))
	names = list(pct_full[(pct_full > 0.75)].index)
	loan = ld[names]
	loan['pct_paid'] = (loan.out_prncp / loan.loan_amnt)
	return loan
=============

# Function 76
def cleaning_func_1(ld):
	# core cleaning code
	import pandas as pd
	# ld = pd.read_csv('../input/loan.csv', low_memory=False, parse_dates=True)
	pct_full = (ld.count() / len(ld))
	names = list(pct_full[(pct_full > 0.75)].index)
	loan = ld[names]
	loan['issue_mo'] = loan.issue_d.str[slice(0, 3, None)]
	return loan
=============

# Function 77
def cleaning_func_2(ld):
	# core cleaning code
	import pandas as pd
	# ld = pd.read_csv('../input/loan.csv', low_memory=False, parse_dates=True)
	pct_full = (ld.count() / len(ld))
	names = list(pct_full[(pct_full > 0.75)].index)
	loan = ld[names]
	loan['issue_year'] = loan.issue_d.str[slice(4, None, None)]
	return loan
=============

# Function 78
def cleaning_func_0(df_loan):
	# core cleaning code
	import pandas as pd
	# df_loan = pd.read_csv('../input/loan.csv', low_memory=False)
	df_loan.loc[((df_loan.loan_status == 'Does not meet the credit policy. Status:Fully Paid'), 'loan_status')] = 'NMCP Fully Paid'
	df_loan.loc[((df_loan.loan_status == 'Does not meet the credit policy. Status:Charged Off'), 'loan_status')] = 'NMCP Charged Off'
	return df_loan
=============

# Function 79
def cleaning_func_1(df_loan):
	# core cleaning code
	import pandas as pd
	# df_loan = pd.read_csv('../input/loan.csv', low_memory=False)
	(df_loan['issue_month'], df_loan['issue_year']) = df_loan['issue_d'].str.split('-', 1).str
	return df_loan
=============

# Function 80
def cleaning_func_2(df_loan):
	# core cleaning code
	import pandas as pd
	# df_loan = pd.read_csv('../input/loan.csv', low_memory=False)
	df_loan['int_round'] = df_loan['int_rate'].round(0).astype(int)
	return df_loan
=============

# Function 81
def cleaning_func_3(df_loan):
	# core cleaning code
	import pandas as pd
	# df_loan = pd.read_csv('../input/loan.csv', low_memory=False)
	(df_loan['issue_month'], df_loan['issue_year']) = df_loan['issue_d'].str.split('-', 1).str
	months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
	df_loan['issue_month'] = pd.Categorical(df_loan['issue_month'], categories=months_order, ordered=True)
	return df_loan
=============

# Function 82
def cleaning_func_0(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df.mths_since_last_delinq = df.mths_since_last_delinq.fillna(df.mths_since_last_delinq.median())
	return df
=============

# Function 83
def cleaning_func_1(df):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df['good_loan'] = np.where((((df.loan_status == 'Fully Paid') | (df.loan_status == 'Current')) | (df.loan_status == 'Does not meet the credit policy. Status:Fully Paid')), 1, 0)
	return df
=============

# Function 84
def cleaning_func_0(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data['bad_loan'] = 0
	return data
=============

# Function 85
def cleaning_func_1(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data['issue_dt'] = pd.to_datetime(data.issue_d)
	return data
=============

# Function 86
def cleaning_func_2(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	bad_indicators = ['Charged Off ', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Default Receiver', 'Late (16-30 days)', 'Late (31-120 days)']
	data.loc[(data.loan_status.isin(bad_indicators), 'bad_loan')] = 1
	return data
=============

# Function 87
def cleaning_func_3(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data['issue_dt'] = pd.to_datetime(data.issue_d)
	data['month'] = data['issue_dt'].dt.month
	return data
=============

# Function 88
def cleaning_func_4(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data['issue_dt'] = pd.to_datetime(data.issue_d)
	data['year'] = data['issue_dt'].dt.year
	return data
=============

# Function 89
def cleaning_func_0(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.loc[(df['loan_status'] != 'Current')]
	return df
=============

# Function 90
def cleaning_func_1(df):
	# additional context code from user definitions

	def duplicate_columns(df, return_dataframe=False, verbose=True):
	    '\n        a function to detect and possibly remove duplicated columns for a pandas dataframe\n    '
	    from pandas.core.common import array_equivalent
	    groups = df.columns.to_series().groupby(df.dtypes).groups
	    duplicated_columns = []
	    for (dtype, col_names) in groups.items():
	        column_values = df[col_names]
	        num_columns = len(col_names)
	        for i in range(num_columns):
	            column_i = column_values.iloc[:, i].values
	            for j in range((i + 1), num_columns):
	                column_j = column_values.iloc[:, j].values
	                if array_equivalent(column_i, column_j):
	                    if verbose:
	                        print('column {} is a duplicate of column {}'.format(col_names[i], col_names[j]))
	                    duplicated_columns.append(col_names[i])
	                    break
	    if (not return_dataframe):
	        return duplicated_columns
	    else:
	        return df.drop(labels=duplicated_columns, axis=1)

	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.loc[(df['loan_status'] != 'Current')]
	df = duplicate_columns(df, return_dataframe=True)
	df['loan_Default'] = int(0)
	return df
=============

# Function 91
def cleaning_func_0(loans):
	# core cleaning code
	import pandas as pd
	date = ['issue_d', 'last_pymnt_d']
	cols = ['issue_d', 'term', 'int_rate', 'loan_amnt', 'total_pymnt', 'last_pymnt_d', 'sub_grade', 'grade', 'loan_status']
	# loans = pd.read_csv('../input/loan.csv', low_memory=False, parse_dates=date, usecols=cols, infer_datetime_format=True)
	latest = loans['issue_d'].max()
	finished_bool = (((loans['issue_d'] < (latest - pd.DateOffset(years=3))) & (loans['term'] == ' 36 months')) | ((loans['issue_d'] < (latest - pd.DateOffset(years=5))) & (loans['term'] == ' 60 months')))
	finished_loans = loans.loc[finished_bool]
	finished_loans['roi'] = (((finished_loans.total_pymnt / finished_loans.loan_amnt) - 1) * 100)
	return finished_loans
=============

# Function 92
def cleaning_func_0(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	return df
=============

# Function 93
def cleaning_func_1(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	return df
=============

# Function 94
def cleaning_func_2(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	terms = []
	df1.term = terms
	return df1
=============

# Function 95
def cleaning_func_3(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	return df
=============

# Function 96
def cleaning_func_4(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	emp_lengths = []
	df1.emp_length = emp_lengths
	return df1
=============

# Function 97
def cleaning_func_5(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	return df1
=============

# Function 98
def cleaning_func_6(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	df1['revol_util_nan'] = (pd.isnull(df1.revol_util) * 1)
	return df1
=============

# Function 99
def cleaning_func_7(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	return df1
=============

# Function 100
def cleaning_func_10(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	df1['mths_since_last_delinq_nan'] = (np.isnan(df1.mths_since_last_delinq) * 1)
	return df1
=============

# Function 101
def cleaning_func_14(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	df1.tot_coll_amt = df1.tot_coll_amt.replace(np.nan, 0)
	return df1
=============

# Function 102
def cleaning_func_15(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	df1['mths_since_last_record_nan'] = (np.isnan(df1.mths_since_last_record) * 1)
	return df1
=============

# Function 103
def cleaning_func_16(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	emp_lengths = []
	df1.emp_length = emp_lengths
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	return df1
=============

# Function 104
def cleaning_func_17(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	emp_lengths = []
	df1.emp_length = emp_lengths
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	df1['emp_length_nan'] = (pd.isnull(df1.emp_length) * 1)
	return df1
=============

# Function 105
def cleaning_func_18(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	df1.tot_cur_bal = df1.tot_cur_bal.replace(np.nan, 0)
	return df1
=============

# Function 106
def cleaning_func_21(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import Imputer
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	imp = Imputer(strategy='median')
	df1.total_rev_hi_lim = imp.fit_transform(df1.total_rev_hi_lim.reshape((- 1), 1))
	return df1
=============

# Function 107
def cleaning_func_25(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import Imputer
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	imp = Imputer(strategy='most_frequent')
	df1.collections_12_mths_ex_med = imp.fit_transform(df1.collections_12_mths_ex_med.reshape((- 1), 1))
	return df1
=============

# Function 108
def cleaning_func_26(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import Imputer
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	imp = Imputer(strategy='mean')
	df1.revol_util = imp.fit_transform(df1.revol_util.values.reshape((- 1), 1))
	return df1
=============

# Function 109
def cleaning_func_27(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import Imputer
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	imp = Imputer(strategy='most_frequent')
	msld = imp.fit_transform(df1.mths_since_last_delinq.values.reshape((- 1), 1))
	df1.mths_since_last_delinq = msld
	return df1
=============

# Function 110
def cleaning_func_30(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	lbl_enc = LabelEncoder()
	df1[(x + '_old')] = df[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1[(x + '_old')] = df[x]
	df1[x] = df1[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1.issue_d = pd.to_datetime(df1.issue_d, format='%b-%Y')
	return df1
=============

# Function 111
def cleaning_func_34(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	lbl_enc = LabelEncoder()
	df1[(x + '_old')] = df[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1[(x + '_old')] = df[x]
	df1[x] = df1[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1.earliest_cr_line = pd.to_datetime(df1.earliest_cr_line, format='%b-%Y')
	return df1
=============

# Function 112
def cleaning_func_35(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import Imputer
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	imp = Imputer(strategy='median')
	mslr = imp.fit_transform(df1.mths_since_last_record.values.reshape((- 1), 1))
	df1.mths_since_last_record = mslr
	return df1
=============

# Function 113
def cleaning_func_37(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import Imputer
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	emp_lengths = []
	df1.emp_length = emp_lengths
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	imp = Imputer(strategy='median')
	df1.emp_length = imp.fit_transform(df1.emp_length.values.reshape((- 1), 1))
	return df1
=============

# Function 114
def cleaning_func_39(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	lbl_enc = LabelEncoder()
	df1[(x + '_old')] = df[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1[(x + '_old')] = df[x]
	df1[x] = df1[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1.issue_d = pd.to_datetime(df1.issue_d, format='%b-%Y')
	df1['issue_d_year'] = df1.issue_d.dt.year
	return df1
=============

# Function 115
def cleaning_func_40(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	lbl_enc = LabelEncoder()
	df1[(x + '_old')] = df[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1[(x + '_old')] = df[x]
	df1[x] = df1[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1['int_rate'] = df1.int_rate.astype(str).astype(float)
	return df1
=============

# Function 116
def cleaning_func_41(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	lbl_enc = LabelEncoder()
	df1[(x + '_old')] = df[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1[(x + '_old')] = df[x]
	df1[x] = df1[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1.earliest_cr_line = pd.to_datetime(df1.earliest_cr_line, format='%b-%Y')
	df1['earliest_cr_line_year'] = df1.earliest_cr_line.dt.year
	return df1
=============

# Function 117
def cleaning_func_42(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	lbl_enc = LabelEncoder()
	df1[(x + '_old')] = df[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1[(x + '_old')] = df[x]
	df1[x] = df1[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1.issue_d = pd.to_datetime(df1.issue_d, format='%b-%Y')
	df1['issue_d_month'] = df1.issue_d.dt.month
	return df1
=============

# Function 118
def cleaning_func_43(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	lbl_enc = LabelEncoder()
	df1[(x + '_old')] = df[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1[(x + '_old')] = df[x]
	df1[x] = df1[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1.earliest_cr_line = pd.to_datetime(df1.earliest_cr_line, format='%b-%Y')
	df1['earliest_cr_line_month'] = df1.earliest_cr_line.dt.month
	return df1
=============

# Function 119
def cleaning_func_44(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	terms = []
	df1.term = terms
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	return df1
=============

# Function 120
def cleaning_func_45(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	terms = []
	df1.term = terms
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	lbl_enc = LabelEncoder()
	df1[(x + '_old')] = df[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1[(x + '_old')] = df[x]
	df1[x] = df1[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1['term'] = df1.term.astype(str).astype(int)
	return df1
=============

# Function 121
def cleaning_func_47(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	lbl_enc = LabelEncoder()
	df1[(x + '_old')] = df[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1[(x + '_old')] = df[x]
	df1[x] = df1[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	return df1
=============

# Function 122
def cleaning_func_48(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	lbl_enc = LabelEncoder()
	df1[(x + '_old')] = df[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1[(x + '_old')] = df[x]
	df1[x] = df1[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df2 = df1[((df1['loan_status'] == 'Fully Paid') | (df1['loan_status'] == 'Charged Off'))]
	targets = []
	df2['target'] = targets
	return df2
=============

# Function 123
def cleaning_func_49(df):
	# core cleaning code
	import pandas as pd
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	# df = pd.read_csv('../input/loan.csv')
	df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
	df = df[(df['pymnt_plan'] == 'n')]
	df = df[(df['application_type'] == 'INDIVIDUAL')]
	df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url', 'application_type', 'grade', 'annual_inc_joint', 'dti_joint'])
	df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'])
	df1 = df1.drop(columns=['mths_since_last_major_derog'])
	lbl_enc = LabelEncoder()
	df1[(x + '_old')] = df[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1[(x + '_old')] = df[x]
	df1[x] = df1[x]
	df1[x] = lbl_enc.fit_transform(df1[x])
	df1['text'] = ((((df1.emp_title + ' ') + df1.title) + ' ') + df1.desc)
	df1['text'] = df1['text'].fillna('nan')
	return df1
=============

# Function 124
def cleaning_func_0(df):
	# core cleaning code
	import pandas as pd
	badLoan = ['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)', 'In Grace Period', 'Does not meet the credit policy. Status:Charged Off']
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_status', 'addr_state'])
	df['isBad'] = [(1 if (x in badLoan) else 0) for x in df.loan_status]
	return df
=============

# Function 125
def cleaning_func_4(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_amnt', ascending=False)
	perStatedf.columns = ['State', 'Num_Loans']
	return perStatedf
=============

# Function 126
def cleaning_func_5(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	return df.groupby('addr_state', as_index=False).count()
=============

# Function 127
def cleaning_func_7(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).sum().sort_values(by='loan_amnt', ascending=False)
	perStatedf.columns = ['State', 'loan_amt']
	return perStatedf
=============

# Function 128
def cleaning_func_9(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_status', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).sum().sort_values(by='isBad', ascending=False)
	perStatedf.columns = ['State', 'badLoans']
	return perStatedf
=============

# Function 129
def cleaning_func_11(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_status', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_status', ascending=False)
	perStatedf.columns = ['State', 'totalLoans']
	return perStatedf
=============

# Function 130
def cleaning_func_14(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_amnt', ascending=False)
	statePop = {'CA': 39144818, 'TX': 27469144, 'FL': 20271878, 'NY': 19795791, 'IL': 12859995, 'PA': 12802503, 'OH': 11613423, 'GA': 10214860, 'NC': 10042802, 'MI': 9922576, 'NJ': 8958013, 'VA': 8382993, 'WA': 7170351, 'AZ': 6828065, 'MA': 6794422, 'IN': 6619680, 'TN': 6600299, 'MO': 6083672, 'MD': 6006401, 'WI': 5771337, 'MN': 5489594, 'CO': 5456574, 'SC': 4896146, 'AL': 4858979, 'LA': 4670724, 'KY': 4425092, 'OR': 4028977, 'OK': 3911338, 'CT': 3890886, 'IA': 3123899, 'UT': 2995919, 'MS': 2992333, 'AK': 2978204, 'KS': 2911641, 'NV': 2890845, 'NM': 2085109, 'NE': 1896190, 'WV': 1844128, 'ID': 1654930, 'HI': 1431603, 'NH': 1330608, 'ME': 1329328, 'RI': 1053298, 'MT': 1032949, 'DE': 945934, 'SD': 858469, 'ND': 756927, 'AK': 738432, 'DC': 672228, 'VT': 626042, 'WY': 586107}
	statePopdf = pd.DataFrame.from_dict(statePop, orient='index').reset_index()
	perStatedf = pd.merge(perStatedf, statePopdf, on=['State'], how='inner')
	perStatedf['PerCaptia'] = (perStatedf.Num_Loans / perStatedf.Pop)
	return perStatedf
=============

# Function 131
def cleaning_func_15(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_amnt', ascending=False)
	statePop = {'CA': 39144818, 'TX': 27469144, 'FL': 20271878, 'NY': 19795791, 'IL': 12859995, 'PA': 12802503, 'OH': 11613423, 'GA': 10214860, 'NC': 10042802, 'MI': 9922576, 'NJ': 8958013, 'VA': 8382993, 'WA': 7170351, 'AZ': 6828065, 'MA': 6794422, 'IN': 6619680, 'TN': 6600299, 'MO': 6083672, 'MD': 6006401, 'WI': 5771337, 'MN': 5489594, 'CO': 5456574, 'SC': 4896146, 'AL': 4858979, 'LA': 4670724, 'KY': 4425092, 'OR': 4028977, 'OK': 3911338, 'CT': 3890886, 'IA': 3123899, 'UT': 2995919, 'MS': 2992333, 'AK': 2978204, 'KS': 2911641, 'NV': 2890845, 'NM': 2085109, 'NE': 1896190, 'WV': 1844128, 'ID': 1654930, 'HI': 1431603, 'NH': 1330608, 'ME': 1329328, 'RI': 1053298, 'MT': 1032949, 'DE': 945934, 'SD': 858469, 'ND': 756927, 'AK': 738432, 'DC': 672228, 'VT': 626042, 'WY': 586107}
	return pd.DataFrame.from_dict(statePop, orient='index')
=============

# Function 132
def cleaning_func_16(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_amnt', ascending=False)
	statePop = {'CA': 39144818, 'TX': 27469144, 'FL': 20271878, 'NY': 19795791, 'IL': 12859995, 'PA': 12802503, 'OH': 11613423, 'GA': 10214860, 'NC': 10042802, 'MI': 9922576, 'NJ': 8958013, 'VA': 8382993, 'WA': 7170351, 'AZ': 6828065, 'MA': 6794422, 'IN': 6619680, 'TN': 6600299, 'MO': 6083672, 'MD': 6006401, 'WI': 5771337, 'MN': 5489594, 'CO': 5456574, 'SC': 4896146, 'AL': 4858979, 'LA': 4670724, 'KY': 4425092, 'OR': 4028977, 'OK': 3911338, 'CT': 3890886, 'IA': 3123899, 'UT': 2995919, 'MS': 2992333, 'AK': 2978204, 'KS': 2911641, 'NV': 2890845, 'NM': 2085109, 'NE': 1896190, 'WV': 1844128, 'ID': 1654930, 'HI': 1431603, 'NH': 1330608, 'ME': 1329328, 'RI': 1053298, 'MT': 1032949, 'DE': 945934, 'SD': 858469, 'ND': 756927, 'AK': 738432, 'DC': 672228, 'VT': 626042, 'WY': 586107}
	statePopdf = pd.DataFrame.from_dict(statePop, orient='index').reset_index()
	return statePopdf
=============

# Function 133
def cleaning_func_17(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_amnt', ascending=False)
	return perStatedf
=============

# Function 134
def cleaning_func_18(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	statePop = {'CA': 39144818, 'TX': 27469144, 'FL': 20271878, 'NY': 19795791, 'IL': 12859995, 'PA': 12802503, 'OH': 11613423, 'GA': 10214860, 'NC': 10042802, 'MI': 9922576, 'NJ': 8958013, 'VA': 8382993, 'WA': 7170351, 'AZ': 6828065, 'MA': 6794422, 'IN': 6619680, 'TN': 6600299, 'MO': 6083672, 'MD': 6006401, 'WI': 5771337, 'MN': 5489594, 'CO': 5456574, 'SC': 4896146, 'AL': 4858979, 'LA': 4670724, 'KY': 4425092, 'OR': 4028977, 'OK': 3911338, 'CT': 3890886, 'IA': 3123899, 'UT': 2995919, 'MS': 2992333, 'AK': 2978204, 'KS': 2911641, 'NV': 2890845, 'NM': 2085109, 'NE': 1896190, 'WV': 1844128, 'ID': 1654930, 'HI': 1431603, 'NH': 1330608, 'ME': 1329328, 'RI': 1053298, 'MT': 1032949, 'DE': 945934, 'SD': 858469, 'ND': 756927, 'AK': 738432, 'DC': 672228, 'VT': 626042, 'WY': 586107}
	statePopdf = pd.DataFrame.from_dict(statePop, orient='index').reset_index()
	perStatedf = df.groupby('addr_state', as_index=False).sum().sort_values(by='loan_amnt', ascending=False)
	return perStatedf
=============

# Function 135
def cleaning_func_19(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	statePop = {'CA': 39144818, 'TX': 27469144, 'FL': 20271878, 'NY': 19795791, 'IL': 12859995, 'PA': 12802503, 'OH': 11613423, 'GA': 10214860, 'NC': 10042802, 'MI': 9922576, 'NJ': 8958013, 'VA': 8382993, 'WA': 7170351, 'AZ': 6828065, 'MA': 6794422, 'IN': 6619680, 'TN': 6600299, 'MO': 6083672, 'MD': 6006401, 'WI': 5771337, 'MN': 5489594, 'CO': 5456574, 'SC': 4896146, 'AL': 4858979, 'LA': 4670724, 'KY': 4425092, 'OR': 4028977, 'OK': 3911338, 'CT': 3890886, 'IA': 3123899, 'UT': 2995919, 'MS': 2992333, 'AK': 2978204, 'KS': 2911641, 'NV': 2890845, 'NM': 2085109, 'NE': 1896190, 'WV': 1844128, 'ID': 1654930, 'HI': 1431603, 'NH': 1330608, 'ME': 1329328, 'RI': 1053298, 'MT': 1032949, 'DE': 945934, 'SD': 858469, 'ND': 756927, 'AK': 738432, 'DC': 672228, 'VT': 626042, 'WY': 586107}
	statePopdf = pd.DataFrame.from_dict(statePop, orient='index').reset_index()
	perStatedf = df.groupby('addr_state', as_index=False).sum().sort_values(by='loan_amnt', ascending=False)
	perStatedf = pd.merge(perStatedf, statePopdf, on=['State'], how='inner')
	perStatedf['PerCaptia'] = (perStatedf.loan_amt / perStatedf.Pop)
	return perStatedf
=============

# Function 136
def cleaning_func_20(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	statePop = {'CA': 39144818, 'TX': 27469144, 'FL': 20271878, 'NY': 19795791, 'IL': 12859995, 'PA': 12802503, 'OH': 11613423, 'GA': 10214860, 'NC': 10042802, 'MI': 9922576, 'NJ': 8958013, 'VA': 8382993, 'WA': 7170351, 'AZ': 6828065, 'MA': 6794422, 'IN': 6619680, 'TN': 6600299, 'MO': 6083672, 'MD': 6006401, 'WI': 5771337, 'MN': 5489594, 'CO': 5456574, 'SC': 4896146, 'AL': 4858979, 'LA': 4670724, 'KY': 4425092, 'OR': 4028977, 'OK': 3911338, 'CT': 3890886, 'IA': 3123899, 'UT': 2995919, 'MS': 2992333, 'AK': 2978204, 'KS': 2911641, 'NV': 2890845, 'NM': 2085109, 'NE': 1896190, 'WV': 1844128, 'ID': 1654930, 'HI': 1431603, 'NH': 1330608, 'ME': 1329328, 'RI': 1053298, 'MT': 1032949, 'DE': 945934, 'SD': 858469, 'ND': 756927, 'AK': 738432, 'DC': 672228, 'VT': 626042, 'WY': 586107}
	return pd.DataFrame.from_dict(statePop, orient='index')
=============

# Function 137
def cleaning_func_21(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	statePop = {'CA': 39144818, 'TX': 27469144, 'FL': 20271878, 'NY': 19795791, 'IL': 12859995, 'PA': 12802503, 'OH': 11613423, 'GA': 10214860, 'NC': 10042802, 'MI': 9922576, 'NJ': 8958013, 'VA': 8382993, 'WA': 7170351, 'AZ': 6828065, 'MA': 6794422, 'IN': 6619680, 'TN': 6600299, 'MO': 6083672, 'MD': 6006401, 'WI': 5771337, 'MN': 5489594, 'CO': 5456574, 'SC': 4896146, 'AL': 4858979, 'LA': 4670724, 'KY': 4425092, 'OR': 4028977, 'OK': 3911338, 'CT': 3890886, 'IA': 3123899, 'UT': 2995919, 'MS': 2992333, 'AK': 2978204, 'KS': 2911641, 'NV': 2890845, 'NM': 2085109, 'NE': 1896190, 'WV': 1844128, 'ID': 1654930, 'HI': 1431603, 'NH': 1330608, 'ME': 1329328, 'RI': 1053298, 'MT': 1032949, 'DE': 945934, 'SD': 858469, 'ND': 756927, 'AK': 738432, 'DC': 672228, 'VT': 626042, 'WY': 586107}
	statePopdf = pd.DataFrame.from_dict(statePop, orient='index').reset_index()
	return statePopdf
=============

# Function 138
def cleaning_func_23(df):
	# core cleaning code
	import pandas as pd
	statePop = {'CA': 39144818, 'TX': 27469144, 'FL': 20271878, 'NY': 19795791, 'IL': 12859995, 'PA': 12802503, 'OH': 11613423, 'GA': 10214860, 'NC': 10042802, 'MI': 9922576, 'NJ': 8958013, 'VA': 8382993, 'WA': 7170351, 'AZ': 6828065, 'MA': 6794422, 'IN': 6619680, 'TN': 6600299, 'MO': 6083672, 'MD': 6006401, 'WI': 5771337, 'MN': 5489594, 'CO': 5456574, 'SC': 4896146, 'AL': 4858979, 'LA': 4670724, 'KY': 4425092, 'OR': 4028977, 'OK': 3911338, 'CT': 3890886, 'IA': 3123899, 'UT': 2995919, 'MS': 2992333, 'AK': 2978204, 'KS': 2911641, 'NV': 2890845, 'NM': 2085109, 'NE': 1896190, 'WV': 1844128, 'ID': 1654930, 'HI': 1431603, 'NH': 1330608, 'ME': 1329328, 'RI': 1053298, 'MT': 1032949, 'DE': 945934, 'SD': 858469, 'ND': 756927, 'AK': 738432, 'DC': 672228, 'VT': 626042, 'WY': 586107}
	statePopdf = pd.DataFrame.from_dict(statePop, orient='index').reset_index()
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_status', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).sum().sort_values(by='isBad', ascending=False)
	perStatedf = pd.merge(perStatedf, statePopdf, on=['State'], how='inner')
	perStatedf['PerCaptia'] = (perStatedf.badLoans / perStatedf.Pop)
	return perStatedf
=============

# Function 139
def cleaning_func_26(df):
	# core cleaning code
	import pandas as pd
	statePop = {'CA': 39144818, 'TX': 27469144, 'FL': 20271878, 'NY': 19795791, 'IL': 12859995, 'PA': 12802503, 'OH': 11613423, 'GA': 10214860, 'NC': 10042802, 'MI': 9922576, 'NJ': 8958013, 'VA': 8382993, 'WA': 7170351, 'AZ': 6828065, 'MA': 6794422, 'IN': 6619680, 'TN': 6600299, 'MO': 6083672, 'MD': 6006401, 'WI': 5771337, 'MN': 5489594, 'CO': 5456574, 'SC': 4896146, 'AL': 4858979, 'LA': 4670724, 'KY': 4425092, 'OR': 4028977, 'OK': 3911338, 'CT': 3890886, 'IA': 3123899, 'UT': 2995919, 'MS': 2992333, 'AK': 2978204, 'KS': 2911641, 'NV': 2890845, 'NM': 2085109, 'NE': 1896190, 'WV': 1844128, 'ID': 1654930, 'HI': 1431603, 'NH': 1330608, 'ME': 1329328, 'RI': 1053298, 'MT': 1032949, 'DE': 945934, 'SD': 858469, 'ND': 756927, 'AK': 738432, 'DC': 672228, 'VT': 626042, 'WY': 586107}
	statePopdf = pd.DataFrame.from_dict(statePop, orient='index').reset_index()
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_status', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).sum().sort_values(by='isBad', ascending=False)
	return perStatedf
=============

# Function 140
def cleaning_func_27(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_status', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_status', ascending=False)
	return df.groupby('addr_state', as_index=False).sum()
=============

# Function 141
def cleaning_func_28(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_status', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_status', ascending=False)
	badLoansdf = df.groupby('addr_state', as_index=False).sum().sort_values(by='isBad', ascending=False)
	perStatedf = pd.merge(perStatedf, badLoansdf, on=['State'], how='inner')
	perStatedf['percentBadLoans'] = ((perStatedf.badLoans / perStatedf.totalLoans) * 100)
	return perStatedf
=============

# Function 142
def cleaning_func_29(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_status', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_status', ascending=False)
	badLoansdf = df.groupby('addr_state', as_index=False).sum().sort_values(by='isBad', ascending=False)
	return badLoansdf
=============

# Function 143
def cleaning_func_0(loan):
	# core cleaning code
	import pandas as pd
	from collections import Counter
	# loan = pd.read_csv('../input/loan.csv')
	loan = loan[(loan.loan_status != 'Current')]
	c = Counter(list(loan.loan_status))
	mmp = {x[0]: 1 for x in c.most_common(20)}
	loan['target'] = loan['loan_status'].map(mmp)
	return loan
=============

# Function 144
def cleaning_func_2(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	return category_two_data
=============

# Function 145
def cleaning_func_3(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	return category_one_data
=============

# Function 146
def cleaning_func_4(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	emp_title = new_data_df[8]
	emp_title = pd.DataFrame(emp_title)
	emp_title.columns = ['Employee Title']
	return emp_title
=============

# Function 147
def cleaning_func_5(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	return new_data_df
=============

# Function 148
def cleaning_func_6(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	title = new_data_df[19]
	title = pd.DataFrame(title)
	title.columns = ['Title']
	return title
=============

# Function 149
def cleaning_func_7(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	return status_labels
=============

# Function 150
def cleaning_func_8(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	status_home_status.columns = ['Home Status', 'status_labels']
	return status_home_status
=============

# Function 151
def cleaning_func_9(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	return home_status
=============

# Function 152
def cleaning_func_11(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	status_state.columns = ['State', 'status_labels']
	return status_state
=============

# Function 153
def cleaning_func_13(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	ver_stat = new_data_df[12]
	ver_stat = pd.DataFrame(ver_stat)
	status_ver_stat = pd.DataFrame(np.hstack((ver_stat, status_labels)))
	status_ver_stat.columns = ['Verification Status', 'status_labels']
	return status_ver_stat
=============

# Function 154
def cleaning_func_14(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	status_installment_grade.columns = ['Installment_grade', 'status_labels']
	return status_installment_grade
=============

# Function 155
def cleaning_func_16(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	status_annual_groups.columns = ['Annual_income_grp', 'status_labels']
	return status_annual_groups
=============

# Function 156
def cleaning_func_17(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	return binned_annual_income
=============

# Function 157
def cleaning_func_18(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	return status_labels
=============

# Function 158
def cleaning_func_19(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	status_installment_groups.columns = ['Installment_amt_grp', 'status_labels']
	return status_installment_groups
=============

# Function 159
def cleaning_func_20(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	return binned_installment_amt
=============

# Function 160
def cleaning_func_21(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data_copy = np.vstack((category_one_data, category_two_data))
	new_data_copy = pd.DataFrame(new_data_copy)
	data_2 = new_data_copy
	col_nos = []
	i = 0
	i = (i + 1)
	data_2 = data_2.drop(data_2.columns[col_nos], axis=1)
	rename_1 = range(0, 49)
	data_2.columns = rename_1
	cols_remove = [0, 10, 11, 17, 18, 19, 20, 21]
	data_2 = data_2.drop(data_2.columns[cols_remove], axis=1)
	cat_cols = [4, 7, 8, 9, 11, 14, 16, 18, 19, 20, 24, 25, 32, 33, 37, 38, 39]
	cat_df = data_2.iloc[(slice(None, None, None), cat_cols)].values
	cat_df = pd.DataFrame(cat_df)
	return cat_df
=============

# Function 161
def cleaning_func_22(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data_copy = np.vstack((category_one_data, category_two_data))
	new_data_copy = pd.DataFrame(new_data_copy)
	data_2 = new_data_copy
	col_nos = []
	i = 0
	i = (i + 1)
	data_2 = data_2.drop(data_2.columns[col_nos], axis=1)
	rename_1 = range(0, 49)
	data_2.columns = rename_1
	return data_2
=============

# Function 162
def cleaning_func_23(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data_copy = np.vstack((category_one_data, category_two_data))
	new_data_copy = pd.DataFrame(new_data_copy)
	data_2 = new_data_copy
	col_nos = []
	i = 0
	i = (i + 1)
	data_2 = data_2.drop(data_2.columns[col_nos], axis=1)
	rename_1 = range(0, 49)
	data_2.columns = rename_1
	cols_remove = [0, 10, 11, 17, 18, 19, 20, 21]
	data_2 = data_2.drop(data_2.columns[cols_remove], axis=1)
	cat_cols = [4, 7, 8, 9, 11, 14, 16, 18, 19, 20, 24, 25, 32, 33, 37, 38, 39]
	return data_2.iloc[(slice(None, None, None), cat_cols)]
=============

# Function 163
def cleaning_func_25(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data_copy = np.vstack((category_one_data, category_two_data))
	new_data_copy = pd.DataFrame(new_data_copy)
	return new_data_copy
=============

# Function 164
def cleaning_func_26(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data_copy = np.vstack((category_one_data, category_two_data))
	new_data_copy = pd.DataFrame(new_data_copy)
	data_2 = new_data_copy
	col_nos = []
	i = 0
	i = (i + 1)
	data_2 = data_2.drop(data_2.columns[col_nos], axis=1)
	rename_1 = range(0, 49)
	data_2.columns = rename_1
	cols_remove = [0, 10, 11, 17, 18, 19, 20, 21]
	data_2 = data_2.drop(data_2.columns[cols_remove], axis=1)
	return data_2
=============

# Function 165
def cleaning_func_27(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data_copy = np.vstack((category_one_data, category_two_data))
	new_data_copy = pd.DataFrame(new_data_copy)
	data_2 = new_data_copy
	col_nos = []
	i = 0
	i = (i + 1)
	data_2 = data_2.drop(data_2.columns[col_nos], axis=1)
	rename_1 = range(0, 49)
	data_2.columns = rename_1
	cols_remove = [0, 10, 11, 17, 18, 19, 20, 21]
	data_2 = data_2.drop(data_2.columns[cols_remove], axis=1)
	cat_cols = [4, 7, 8, 9, 11, 14, 16, 18, 19, 20, 24, 25, 32, 33, 37, 38, 39]
	cat_df = data_2.iloc[(slice(None, None, None), cat_cols)].values
	cat_df = pd.DataFrame(cat_df)
	c = [11, 12, 13, 15]
	cat_df = cat_df.drop(cat_df.columns[c], axis=1)
	r = range(0, 13)
	cat_df.columns = r
	return cat_df
=============

# Function 166
def cleaning_func_29(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data_copy = np.vstack((category_one_data, category_two_data))
	new_data_copy = pd.DataFrame(new_data_copy)
	data_2 = new_data_copy
	col_nos = []
	i = 0
	i = (i + 1)
	data_2 = data_2.drop(data_2.columns[col_nos], axis=1)
	rename_1 = range(0, 49)
	data_2.columns = rename_1
	cols_remove = [0, 10, 11, 17, 18, 19, 20, 21]
	data_2 = data_2.drop(data_2.columns[cols_remove], axis=1)
	rename_2 = range(0, 41)
	data_2.columns = rename_2
	cat_plus_time_cols = [4, 7, 8, 9, 11, 12, 14, 16, 17, 18, 19, 20, 24, 25, 32, 33, 34, 36, 37, 38, 39]
	data_2_copy = data_2
	non_cat = data_2_copy.drop(data_2_copy.columns[cat_plus_time_cols], axis=1)
	rename = range(0, 20)
	non_cat.columns = rename
	return non_cat
=============

# Function 167
def cleaning_func_30(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data_copy = np.vstack((category_one_data, category_two_data))
	new_data_copy = pd.DataFrame(new_data_copy)
	data_2 = new_data_copy
	col_nos = []
	i = 0
	i = (i + 1)
	data_2 = data_2.drop(data_2.columns[col_nos], axis=1)
	rename_1 = range(0, 49)
	data_2.columns = rename_1
	cols_remove = [0, 10, 11, 17, 18, 19, 20, 21]
	data_2 = data_2.drop(data_2.columns[cols_remove], axis=1)
	rename_2 = range(0, 41)
	data_2.columns = rename_2
	cat_plus_time_cols = [4, 7, 8, 9, 11, 12, 14, 16, 17, 18, 19, 20, 24, 25, 32, 33, 34, 36, 37, 38, 39]
	data_2_copy = data_2
	non_cat = data_2_copy.drop(data_2_copy.columns[cat_plus_time_cols], axis=1)
	rename = range(0, 20)
	non_cat.columns = rename
	non_cat = non_cat.drop(non_cat.columns[7], axis=1)
	renaming_df = range(0, 19)
	non_cat.columns = renaming_df
	return non_cat
=============

# Function 168
def cleaning_func_32(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data_copy = np.vstack((category_one_data, category_two_data))
	new_data_copy = pd.DataFrame(new_data_copy)
	data_2 = new_data_copy
	col_nos = []
	i = 0
	i = (i + 1)
	data_2 = data_2.drop(data_2.columns[col_nos], axis=1)
	rename_1 = range(0, 49)
	data_2.columns = rename_1
	cols_remove = [0, 10, 11, 17, 18, 19, 20, 21]
	data_2 = data_2.drop(data_2.columns[cols_remove], axis=1)
	rename_2 = range(0, 41)
	data_2.columns = rename_2
	cat_plus_time_cols = [4, 7, 8, 9, 11, 12, 14, 16, 17, 18, 19, 20, 24, 25, 32, 33, 34, 36, 37, 38, 39]
	data_2_copy = data_2
	return data_2_copy
=============

# Function 169
def cleaning_func_33(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	return plot_stack
=============

# Function 170
def cleaning_func_34(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	plot_stack = plot_stack.drop(plot_stack.columns[2], axis=1)
	plot_stack.columns = ['Installment_amt_grp', 'Charged Off', 'Fully Paid']
	return plot_stack
=============

# Function 171
def cleaning_func_35(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	return status_installment_groups
=============

# Function 172
def cleaning_func_36(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	return Fully_paid
=============

# Function 173
def cleaning_func_37(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	return Charged_off
=============

# Function 174
def cleaning_func_38(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	return status_installment_grade
=============

# Function 175
def cleaning_func_39(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	return Charged_off_grade
=============

# Function 176
def cleaning_func_40(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	return Fully_Paid_grade
=============

# Function 177
def cleaning_func_41(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	plot_stack_1 = plot_stack_1.drop(plot_stack_1.columns[2], axis=1)
	plot_stack_1.columns = ['Installment_grade_grp', 'Charged Off', 'Fully Paid']
	return plot_stack_1
=============

# Function 178
def cleaning_func_42(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	return installment_grade
=============

# Function 179
def cleaning_func_43(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	return plot_stack_1
=============

# Function 180
def cleaning_func_44(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	return plot_stack_3
=============

# Function 181
def cleaning_func_45(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	return Fully_Paid_home_status
=============

# Function 182
def cleaning_func_46(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	plot_stack_3 = plot_stack_3.drop(plot_stack_3.columns[2], axis=1)
	plot_stack_3.columns = ['Home Status', 'Charged Off', 'Fully Paid']
	return plot_stack_3
=============

# Function 183
def cleaning_func_47(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	return Charged_off_home_status
=============

# Function 184
def cleaning_func_48(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	return plot_home_status_44
=============

# Function 185
def cleaning_func_49(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	return home_status
=============

# Function 186
def cleaning_func_50(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	return plot_home_status_55
=============

# Function 187
def cleaning_func_51(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	return plot_home_status_55
=============

# Function 188
def cleaning_func_52(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	return status_home_status
=============

# Function 189
def cleaning_func_53(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	return plot_annual_income_77
=============

# Function 190
def cleaning_func_54(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	return status_annual_groups
=============

# Function 191
def cleaning_func_55(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	return plot_annual_income_66
=============

# Function 192
def cleaning_func_56(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	return plot_stack_4
=============

# Function 193
def cleaning_func_57(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	return binned_annual_income
=============

# Function 194
def cleaning_func_58(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	plot_stack_4 = plot_stack_4.drop(plot_stack_4.columns[2], axis=1)
	plot_stack_4.columns = ['Annual Income Group', 'Charged Off', 'Fully Paid']
	return plot_stack_4
=============

# Function 195
def cleaning_func_59(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	return Fully_Paid_annual_income
=============

# Function 196
def cleaning_func_60(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	return Charged_off_annual_income
=============

# Function 197
def cleaning_func_61(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	return plot_state_88
=============

# Function 198
def cleaning_func_62(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	plot_state_99 = plot_state_99.drop([7, 21, 28])
	plot_stack_5 = np.hstack((plot_state_88, plot_state_99))
	plot_stack_5 = pd.DataFrame(plot_stack_5)
	return plot_stack_5
=============

# Function 199
def cleaning_func_63(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	return Charged_off_state
=============

# Function 200
def cleaning_func_64(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	return plot_state_88
=============

# Function 201
def cleaning_func_65(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	plot_state_99 = plot_state_99.drop([7, 21, 28])
	plot_stack_5 = np.hstack((plot_state_88, plot_state_99))
	plot_stack_5 = pd.DataFrame(plot_stack_5)
	plot_stack_5 = plot_stack_5.drop(plot_stack_5.columns[2], axis=1)
	plot_stack_5.columns = ['state', 'Charged Off', 'Fully Paid']
	return plot_stack_5
=============

# Function 202
def cleaning_func_68(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	return Fully_Paid_state
=============

# Function 203
def cleaning_func_69(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	return state
=============

# Function 204
def cleaning_func_70(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	return status_state
=============

# Function 205
def cleaning_func_71(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	plot_state_99 = plot_state_99.drop([7, 21, 28])
	plot_stack_5 = np.hstack((plot_state_88, plot_state_99))
	plot_stack_5 = pd.DataFrame(plot_stack_5)
	plot_stack_5 = plot_stack_5.drop(plot_stack_5.columns[2], axis=1)
	return plot_stack_5
=============

# Function 206
def cleaning_func_72(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	plot_state_99 = plot_state_99.drop([7, 21, 28])
	plot_stack_5 = np.hstack((plot_state_88, plot_state_99))
	plot_stack_5 = pd.DataFrame(plot_stack_5)
	plot_stack_5 = plot_stack_5.drop(plot_stack_5.columns[2], axis=1)
	totals = [(i + j) for (i, j) in zip(plot_stack_5['Charged Off'], plot_stack_5['Fully Paid'])]
	C_Off = [((i / j) * 100) for (i, j) in zip(plot_stack_5['Charged Off'], totals)]
	C_Off = pd.DataFrame(C_Off)
	return C_Off
=============

# Function 207
def cleaning_func_73(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	plot_state_99 = plot_state_99.drop([7, 21, 28])
	plot_stack_5 = np.hstack((plot_state_88, plot_state_99))
	plot_stack_5 = pd.DataFrame(plot_stack_5)
	plot_stack_5 = plot_stack_5.drop(plot_stack_5.columns[2], axis=1)
	totals = [(i + j) for (i, j) in zip(plot_stack_5['Charged Off'], plot_stack_5['Fully Paid'])]
	C_Off = [((i / j) * 100) for (i, j) in zip(plot_stack_5['Charged Off'], totals)]
	C_Off = pd.DataFrame(C_Off)
	temp_plot = np.hstack((plot_stack_5, C_Off))
	temp_plot = pd.DataFrame(temp_plot)
	temp_plot.columns = ['state', 'Charged Off', 'Fully Paid', '% Charged Off']
	return temp_plot
=============

# Function 208
def cleaning_func_74(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	plot_state_99 = plot_state_99.drop([7, 21, 28])
	plot_stack_5 = np.hstack((plot_state_88, plot_state_99))
	plot_stack_5 = pd.DataFrame(plot_stack_5)
	plot_stack_5 = plot_stack_5.drop(plot_stack_5.columns[2], axis=1)
	totals = [(i + j) for (i, j) in zip(plot_stack_5['Charged Off'], plot_stack_5['Fully Paid'])]
	C_Off = [((i / j) * 100) for (i, j) in zip(plot_stack_5['Charged Off'], totals)]
	C_Off = pd.DataFrame(C_Off)
	temp_plot = np.hstack((plot_stack_5, C_Off))
	temp_plot = pd.DataFrame(temp_plot)
	temp_plot = np.array(temp_plot.sort_values(by='% Charged Off', ascending=False))
	temp_plot = pd.DataFrame(temp_plot)
	temp_plot.columns = ['state', 'Charged Off', 'Fully Paid', '% Charged Off']
	return temp_plot
=============

# Function 209
def cleaning_func_75(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	plot_state_99 = plot_state_99.drop([7, 21, 28])
	plot_stack_5 = np.hstack((plot_state_88, plot_state_99))
	plot_stack_5 = pd.DataFrame(plot_stack_5)
	plot_stack_5 = plot_stack_5.drop(plot_stack_5.columns[2], axis=1)
	totals = [(i + j) for (i, j) in zip(plot_stack_5['Charged Off'], plot_stack_5['Fully Paid'])]
	C_Off = [((i / j) * 100) for (i, j) in zip(plot_stack_5['Charged Off'], totals)]
	C_Off = pd.DataFrame(C_Off)
	temp_plot = np.hstack((plot_stack_5, C_Off))
	temp_plot = pd.DataFrame(temp_plot)
	return temp_plot
=============

# Function 210
def cleaning_func_77(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	plot_state_99 = plot_state_99.drop([7, 21, 28])
	plot_stack_5 = np.hstack((plot_state_88, plot_state_99))
	plot_stack_5 = pd.DataFrame(plot_stack_5)
	ver_stat = new_data_df[12]
	ver_stat = pd.DataFrame(ver_stat)
	status_ver_stat = pd.DataFrame(np.hstack((ver_stat, status_labels)))
	Charged_off_ver_stat = status_ver_stat[(status_ver_stat.status_labels == 1)]
	return Charged_off_ver_stat
=============

# Function 211
def cleaning_func_78(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	plot_state_99 = plot_state_99.drop([7, 21, 28])
	plot_stack_5 = np.hstack((plot_state_88, plot_state_99))
	plot_stack_5 = pd.DataFrame(plot_stack_5)
	ver_stat = new_data_df[12]
	ver_stat = pd.DataFrame(ver_stat)
	status_ver_stat = pd.DataFrame(np.hstack((ver_stat, status_labels)))
	Charged_off_ver_stat = status_ver_stat[(status_ver_stat.status_labels == 1)]
	temp_71 = Charged_off_ver_stat.iloc[(slice(None, None, None), 0)].values
	plot_ver_stat = np.array(np.unique(temp_71, return_counts=True))
	plot_ver_stat_101 = pd.DataFrame(plot_ver_stat.T)
	Fully_Paid_ver_stat = status_ver_stat[(status_ver_stat.status_labels == 0)]
	return Fully_Paid_ver_stat
=============

# Function 212
def cleaning_func_79(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	plot_state_99 = plot_state_99.drop([7, 21, 28])
	plot_stack_5 = np.hstack((plot_state_88, plot_state_99))
	plot_stack_5 = pd.DataFrame(plot_stack_5)
	ver_stat = new_data_df[12]
	ver_stat = pd.DataFrame(ver_stat)
	status_ver_stat = pd.DataFrame(np.hstack((ver_stat, status_labels)))
	Charged_off_ver_stat = status_ver_stat[(status_ver_stat.status_labels == 1)]
	temp_71 = Charged_off_ver_stat.iloc[(slice(None, None, None), 0)].values
	plot_ver_stat = np.array(np.unique(temp_71, return_counts=True))
	plot_ver_stat_101 = pd.DataFrame(plot_ver_stat.T)
	return plot_ver_stat_101
=============

# Function 213
def cleaning_func_80(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	plot_state_99 = plot_state_99.drop([7, 21, 28])
	plot_stack_5 = np.hstack((plot_state_88, plot_state_99))
	plot_stack_5 = pd.DataFrame(plot_stack_5)
	ver_stat = new_data_df[12]
	ver_stat = pd.DataFrame(ver_stat)
	return ver_stat
=============

# Function 214
def cleaning_func_81(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	plot_state_99 = plot_state_99.drop([7, 21, 28])
	plot_stack_5 = np.hstack((plot_state_88, plot_state_99))
	plot_stack_5 = pd.DataFrame(plot_stack_5)
	ver_stat = new_data_df[12]
	ver_stat = pd.DataFrame(ver_stat)
	status_ver_stat = pd.DataFrame(np.hstack((ver_stat, status_labels)))
	Charged_off_ver_stat = status_ver_stat[(status_ver_stat.status_labels == 1)]
	temp_71 = Charged_off_ver_stat.iloc[(slice(None, None, None), 0)].values
	plot_ver_stat = np.array(np.unique(temp_71, return_counts=True))
	plot_ver_stat_101 = pd.DataFrame(plot_ver_stat.T)
	Fully_Paid_ver_stat = status_ver_stat[(status_ver_stat.status_labels == 0)]
	temp_72 = Fully_Paid_ver_stat.iloc[(slice(None, None, None), 0)].values
	plot_ver_stat_2 = np.array(np.unique(temp_72, return_counts=True))
	plot_ver_stat_111 = pd.DataFrame(plot_ver_stat_2.T)
	plot_stack_6 = np.hstack((plot_ver_stat_101, plot_ver_stat_111))
	plot_stack_6 = pd.DataFrame(plot_stack_6)
	plot_stack_6 = plot_stack_6.drop(plot_stack_6.columns[2], axis=1)
	plot_stack_6.columns = ['Verification Status', 'Charged Off', 'Fully Paid']
	return plot_stack_6
=============

# Function 215
def cleaning_func_82(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	plot_state_99 = plot_state_99.drop([7, 21, 28])
	plot_stack_5 = np.hstack((plot_state_88, plot_state_99))
	plot_stack_5 = pd.DataFrame(plot_stack_5)
	ver_stat = new_data_df[12]
	ver_stat = pd.DataFrame(ver_stat)
	status_ver_stat = pd.DataFrame(np.hstack((ver_stat, status_labels)))
	Charged_off_ver_stat = status_ver_stat[(status_ver_stat.status_labels == 1)]
	temp_71 = Charged_off_ver_stat.iloc[(slice(None, None, None), 0)].values
	plot_ver_stat = np.array(np.unique(temp_71, return_counts=True))
	plot_ver_stat_101 = pd.DataFrame(plot_ver_stat.T)
	Fully_Paid_ver_stat = status_ver_stat[(status_ver_stat.status_labels == 0)]
	temp_72 = Fully_Paid_ver_stat.iloc[(slice(None, None, None), 0)].values
	plot_ver_stat_2 = np.array(np.unique(temp_72, return_counts=True))
	plot_ver_stat_111 = pd.DataFrame(plot_ver_stat_2.T)
	plot_stack_6 = np.hstack((plot_ver_stat_101, plot_ver_stat_111))
	plot_stack_6 = pd.DataFrame(plot_stack_6)
	return plot_stack_6
=============

# Function 216
def cleaning_func_83(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	plot_state_99 = plot_state_99.drop([7, 21, 28])
	plot_stack_5 = np.hstack((plot_state_88, plot_state_99))
	plot_stack_5 = pd.DataFrame(plot_stack_5)
	ver_stat = new_data_df[12]
	ver_stat = pd.DataFrame(ver_stat)
	status_ver_stat = pd.DataFrame(np.hstack((ver_stat, status_labels)))
	return status_ver_stat
=============

# Function 217
def cleaning_func_84(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data_1 = pd.DataFrame(data)
	category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
	category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
	new_data = np.vstack((category_one_data, category_two_data))
	new_data = new_data[(slice(None, None, None), slice(2, (- 30), None))]
	new_data_df = pd.DataFrame(new_data)
	installment_amt = new_data[(slice(None, None, None), 5)]
	bins = np.linspace(installment_amt.min(), installment_amt.max(), 10)
	installment_amt = installment_amt.astype(float).reshape(installment_amt.size, 1)
	binned_installment_amt = pd.DataFrame(np.digitize(installment_amt, bins))
	status_new = new_data_df[14]
	factored_status = np.array(pd.factorize(status_new))
	status_labels = pd.DataFrame(factored_status[0])
	status_installment_groups = pd.DataFrame(np.hstack((binned_installment_amt, status_labels)))
	Charged_off = status_installment_groups[(status_installment_groups.status_labels == 1)]
	temp_1 = Charged_off.iloc[(slice(None, None, None), 0)].values
	plot_1 = np.array(np.unique(temp_1, return_counts=True))
	plot_1 = plot_1[(slice(None, None, None), slice(None, (- 1), None))]
	plot_11 = plot_1.T
	Fully_paid = status_installment_groups[(status_installment_groups.status_labels == 0)]
	temp_2 = Fully_paid.iloc[(slice(None, None, None), 0)].values
	plot_2 = np.array(np.unique(temp_2, return_counts=True))
	plot_22 = plot_2.T
	plot_stack = np.hstack((plot_11, plot_22))
	plot_stack = pd.DataFrame(plot_stack)
	installment_grade = new_data[(slice(None, None, None), 6)]
	installment_grade = pd.DataFrame(installment_grade)
	status_installment_grade = pd.DataFrame(np.hstack((installment_grade, status_labels)))
	Charged_off_grade = status_installment_grade[(status_installment_grade.status_labels == 1)]
	temp_11 = Charged_off_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade = np.array(np.unique(temp_11, return_counts=True))
	plot_grade_11 = plot_grade.T
	Fully_Paid_grade = status_installment_grade[(status_installment_grade.status_labels == 0)]
	temp_22 = Fully_Paid_grade.iloc[(slice(None, None, None), 0)].values
	plot_grade_2 = np.array(np.unique(temp_22, return_counts=True))
	plot_grade_22 = plot_grade_2.T
	plot_stack_1 = np.hstack((plot_grade_11, plot_grade_22))
	plot_stack_1 = pd.DataFrame(plot_stack_1)
	home_status = new_data_df[10]
	home_status = pd.DataFrame(home_status)
	status_home_status = pd.DataFrame(np.hstack((home_status, status_labels)))
	Charged_off_home_status = status_home_status[(status_home_status.status_labels == 1)]
	temp_41 = Charged_off_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status = np.array(np.unique(temp_41, return_counts=True))
	plot_home_status_44 = pd.DataFrame(plot_home_status.T)
	Fully_Paid_home_status = status_home_status[(status_home_status.status_labels == 0)]
	temp_42 = Fully_Paid_home_status.iloc[(slice(None, None, None), 0)].values
	plot_home_status_2 = np.array(np.unique(temp_42, return_counts=True))
	plot_home_status_55 = pd.DataFrame(plot_home_status_2.T)
	plot_home_status_55 = plot_home_status_55.drop(0)
	plot_stack_3 = np.hstack((plot_home_status_44, plot_home_status_55))
	plot_stack_3 = pd.DataFrame(plot_stack_3)
	annual_income = new_data[(slice(None, None, None), 11)]
	bins_2 = np.array([40000, 70000, 100000, 150000])
	annual_income = annual_income.astype(float).reshape(annual_income.size, 1)
	binned_annual_income = pd.DataFrame(np.digitize(annual_income, bins_2))
	status_annual_groups = pd.DataFrame(np.hstack((binned_annual_income, status_labels)))
	Charged_off_annual_income = status_annual_groups[(status_annual_groups.status_labels == 1)]
	temp_51 = Charged_off_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income = np.array(np.unique(temp_51, return_counts=True))
	plot_annual_income_66 = pd.DataFrame(plot_annual_income.T)
	Fully_Paid_annual_income = status_annual_groups[(status_annual_groups.status_labels == 0)]
	temp_52 = Fully_Paid_annual_income.iloc[(slice(None, None, None), 0)].values
	plot_annual_income_2 = np.array(np.unique(temp_52, return_counts=True))
	plot_annual_income_77 = pd.DataFrame(plot_annual_income_2.T)
	plot_stack_4 = np.hstack((plot_annual_income_66, plot_annual_income_77))
	plot_stack_4 = pd.DataFrame(plot_stack_4)
	state = new_data_df[21]
	state = pd.DataFrame(state)
	status_state = pd.DataFrame(np.hstack((state, status_labels)))
	Charged_off_state = status_state[(status_state.status_labels == 1)]
	temp_61 = Charged_off_state.iloc[(slice(None, None, None), 0)].values
	plot_state = np.array(np.unique(temp_61, return_counts=True))
	plot_state_88 = pd.DataFrame(plot_state.T)
	Fully_Paid_state = status_state[(status_state.status_labels == 0)]
	temp_62 = Fully_Paid_state.iloc[(slice(None, None, None), 0)].values
	plot_state_2 = np.array(np.unique(temp_62, return_counts=True))
	plot_state_99 = pd.DataFrame(plot_state_2.T)
	plot_state_88 = plot_state_88.drop(7)
	plot_state_99 = plot_state_99.drop([7, 21, 28])
	plot_stack_5 = np.hstack((plot_state_88, plot_state_99))
	plot_stack_5 = pd.DataFrame(plot_stack_5)
	ver_stat = new_data_df[12]
	ver_stat = pd.DataFrame(ver_stat)
	status_ver_stat = pd.DataFrame(np.hstack((ver_stat, status_labels)))
	Charged_off_ver_stat = status_ver_stat[(status_ver_stat.status_labels == 1)]
	temp_71 = Charged_off_ver_stat.iloc[(slice(None, None, None), 0)].values
	plot_ver_stat = np.array(np.unique(temp_71, return_counts=True))
	plot_ver_stat_101 = pd.DataFrame(plot_ver_stat.T)
	Fully_Paid_ver_stat = status_ver_stat[(status_ver_stat.status_labels == 0)]
	temp_72 = Fully_Paid_ver_stat.iloc[(slice(None, None, None), 0)].values
	plot_ver_stat_2 = np.array(np.unique(temp_72, return_counts=True))
	plot_ver_stat_111 = pd.DataFrame(plot_ver_stat_2.T)
	return plot_ver_stat_111
=============

# Function 218
def cleaning_func_0(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	df['emp_length_int'] = np.nan
	return df
=============

# Function 219
def cleaning_func_1(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	df['income_category'] = np.nan
	return df
=============

# Function 220
def cleaning_func_2(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	df['loan_condition'] = np.nan
	return df
=============

# Function 221
def cleaning_func_3(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	df['loan_condition_int'] = np.nan
	return df
=============

# Function 222
def cleaning_func_4(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	df['interest_payments'] = np.nan
	return df
=============

# Function 223
def cleaning_func_5(df):
	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	df['region'] = np.nan
	return df
=============

# Function 224
def cleaning_func_6(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	df['complete_date'] = pd.to_datetime(df['issue_d'])
	return df
=============

# Function 225
def cleaning_func_7(df):
	# additional context code from user definitions

	def loan_condition(status):
	    if (status in bad_loan):
	        return 'Bad Loan'
	    else:
	        return 'Good Loan'

	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	bad_loan = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
	df['loan_condition'] = df['loan_status'].apply(loan_condition)
	return df
=============

# Function 226
def cleaning_func_8(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	dt_series = pd.to_datetime(df['issue_d'])
	df['year'] = dt_series.dt.year
	return df
=============

# Function 227
def cleaning_func_9(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	df['complete_date'] = pd.to_datetime(df['issue_d'])
	group_dates = df.groupby(['complete_date', 'region'], as_index=False).sum()
	group_dates['issue_d'] = [month.to_period('M') for month in group_dates['complete_date']]
	return group_dates
=============

# Function 228
def cleaning_func_10(df):
	# additional context code from user definitions

	def finding_regions(state):
	    if (state in west):
	        return 'West'
	    elif (state in south_west):
	        return 'SouthWest'
	    elif (state in south_east):
	        return 'SouthEast'
	    elif (state in mid_west):
	        return 'MidWest'
	    elif (state in north_east):
	        return 'NorthEast'

	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	west = ['CA', 'OR', 'UT', 'WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID']
	south_west = ['AZ', 'TX', 'NM', 'OK']
	south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN']
	mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
	north_east = ['CT', 'NY', 'PA', 'NJ', 'RI', 'MA', 'MD', 'VT', 'NH', 'ME']
	df['region'] = df['addr_state'].apply(finding_regions)
	return df
=============

# Function 229
def cleaning_func_11(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	complete_df = df.copy()
	complete_df[col] = complete_df[col].fillna(0)
	complete_df[col] = complete_df[col]
	complete_df[col].fillna = complete_df[col].fillna
	complete_df[col] = complete_df[col].fillna(0)
	complete_df['last_credit_pull_d'] = complete_df.groupby('region')['last_credit_pull_d'].transform((lambda x: x.fillna(x.mode)))
	return complete_df
=============

# Function 230
def cleaning_func_12(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	return df
=============

# Function 231
def cleaning_func_13(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	complete_df = df.copy()
	complete_df[col] = complete_df[col].fillna(0)
	complete_df[col] = complete_df[col]
	complete_df[col].fillna = complete_df[col].fillna
	complete_df[col] = complete_df[col].fillna(0)
	complete_df['total_acc'] = complete_df.groupby('region')['total_acc'].transform((lambda x: x.fillna(x.median())))
	return complete_df
=============

# Function 232
def cleaning_func_14(col,df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	return col
=============

# Function 233
def cleaning_func_15(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	complete_df = df.copy()
	complete_df[col] = complete_df[col].fillna(0)
	complete_df[col] = complete_df[col]
	complete_df[col].fillna = complete_df[col].fillna
	complete_df[col] = complete_df[col].fillna(0)
	complete_df['delinq_2yrs'] = complete_df.groupby('region')['delinq_2yrs'].transform((lambda x: x.fillna(x.mean())))
	return complete_df
=============

# Function 234
def cleaning_func_16(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	complete_df = df.copy()
	complete_df[col] = complete_df[col].fillna(0)
	complete_df[col] = complete_df[col]
	complete_df[col].fillna = complete_df[col].fillna
	complete_df[col] = complete_df[col].fillna(0)
	complete_df['last_pymnt_d'] = complete_df.groupby('region')['last_pymnt_d'].transform((lambda x: x.fillna(x.mode)))
	return complete_df
=============

# Function 235
def cleaning_func_17(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	complete_df = df.copy()
	complete_df[col] = complete_df[col].fillna(0)
	complete_df[col] = complete_df[col]
	complete_df[col].fillna = complete_df[col].fillna
	complete_df[col] = complete_df[col].fillna(0)
	complete_df['earliest_cr_line'] = complete_df.groupby('region')['earliest_cr_line'].transform((lambda x: x.fillna(x.mode)))
	return complete_df
=============

# Function 236
def cleaning_func_18(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	complete_df = df.copy()
	complete_df[col] = complete_df[col].fillna(0)
	complete_df[col] = complete_df[col]
	complete_df[col].fillna = complete_df[col].fillna
	complete_df[col] = complete_df[col].fillna(0)
	complete_df['pub_rec'] = complete_df.groupby('region')['pub_rec'].transform((lambda x: x.fillna(x.median())))
	return complete_df
=============

# Function 237
def cleaning_func_19(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	complete_df = df.copy()
	complete_df[col] = complete_df[col].fillna(0)
	complete_df[col] = complete_df[col]
	complete_df[col].fillna = complete_df[col].fillna
	complete_df[col] = complete_df[col].fillna(0)
	complete_df['next_pymnt_d'] = complete_df.groupby('region')['next_pymnt_d'].transform((lambda x: x.fillna(x.mode)))
	return complete_df
=============

# Function 238
def cleaning_func_20(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	group_dates = df.groupby(['complete_date', 'region'], as_index=False).sum()
	group_dates = group_dates.groupby(['issue_d', 'region'], as_index=False).sum()
	return group_dates
=============

# Function 239
def cleaning_func_22(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	group_dates = df.groupby(['complete_date', 'region'], as_index=False).sum()
	return group_dates
=============

# Function 240
def cleaning_func_23(col,df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	col.loc[((col['annual_income'] <= 100000), 'income_category')] = 'Low'
	col.loc[(((col['annual_income'] > 100000) & (col['annual_income'] <= 200000)), 'income_category')] = 'Medium'
	col.loc[((col['annual_income'] > 200000), 'income_category')] = 'High'
	return col
=============

# Function 241
def cleaning_func_24(col,df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	col.loc[((col['annual_income'] <= 100000), 'income_category')] = 'Low'
	col.loc[(((col['annual_income'] > 100000) & (col['annual_income'] <= 200000)), 'income_category')] = 'Medium'
	col.loc[((col['annual_income'] > 200000), 'income_category')] = 'High'
	col.loc[((df['loan_condition'] == 'Bad Loan'), 'loan_condition_int')] = 0
	col.loc[((df['loan_condition'] == 'Good Loan'), 'loan_condition_int')] = 1
	return col
=============

# Function 242
def cleaning_func_25(col,df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	col.loc[((col['annual_income'] <= 100000), 'income_category')] = 'Low'
	col.loc[(((col['annual_income'] > 100000) & (col['annual_income'] <= 200000)), 'income_category')] = 'Medium'
	col.loc[((col['annual_income'] > 200000), 'income_category')] = 'High'
	col.loc[((df['loan_condition'] == 'Bad Loan'), 'loan_condition_int')] = 0
	col.loc[((df['loan_condition'] == 'Good Loan'), 'loan_condition_int')] = 1
	col.loc[((col['interest_rate'] <= 13.23), 'interest_payments')] = 'Low'
	col.loc[((col['interest_rate'] > 13.23), 'interest_payments')] = 'High'
	return col
=============

# Function 243
def cleaning_func_26(col,df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	col.loc[((col['annual_income'] <= 100000), 'income_category')] = 'Low'
	col.loc[(((col['annual_income'] > 100000) & (col['annual_income'] <= 200000)), 'income_category')] = 'Medium'
	col.loc[((col['annual_income'] > 200000), 'income_category')] = 'High'
	complete_df = df.copy()
	complete_df[col] = complete_df[col].fillna(0)
	complete_df[col] = complete_df[col]
	complete_df[col].fillna = complete_df[col].fillna
	complete_df[col] = complete_df[col].fillna(0)
	complete_df['annual_income'] = complete_df.groupby('region')['annual_income'].transform((lambda x: x.fillna(x.mean())))
	return complete_df
=============

# Function 244
def cleaning_func_27(col,df):
	# additional context code from user definitions

	def finding_regions(state):
	    if (state in west):
	        return 'West'
	    elif (state in south_west):
	        return 'SouthWest'
	    elif (state in south_east):
	        return 'SouthEast'
	    elif (state in mid_west):
	        return 'MidWest'
	    elif (state in north_east):
	        return 'NorthEast'


	def loan_condition(status):
	    if (status in bad_loan):
	        return 'Bad Loan'
	    else:
	        return 'Good Loan'

	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	bad_loan = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
	df['loan_condition'] = np.nan
	df['loan_condition'] = df['loan_status'].apply(loan_condition)
	west = ['CA', 'OR', 'UT', 'WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID']
	south_west = ['AZ', 'TX', 'NM', 'OK']
	south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN']
	mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
	north_east = ['CT', 'NY', 'PA', 'NJ', 'RI', 'MA', 'MD', 'VT', 'NH', 'ME']
	df['region'] = np.nan
	df['region'] = df['addr_state'].apply(finding_regions)
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	badloans_df = df.loc[(df['loan_condition'] == 'Bad Loan')]
	number_of_loanstatus = pd.crosstab(badloans_df['region'], badloans_df['loan_status'])
	number_of_loanstatus['Total'] = number_of_loanstatus.sum(axis=1)
	return number_of_loanstatus
=============

# Function 245
def cleaning_func_28(col,df):
	# additional context code from user definitions

	def finding_regions(state):
	    if (state in west):
	        return 'West'
	    elif (state in south_west):
	        return 'SouthWest'
	    elif (state in south_east):
	        return 'SouthEast'
	    elif (state in mid_west):
	        return 'MidWest'
	    elif (state in north_east):
	        return 'NorthEast'


	def loan_condition(status):
	    if (status in bad_loan):
	        return 'Bad Loan'
	    else:
	        return 'Good Loan'

	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	bad_loan = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
	df['loan_condition'] = np.nan
	df['loan_condition'] = df['loan_status'].apply(loan_condition)
	west = ['CA', 'OR', 'UT', 'WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID']
	south_west = ['AZ', 'TX', 'NM', 'OK']
	south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN']
	mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
	north_east = ['CT', 'NY', 'PA', 'NJ', 'RI', 'MA', 'MD', 'VT', 'NH', 'ME']
	df['region'] = np.nan
	df['region'] = df['addr_state'].apply(finding_regions)
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	return col
=============

# Function 246
def cleaning_func_29(col,df):
	# additional context code from user definitions

	def finding_regions(state):
	    if (state in west):
	        return 'West'
	    elif (state in south_west):
	        return 'SouthWest'
	    elif (state in south_east):
	        return 'SouthEast'
	    elif (state in mid_west):
	        return 'MidWest'
	    elif (state in north_east):
	        return 'NorthEast'


	def loan_condition(status):
	    if (status in bad_loan):
	        return 'Bad Loan'
	    else:
	        return 'Good Loan'

	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	bad_loan = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
	df['loan_condition'] = np.nan
	df['loan_condition'] = df['loan_status'].apply(loan_condition)
	west = ['CA', 'OR', 'UT', 'WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID']
	south_west = ['AZ', 'TX', 'NM', 'OK']
	south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN']
	mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
	north_east = ['CT', 'NY', 'PA', 'NJ', 'RI', 'MA', 'MD', 'VT', 'NH', 'ME']
	df['region'] = np.nan
	df['region'] = df['addr_state'].apply(finding_regions)
	return df
=============

# Function 247
def cleaning_func_30(col,df):
	# additional context code from user definitions

	def finding_regions(state):
	    if (state in west):
	        return 'West'
	    elif (state in south_west):
	        return 'SouthWest'
	    elif (state in south_east):
	        return 'SouthEast'
	    elif (state in mid_west):
	        return 'MidWest'
	    elif (state in north_east):
	        return 'NorthEast'


	def loan_condition(status):
	    if (status in bad_loan):
	        return 'Bad Loan'
	    else:
	        return 'Good Loan'

	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	bad_loan = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
	df['loan_condition'] = np.nan
	df['loan_condition'] = df['loan_status'].apply(loan_condition)
	west = ['CA', 'OR', 'UT', 'WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID']
	south_west = ['AZ', 'TX', 'NM', 'OK']
	south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN']
	mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
	north_east = ['CT', 'NY', 'PA', 'NJ', 'RI', 'MA', 'MD', 'VT', 'NH', 'ME']
	df['region'] = np.nan
	df['region'] = df['addr_state'].apply(finding_regions)
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	badloans_df = df.loc[(df['loan_condition'] == 'Bad Loan')]
	return badloans_df
=============

# Function 248
def cleaning_func_31(col,df):
	# additional context code from user definitions

	def finding_regions(state):
	    if (state in west):
	        return 'West'
	    elif (state in south_west):
	        return 'SouthWest'
	    elif (state in south_east):
	        return 'SouthEast'
	    elif (state in mid_west):
	        return 'MidWest'
	    elif (state in north_east):
	        return 'NorthEast'


	def loan_condition(status):
	    if (status in bad_loan):
	        return 'Bad Loan'
	    else:
	        return 'Good Loan'

	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	bad_loan = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
	df['loan_condition'] = np.nan
	df['loan_condition'] = df['loan_status'].apply(loan_condition)
	west = ['CA', 'OR', 'UT', 'WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID']
	south_west = ['AZ', 'TX', 'NM', 'OK']
	south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN']
	mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
	north_east = ['CT', 'NY', 'PA', 'NJ', 'RI', 'MA', 'MD', 'VT', 'NH', 'ME']
	df['region'] = np.nan
	df['region'] = df['addr_state'].apply(finding_regions)
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	badloans_df = df.loc[(df['loan_condition'] == 'Bad Loan')]
	loan_status_cross = pd.crosstab(badloans_df['region'], badloans_df['loan_status']).apply((lambda x: ((x / x.sum()) * 100)))
	loan_status_cross['In Grace Period'] = loan_status_cross['In Grace Period'].apply((lambda x: round(x, 2)))
	return loan_status_cross
=============

# Function 249
def cleaning_func_38(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	group_dates = df.groupby(['complete_date', 'region'], as_index=False).sum()
	group_dates = group_dates.groupby(['issue_d', 'region'], as_index=False).sum()
	group_dates = group_dates.groupby(['issue_d', 'region'], as_index=False).sum()
	group_dates['loan_amount'] = (group_dates['loan_amount'] / 1000)
	by_loan_amount = df.groupby(['region', 'addr_state'], as_index=False).loan_amount.sum()
	by_interest_rate = df.groupby(['region', 'addr_state'], as_index=False).interest_rate.mean()
	return by_interest_rate
=============

# Function 250
def cleaning_func_39(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	group_dates = df.groupby(['complete_date', 'region'], as_index=False).sum()
	group_dates = group_dates.groupby(['issue_d', 'region'], as_index=False).sum()
	group_dates = group_dates.groupby(['issue_d', 'region'], as_index=False).sum()
	group_dates['loan_amount'] = (group_dates['loan_amount'] / 1000)
	by_loan_amount = df.groupby(['region', 'addr_state'], as_index=False).loan_amount.sum()
	by_interest_rate = df.groupby(['region', 'addr_state'], as_index=False).interest_rate.mean()
	by_income = df.groupby(['region', 'addr_state'], as_index=False).annual_income.mean()
	return by_income
=============

# Function 251
def cleaning_func_40(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	group_dates = df.groupby(['complete_date', 'region'], as_index=False).sum()
	group_dates = group_dates.groupby(['issue_d', 'region'], as_index=False).sum()
	group_dates = group_dates.groupby(['issue_d', 'region'], as_index=False).sum()
	group_dates['loan_amount'] = (group_dates['loan_amount'] / 1000)
	by_loan_amount = df.groupby(['region', 'addr_state'], as_index=False).loan_amount.sum()
	by_interest_rate = df.groupby(['region', 'addr_state'], as_index=False).interest_rate.mean()
	by_income = df.groupby(['region', 'addr_state'], as_index=False).annual_income.mean()
	states = by_loan_amount['addr_state'].values.tolist()
	average_loan_amounts = by_loan_amount['loan_amount'].values.tolist()
	average_interest_rates = by_interest_rate['interest_rate'].values.tolist()
	average_annual_income = by_income['annual_income'].values.tolist()
	from collections import OrderedDict
	metrics_data = OrderedDict([('state_codes', states), ('issued_loans', average_loan_amounts), ('interest_rate', average_interest_rates), ('annual_income', average_annual_income)])
	metrics_df = pd.DataFrame.from_dict(metrics_data)
	return metrics_df
=============

# Function 252
def cleaning_func_41(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	group_dates = df.groupby(['complete_date', 'region'], as_index=False).sum()
	group_dates = group_dates.groupby(['issue_d', 'region'], as_index=False).sum()
	group_dates = group_dates.groupby(['issue_d', 'region'], as_index=False).sum()
	group_dates['loan_amount'] = (group_dates['loan_amount'] / 1000)
	by_loan_amount = df.groupby(['region', 'addr_state'], as_index=False).loan_amount.sum()
	return by_loan_amount
=============

# Function 253
def cleaning_func_42(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	group_dates = df.groupby(['complete_date', 'region'], as_index=False).sum()
	group_dates = group_dates.groupby(['issue_d', 'region'], as_index=False).sum()
	group_dates = group_dates.groupby(['issue_d', 'region'], as_index=False).sum()
	group_dates['loan_amount'] = (group_dates['loan_amount'] / 1000)
	by_loan_amount = df.groupby(['region', 'addr_state'], as_index=False).loan_amount.sum()
	by_interest_rate = df.groupby(['region', 'addr_state'], as_index=False).interest_rate.mean()
	by_income = df.groupby(['region', 'addr_state'], as_index=False).annual_income.mean()
	states = by_loan_amount['addr_state'].values.tolist()
	average_loan_amounts = by_loan_amount['loan_amount'].values.tolist()
	average_interest_rates = by_interest_rate['interest_rate'].values.tolist()
	average_annual_income = by_income['annual_income'].values.tolist()
	from collections import OrderedDict
	metrics_data = OrderedDict([('state_codes', states), ('issued_loans', average_loan_amounts), ('interest_rate', average_interest_rates), ('annual_income', average_annual_income)])
	metrics_df = pd.DataFrame.from_dict(metrics_data)
	metrics_df = metrics_df.round(decimals=2)
	metrics_df[col] = metrics_df[col].astype(str)
	metrics_df[col] = metrics_df[col]
	metrics_df[col].astype = metrics_df[col].astype
	metrics_df[col] = metrics_df[col].astype(str)
	metrics_df['text'] = ((((((metrics_df['state_codes'] + '<br>') + 'Average loan interest rate: ') + metrics_df['interest_rate']) + '<br>') + 'Average annual income: ') + metrics_df['annual_income'])
	return metrics_df
=============

# Function 254
def cleaning_func_43(col,df):
	# additional context code from user definitions

	def loan_condition(status):
	    if (status in bad_loan):
	        return 'Bad Loan'
	    else:
	        return 'Good Loan'

	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	bad_loan = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
	df['loan_condition'] = np.nan
	df['loan_condition'] = df['loan_status'].apply(loan_condition)
	df['emp_length_int'] = np.nan
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	by_loan_amount = df.groupby(['region', 'addr_state'], as_index=False).loan_amount.sum()
	states = by_loan_amount['addr_state'].values.tolist()
	from collections import OrderedDict
	col.loc[((col['annual_income'] <= 100000), 'income_category')] = 'Low'
	col.loc[(((col['annual_income'] > 100000) & (col['annual_income'] <= 200000)), 'income_category')] = 'Medium'
	col.loc[((col['annual_income'] > 200000), 'income_category')] = 'High'
	col.loc[((df['loan_condition'] == 'Bad Loan'), 'loan_condition_int')] = 0
	col.loc[((df['loan_condition'] == 'Good Loan'), 'loan_condition_int')] = 1
	by_emp_length = df.groupby(['region', 'addr_state'], as_index=False).emp_length_int.mean().sort_values(by='addr_state')
	loan_condition_bystate = pd.crosstab(df['addr_state'], df['loan_condition'])
	return loan_condition_bystate
=============

# Function 255
def cleaning_func_44(col,df):
	# additional context code from user definitions

	def loan_condition(status):
	    if (status in bad_loan):
	        return 'Bad Loan'
	    else:
	        return 'Good Loan'

	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	bad_loan = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
	df['loan_condition'] = np.nan
	df['loan_condition'] = df['loan_status'].apply(loan_condition)
	df['emp_length_int'] = np.nan
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	by_loan_amount = df.groupby(['region', 'addr_state'], as_index=False).loan_amount.sum()
	states = by_loan_amount['addr_state'].values.tolist()
	from collections import OrderedDict
	col.loc[((col['annual_income'] <= 100000), 'income_category')] = 'Low'
	col.loc[(((col['annual_income'] > 100000) & (col['annual_income'] <= 200000)), 'income_category')] = 'Medium'
	col.loc[((col['annual_income'] > 200000), 'income_category')] = 'High'
	col.loc[((df['loan_condition'] == 'Bad Loan'), 'loan_condition_int')] = 0
	col.loc[((df['loan_condition'] == 'Good Loan'), 'loan_condition_int')] = 1
	by_emp_length = df.groupby(['region', 'addr_state'], as_index=False).emp_length_int.mean().sort_values(by='addr_state')
	return by_emp_length
=============

# Function 256
def cleaning_func_45(col,df):
	# additional context code from user definitions

	def loan_condition(status):
	    if (status in bad_loan):
	        return 'Bad Loan'
	    else:
	        return 'Good Loan'

	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	bad_loan = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
	df['loan_condition'] = np.nan
	df['loan_condition'] = df['loan_status'].apply(loan_condition)
	df['emp_length_int'] = np.nan
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	by_loan_amount = df.groupby(['region', 'addr_state'], as_index=False).loan_amount.sum()
	states = by_loan_amount['addr_state'].values.tolist()
	from collections import OrderedDict
	col.loc[((col['annual_income'] <= 100000), 'income_category')] = 'Low'
	col.loc[(((col['annual_income'] > 100000) & (col['annual_income'] <= 200000)), 'income_category')] = 'Medium'
	col.loc[((col['annual_income'] > 200000), 'income_category')] = 'High'
	col.loc[((df['loan_condition'] == 'Bad Loan'), 'loan_condition_int')] = 0
	col.loc[((df['loan_condition'] == 'Good Loan'), 'loan_condition_int')] = 1
	by_emp_length = df.groupby(['region', 'addr_state'], as_index=False).emp_length_int.mean().sort_values(by='addr_state')
	loan_condition_bystate = pd.crosstab(df['addr_state'], df['loan_condition'])
	cross_condition = pd.crosstab(df['addr_state'], df['loan_condition'])
	return cross_condition
=============

# Function 257
def cleaning_func_46(col,df):
	# additional context code from user definitions

	def loan_condition(status):
	    if (status in bad_loan):
	        return 'Bad Loan'
	    else:
	        return 'Good Loan'

	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	bad_loan = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
	df['loan_condition'] = np.nan
	df['loan_condition'] = df['loan_status'].apply(loan_condition)
	df['emp_length_int'] = np.nan
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	by_loan_amount = df.groupby(['region', 'addr_state'], as_index=False).loan_amount.sum()
	states = by_loan_amount['addr_state'].values.tolist()
	from collections import OrderedDict
	col.loc[((col['annual_income'] <= 100000), 'income_category')] = 'Low'
	col.loc[(((col['annual_income'] > 100000) & (col['annual_income'] <= 200000)), 'income_category')] = 'Medium'
	col.loc[((col['annual_income'] > 200000), 'income_category')] = 'High'
	col.loc[((df['loan_condition'] == 'Bad Loan'), 'loan_condition_int')] = 0
	col.loc[((df['loan_condition'] == 'Good Loan'), 'loan_condition_int')] = 1
	return col
=============

# Function 258
def cleaning_func_47(col,df):
	# additional context code from user definitions

	def loan_condition(status):
	    if (status in bad_loan):
	        return 'Bad Loan'
	    else:
	        return 'Good Loan'

	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	bad_loan = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
	df['loan_condition'] = np.nan
	df['loan_condition'] = df['loan_status'].apply(loan_condition)
	df['emp_length_int'] = np.nan
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	by_loan_amount = df.groupby(['region', 'addr_state'], as_index=False).loan_amount.sum()
	states = by_loan_amount['addr_state'].values.tolist()
	from collections import OrderedDict
	col.loc[((col['annual_income'] <= 100000), 'income_category')] = 'Low'
	col.loc[(((col['annual_income'] > 100000) & (col['annual_income'] <= 200000)), 'income_category')] = 'Medium'
	col.loc[((col['annual_income'] > 200000), 'income_category')] = 'High'
	col.loc[((df['loan_condition'] == 'Bad Loan'), 'loan_condition_int')] = 0
	col.loc[((df['loan_condition'] == 'Good Loan'), 'loan_condition_int')] = 1
	by_emp_length = df.groupby(['region', 'addr_state'], as_index=False).emp_length_int.mean().sort_values(by='addr_state')
	loan_condition_bystate = pd.crosstab(df['addr_state'], df['loan_condition'])
	cross_condition = pd.crosstab(df['addr_state'], df['loan_condition'])
	percentage_loan_contributor = pd.crosstab(df['addr_state'], df['loan_condition']).apply((lambda x: ((x / x.sum()) * 100)))
	condition_ratio = (cross_condition['Bad Loan'] / cross_condition['Good Loan'])
	by_dti = df.groupby(['region', 'addr_state'], as_index=False).dti.mean()
	return by_dti
=============

# Function 259
def cleaning_func_48(col,df):
	# additional context code from user definitions

	def loan_condition(status):
	    if (status in bad_loan):
	        return 'Bad Loan'
	    else:
	        return 'Good Loan'

	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	bad_loan = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
	df['loan_condition'] = np.nan
	df['loan_condition'] = df['loan_status'].apply(loan_condition)
	df['emp_length_int'] = np.nan
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	by_loan_amount = df.groupby(['region', 'addr_state'], as_index=False).loan_amount.sum()
	states = by_loan_amount['addr_state'].values.tolist()
	from collections import OrderedDict
	col.loc[((col['annual_income'] <= 100000), 'income_category')] = 'Low'
	col.loc[(((col['annual_income'] > 100000) & (col['annual_income'] <= 200000)), 'income_category')] = 'Medium'
	col.loc[((col['annual_income'] > 200000), 'income_category')] = 'High'
	col.loc[((df['loan_condition'] == 'Bad Loan'), 'loan_condition_int')] = 0
	col.loc[((df['loan_condition'] == 'Good Loan'), 'loan_condition_int')] = 1
	by_emp_length = df.groupby(['region', 'addr_state'], as_index=False).emp_length_int.mean().sort_values(by='addr_state')
	loan_condition_bystate = pd.crosstab(df['addr_state'], df['loan_condition'])
	cross_condition = pd.crosstab(df['addr_state'], df['loan_condition'])
	percentage_loan_contributor = pd.crosstab(df['addr_state'], df['loan_condition']).apply((lambda x: ((x / x.sum()) * 100)))
	condition_ratio = (cross_condition['Bad Loan'] / cross_condition['Good Loan'])
	by_dti = df.groupby(['region', 'addr_state'], as_index=False).dti.mean()
	state_codes = sorted(states)
	default_ratio = condition_ratio.values.tolist()
	average_dti = by_dti['dti'].values.tolist()
	average_emp_length = by_emp_length['emp_length_int'].values.tolist()
	number_of_badloans = loan_condition_bystate['Bad Loan'].values.tolist()
	percentage_ofall_badloans = percentage_loan_contributor['Bad Loan'].values.tolist()
	risk_data = OrderedDict([('state_codes', state_codes), ('default_ratio', default_ratio), ('badloans_amount', number_of_badloans), ('percentage_of_badloans', percentage_ofall_badloans), ('average_dti', average_dti), ('average_emp_length', average_emp_length)])
	risk_df = pd.DataFrame.from_dict(risk_data)
	return risk_df
=============

# Function 260
def cleaning_func_50(col,df):
	# additional context code from user definitions

	def loan_condition(status):
	    if (status in bad_loan):
	        return 'Bad Loan'
	    else:
	        return 'Good Loan'

	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	bad_loan = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
	df['loan_condition'] = np.nan
	df['loan_condition'] = df['loan_status'].apply(loan_condition)
	df['emp_length_int'] = np.nan
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	by_loan_amount = df.groupby(['region', 'addr_state'], as_index=False).loan_amount.sum()
	states = by_loan_amount['addr_state'].values.tolist()
	from collections import OrderedDict
	col.loc[((col['annual_income'] <= 100000), 'income_category')] = 'Low'
	col.loc[(((col['annual_income'] > 100000) & (col['annual_income'] <= 200000)), 'income_category')] = 'Medium'
	col.loc[((col['annual_income'] > 200000), 'income_category')] = 'High'
	col.loc[((df['loan_condition'] == 'Bad Loan'), 'loan_condition_int')] = 0
	col.loc[((df['loan_condition'] == 'Good Loan'), 'loan_condition_int')] = 1
	by_emp_length = df.groupby(['region', 'addr_state'], as_index=False).emp_length_int.mean().sort_values(by='addr_state')
	loan_condition_bystate = pd.crosstab(df['addr_state'], df['loan_condition'])
	cross_condition = pd.crosstab(df['addr_state'], df['loan_condition'])
	percentage_loan_contributor = pd.crosstab(df['addr_state'], df['loan_condition']).apply((lambda x: ((x / x.sum()) * 100)))
	condition_ratio = (cross_condition['Bad Loan'] / cross_condition['Good Loan'])
	by_dti = df.groupby(['region', 'addr_state'], as_index=False).dti.mean()
	state_codes = sorted(states)
	default_ratio = condition_ratio.values.tolist()
	average_dti = by_dti['dti'].values.tolist()
	average_emp_length = by_emp_length['emp_length_int'].values.tolist()
	number_of_badloans = loan_condition_bystate['Bad Loan'].values.tolist()
	percentage_ofall_badloans = percentage_loan_contributor['Bad Loan'].values.tolist()
	risk_data = OrderedDict([('state_codes', state_codes), ('default_ratio', default_ratio), ('badloans_amount', number_of_badloans), ('percentage_of_badloans', percentage_ofall_badloans), ('average_dti', average_dti), ('average_emp_length', average_emp_length)])
	risk_df = pd.DataFrame.from_dict(risk_data)
	risk_df = risk_df.round(decimals=3)
	risk_df[col] = risk_df[col].astype(str)
	risk_df[col] = risk_df[col]
	risk_df[col].astype = risk_df[col].astype
	risk_df[col] = risk_df[col].astype(str)
	risk_df['text'] = (((((((((((((risk_df['state_codes'] + '<br>') + 'Number of Bad Loans: ') + risk_df['badloans_amount']) + '<br>') + 'Percentage of all Bad Loans: ') + risk_df['percentage_of_badloans']) + '%') + '<br>') + 'Average Debt-to-Income Ratio: ') + risk_df['average_dti']) + '<br>') + 'Average Length of Employment: ') + risk_df['average_emp_length'])
	return risk_df
=============

# Function 261
def cleaning_func_51(col,df):
	# additional context code from user definitions

	def loan_condition(status):
	    if (status in bad_loan):
	        return 'Bad Loan'
	    else:
	        return 'Good Loan'

	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	bad_loan = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
	df['loan_condition'] = np.nan
	df['loan_condition'] = df['loan_status'].apply(loan_condition)
	df['emp_length_int'] = np.nan
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	by_loan_amount = df.groupby(['region', 'addr_state'], as_index=False).loan_amount.sum()
	return by_loan_amount
=============

# Function 262
def cleaning_func_53(col,df):
	# additional context code from user definitions

	def loan_condition(status):
	    if (status in bad_loan):
	        return 'Bad Loan'
	    else:
	        return 'Good Loan'

	# core cleaning code
	import pandas as pd
	import numpy as np
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	bad_loan = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
	df['loan_condition'] = np.nan
	df['loan_condition'] = df['loan_status'].apply(loan_condition)
	df['emp_length_int'] = np.nan
	col.loc[((col['emp_length'] == '10+ years'), 'emp_length_int')] = 10
	col.loc[((col['emp_length'] == '9 years'), 'emp_length_int')] = 9
	col.loc[((col['emp_length'] == '8 years'), 'emp_length_int')] = 8
	col.loc[((col['emp_length'] == '7 years'), 'emp_length_int')] = 7
	col.loc[((col['emp_length'] == '6 years'), 'emp_length_int')] = 6
	col.loc[((col['emp_length'] == '5 years'), 'emp_length_int')] = 5
	col.loc[((col['emp_length'] == '4 years'), 'emp_length_int')] = 4
	col.loc[((col['emp_length'] == '3 years'), 'emp_length_int')] = 3
	col.loc[((col['emp_length'] == '2 years'), 'emp_length_int')] = 2
	col.loc[((col['emp_length'] == '1 year'), 'emp_length_int')] = 1
	col.loc[((col['emp_length'] == '< 1 year'), 'emp_length_int')] = 0.5
	col.loc[((col['emp_length'] == 'n/a'), 'emp_length_int')] = 0
	by_loan_amount = df.groupby(['region', 'addr_state'], as_index=False).loan_amount.sum()
	states = by_loan_amount['addr_state'].values.tolist()
	from collections import OrderedDict
	col.loc[((col['annual_income'] <= 100000), 'income_category')] = 'Low'
	col.loc[(((col['annual_income'] > 100000) & (col['annual_income'] <= 200000)), 'income_category')] = 'Medium'
	col.loc[((col['annual_income'] > 200000), 'income_category')] = 'High'
	col.loc[((df['loan_condition'] == 'Bad Loan'), 'loan_condition_int')] = 0
	col.loc[((df['loan_condition'] == 'Good Loan'), 'loan_condition_int')] = 1
	by_emp_length = df.groupby(['region', 'addr_state'], as_index=False).emp_length_int.mean().sort_values(by='addr_state')
	loan_condition_bystate = pd.crosstab(df['addr_state'], df['loan_condition'])
	cross_condition = pd.crosstab(df['addr_state'], df['loan_condition'])
	percentage_loan_contributor = pd.crosstab(df['addr_state'], df['loan_condition']).apply((lambda x: ((x / x.sum()) * 100)))
	return percentage_loan_contributor
=============

# Function 263
def cleaning_func_0(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	return data
=============

# Function 264
def cleaning_func_1(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data.next_pymnt_d = pd.to_datetime(data.next_pymnt_d)
	return data
=============

# Function 265
def cleaning_func_2(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data.issue_d = pd.to_datetime(data.issue_d)
	return data
=============

# Function 266
def cleaning_func_3(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data.last_pymnt_d = pd.to_datetime(data.last_pymnt_d)
	return data
=============

# Function 267
def cleaning_func_4(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data.last_credit_pull_d = pd.to_datetime(data.last_credit_pull_d)
	return data
=============

# Function 268
def cleaning_func_5(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	earliest_cr_line = pd.to_datetime(data.earliest_cr_line)
	data.earliest_cr_line = earliest_cr_line.dt.year
	return data
=============

# Function 269
def cleaning_func_7(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data['rating'] = np.where((data.loan_status != 'Current'), 1, 0)
	return data
=============

# Function 270
def cleaning_func_8(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data['recovery'] = np.where((data.recoveries != 0.0), 1, 0)
	return data
=============

# Function 271
def cleaning_func_9(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data.emp_length = data.emp_length.replace(np.nan, 0)
	return data
=============

# Function 272
def cleaning_func_10(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data.dti_joint = data.dti_joint.replace(np.nan, 0)
	return data
=============

# Function 273
def cleaning_func_11(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data.annual_inc_joint = data.annual_inc_joint.replace(np.nan, 0)
	return data
=============

# Function 274
def cleaning_func_12(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data.verification_status_joint = data.verification_status_joint.replace(np.nan, 'None')
	return data
=============

# Function 275
def cleaning_func_13(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data[e] = data[e].replace(np.nan, 0)
	data[e] = data[e]
	data[e].replace = data[e].replace
	np.nan = np.nan
	data[e] = data[e].replace(np.nan, 0)
	data.loc[(data.mths_since_last_delinq.notnull(), 'delinq')] = 1
	data.loc[(data.mths_since_last_delinq.isnull(), 'delinq')] = 0
	return data
=============

# Function 276
def cleaning_func_14(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data[e] = data[e].replace(np.nan, 0)
	data[e] = data[e]
	data[e].replace = data[e].replace
	np.nan = np.nan
	data[e] = data[e].replace(np.nan, 0)
	data.loc[(data.mths_since_last_delinq.notnull(), 'delinq')] = 1
	data.loc[(data.mths_since_last_delinq.isnull(), 'delinq')] = 0
	data.loc[(data.mths_since_last_major_derog.notnull(), 'derog')] = 1
	data.loc[(data.mths_since_last_major_derog.isnull(), 'derog')] = 0
	return data
=============

# Function 277
def cleaning_func_15(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data[e] = data[e].replace(np.nan, 0)
	data[e] = data[e]
	data[e].replace = data[e].replace
	np.nan = np.nan
	data[e] = data[e].replace(np.nan, 0)
	data.loc[(data.mths_since_last_delinq.notnull(), 'delinq')] = 1
	data.loc[(data.mths_since_last_delinq.isnull(), 'delinq')] = 0
	data.loc[(data.mths_since_last_major_derog.notnull(), 'derog')] = 1
	data.loc[(data.mths_since_last_major_derog.isnull(), 'derog')] = 0
	data.loc[(data.mths_since_last_record.notnull(), 'public_record')] = 1
	data.loc[(data.mths_since_last_record.isnull(), 'public_record')] = 0
	return data
=============

# Function 278
def cleaning_func_16(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data[e] = data[e].replace(np.nan, 0)
	data[e] = data[e]
	data[e].replace = data[e].replace
	np.nan = np.nan
	data[e] = data[e].replace(np.nan, 0)
	data.loc[(data.mths_since_last_delinq.notnull(), 'delinq')] = 1
	data.loc[(data.mths_since_last_delinq.isnull(), 'delinq')] = 0
	data[e] = data[e].replace(np.nan, 0)
	data[e] = data[e]
	data[e].replace = data[e].replace
	np.nan = np.nan
	data[e] = data[e].replace(np.nan, 0)
	data.revol_util = data.revol_util.replace(np.nan, 0)
	return data
=============

# Function 279
def cleaning_func_17(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data.last_pymnt_d = pd.to_datetime(data.last_pymnt_d)
	data[e] = data[e].replace(np.nan, 0)
	data[e] = data[e]
	data[e].replace = data[e].replace
	np.nan = np.nan
	data[e] = data[e].replace(np.nan, 0)
	data.loc[(data.mths_since_last_delinq.notnull(), 'delinq')] = 1
	data.loc[(data.mths_since_last_delinq.isnull(), 'delinq')] = 0
	data.loc[(data.mths_since_last_major_derog.notnull(), 'derog')] = 1
	data.loc[(data.mths_since_last_major_derog.isnull(), 'derog')] = 0
	data.loc[(data.mths_since_last_record.notnull(), 'public_record')] = 1
	data.loc[(data.mths_since_last_record.isnull(), 'public_record')] = 0
	data[e] = data[e].replace(np.nan, 0)
	data[e] = data[e]
	data[e].replace = data[e].replace
	np.nan = np.nan
	data[e] = data[e].replace(np.nan, 0)
	data.loc[(data.last_pymnt_d.notnull(), 'pymnt_received')] = 1
	data.loc[(data.last_pymnt_d.isnull(), 'pymnt_received')] = 0
	return data
=============

# Function 280
def cleaning_func_0(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv')
	del_cols = ['id', 'member_id', 'policy_code', 'url', 'zip_code', 'addr_state', 'pymnt_plan', 'emp_title', 'application_type', 'acc_now_delinq', 'title', 'collections_12_mths_ex_med', 'collection_recovery_fee']
	loan = loan.drop(del_cols, axis=1)
	loan = loan[(loan['loan_status'] != 'Current')]
	loan['empl_exp'] = 'experienced'
	return loan
=============

# Function 281
def cleaning_func_1(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv')
	del_cols = ['id', 'member_id', 'policy_code', 'url', 'zip_code', 'addr_state', 'pymnt_plan', 'emp_title', 'application_type', 'acc_now_delinq', 'title', 'collections_12_mths_ex_med', 'collection_recovery_fee']
	loan = loan.drop(del_cols, axis=1)
	return loan
=============

# Function 282
def cleaning_func_2(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv')
	del_cols = ['id', 'member_id', 'policy_code', 'url', 'zip_code', 'addr_state', 'pymnt_plan', 'emp_title', 'application_type', 'acc_now_delinq', 'title', 'collections_12_mths_ex_med', 'collection_recovery_fee']
	loan = loan.drop(del_cols, axis=1)
	loan = loan[(loan['loan_status'] != 'Current')]
	return loan
=============

# Function 283
def cleaning_func_3(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv')
	del_cols = ['id', 'member_id', 'policy_code', 'url', 'zip_code', 'addr_state', 'pymnt_plan', 'emp_title', 'application_type', 'acc_now_delinq', 'title', 'collections_12_mths_ex_med', 'collection_recovery_fee']
	loan = loan.drop(del_cols, axis=1)
	loan = loan[(loan['loan_status'] != 'Current')]
	loan = loan.drop('emp_length', axis=1)
	loan['target'] = 0
	return loan
=============

# Function 284
def cleaning_func_4(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv')
	del_cols = ['id', 'member_id', 'policy_code', 'url', 'zip_code', 'addr_state', 'pymnt_plan', 'emp_title', 'application_type', 'acc_now_delinq', 'title', 'collections_12_mths_ex_med', 'collection_recovery_fee']
	loan = loan.drop(del_cols, axis=1)
	loan = loan[(loan['loan_status'] != 'Current')]
	loan = loan.drop('emp_length', axis=1)
	mask = (loan.loan_status == 'Charged Off')
	loan.loc[(mask, 'target')] = 1
	return loan
=============

# Function 285
def cleaning_func_5(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv')
	del_cols = ['id', 'member_id', 'policy_code', 'url', 'zip_code', 'addr_state', 'pymnt_plan', 'emp_title', 'application_type', 'acc_now_delinq', 'title', 'collections_12_mths_ex_med', 'collection_recovery_fee']
	loan = loan.drop(del_cols, axis=1)
	loan = loan[(loan['loan_status'] != 'Current')]
	loan.loc[((loan['emp_length'] == '< 1 year'), 'empl_exp')] = 'inexp'
	loan.loc[((loan['emp_length'] == '1 year'), 'empl_exp')] = 'new'
	loan.loc[((loan['emp_length'] == '2 years'), 'empl_exp')] = 'new'
	loan.loc[((loan['emp_length'] == '3 years'), 'empl_exp')] = 'new'
	loan.loc[((loan['emp_length'] == '4 years'), 'empl_exp')] = 'intermed'
	loan.loc[((loan['emp_length'] == '5 years'), 'empl_exp')] = 'intermed'
	loan.loc[((loan['emp_length'] == '6 years'), 'empl_exp')] = 'intermed'
	loan.loc[((loan['emp_length'] == '7 years'), 'empl_exp')] = 'seasoned'
	loan.loc[((loan['emp_length'] == '8 years'), 'empl_exp')] = 'seasoned'
	loan.loc[((loan['emp_length'] == '9 years'), 'empl_exp')] = 'seasoned'
	loan.loc[((loan['emp_length'] == 'n/a'), 'empl_exp')] = 'unknown'
	return loan
=============

# Function 286
def cleaning_func_0(df_loan):
	# core cleaning code
	import pandas as pd
	# df_loan = pd.read_csv('../input/loan.csv', low_memory=False)
	df_loan['issue_date'] = df_loan['issue_d']
	return df_loan
=============

# Function 287
def cleaning_func_1(df_loan):
	# core cleaning code
	import pandas as pd
	# df_loan = pd.read_csv('../input/loan.csv', low_memory=False)
	df_loan['issue_d'] = pd.to_datetime(df_loan['issue_d'])
	return df_loan
=============

# Function 288
def cleaning_func_3(df_loan):
	# core cleaning code
	import pandas as pd
	# df_loan = pd.read_csv('../input/loan.csv', low_memory=False)
	df_loan['issue_d'] = pd.to_datetime(df_loan['issue_d'])
	df_loan.index = df_loan['issue_d']
	return df_loan
=============

# Function 289
def cleaning_func_4(df_loan):
	# core cleaning code
	import pandas as pd
	# df_loan = pd.read_csv('../input/loan.csv', low_memory=False)
	df_loan_dt = df_loan[slice('2010-01-01', '2015-12-01', None)]
	df_loan_dt['emp_title'] = df_loan_dt['emp_title'].replace({'RN': 'Registered Nurse'})
	df_loan_dt['emp_title'] = df_loan_dt['emp_title'].replace({'manager': 'Manager'})
	df_loan_dt['emp_title'] = df_loan_dt['emp_title'].replace({'driver': 'Driver'})
	df_loan_dt['emp_title'] = df_loan_dt['emp_title'].replace({'supervisor': 'Supervisor'})
	df_loan_dt['emp_title'] = df_loan_dt['emp_title'].replace({'owner': 'Owner'})
	return df_loan_dt
=============

