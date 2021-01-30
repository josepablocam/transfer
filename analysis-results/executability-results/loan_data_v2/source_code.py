# Function 0
def cleaning_func_0(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['90day_worse_rating'] = np.where(loan['mths_since_last_major_derog'].isnull(), 0, 1)
	return loan
=============

# Function 1
def cleaning_func_1(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['revol_util'] = loan['revol_util'].fillna(loan['revol_util'].median())
	return loan
=============

# Function 2
def cleaning_func_2(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['emp_title'] = np.where(loan['emp_title'].isnull(), 'Job title not given', loan['emp_title'])
	return loan
=============

# Function 3
def cleaning_func_3(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['acc_now_delinq'] = np.where(loan['acc_now_delinq'].isnull(), 0, loan['acc_now_delinq'])
	return loan
=============

# Function 4
def cleaning_func_4(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['delinq_2yrs'] = np.where(loan['delinq_2yrs'].isnull(), 0, loan['delinq_2yrs'])
	return loan
=============

# Function 5
def cleaning_func_5(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['tot_coll_amt'] = loan['tot_coll_amt'].fillna(loan['tot_coll_amt'].median())
	return loan
=============

# Function 6
def cleaning_func_6(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['title'] = np.where(loan['title'].isnull(), 0, loan['title'])
	return loan
=============

# Function 7
def cleaning_func_7(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['total_rev_hi_lim'] = loan['total_rev_hi_lim'].fillna(loan['total_rev_hi_lim'].median())
	return loan
=============

# Function 8
def cleaning_func_8(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['inq_last_6mths'] = np.where(loan['inq_last_6mths'].isnull(), 0, loan['inq_last_6mths'])
	return loan
=============

# Function 9
def cleaning_func_9(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['total_acc'] = np.where(loan['total_acc'].isnull(), 0, loan['total_acc'])
	return loan
=============

# Function 10
def cleaning_func_10(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['annual_inc'] = loan['annual_inc'].fillna(loan['annual_inc'].median())
	return loan
=============

# Function 11
def cleaning_func_11(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['open_acc'] = np.where(loan['open_acc'].isnull(), 0, loan['open_acc'])
	return loan
=============

# Function 12
def cleaning_func_12(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['collections_12_mths_ex_med'] = np.where(loan['collections_12_mths_ex_med'].isnull(), 0, loan['collections_12_mths_ex_med'])
	return loan
=============

# Function 13
def cleaning_func_13(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['tot_cur_bal'] = loan['tot_cur_bal'].fillna(loan['tot_cur_bal'].median())
	return loan
=============

# Function 14
def cleaning_func_14(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['pub_rec'] = np.where(loan['pub_rec'].isnull(), 0, loan['pub_rec'])
	return loan
=============

# Function 15
def cleaning_func_15(loan):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv', low_memory=False)
	loan['mths_since_last_delinq'] = np.where(loan['mths_since_last_delinq'].isnull(), 188, loan['mths_since_last_delinq'])
	return loan
=============

# Function 16
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

# Function 17
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

# Function 18
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

# Function 19
def cleaning_func_0(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data['bad_loan'] = 0
	return data
=============

# Function 20
def cleaning_func_1(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	bad_indicators = ['Charged Off ', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Default Receiver', 'Late (16-30 days)', 'Late (31-120 days)']
	data.loc[(data.loan_status.isin(bad_indicators), 'bad_loan')] = 1
	return data
=============

# Function 21
def cleaning_func_2(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data['issue_dt'] = pd.to_datetime(data.issue_d)
	return data
=============

# Function 22
def cleaning_func_3(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data['issue_dt'] = pd.to_datetime(data.issue_d)
	data['month'] = data['issue_dt'].dt.month
	return data
=============

# Function 23
def cleaning_func_4(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	data['issue_dt'] = pd.to_datetime(data.issue_d)
	data['year'] = data['issue_dt'].dt.year
	return data
=============

# Function 24
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

# Function 25
def cleaning_func_0(df):
	# core cleaning code
	import pandas as pd
	badLoan = ['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)', 'In Grace Period', 'Does not meet the credit policy. Status:Charged Off']
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_status', 'addr_state'])
	df['isBad'] = [(1 if (x in badLoan) else 0) for x in df.loan_status]
	return df
=============

# Function 26
def cleaning_func_4(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_amnt', ascending=False)
	perStatedf.columns = ['State', 'Num_Loans']
	return perStatedf
=============

# Function 27
def cleaning_func_5(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	return df.groupby('addr_state', as_index=False).count()
=============

# Function 28
def cleaning_func_6(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).sum().sort_values(by='loan_amnt', ascending=False)
	perStatedf.columns = ['State', 'loan_amt']
	return perStatedf
=============

# Function 29
def cleaning_func_8(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_status', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).sum().sort_values(by='isBad', ascending=False)
	perStatedf.columns = ['State', 'badLoans']
	return perStatedf
=============

# Function 30
def cleaning_func_10(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_status', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_status', ascending=False)
	perStatedf.columns = ['State', 'totalLoans']
	return perStatedf
=============

# Function 31
def cleaning_func_14(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_amnt', ascending=False)
	return perStatedf
=============

# Function 32
def cleaning_func_15(df):
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

# Function 33
def cleaning_func_16(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_amnt', ascending=False)
	statePop = {'CA': 39144818, 'TX': 27469144, 'FL': 20271878, 'NY': 19795791, 'IL': 12859995, 'PA': 12802503, 'OH': 11613423, 'GA': 10214860, 'NC': 10042802, 'MI': 9922576, 'NJ': 8958013, 'VA': 8382993, 'WA': 7170351, 'AZ': 6828065, 'MA': 6794422, 'IN': 6619680, 'TN': 6600299, 'MO': 6083672, 'MD': 6006401, 'WI': 5771337, 'MN': 5489594, 'CO': 5456574, 'SC': 4896146, 'AL': 4858979, 'LA': 4670724, 'KY': 4425092, 'OR': 4028977, 'OK': 3911338, 'CT': 3890886, 'IA': 3123899, 'UT': 2995919, 'MS': 2992333, 'AK': 2978204, 'KS': 2911641, 'NV': 2890845, 'NM': 2085109, 'NE': 1896190, 'WV': 1844128, 'ID': 1654930, 'HI': 1431603, 'NH': 1330608, 'ME': 1329328, 'RI': 1053298, 'MT': 1032949, 'DE': 945934, 'SD': 858469, 'ND': 756927, 'AK': 738432, 'DC': 672228, 'VT': 626042, 'WY': 586107}
	return pd.DataFrame.from_dict(statePop, orient='index')
=============

# Function 34
def cleaning_func_17(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_amnt', ascending=False)
	statePop = {'CA': 39144818, 'TX': 27469144, 'FL': 20271878, 'NY': 19795791, 'IL': 12859995, 'PA': 12802503, 'OH': 11613423, 'GA': 10214860, 'NC': 10042802, 'MI': 9922576, 'NJ': 8958013, 'VA': 8382993, 'WA': 7170351, 'AZ': 6828065, 'MA': 6794422, 'IN': 6619680, 'TN': 6600299, 'MO': 6083672, 'MD': 6006401, 'WI': 5771337, 'MN': 5489594, 'CO': 5456574, 'SC': 4896146, 'AL': 4858979, 'LA': 4670724, 'KY': 4425092, 'OR': 4028977, 'OK': 3911338, 'CT': 3890886, 'IA': 3123899, 'UT': 2995919, 'MS': 2992333, 'AK': 2978204, 'KS': 2911641, 'NV': 2890845, 'NM': 2085109, 'NE': 1896190, 'WV': 1844128, 'ID': 1654930, 'HI': 1431603, 'NH': 1330608, 'ME': 1329328, 'RI': 1053298, 'MT': 1032949, 'DE': 945934, 'SD': 858469, 'ND': 756927, 'AK': 738432, 'DC': 672228, 'VT': 626042, 'WY': 586107}
	statePopdf = pd.DataFrame.from_dict(statePop, orient='index').reset_index()
	return statePopdf
=============

# Function 35
def cleaning_func_18(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	statePop = {'CA': 39144818, 'TX': 27469144, 'FL': 20271878, 'NY': 19795791, 'IL': 12859995, 'PA': 12802503, 'OH': 11613423, 'GA': 10214860, 'NC': 10042802, 'MI': 9922576, 'NJ': 8958013, 'VA': 8382993, 'WA': 7170351, 'AZ': 6828065, 'MA': 6794422, 'IN': 6619680, 'TN': 6600299, 'MO': 6083672, 'MD': 6006401, 'WI': 5771337, 'MN': 5489594, 'CO': 5456574, 'SC': 4896146, 'AL': 4858979, 'LA': 4670724, 'KY': 4425092, 'OR': 4028977, 'OK': 3911338, 'CT': 3890886, 'IA': 3123899, 'UT': 2995919, 'MS': 2992333, 'AK': 2978204, 'KS': 2911641, 'NV': 2890845, 'NM': 2085109, 'NE': 1896190, 'WV': 1844128, 'ID': 1654930, 'HI': 1431603, 'NH': 1330608, 'ME': 1329328, 'RI': 1053298, 'MT': 1032949, 'DE': 945934, 'SD': 858469, 'ND': 756927, 'AK': 738432, 'DC': 672228, 'VT': 626042, 'WY': 586107}
	statePopdf = pd.DataFrame.from_dict(statePop, orient='index').reset_index()
	perStatedf = df.groupby('addr_state', as_index=False).sum().sort_values(by='loan_amnt', ascending=False)
	return perStatedf
=============

# Function 36
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

# Function 37
def cleaning_func_20(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	statePop = {'CA': 39144818, 'TX': 27469144, 'FL': 20271878, 'NY': 19795791, 'IL': 12859995, 'PA': 12802503, 'OH': 11613423, 'GA': 10214860, 'NC': 10042802, 'MI': 9922576, 'NJ': 8958013, 'VA': 8382993, 'WA': 7170351, 'AZ': 6828065, 'MA': 6794422, 'IN': 6619680, 'TN': 6600299, 'MO': 6083672, 'MD': 6006401, 'WI': 5771337, 'MN': 5489594, 'CO': 5456574, 'SC': 4896146, 'AL': 4858979, 'LA': 4670724, 'KY': 4425092, 'OR': 4028977, 'OK': 3911338, 'CT': 3890886, 'IA': 3123899, 'UT': 2995919, 'MS': 2992333, 'AK': 2978204, 'KS': 2911641, 'NV': 2890845, 'NM': 2085109, 'NE': 1896190, 'WV': 1844128, 'ID': 1654930, 'HI': 1431603, 'NH': 1330608, 'ME': 1329328, 'RI': 1053298, 'MT': 1032949, 'DE': 945934, 'SD': 858469, 'ND': 756927, 'AK': 738432, 'DC': 672228, 'VT': 626042, 'WY': 586107}
	statePopdf = pd.DataFrame.from_dict(statePop, orient='index').reset_index()
	return df.groupby('addr_state', as_index=False).sum()
=============

# Function 38
def cleaning_func_21(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
	statePop = {'CA': 39144818, 'TX': 27469144, 'FL': 20271878, 'NY': 19795791, 'IL': 12859995, 'PA': 12802503, 'OH': 11613423, 'GA': 10214860, 'NC': 10042802, 'MI': 9922576, 'NJ': 8958013, 'VA': 8382993, 'WA': 7170351, 'AZ': 6828065, 'MA': 6794422, 'IN': 6619680, 'TN': 6600299, 'MO': 6083672, 'MD': 6006401, 'WI': 5771337, 'MN': 5489594, 'CO': 5456574, 'SC': 4896146, 'AL': 4858979, 'LA': 4670724, 'KY': 4425092, 'OR': 4028977, 'OK': 3911338, 'CT': 3890886, 'IA': 3123899, 'UT': 2995919, 'MS': 2992333, 'AK': 2978204, 'KS': 2911641, 'NV': 2890845, 'NM': 2085109, 'NE': 1896190, 'WV': 1844128, 'ID': 1654930, 'HI': 1431603, 'NH': 1330608, 'ME': 1329328, 'RI': 1053298, 'MT': 1032949, 'DE': 945934, 'SD': 858469, 'ND': 756927, 'AK': 738432, 'DC': 672228, 'VT': 626042, 'WY': 586107}
	return pd.DataFrame.from_dict(statePop, orient='index')
=============

# Function 39
def cleaning_func_23(df):
	# core cleaning code
	import pandas as pd
	statePop = {'CA': 39144818, 'TX': 27469144, 'FL': 20271878, 'NY': 19795791, 'IL': 12859995, 'PA': 12802503, 'OH': 11613423, 'GA': 10214860, 'NC': 10042802, 'MI': 9922576, 'NJ': 8958013, 'VA': 8382993, 'WA': 7170351, 'AZ': 6828065, 'MA': 6794422, 'IN': 6619680, 'TN': 6600299, 'MO': 6083672, 'MD': 6006401, 'WI': 5771337, 'MN': 5489594, 'CO': 5456574, 'SC': 4896146, 'AL': 4858979, 'LA': 4670724, 'KY': 4425092, 'OR': 4028977, 'OK': 3911338, 'CT': 3890886, 'IA': 3123899, 'UT': 2995919, 'MS': 2992333, 'AK': 2978204, 'KS': 2911641, 'NV': 2890845, 'NM': 2085109, 'NE': 1896190, 'WV': 1844128, 'ID': 1654930, 'HI': 1431603, 'NH': 1330608, 'ME': 1329328, 'RI': 1053298, 'MT': 1032949, 'DE': 945934, 'SD': 858469, 'ND': 756927, 'AK': 738432, 'DC': 672228, 'VT': 626042, 'WY': 586107}
	statePopdf = pd.DataFrame.from_dict(statePop, orient='index').reset_index()
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_status', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).sum().sort_values(by='isBad', ascending=False)
	return perStatedf
=============

# Function 40
def cleaning_func_24(df):
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

# Function 41
def cleaning_func_27(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_status', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_status', ascending=False)
	return perStatedf
=============

# Function 42
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

# Function 43
def cleaning_func_29(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', usecols=['loan_status', 'addr_state'])
	perStatedf = df.groupby('addr_state', as_index=False).count().sort_values(by='loan_status', ascending=False)
	badLoansdf = df.groupby('addr_state', as_index=False).sum().sort_values(by='isBad', ascending=False)
	return badLoansdf
=============

# Function 44
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

# Function 45
def cleaning_func_0(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data.last_credit_pull_d = pd.to_datetime(data.last_credit_pull_d)
	return data
=============

# Function 46
def cleaning_func_1(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	return data
=============

# Function 47
def cleaning_func_2(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data.next_pymnt_d = pd.to_datetime(data.next_pymnt_d)
	return data
=============

# Function 48
def cleaning_func_3(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data.issue_d = pd.to_datetime(data.issue_d)
	return data
=============

# Function 49
def cleaning_func_4(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data.last_pymnt_d = pd.to_datetime(data.last_pymnt_d)
	return data
=============

# Function 50
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

# Function 51
def cleaning_func_6(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data['rating'] = np.where((data.loan_status != 'Current'), 1, 0)
	return data
=============

# Function 52
def cleaning_func_8(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data.emp_length = data.emp_length.replace(np.nan, 0)
	return data
=============

# Function 53
def cleaning_func_9(data):
	# core cleaning code
	import numpy as np
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', parse_dates=True)
	data = data[(data.loan_status != 'Fully Paid')]
	data = data[(data.loan_status != 'Does not meet the credit policy. Status:Fully Paid')]
	data['recovery'] = np.where((data.recoveries != 0.0), 1, 0)
	return data
=============

# Function 54
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

# Function 55
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

# Function 56
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

# Function 57
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
	data.loc[(data.mths_since_last_major_derog.notnull(), 'derog')] = 1
	data.loc[(data.mths_since_last_major_derog.isnull(), 'derog')] = 0
	return data
=============

# Function 58
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
	return data
=============

# Function 59
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

# Function 60
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
	data.loc[(data.mths_since_last_major_derog.notnull(), 'derog')] = 1
	data.loc[(data.mths_since_last_major_derog.isnull(), 'derog')] = 0
	data.loc[(data.mths_since_last_record.notnull(), 'public_record')] = 1
	data[e] = data[e].replace(np.nan, 0)
	data[e] = data[e]
	data[e].replace = data[e].replace
	np.nan = np.nan
	data[e] = data[e].replace(np.nan, 0)
	data.revol_util = data.revol_util.replace(np.nan, 0)
	return data
=============

# Function 61
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
	data[e] = data[e].replace(np.nan, 0)
	data[e] = data[e]
	data[e].replace = data[e].replace
	np.nan = np.nan
	data[e] = data[e].replace(np.nan, 0)
	data.loc[(data.last_pymnt_d.notnull(), 'pymnt_received')] = 1
	data.loc[(data.last_pymnt_d.isnull(), 'pymnt_received')] = 0
	return data
=============

# Function 62
def cleaning_func_0(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv')
	del_cols = ['id', 'member_id', 'policy_code', 'url', 'zip_code', 'addr_state', 'pymnt_plan', 'emp_title', 'application_type', 'acc_now_delinq', 'title', 'collections_12_mths_ex_med', 'collection_recovery_fee']
	loan = loan.drop(del_cols, axis=1)
	return loan
=============

# Function 63
def cleaning_func_1(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv')
	del_cols = ['id', 'member_id', 'policy_code', 'url', 'zip_code', 'addr_state', 'pymnt_plan', 'emp_title', 'application_type', 'acc_now_delinq', 'title', 'collections_12_mths_ex_med', 'collection_recovery_fee']
	loan = loan.drop(del_cols, axis=1)
	loan = loan[(loan['loan_status'] != 'Current')]
	loan['empl_exp'] = 'experienced'
	return loan
=============

# Function 64
def cleaning_func_2(loan):
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

# Function 65
def cleaning_func_3(loan):
	# core cleaning code
	import pandas as pd
	# loan = pd.read_csv('../input/loan.csv')
	del_cols = ['id', 'member_id', 'policy_code', 'url', 'zip_code', 'addr_state', 'pymnt_plan', 'emp_title', 'application_type', 'acc_now_delinq', 'title', 'collections_12_mths_ex_med', 'collection_recovery_fee']
	loan = loan.drop(del_cols, axis=1)
	loan = loan[(loan['loan_status'] != 'Current')]
	return loan
=============

# Function 66
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

# Function 67
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

