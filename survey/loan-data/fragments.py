# Task 1
# Identify non-current loans based on loan_status
# Transfer fragments (treatment)
# Fragment 0
def f0(data):
    # core cleaning code
    import pandas as pd
    # data = pd.read_csv('../input/loan.csv', parse_dates=True)
    data = data[(data.loan_status != 'Fully Paid')]
    return data


# Fragment 1
def f1(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv', low_memory=False)
    df = df.loc[(df['loan_status'] != 'Current')]
    return df


# Fragment 2
def f2(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv')
    df = df[((df.loan_status == 'Fully Paid') |
             (df.loan_status == 'Charged Off'))]
    return df


# Fragment 3
def f3(dataset):
    # core cleaning code
    import pandas as pd
    # dataset = pd.read_csv('../input/loan.csv', low_memory=False)
    dataset = dataset.fillna(0)
    dataset['loan_status'] = dataset['loan_status'].astype(
        'category'
    ).cat.codes
    return dataset


# Fragment 4
def f4(df):
    # core cleaning code
    import pandas as pd
    badLoan = [
        'Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)',
        'In Grace Period',
        'Does not meet the credit policy. Status:Charged Off'
    ]
    # df = pd.read_csv('../input/loan.csv', usecols=['loan_status', 'addr_state'])
    df['isBad'] = [(1 if (x in badLoan) else 0) for x in df.loan_status]
    return df


# Random fragments (control)
# Fragment 0
def f0(loan):
    # core cleaning code
    import pandas as pd
    # loan = pd.read_csv('../input/loan.csv')
    del_cols = [
        'id', 'member_id', 'policy_code', 'url', 'zip_code', 'addr_state',
        'pymnt_plan', 'emp_title', 'application_type', 'acc_now_delinq',
        'title', 'collections_12_mths_ex_med', 'collection_recovery_fee'
    ]
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


# Fragment 1
def f1(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv', low_memory=False)
    df = df.rename(
        columns={
            'loan_amnt': 'loan_amount',
            'funded_amnt': 'funded_amount',
            'funded_amnt_inv': 'investor_funds',
            'int_rate': 'interest_rate',
            'annual_inc': 'annual_income'
        }
    )
    group_dates = df.groupby(['complete_date', 'region'], as_index=False).sum()
    group_dates = group_dates.groupby(['issue_d', 'region'],
                                      as_index=False).sum()
    group_dates = group_dates.groupby(['issue_d', 'region'],
                                      as_index=False).sum()
    group_dates['loan_amount'] = (group_dates['loan_amount'] / 1000)
    by_loan_amount = df.groupby(['region', 'addr_state'],
                                as_index=False).loan_amount.sum()
    return by_loan_amount


# Fragment 2
def f2(dataset):
    # core cleaning code
    import pandas as pd
    # dataset = pd.read_csv('../input/loan.csv', low_memory=False)
    dataset = dataset.fillna(0)
    dataset['verification_status_joint'] = dataset[
        'verification_status_joint'].astype('category').cat.codes
    return dataset


# Fragment 3
def f3(dataset):
    # core cleaning code
    import pandas as pd
    # dataset = pd.read_csv('../input/loan.csv', low_memory=False)
    dataset = dataset.fillna(0)
    dataset['loan_status'] = dataset['loan_status'].astype(
        'category'
    ).cat.codes
    return dataset


# Fragment 4
def f4(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv', low_memory=False)
    df = df.loc[(df['loan_status'] != 'Current')]
    return df


# Task 2
# Round the interest rate column (`int_rate`) to nearest integer
# Transfer fragments (treatment)
# Fragment 0
def f0(df_loan):
    # core cleaning code
    import pandas as pd
    # df_loan = pd.read_csv('../input/loan.csv', low_memory=False)
    df_loan['int_round'] = df_loan['int_rate'].round(0).astype(int)
    return df_loan


# Fragment 1
def f1(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv', low_memory=False)
    df = df.rename(
        columns={
            'loan_amnt': 'loan_amount',
            'funded_amnt': 'funded_amount',
            'funded_amnt_inv': 'investor_funds',
            'int_rate': 'interest_rate',
            'annual_inc': 'annual_income'
        }
    )
    return df


# Fragment 2
def f2(data):
    # core cleaning code
    import pandas as pd
    # data = pd.read_csv('../input/loan.csv', low_memory=False)
    data['emp_length'] = data['emp_length'].astype(int)
    return data


# Fragment 3
def f3(dataset):
    # core cleaning code
    import pandas as pd
    # dataset = pd.read_csv('../input/loan.csv', low_memory=False)
    dataset = dataset.fillna(0)
    dataset['application_type'] = dataset['application_type'].astype(
        'category'
    ).cat.codes
    return dataset


# Fragment 4
def f4(dataset):
    # core cleaning code
    import pandas as pd
    # dataset = pd.read_csv('../input/loan.csv', low_memory=False)
    dataset = dataset.fillna(0)
    dataset['loan_status'] = dataset['loan_status'].astype(
        'category'
    ).cat.codes
    return dataset


# Random fragments (control)
# Fragment 0
def f0(loan):
    # core cleaning code
    import numpy as np
    import pandas as pd
    # loan = pd.read_csv('../input/loan.csv', low_memory=False)
    loan['title'] = np.where(loan['title'].isnull(), 0, loan['title'])
    return loan


# Fragment 1
def f1(dataset):
    # core cleaning code
    import pandas as pd
    # dataset = pd.read_csv('../input/loan.csv', low_memory=False)
    dataset = dataset.fillna(0)
    dataset['verification_status'] = dataset['verification_status'].astype(
        'category'
    ).cat.codes
    return dataset


# Fragment 2
def f2(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv', low_memory=False)
    df = df.rename(
        columns={
            'loan_amnt': 'loan_amount',
            'funded_amnt': 'funded_amount',
            'funded_amnt_inv': 'investor_funds',
            'int_rate': 'interest_rate',
            'annual_inc': 'annual_income'
        }
    )
    group_dates = df.groupby(['complete_date', 'region'], as_index=False).sum()
    group_dates = group_dates.groupby(['issue_d', 'region'],
                                      as_index=False).sum()
    group_dates = group_dates.groupby(['issue_d', 'region'],
                                      as_index=False).sum()
    group_dates['loan_amount'] = (group_dates['loan_amount'] / 1000)
    by_loan_amount = df.groupby(['region', 'addr_state'],
                                as_index=False).loan_amount.sum()
    by_interest_rate = df.groupby(['region', 'addr_state'],
                                  as_index=False).interest_rate.mean()
    return by_interest_rate


# Fragment 3
def f3(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
    statePop = {
        'CA': 39144818,
        'TX': 27469144,
        'FL': 20271878,
        'NY': 19795791,
        'IL': 12859995,
        'PA': 12802503,
        'OH': 11613423,
        'GA': 10214860,
        'NC': 10042802,
        'MI': 9922576,
        'NJ': 8958013,
        'VA': 8382993,
        'WA': 7170351,
        'AZ': 6828065,
        'MA': 6794422,
        'IN': 6619680,
        'TN': 6600299,
        'MO': 6083672,
        'MD': 6006401,
        'WI': 5771337,
        'MN': 5489594,
        'CO': 5456574,
        'SC': 4896146,
        'AL': 4858979,
        'LA': 4670724,
        'KY': 4425092,
        'OR': 4028977,
        'OK': 3911338,
        'CT': 3890886,
        'IA': 3123899,
        'UT': 2995919,
        'MS': 2992333,
        'AK': 2978204,
        'KS': 2911641,
        'NV': 2890845,
        'NM': 2085109,
        'NE': 1896190,
        'WV': 1844128,
        'ID': 1654930,
        'HI': 1431603,
        'NH': 1330608,
        'ME': 1329328,
        'RI': 1053298,
        'MT': 1032949,
        'DE': 945934,
        'SD': 858469,
        'ND': 756927,
        'AK': 738432,
        'DC': 672228,
        'VT': 626042,
        'WY': 586107
    }
    statePopdf = pd.DataFrame.from_dict(statePop, orient='index').reset_index()
    return statePopdf


# Fragment 4
def f4(df):
    # core cleaning code
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # df = pd.read_csv('../input/loan.csv')
    df = df[((df.loan_status == 'Fully Paid') |
             (df.loan_status == 'Charged Off'))]
    df = df[(df['pymnt_plan'] == 'n')]
    df = df[(df['application_type'] == 'INDIVIDUAL')]
    df1 = df.drop(
        columns=[
            'policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv',
            'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url',
            'application_type', 'grade', 'annual_inc_joint', 'dti_joint'
        ]
    )
    df1 = df1.drop(
        columns=[
            'verification_status_joint', 'open_acc_6m', 'open_il_6m',
            'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il',
            'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
            'inq_fi', 'total_cu_tl', 'inq_last_12m'
        ]
    )
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


# Task 3
# Compute the issue month and year associated with each loan
# Transfer fragments (treatment)
# Fragment 0
def f0(df_loan):
    # core cleaning code
    import pandas as pd
    # df_loan = pd.read_csv('../input/loan.csv', low_memory=False)
    (df_loan['issue_month'],
     df_loan['issue_year']) = df_loan['issue_d'].str.split('-', 1).str
    return df_loan


# Fragment 1
def f1(data):
    # core cleaning code
    import pandas as pd
    # data = pd.read_csv('../input/loan.csv')
    data['issue_dt'] = pd.to_datetime(data.issue_d)
    return data


# Fragment 2
def f2(df_loan):
    # core cleaning code
    import pandas as pd
    # df_loan = pd.read_csv('../input/loan.csv', low_memory=False)
    df_loan['issue_d'] = pd.to_datetime(df_loan['issue_d'])
    return df_loan


# Fragment 3
def f3(data):
    # core cleaning code
    import pandas as pd
    # data = pd.read_csv('../input/loan.csv', low_memory=False)
    data.earliest_cr_line = pd.to_datetime(data.earliest_cr_line)
    return data


# Fragment 4
def f4(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv', low_memory=False)
    df = df.rename(
        columns={
            'loan_amnt': 'loan_amount',
            'funded_amnt': 'funded_amount',
            'funded_amnt_inv': 'investor_funds',
            'int_rate': 'interest_rate',
            'annual_inc': 'annual_income'
        }
    )
    df['complete_date'] = pd.to_datetime(df['issue_d'])
    return df


# Random fragments (control)
# Fragment 0
def f0(dataset):
    # core cleaning code
    import pandas as pd
    # dataset = pd.read_csv('../input/loan.csv', low_memory=False)
    dataset = dataset.fillna(0)
    dataset['term'] = dataset['term'].astype('category').cat.codes
    return dataset


# Fragment 1
def f1(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv')
    df = df[((df.loan_status == 'Fully Paid') |
             (df.loan_status == 'Charged Off'))]
    df = df[(df['pymnt_plan'] == 'n')]
    df = df[(df['application_type'] == 'INDIVIDUAL')]
    df1 = df.drop(
        columns=[
            'policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv',
            'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url',
            'application_type', 'grade', 'annual_inc_joint', 'dti_joint'
        ]
    )
    emp_lengths = []
    df1.emp_length = emp_lengths
    return df1


# Fragment 2
def f2(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv', low_memory=False)
    df = df.rename(
        columns={
            'loan_amnt': 'loan_amount',
            'funded_amnt': 'funded_amount',
            'funded_amnt_inv': 'investor_funds',
            'int_rate': 'interest_rate',
            'annual_inc': 'annual_income'
        }
    )
    group_dates = df.groupby(['complete_date', 'region'], as_index=False).sum()
    group_dates = group_dates.groupby(['issue_d', 'region'],
                                      as_index=False).sum()
    return group_dates


# Fragment 3
def f3(dataset):
    # core cleaning code
    import numpy as np
    import pandas as pd
    # dataset = pd.read_csv('../input/loan.csv', low_memory=False)
    dataset = dataset.fillna(0)
    dataset['loan_status'] = dataset['loan_status'].astype(
        'category'
    ).cat.codes
    non_numerics = [
        x for x in dataset.columns if (
            not ((dataset[x].dtype == np.float64) or
                 (dataset[x].dtype == np.int8) or
                 (dataset[x].dtype == np.int64))
        )
    ]
    df = dataset
    return df


# Fragment 4
def f4(data):
    # core cleaning code
    import numpy as np
    import pandas as pd
    # data = pd.read_csv('../input/loan.csv')
    data_1 = pd.DataFrame(data)
    category_one_data = data_1[(data_1.loan_status == 'Fully Paid')]
    category_two_data = data_1[(data_1.loan_status == 'Charged Off')]
    new_data = np.vstack((category_one_data, category_two_data))
    new_data = new_data[(slice(None, None, None), slice(2, (-30), None))]
    new_data_df = pd.DataFrame(new_data)
    title = new_data_df[19]
    title = pd.DataFrame(title)
    title.columns = ['Title']
    return title


# Task 4
# Fill in missing values in the months since last delinquency column (`mths_since_last_delinq`)
# Transfer fragments (treatment)
# Fragment 0
def f0(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv', low_memory=False)
    df.mths_since_last_delinq = df.mths_since_last_delinq.fillna(
        df.mths_since_last_delinq.median()
    )
    return df


# Fragment 1
def f1(loan):
    # core cleaning code
    import pandas as pd
    # loan = pd.read_csv('../input/loan.csv', low_memory=False)
    loan['annual_inc'] = loan['annual_inc'].fillna(loan['annual_inc'].median())
    return loan


# Fragment 2
def f2(loan):
    # core cleaning code
    import pandas as pd
    # loan = pd.read_csv('../input/loan.csv', low_memory=False)
    loan['total_rev_hi_lim'] = loan['total_rev_hi_lim'].fillna(
        loan['total_rev_hi_lim'].median()
    )
    return loan


# Fragment 3
def f3(loan):
    # core cleaning code
    import pandas as pd
    # loan = pd.read_csv('../input/loan.csv', low_memory=False)
    loan['tot_coll_amt'] = loan['tot_coll_amt'].fillna(
        loan['tot_coll_amt'].median()
    )
    return loan


# Fragment 4
def f4(loan):
    # core cleaning code
    import numpy as np
    import pandas as pd
    # loan = pd.read_csv('../input/loan.csv', low_memory=False)
    loan['mths_since_last_delinq'] = np.where(
        loan['mths_since_last_delinq'].isnull(), 188,
        loan['mths_since_last_delinq']
    )
    return loan


# Random fragments (control)
# Fragment 0
def f0():
    # core cleaning code
    import pandas as pd
    statePop = {
        'CA': 39144818,
        'TX': 27469144,
        'FL': 20271878,
        'NY': 19795791,
        'IL': 12859995,
        'PA': 12802503,
        'OH': 11613423,
        'GA': 10214860,
        'NC': 10042802,
        'MI': 9922576,
        'NJ': 8958013,
        'VA': 8382993,
        'WA': 7170351,
        'AZ': 6828065,
        'MA': 6794422,
        'IN': 6619680,
        'TN': 6600299,
        'MO': 6083672,
        'MD': 6006401,
        'WI': 5771337,
        'MN': 5489594,
        'CO': 5456574,
        'SC': 4896146,
        'AL': 4858979,
        'LA': 4670724,
        'KY': 4425092,
        'OR': 4028977,
        'OK': 3911338,
        'CT': 3890886,
        'IA': 3123899,
        'UT': 2995919,
        'MS': 2992333,
        'AK': 2978204,
        'KS': 2911641,
        'NV': 2890845,
        'NM': 2085109,
        'NE': 1896190,
        'WV': 1844128,
        'ID': 1654930,
        'HI': 1431603,
        'NH': 1330608,
        'ME': 1329328,
        'RI': 1053298,
        'MT': 1032949,
        'DE': 945934,
        'SD': 858469,
        'ND': 756927,
        'AK': 738432,
        'DC': 672228,
        'VT': 626042,
        'WY': 586107
    }
    statePopdf = pd.DataFrame.from_dict(statePop, orient='index').reset_index()
    statePopdf.columns = ['State', 'Pop']
    return statePopdf


# Fragment 1
def f1(data):
    # core cleaning code
    import numpy as np
    import pandas as pd
    # data = pd.read_csv('../input/loan.csv', parse_dates=True)
    data = data[(data.loan_status != 'Fully Paid')]
    data = data[(
        data.loan_status !=
        'Does not meet the credit policy. Status:Fully Paid'
    )]
    data['recovery'] = np.where((data.recoveries != 0.0), 1, 0)
    return data


# Fragment 2
def f2(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv')
    new_df = df[(df['addr_state'] == x)]
    new_df['weighted'] = ((new_df['int_rate'] / 100) * new_df['funded_amnt'])
    return new_df


# Fragment 3
def f3(data):
    # core cleaning code
    import numpy as np
    import pandas as pd
    # data = pd.read_csv('../input/loan.csv', parse_dates=True)
    data = data[(data.loan_status != 'Fully Paid')]
    data = data[(
        data.loan_status !=
        'Does not meet the credit policy. Status:Fully Paid'
    )]
    data[e] = data[e].replace(np.nan, 0)
    data[e] = data[e]
    data[e].replace = data[e].replace
    np.nan = np.nan
    data[e] = data[e].replace(np.nan, 0)
    data.loc[(data.mths_since_last_delinq.notnull(), 'delinq')] = 1
    data.loc[(data.mths_since_last_delinq.isnull(), 'delinq')] = 0
    return data


# Fragment 4
def f4(loan):
    # core cleaning code
    import numpy as np
    import pandas as pd
    # loan = pd.read_csv('../input/loan.csv', low_memory=False)
    loan['total_acc'] = np.where(
        loan['total_acc'].isnull(), 0, loan['total_acc']
    )
    return loan


# Task 5
# Drop columns with many missing values
# Transfer fragments (treatment)
# Fragment 0
def f0(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv')
    df = df.drop([
        'id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade',
        'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv',
        'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
        'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title',
        'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type',
        'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'
    ],
                 axis=1)
    df = df.dropna(thresh=(len(df) / 2), axis=1)
    return df


# Fragment 1
def f1(df):
    # additional context code from user definitions

    def status_binary(text):
        if (text == 'Fully Paid'):
            return 0
        elif ((text == 'Current') or (text == 'Issued')):
            return (-1)
        else:
            return 1

    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv')
    df = df.drop([
        'id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade',
        'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv',
        'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
        'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title',
        'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type',
        'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'
    ],
                 axis=1)
    df = df.dropna(thresh=(len(df) / 2), axis=1)
    df = df.dropna()
    df['loan_status'] = df['loan_status'].apply(status_binary)
    return df


# Fragment 2
def f2(df):
    # additional context code from user definitions

    def status_binary(text):
        if (text == 'Fully Paid'):
            return 0
        elif ((text == 'Current') or (text == 'Issued')):
            return (-1)
        else:
            return 1

    # core cleaning code
    import pandas as pd
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv')
    df = df.drop([
        'id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade',
        'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv',
        'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
        'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title',
        'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type',
        'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'
    ],
                 axis=1)
    df = df.dropna(thresh=(len(df) / 2), axis=1)
    df = df.dropna()
    df['loan_status'] = df['loan_status'].apply(status_binary)
    df = df[(df['loan_status'] != (-1))]
    dummy_df = pd.get_dummies(
        df[['home_ownership', 'verification_status', 'purpose', 'term']]
    )
    return dummy_df


# Fragment 3
def f3(df):
    # additional context code from user definitions

    def status_binary(text):
        if (text == 'Fully Paid'):
            return 0
        elif ((text == 'Current') or (text == 'Issued')):
            return (-1)
        else:
            return 1

    # core cleaning code
    import pandas as pd
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv')
    df = df.drop([
        'id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade',
        'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv',
        'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
        'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title',
        'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type',
        'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'
    ],
                 axis=1)
    df = df.dropna(thresh=(len(df) / 2), axis=1)
    df = df.dropna()
    df['loan_status'] = df['loan_status'].apply(status_binary)
    df = df[(df['loan_status'] != (-1))]
    dummy_df = pd.get_dummies(
        df[['home_ownership', 'verification_status', 'purpose', 'term']]
    )
    df = df.drop(['home_ownership', 'verification_status', 'purpose', 'term'],
                 axis=1)
    df = pd.concat([df, dummy_df], axis=1)
    return df


# Fragment 4
def f4(df):
    # additional context code from user definitions

    def status_binary(text):
        if (text == 'Fully Paid'):
            return 0
        elif ((text == 'Current') or (text == 'Issued')):
            return (-1)
        else:
            return 1

    # core cleaning code
    import pandas as pd
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv')
    df = df.drop([
        'id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'sub_grade',
        'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv',
        'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
        'last_pymnt_d', 'last_pymnt_amnt', 'desc', 'url', 'title',
        'initial_list_status', 'pymnt_plan', 'policy_code', 'application_type',
        'earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'addr_state'
    ],
                 axis=1)
    df = df.dropna(thresh=(len(df) / 2), axis=1)
    df = df.dropna()
    df['loan_status'] = df['loan_status'].apply(status_binary)
    df = df[(df['loan_status'] != (-1))]
    dummy_df = pd.get_dummies(
        df[['home_ownership', 'verification_status', 'purpose', 'term']]
    )
    df = df.drop(['home_ownership', 'verification_status', 'purpose', 'term'],
                 axis=1)
    df = pd.concat([df, dummy_df], axis=1)
    mapping_dict = {
        'emp_length': {
            '10+ years': 10,
            '9 years': 9,
            '8 years': 8,
            '7 years': 7,
            '6 years': 6,
            '5 years': 5,
            '4 years': 4,
            '3 years': 3,
            '2 years': 2,
            '1 year': 1,
            '< 1 year': 0,
            'n/a': 0
        },
        'grade': {
            'A': 1,
            'B': 2,
            'C': 3,
            'D': 4,
            'E': 5,
            'F': 6,
            'G': 7
        }
    }
    df = df.replace(mapping_dict)
    return df


# Random fragments (control)
# Fragment 0
def f0(df):
    # core cleaning code
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # df = pd.read_csv('../input/loan.csv')
    df = df[((df.loan_status == 'Fully Paid') |
             (df.loan_status == 'Charged Off'))]
    df = df[(df['pymnt_plan'] == 'n')]
    df = df[(df['application_type'] == 'INDIVIDUAL')]
    df1 = df.drop(
        columns=[
            'policy_code', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv',
            'pymnt_plan', 'initial_list_status', 'member_id', 'id', 'url',
            'application_type', 'grade', 'annual_inc_joint', 'dti_joint'
        ]
    )
    df1 = df1.drop(
        columns=[
            'verification_status_joint', 'open_acc_6m', 'open_il_6m',
            'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il',
            'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
            'inq_fi', 'total_cu_tl', 'inq_last_12m'
        ]
    )
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


# Fragment 1
def f1(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv', usecols=['loan_amnt', 'addr_state'])
    perStatedf = df.groupby(
        'addr_state', as_index=False
    ).count().sort_values(
        by='loan_amnt', ascending=False
    )
    statePop = {
        'CA': 39144818,
        'TX': 27469144,
        'FL': 20271878,
        'NY': 19795791,
        'IL': 12859995,
        'PA': 12802503,
        'OH': 11613423,
        'GA': 10214860,
        'NC': 10042802,
        'MI': 9922576,
        'NJ': 8958013,
        'VA': 8382993,
        'WA': 7170351,
        'AZ': 6828065,
        'MA': 6794422,
        'IN': 6619680,
        'TN': 6600299,
        'MO': 6083672,
        'MD': 6006401,
        'WI': 5771337,
        'MN': 5489594,
        'CO': 5456574,
        'SC': 4896146,
        'AL': 4858979,
        'LA': 4670724,
        'KY': 4425092,
        'OR': 4028977,
        'OK': 3911338,
        'CT': 3890886,
        'IA': 3123899,
        'UT': 2995919,
        'MS': 2992333,
        'AK': 2978204,
        'KS': 2911641,
        'NV': 2890845,
        'NM': 2085109,
        'NE': 1896190,
        'WV': 1844128,
        'ID': 1654930,
        'HI': 1431603,
        'NH': 1330608,
        'ME': 1329328,
        'RI': 1053298,
        'MT': 1032949,
        'DE': 945934,
        'SD': 858469,
        'ND': 756927,
        'AK': 738432,
        'DC': 672228,
        'VT': 626042,
        'WY': 586107
    }
    return pd.DataFrame.from_dict(statePop, orient='index')


# Fragment 2
def f2():
    # core cleaning code
    import pandas as pd
    statePop = {
        'CA': 39144818,
        'TX': 27469144,
        'FL': 20271878,
        'NY': 19795791,
        'IL': 12859995,
        'PA': 12802503,
        'OH': 11613423,
        'GA': 10214860,
        'NC': 10042802,
        'MI': 9922576,
        'NJ': 8958013,
        'VA': 8382993,
        'WA': 7170351,
        'AZ': 6828065,
        'MA': 6794422,
        'IN': 6619680,
        'TN': 6600299,
        'MO': 6083672,
        'MD': 6006401,
        'WI': 5771337,
        'MN': 5489594,
        'CO': 5456574,
        'SC': 4896146,
        'AL': 4858979,
        'LA': 4670724,
        'KY': 4425092,
        'OR': 4028977,
        'OK': 3911338,
        'CT': 3890886,
        'IA': 3123899,
        'UT': 2995919,
        'MS': 2992333,
        'AK': 2978204,
        'KS': 2911641,
        'NV': 2890845,
        'NM': 2085109,
        'NE': 1896190,
        'WV': 1844128,
        'ID': 1654930,
        'HI': 1431603,
        'NH': 1330608,
        'ME': 1329328,
        'RI': 1053298,
        'MT': 1032949,
        'DE': 945934,
        'SD': 858469,
        'ND': 756927,
        'AK': 738432,
        'DC': 672228,
        'VT': 626042,
        'WY': 586107
    }
    return pd.DataFrame.from_dict(statePop, orient='index')


# Fragment 3
def f3(data):
    # core cleaning code
    import pandas as pd
    # data = pd.read_csv('../input/loan.csv', low_memory=False)
    data.issue_d = pd.Series(data.issue_d).str.replace('-2015', '')
    return data


# Fragment 4
def f4(data):
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
