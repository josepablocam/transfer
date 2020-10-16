# Task 1: 
# Identify non-current loans based on loan_status
# db.query(["loan_status"])
def f0(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv', low_memory=False)
    df = df.loc[(df['loan_status'] != 'Current')]
    return df


def f1(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv')
    df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
    return df


def f2(data):
    # core cleaning code
    import pandas as pd
    # data = pd.read_csv('../input/loan.csv', parse_dates=True)
    data = data[(data.loan_status != 'Fully Paid')]
    return data


def f3(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv')
    df = df[((df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off'))]
    df = df[(df['pymnt_plan'] == 'n')]
    return df


def f4(df_loan):
    # core cleaning code
    import pandas as pd
    # df_loan = pd.read_csv('../input/loan.csv', low_memory=False)
    df_loan.loc[((df_loan.loan_status == 'Does not meet the credit policy. Status:Fully Paid'), 'loan_status')] = 'NMCP Fully Paid'
    df_loan.loc[((df_loan.loan_status == 'Does not meet the credit policy. Status:Charged Off'), 'loan_status')] = 'NMCP Charged Off'
    return df_loan



# Task 2: Round the interest rate column (`int_rate`) to nearest integer
# db.query(["int_rate", pd.DataFrame.astype])[:5]
def f0(df_loan):
    # core cleaning code
    import pandas as pd
    # df_loan = pd.read_csv('../input/loan.csv', low_memory=False)
    df_loan['int_round'] = df_loan['int_rate'].round(0).astype(int)
    return df_loan


def f1(data):
    # core cleaning code
    import pandas as pd
    # data = pd.read_csv('../input/loan.csv', low_memory=False)
    data['emp_length'] = data['emp_length'].astype(int)
    return data


def f2(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv', low_memory=False)
    df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 
                'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 
                'annual_inc': 'annual_income'})
    return df


def f3(dataset):
    # core cleaning code
    import pandas as pd
    # dataset = pd.read_csv('../input/loan.csv', low_memory=False)
    dataset = dataset.fillna(0)
    dataset['pymnt_plan'] = dataset['pymnt_plan'].astype('category').cat.codes
    return dataset


def f4(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv')
    new_df = df[(df['addr_state'] == x)]
    new_df['weighted'] = ((new_df['int_rate'] / 100) * new_df['funded_amnt'])
    return new_df


# Task 3:
# Compute the issue month and year associated with each loan
# db.query(["issue_month", pd.to_datetime])

def f0(df_loan):
    # core cleaning code
    import pandas as pd
    # df_loan = pd.read_csv('../input/loan.csv', low_memory=False)
    df_loan['issue_d'] = pd.to_datetime(df_loan['issue_d'])
    return df_loan


def f1(df_loan):
    # core cleaning code
    import pandas as pd
    # df_loan = pd.read_csv('../input/loan.csv', low_memory=False)
    (df_loan['issue_month'], df_loan['issue_year']) = df_loan['issue_d'].str.split('-', 1).str
    return df_loan


def f2(data):
    # core cleaning code
    import pandas as pd
    # data = pd.read_csv('../input/loan.csv', low_memory=False)
    data.earliest_cr_line = pd.to_datetime(data.earliest_cr_line)
    return data


def f3(data):
    # core cleaning code
    import pandas as pd
    # data = pd.read_csv('../input/loan.csv')
    data['issue_dt'] = pd.to_datetime(data.issue_d)
    return data


def f4(data):
    # core cleaning code
    import pandas as pd
    # data = pd.read_csv('../input/loan.csv')
    data['issue_dt'] = pd.to_datetime(data.issue_d)
    data['month'] = data['issue_dt'].dt.month
    return data


# Task 4
# Fill in missing values in the months since last delinquency column (`mths_since_last_delinq`)
# db.query(["mths_since_last_delinq", pd.Series.fillna])

def f0(df):
    # core cleaning code
    import pandas as pd
    # df = pd.read_csv('../input/loan.csv', low_memory=False)
    df.mths_since_last_delinq = df.mths_since_last_delinq.fillna(df.mths_since_last_delinq.median())
    return df


def f1(loan):
    # core cleaning code
    import pandas as pd
    # loan = pd.read_csv('../input/loan.csv', low_memory=False)
    loan['annual_inc'] = loan['annual_inc'].fillna(loan['annual_inc'].median())
    return loan


def f2(loan):
    # core cleaning code
    import pandas as pd
    # loan = pd.read_csv('../input/loan.csv', low_memory=False)
    loan['total_rev_hi_lim'] = loan['total_rev_hi_lim'].fillna(loan['total_rev_hi_lim'].median())
    return loan


def f3(loan):
    # core cleaning code
    import pandas as pd
    # loan = pd.read_csv('../input/loan.csv', low_memory=False)
    loan['tot_coll_amt'] = loan['tot_coll_amt'].fillna(loan['tot_coll_amt'].median())
    return loan


def f4(loan):
    # core cleaning code
    import numpy as np
    import pandas as pd
    # loan = pd.read_csv('../input/loan.csv', low_memory=False)
    loan['mths_since_last_delinq'] = np.where(
        loan['mths_since_last_delinq'].isnull(), 188, loan['mths_since_last_delinq']
    )
    return loan

# Task 5
# Drop columns with many missing values
#  db.query([pd.DataFrame.dropna])

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
            '10+ years': 10, '9 years': 9,
            '8 years': 8, '7 years': 7, '6 years': 6, '5 years': 5,
            '4 years': 4, '3 years': 3,'2 years': 2,
            '1 year': 1, '< 1 year': 0, 'n/a': 0
        },
        'grade': {'A': 1, 'B': 2, 'C': 3,'D': 4, 'E': 5, 'F': 6, 'G': 7
        }
    }
    df = df.replace(mapping_dict)
    return df
