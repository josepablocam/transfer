import pandas as pd
from demo import *

df = pd.read_csv("demo-data/loan.csv", nrows=1000)

db = start()


# just utility so we don't clobber original dataframe
def cp(d):
    return df.copy()


def code(db_node):
    print(db.get_code(db_node))


def run(db_node):
    func = db.get_executable(db_node)
    cp_df = cp(df)
    return func(cp_df)


# Concrete Task 1: Identify delinquent loan groups based on loan_status
# Goal: semantic partitioning of column values
code(db.uses("loan_status")[0])
run(db.uses("loan_status")[0]).isBad.value_counts()

# Task 2: Round interest rates to nearest integer and convert from float to int
# Goal: semantic type conversions
code(db.calls(pd.DataFrame.astype)[1])
# creates new column name
run(db.calls(pd.DataFrame.astype)[1]).int_round.dtype

# Task 3: Compute the issue month and year associated with each loan
# Goal: semantic type parsing
code(db.calls(pd.to_datetime)[6])
run(db.calls(pd.to_datetime)[6]).groupby("year").size()

# Task 4: Fill missing mths_since_last_delinq with the median value
# Goal: basic imputation, discover what simple impute values
# people use for that (i.e. mean, median etc)
code(db.calls(pd.Series.fillna)[0])
# Will want to replace the "annual_inc" in code with mths_since_last_delinq
# and run
run(db.calls(pd.Series.fillna)[0])

# Task 5: Drop columns with more than 50% missing values
# Goal: another form of basic "imputation" (i.e. removing missing values)
code(db.calls(df.dropna)[0])
run(db.calls(df.dropna)[0]).columns.shape
