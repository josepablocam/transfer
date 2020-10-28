import random
import re
import sys
sys.path.append("../../")

import pandas as pd
import numpy as np
from demo import *

df = pd.read_csv("../../demo-data/loan.csv", nrows=1000)

db = start("../../sample_db.pkl")


# just utility so we don't clobber original dataframe
def cp(d):
    return df.copy()


def code(db_node):
    return db.get_code(db_node)


def run(db_node):
    func = db.get_executable(db_node)
    cp_df = cp(df)
    return func(cp_df)


ALL_FUNCS = db.extracted_functions()
ALL_CODE_FRAGMENTS = [code(p) for p in ALL_FUNCS]


def survey_task(
    db, query, n, max_loc=None, random_state=None, rename_funcs=True
):
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    if query is None:
        # random querying -- effectively
        all_funcs = ALL_CODE_FRAGMENTS
        n = min(len(all_funcs), n)
        if max_loc is not None:
            all_funcs = [c for c in all_funcs if len(c.split("\n")) <= max_loc]
        query_results = np.random.choice(
            all_funcs,
            size=n,
            replace=False,
        )
    else:
        query_results = db.query(query)[:n]
    code_fragments = []
    for ix, prog in enumerate(query_results):
        if not isinstance(prog, str):
            prog = code(prog)
        if rename_funcs:
            prog = re.sub(r'cleaning_func_[0-9]+', 'f{}'.format(ix), prog)
        print("# Fragment {}".format(ix))
        print(prog)
        print("\n")
        code_fragments.append(prog)
    return code_fragments


class Task(object):
    def __init__(self, title, description, query):
        self.title = title
        self.description = description
        self.query = query

    def generate(self, db, n, random_state):
        print("# Task {}".format(self.title))
        print("# {}".format(self.description))
        print("# Transfer fragments (treatment)")
        survey_task(db, self.query, n, random_state=random_state)
        print("\n")
        print("# Random fragments (control)")
        survey_task(db, None, n, max_loc=15, random_state=random_state)


task1 = Task(
    title="1",
    description="Identify non-current loans based on loan_status",
    query=["loan_status"],
)

task2 = Task(
    title="2",
    description=
    "Round the interest rate column (`int_rate`) to nearest integer",
    query=["int_rate", pd.DataFrame.astype],
)

task3 = Task(
    title="3",
    description="Compute the issue month and year associated with each loan",
    query=["issue_month", pd.to_datetime],
)

task4 = Task(
    title="4",
    description=
    "Fill in missing values in the months since last delinquency column (`mths_since_last_delinq`)",
    query=["mths_since_last_delinq", pd.Series.fillna],
)

task5 = Task(
    title="5",
    description="Drop columns with many missing values",
    query=[pd.DataFrame.dropna],
)

tasks = [task1, task2, task3, task4, task5]


def main():
    seed = 42
    for ix, t in enumerate(tasks):
        t.generate(db, 5, seed + ix)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
