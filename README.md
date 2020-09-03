# Transfer

`transfer` is a tool to extract data-wrangling
code snippets from collections of programs as a way
of producing executable "documentation" of a database.
The core idea is: given a set of programs that
work with a given dataset, we can learn information
about that dataset by inspecting certain "types"
of code fragments in those programs. For example,
we can learn type casts (which provide semantic info
about a column), we can learn human-readable labels
for numeric values, we can learn common groupings of
values, etc.

`transfer` collects such snippets and organizes them
into a graph database, where nodes are database columns, library functions, or extracted code snippets, and edges are relationships such as defines, uses, calls, or wrangles for (more detail below).

# Installation
The easiest way to install `transfer` is through a docker
container.

First, build the container

`docker build . -t transfer`

Next, launch the container and build the demo
database.

`docker run -it transfer`

You should see a `neo4j` message indicating startup.
You can now build the demo database by calling

`bash build_demo.sh`

You should see prints to stdout indicating what is being
populated into the database.

You can then interact with `transfer` by launching
`python` or `ipython` and running

```
from demo import *
db = start()
```

which will create a database object by the name `db`.

In the following section, we show how to use this object to
interact with `transfer`.

# Interaction with `transfer`

We use the `db` object created in the prior section and show some simple usage.

* List all columns (from the dataset) stored in the `transfer` graph.

```
$ db.columns()
[(d5d4d7f:COLUMN {name:"earliest_cr_line"}), (d310d83:COLUMN {name:"term"}), ...]
```

* List third-party library functions called

```
$ db.functions()
[(f1fa693:FUNCTION {name:"seaborn.axisgrid.FacetGrid.map"}),
 (a3f2b2c:FUNCTION {name:"pandas.core.series.Series.to_frame"}), ...]
```

* List `transfer` code snippets (wrapped in functions)

```
$ db.extracted_functions()
[(ffa9705:EXTRACTED_FUNCTION {lines_of_code:7,name:"cleaning_func_0_0"}),
 (c973819:EXTRACTED_FUNCTION {lines_of_code:9,name:"cleaning_func_1_1"}), ...]
```

* List `transfer` code snippets that define a column

```
$ db.defines('loan_status')
[(ef26c4c:EXTRACTED_FUNCTION {lines_of_code:7,name:"cleaning_func_2_104"}),
 (faaa7fa:EXTRACTED_FUNCTION {lines_of_code:7,name:"cleaning_func_0_96"}), ...]
```

and then we can get the source code for one of these snippets

```
$ print(db.get_code(db.defines('loan_status')[0]))
def cleaning_func_2(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv')
	bad_indicators = ['Charged Off ', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Default Receiver', 'Late (16-30 days)', 'Late (31-120 days)']
	data.loc[(data.loan_status.isin(bad_indicators), 'bad_loan')] = 1
	return data
```

* List a `transfer` code snippets that use a column (and print its code)

```
$ print(db.get_code(db.uses('emp_length')[5]))
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
```

* List code snippets that are used *before* calling a third party function (i.e. they wrangle data for that call).

```
$ import pandas as pd
$ db.wrangles_for(pd.DataFrame.groupby)
print(db.get_code(db.wrangles_for(pd.DataFrame.groupby)[1]))
def cleaning_func_22(df):
	# core cleaning code
	import pandas as pd
	# df = pd.read_csv('../input/loan.csv', low_memory=False)
	df = df.rename(columns={'loan_amnt': 'loan_amount', 'funded_amnt': 'funded_amount', 'funded_amnt_inv': 'investor_funds', 'int_rate': 'interest_rate', 'annual_inc': 'annual_income'})
	group_dates = df.groupby(['complete_date', 'region'], as_index=False).sum()
	return group_dates
```
Note that we passed in the python function object `pd.DataFrame.groupby` to the `wrangles_for` call.

* List `transfer` code snippets that make a call to a particular third-party library function.

```
$ print(db.get_code(db.calls(pd.DataFrame.fillna)[0]))
def cleaning_func_12(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/loan.csv', low_memory=False)
	dataset = dataset.fillna(0)
	dataset['grade'] = dataset['grade'].astype('category').cat.codes
	return dataset
```

Note that like in `wrangles_for` we pass in the actual python function object to `db.calls`.

* Executable code snippets

A goal of `transfer` is that the code snippets produced are executable. So we can try that out as follows:


```
$ fn = db.get_executable(db.uses("emp_length")[5])
$ import pandas as pd
$ df = pd.read_csv("demo-data/loan.csv", nrows=1000)
$ df["emp_length"].value_counts()
10+ years    234
2 years      110
5 years       97
3 years       91
4 years       90
1 year        82
< 1 year      81
6 years       63
7 years       58
8 years       44
9 years       33

$ fn(df.copy())["emp_length_class"].value_counts()
<=3 years     364
4-6 years     250
>=10 years    234
7-9 years     135
```

Note that not *every* function is going to execute successfully, but it is certainly
our goal to have that be the case. Additionally, note that the function executed
has intermediate steps/variables created by `transfer` during analysis/extraction,
while the source code printed through `db.get_code` does some string replacement
to remove these (since they reduce readability).
