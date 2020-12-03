
# Extracted Summary
`Scripts` contains the count of Kernels that we can parse with Python 3
and that we have a chance of executing. `Successfully Executed`
shows the count of such scripts that we can execute and trace. Many
scripts fail to execute as the use functions that have been
deprecated in the version of libraries we are using (we use
only one set of libraries, rather than per-script tailored dependencies).
`Snippets Extracted` is the count of snippets that we extract and
store in our graph db for that dataset.


| Dataset             | Scripts | Successfully Executed | Snippets Extracted |
|---------------------|---------|-----------------------|--------------------|
| loan_data           | 273     | 58                    | 294                |
| house_sales         | 119     | 52                    | 59                 |
| university_rankings | 86      | 69                    | 180                |

We now show some interesting examples extracted.

## Examples from `loan_data`

```
In [4]: print(db.get_code(db.extracted_functions()[0]))
def cleaning_func_0(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', low_memory=False)
	data.earliest_cr_line = pd.to_datetime(data.earliest_cr_line)
	return data
```

```
In [5]: print(db.get_code(db.extracted_functions()[1]))
def cleaning_func_1(data):
	# core cleaning code
	import pandas as pd
	# data = pd.read_csv('../input/loan.csv', low_memory=False)
	data['term'] = data['term'].apply((lambda x: x.lstrip()))
	return data
```


## Examples from `house_sales`

```
In [4]: print(db.get_code(db.extracted_functions()[0]))
def cleaning_func_0(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/kc_house_data.csv')
	dataset['grade'] = dataset['grade'].astype('category', ordered=False)
	return dataset
```

```
In [6]: print(db.get_code(db.extracted_functions()[5]))
def cleaning_func_5(dataset):
	# core cleaning code
	import pandas as pd
	# dataset = pd.read_csv('../input/kc_house_data.csv')
	dataset = dataset.drop(['id', 'date'], axis=1)
	dataset['basement_is_present'] = dataset['sqft_basement'].apply((lambda x: (1 if (x > 0) else 0)))
	dataset['basement_is_present'] = dataset['basement_is_present'].astype('category', ordered=False)
	return dataset
```

## Examples from `university_rankings`

This dataset is interesting as it contains more than a single table which
can be used by the scripts.

```
In [4]: print(db.get_code(db.extracted_functions()[0]))
def cleaning_func_0(timesData):
	# core cleaning code
	my_university_name = ['University of Illinois at Urbana-Champaign']
	import pandas as pd
	# timesData = pd.read_csv('../input/timesData.csv')
	times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank', 'year']]
	times_plot_data['source'] = 'Times'
	return times_plot_data
```

```
In [8]: print(db.get_code(db.extracted_functions()[8]))
def cleaning_func_15(cwurData,shanghaiData,timesData):
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
```
