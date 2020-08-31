from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import pickle

from transfer.lift_donations import DonatedFunction
from transfer.utils import print_df, plot_df_table


def get_columns_df(results):
    data = []
    for entry in results:
        info = dict(
            input_cols=entry['cols_used'],
            output_cols=entry['cols_defined'],
            input_types=[
                col_info['type']
                for col_info in entry['cols_used_info'].values()
            ],
            output_types=[
                col_info['type']
                for col_info in entry['cols_defined_info'].values()
            ],
        )
        data.append(info)

    df = pd.DataFrame(data)
    type_to_str = lambda x: ['{}'.format(e) for e in x]
    df['input_types'] = df['input_types'].map(type_to_str)
    df['output_types'] = df['output_types'].map(type_to_str)
    for col in df.columns:
        df[col] = df[col].map(lambda x: tuple(sorted(x)))
    return df


def cleanup_object_cols_for_plot(df):
    df = df.copy()
    obj_cols = df.columns[df.dtypes == np.dtype('object')]
    for col in obj_cols:
        df[col] = df[col].map(lambda x: '{}'.format(x))
    return df


# compare re-definitions vs new derived columns
def summarize_definition_categories(columns_df):
    df = columns_df.copy()
    df['initialization'] = df['input_cols'].map(len) == 0
    df['redefinition'] = [
        not set(i).isdisjoint(o)
        for i, o in zip(df['input_cols'], df['output_cols'])
    ]
    df['derivation'] = [
        not set(i).issuperset(o)
        for i, o in zip(df['input_cols'], df['output_cols'])
    ]
    df['derivation'] = df['derivation'] & ~df['initialization']
    output = {}

    # summary counts
    count_info = dict(
        initializations=df['initialization'].sum(),
        redefinitions=df['redefinition'].sum(),
        derivations=df['derivation'].sum()
    )
    count_df = pd.DataFrame([count_info])
    output['Column Assignment Counts'] = count_df

    # column initializations
    inits = df[df['initialization']]
    init_cols = []
    for _, row in inits.iterrows():
        init_cols.extend(row.output_cols)
    inits_summary = pd.Series(init_cols).value_counts().to_frame(
        name='count'
    ).reset_index()
    inits_summary = inits_summary.rename(columns={'index': 'column'})
    output['Column Initializations'] = inits_summary

    # column redefinitions
    redefs = df[df['redefinition']]
    redefined_cols = []
    for _, row in redefs.iterrows():
        redefined_cols.extend(
            set(row.output_cols).intersection(row.input_cols)
        )
    redefined_summary = pd.Series(redefined_cols).value_counts().to_frame(
        name='count'
    ).reset_index()
    redefined_summary = redefined_summary.rename(columns={'index': 'column'})
    output['Column Redefinitions'] = redefined_summary

    derivs = df[df['derivation']].copy()
    derivs['derived_cols'] = [
        set(o).difference(i)
        for i, o in zip(derivs['input_cols'], derivs['output_cols'])
    ]
    derivs['derived_cols'] = derivs['derived_cols'].map(
        lambda x: tuple(sorted(x))
    )
    derivs_summary = derivs.groupby(['input_cols', 'derived_cols']
                                    ).size().to_frame(name='ct').reset_index()
    output['Column Derivations'] = derivs_summary

    return output


def summarize_columns_type_conversions(columns_df):
    types_df = columns_df.groupby(['input_types', 'output_types']
                                  ).size().to_frame(name='count').reset_index()
    output = {}
    output['Type Conversions'] = types_df
    return output


def stat_distribution(results, stat, plot=False):
    data = {'cols_used_info': [], 'cols_defined_info': []}
    for entry in results:
        for group_key, group_vals in data.items():
            for vals in entry[group_key].values():
                mi = vals[stat]
                if not np.isnan(mi):
                    group_vals.append(mi)

    used_stat = np.array(data['cols_used_info'])
    defined_stat = np.array(data['cols_defined_info'])

    # plots
    if plot:
        fig, ax = plt.subplots(1)
        pd.Series(used_stat).plot(ax=ax, kind='kde', label='used')
        pd.Series(defined_stat).plot(ax=ax, kind='kde', label='defined')
        plt.legend(loc='best')
        ax.set_ylabel('Density')
        ax.set_xlabel(stat)
        plt.tight_layout()
        return used_stat, defined_stat, ax
    else:
        return used_stat, defined_stat


def mi_distribution(results):
    used, _def, plot = stat_distribution(results, 'mi', plot=True)
    plot.set_title('Distribution of MI')
    return used, _def, plot


def count_distribution(results):
    used, _def, plot = stat_distribution(results, 'ct_unique_vals', plot=True)
    plot.set_title('Distribution of Column Unique Value Counts')
    return used, _def, plot


def summarize(results):
    plots = []

    cols_df = get_columns_df(results)

    display('Column assignment categories')
    cat_outputs = summarize_definition_categories(cols_df)
    for key, val in cat_outputs.items():
        display(key)
        display(val)

    types_outputs = summarize_columns_type_conversions(cols_df)
    for key, val in types_outputs.items():
        display(key)
        display(val)

    display('Stat distributions')
    _, _, mi_plot = mi_distribution(results)
    _, _, ct_plot = count_distribution(results)
    display(mi_plot)
    display(ct_plot)


def main(input_path):
    with open(input_path, 'rb') as f:
        results = pickle.load(f)
    summarize(results)
