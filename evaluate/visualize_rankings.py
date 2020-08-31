from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import seaborn as sns

from transfer.utils import print_df, plot_df_table


def distribution_corrs(df):
    df_plot = df.groupby(['model', 'test_node_name'])['spearman_corr'].mean()
    df_plot = df_plot.reset_index()
    g = sns.FacetGrid(data=df_plot, row='model')
    g.map(sns.distplot, 'spearman_corr')
    g.set_xlabels('Spearman Rank Correlation')
    g.set_ylabels('Frequency')
    plt.tight_layout()
    return g


def summarize_corrs(df):
    df_corr = df.groupby('model')[['spearman_corr',
                                   'spearman_pval']].mean().reset_index()
    print_df(df_corr)
    return plot_df_table(df_corr)


def main(args):
    df = pd.read_pickle(args.input_path)
    plots = []

    p = summarize_corrs(df)
    plots.append(p)

    p = distribution_corrs(df)
    plots.append(p)

    if args.output_path:
        with PdfPages(args.output_path) as pdf:
            for p in plots:
                try:
                    fig = p.fig
                except AttributeError:
                    fig = p.get_figure()
                pdf.savefig(fig)


if __name__ == '__main__':
    parser = ArgumentParser(description='Plot ranking evaluation metrics')
    parser.add_argument(
        'input_path', type=str, help='Path to pickled evaluation dataframe'
    )
    parser.add_argument(
        'output_path', type=str, help='Path to save down pdf of results'
    )
    args = parser.parse_args()
    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
