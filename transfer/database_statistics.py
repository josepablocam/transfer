from argparse import ArgumentParser
import pickle

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from .build_db import FunctionDatabase, NodeTypes, RelationshipTypes
from .utils import print_df, plot_df_table


def get_node_label(node):
    return list(node.labels())[0]

def get_dist_counts(df, _groupby, sorted=True):
    dist = df.groupby(_groupby).size().to_frame('ct').reset_index()
    if sorted:
        dist = dist.sort_values('ct')
    return dist

def plot_relationship_counts(df_rel_end_freq):
    plot_df_rel_end_freq = df_rel_end_freq.plot(kind='bar', x='end_node', y='ct')
    plot_df_rel_end_freq.set_xlabel('End Node')
    plot_df_rel_end_freq.set_ylabel('Count of Extracted Functions')
    return plot_df_rel_end_freq

def compute_basic_db_distributions(db, top_n=None):
    db.startup()
    plots = []

    funs = db.extracted_functions()
    cols = db.columns()
    third_party_funs = db.functions()
    nodes = [funs, cols, third_party_funs]
    # construct a table of node counts
    node_types = [get_node_label(grp[0]) for grp in nodes]
    node_cts = [len(grp) for grp in nodes]
    node_ct_df = pd.DataFrame(list(zip(node_types, node_cts)), columns=['node_type', 'ct'])
    print_df(node_ct_df)

    table_plot = plot_df_table(node_ct_df)
    table_plot.set_title('Node Type Distribution')
    plots.append(table_plot)

    rel_data = []
    for f in funs:
        rels = db.get_extracted_function_relationships_from_node(f)
        rel_data.extend(rels)

    rel_df = pd.DataFrame(rel_data, columns=['relationship_type', 'end_node'])
    rel_df['relationship_type_str'] = rel_df['relationship_type'].map(lambda x: x.name)
    freq_rel_type = get_dist_counts(rel_df, 'relationship_type_str')
    print_df(freq_rel_type)
    plot_freq_rel_type = freq_rel_type.plot(kind='bar', x='relationship_type_str', y='ct')
    plot_freq_rel_type.set_xlabel('Relationship Type')
    plot_freq_rel_type.set_ylabel('Count of Relationships')
    plot_freq_rel_type.set_title('Relationships Distribution')
    plt.tight_layout()
    plots.append(plot_freq_rel_type)

    # for each relationship, provide a distibution of end node counts
    rel_types = rel_df['relationship_type'].unique()
    for rt in rel_types:
        print("Distribution for {}".format(rt))
        subset_rel_df = rel_df[rel_df['relationship_type'] == rt]
        df_rel_end_freq = get_dist_counts(subset_rel_df, 'end_node')
        print_df(df_rel_end_freq)
        # all counts
        plot_df_rel_end_freq = plot_relationship_counts(df_rel_end_freq)
        plot_df_rel_end_freq.set_title('Distribution for {}'.format(rt))
        plots.append(plot_df_rel_end_freq)
        # top N counts
        if top_n:
            top_n_df = df_rel_end_freq.iloc[-10:]
            plot_df_rel_end_freq_top_n = plot_relationship_counts(top_n_df)
            plot_df_rel_end_freq_top_n.set_title('Top {} Distribution for {}'.format(top_n, rt))
            plt.tight_layout()
            plots.append(plot_df_rel_end_freq_top_n)

    db.shutdown()
    return rel_df, plots

def main(args):
    with open(args.database_file, 'rb') as f:
        db = pickle.load(f)
    df, plots = compute_basic_db_distributions(db, top_n=args.top)

    if args.block:
        plt.show(block=True)

    if args.output_csv_file:
        df.to_csv(args.output_csv_file, index=False)

    if args.output_plot_file:
        with PdfPages(args.output_plot_file) as pdf:
            for p in plots:
                fig = p.get_figure()
                pdf.savefig(fig)

if __name__ == '__main__':
    parser = ArgumentParser('Produce basic statistics for database built')
    parser.add_argument('database_file', type=str, help='Pickled database interface file')
    parser.add_argument('-t', '--top', type=int, help='Plot top n categories for relationship plots')
    parser.add_argument('-b', '--block', action='store_true', help='Block and display plots')
    parser.add_argument('-c', '--output_csv_file', type=str, help='Path for csv output of basic counts dataframe')
    parser.add_argument('-p', '--output_plot_file', type=str, help='Path for pdf output of plots')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
