import os
import tabulate


def build_script_paths(script_path, output_dir=None):
    script_dir = os.path.dirname(script_path)
    if output_dir is None:
        output_dir = script_dir
    basename = '.'.join(os.path.basename(script_path).split('.')[:-1])
    paths = dict(
        script_path=script_path,
        lifted_path=os.path.join(script_dir, basename + '_lifted.py'),
        trace_path=os.path.join(output_dir, basename + '_tracer.pkl'),
        graph_path=os.path.join(output_dir, basename + '_graph.pkl'),
        donations_path=os.path.join(output_dir, basename + '_donations.pkl'),
        functions_path=os.path.join(output_dir, basename + '_functions.pkl'),
    )
    return paths


def print_df(df):
    print(
        tabulate.tabulate(
            df, headers='keys', tablefmt='grid', showindex=False
        )
    )


def plot_df_table(df):
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(1)
    ax.table(
        cellText=df.values.astype(str).tolist(),
        rowLabels=None,
        colLabels=df.columns.tolist(),
        loc='center'
    )
    plt.axis('off')
    return ax


def sort_by_values(orig_ls, vals, reverse=False):
    ls_and_vals = list(zip(orig_ls, vals))
    ls_and_vals = sorted(ls_and_vals, key=lambda x: x[1], reverse=reverse)
    return [e for e, _ in ls_and_vals]
