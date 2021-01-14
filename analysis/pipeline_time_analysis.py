#!/usr/bin/env python3
from argparse import ArgumentParser
import os
import sys

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 14})
import pandas as pd
import seaborn as sns


def get_times(path):
    with open(path, "r") as fin:
        lines = [entry.split(":") for entry in fin.readlines()]

    data = [(e[0], float(e[1])) for e in lines]
    return pd.DataFrame(data, columns=["script", "time"])


def compare_times(plain_paths, full_paths, max_plot=None):
    plain_dfs = []
    for p in plain_paths:
        plain_dfs.append(get_times(p))
    df_plain = pd.concat(plain_dfs)

    full_dfs = []
    for p in full_paths:
        full_dfs.append(get_times(p))
    df_full = pd.concat(full_dfs)

    df_full = pd.merge(
        df_full,
        df_plain,
        how="inner",
        on="script",
        suffixes=("_full", "_plain")
    )

    # valid timing numbers only
    valid = (df_full["time_full"] > 0) & (df_full["time_plain"] > 0)
    print("Valid: {} / {}".format(valid.sum(), valid.shape[0]))
    df_full = df_full[valid]
    df_full["ratio"] = df_full["time_full"] / df_full["time_plain"]
    # if smaller, clamp to 1.0 can't actually be smaller, so noise
    df_full["ratio"] = df_full["ratio"].map(lambda x: 1.0 if x < 1.0 else x)

    summary_df = df_full["ratio"].describe()
    print("N={}".format(df_full.shape[0]))
    print("Summary")
    print(summary_df)

    fig, ax = plt.subplots(1)
    sns.ecdfplot(data=df_full, x="ratio", label="ECDF")
    sns.scatterplot(
        x=df_full["ratio"], y=[0] * df_full.shape[0], label="Observations"
    )
    median = df_full["ratio"].median()
    median_label = "Median={:.2f}".format(median)
    ax.axvline(
        x=median,
        ymin=0.0,
        ymax=1.0,
        label=median_label,
        linestyle="dashed",
    )
    ax.set_ylim(-0.01, 1.0)
    if max_plot is not None and df_full["ratio"].max() > max_plot:
        print("Clamping x-axis to {}".format(max_plot))
        over_max = df_full["ratio"][df_full["ratio"] > max_plot]
        print("Removes: {}".format(over_max.values.tolist()))
        ax.set_xlim(1.0, max_plot)

    ax.set_xlabel("Execution Time Ratio")
    ax.set_ylabel("Empirical Cumulative Distribution")
    plt.legend(loc="best")
    plt.tight_layout()
    return summary_df, ax


def get_args():
    parser = ArgumentParser(description="Compute time ratios")
    parser.add_argument(
        "--plain",
        type=str,
        nargs="+",
        help="List of time files for plain executions"
    )
    parser.add_argument(
        "--full",
        type=str,
        nargs="+",
        help="List of time files for full (pipeline) executions"
    )
    parser.add_argument(
        "--max_plot",
        type=float,
        help=
        "If not none, only show points below max for plotting (but report)",
    )
    parser.add_argument("--output_dir", type=str, help="Output directory")
    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    summary, ax = compare_times(args.plain, args.full, max_plot=args.max_plot)

    summary_path = os.path.join(args.output_dir, "time_summary.csv")
    summary.to_csv(summary_path)

    plot_path = os.path.join(args.output_dir, "time_ecdf.pdf")
    ax.get_figure().savefig(plot_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
