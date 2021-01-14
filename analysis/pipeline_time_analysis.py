#!/usr/bin/env python3
from argparse import ArgumentParser
import os
import sys

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 14})
import numpy as np
import pandas as pd
import seaborn as sns


def get_times(path):
    results = []
    with open(path, "r") as fin:
        lines = [entry.split(":") for entry in fin.readlines()]

    data = [(e[0], float(e[1])) for e in lines]
    return pd.DataFrame(data, columns=["script", "time"])


def compare_times(plain_paths, full_paths):
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
        suffixes=("_full", "_plain"))

    # valid timing numbers only
    df_full = df_full[(df_full["time_full"] > 0) & (df_full["time_plain"] > 0)]
    df_full["ratio"] = df_full["time_full"] / df["time_plain"]
    # if smaller, clamp to 1.0 can't actually be smaller, so noise
    df_full["ratio"] = df_full["ratio"].map(lambda x: 1.0 if x < 1.0 else x)

    summary_df = df_full["ratio"].summarize()
    print("N={}".format(df_full.shape[0]))
    print("Summary")
    print(summary_df)

    fig, ax = plt.subplots(1)
    sns.displot(df_full, x="ratio", kind="kde", ax=ax)
    ax.set_xlabel("Execution Time Ratio")
    ax.set_ylabel("Density")
    return summary_df, ax


def get_args():
    parser = ArgumentParser(description="Compute time ratios")
    parser.add_argument(
        "--plain",
        type=str,
        nargs="+",
        help="List of time files for plain executions")
    parser.add_argument(
        "--full",
        type=str,
        nargs="+",
        help="List of time files for full (pipeline) executions")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    summary, ax = compare_times(args.plain, args.full)

    summary_path = os.path.join(args.output_dir, "time_summary.csv")
    summary.to_csv(summary_path)

    plot_path = os.path.join(args.output_dir, "time_kde.pdf")
    ax.get_figure().savefig(plot_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
        sys.exit(1)
