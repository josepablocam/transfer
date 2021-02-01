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
    df = pd.DataFrame(data, columns=["script", "time"])
    df["dataset"] = path.split("/")[1]
    # to match timeout naming convention
    df["script"] = df["script"].map(lambda x: x.replace("/", "_"))
    return df


def get_times_df(paths, rename_map=None):
    dfs = []
    for p in paths:
        dfs.append(get_times(p))
    df = pd.concat(dfs)
    if rename_map is not None:
        df["dataset"] = df["dataset"].map(lambda x: rename_map.get(x, x))
    return df


def compare_times(
    df_plain,
    df_full,
    max_plot=None,
    timedout=None,
    timeout=None,
):
    if timedout is not None:
        print(
            "Setting time to {} for {} timedout scripts".format(
                timeout, len(timedout)
            )
        )
        assert timeout > 0
        assert all(s not in df_full["script"].values for s in timedout)
        df_timedout = pd.DataFrame(
            [(s, timeout) for s in timedout],
            columns=["script", "time"],
        )
        df_full = pd.concat([df_full, df_timedout])

    df_full = pd.merge(
        df_full,
        df_plain,
        how="inner",
        on=["dataset", "script"],
        suffixes=("_full", "_plain")
    )

    # valid timing numbers only
    valid = (df_full["time_full"] > 0) & (df_full["time_plain"] > 0)
    print("Valid: {} / {}".format(valid.sum(), valid.shape[0]))
    df_full = df_full[valid]
    df_full["ratio"] = df_full["time_full"] / df_full["time_plain"]
    # if smaller, clamp to 1.0 can't actually be smaller, so noise
    df_full["ratio"] = df_full["ratio"].map(lambda x: 1.0 if x < 1.0 else x)

    summary_df = df_full.groupby("dataset")["ratio"].describe()
    print("Summary")
    print(summary_df)

    fig, ax = plt.subplots(1)
    # ax.set_aspect(1)

    unique_datasets = df_full["dataset"].unique().tolist()
    assert len(unique_datasets) == 3
    palette = dict(zip(unique_datasets, sns.color_palette()))

    sns.ecdfplot(data=df_full, x="ratio", hue="dataset")
    sns.scatterplot(
        data=df_full,
        x="ratio",
        y=[0] * df_full.shape[0],
        hue="dataset",
        palette=palette,
    )
    # mark one per
    for d in unique_datasets:
        median = df_full[df_full["dataset"] == d]["ratio"].median()
        median_label = "Median {}={:.2f}".format(d, median)
        ax.axvline(
            x=median,
            ymin=0.0,
            ymax=1.0,
            label=median_label,
            linestyle="dashed",
            color=palette[d],
        )

    ax.set_ylim(-0.01, 1.0)
    if max_plot is not None and df_full["ratio"].max() > max_plot:
        print("Clamping x-axis to {}".format(max_plot))
        over_max = df_full["ratio"][df_full["ratio"] > max_plot]
        print("Removes: {}".format(over_max.values.tolist()))
        ax.set_xlim(1.0, max_plot)

    ax.set_xlabel("Execution Time Ratio")
    ax.set_ylabel("Empirical Cumulative Distribution")
    plt.legend(loc="best", prop={"size": 12})
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
    parser.add_argument(
        "--timedout",
        type=str,
        nargs="+",
        help="Scripts that time out when instrumented (set time to max)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Seconds to use as time for timedout",
    )
    parser.add_argument(
        "--rename", type=str, nargs="+", help="List of renamings (orig:new)"
    )
    parser.add_argument("--output_dir", type=str, help="Output directory")
    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    rename_map = None
    if args.rename is not None:
        rename_map = dict([r.split(":") for r in args.rename])

    df_plain = get_times_df(args.plain, rename_map)
    df_full = get_times_df(args.full, rename_map)

    summary, ax = compare_times(
        df_plain,
        df_full,
        max_plot=args.max_plot,
        timedout=args.timedout,
        timeout=args.timeout,
    )

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
