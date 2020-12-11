#!/usr/bin/env python3
from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy


def get_task_id(df):
    return df["question"].map(lambda x: int(x.split("_")[1].replace("t", "")))


def get_question_num(df):
    return df["question"].map(lambda x: int(x.split("_")[2].replace("q", "")))


def prepare_data(df):
    property_map = dict(df.iloc[0])
    responses_df = df.iloc[2:].copy()
    responses_df["survey_id"] = list(range(0, responses_df.shape[0]))
    responses_df["StartDate"] = pd.to_datetime(responses_df["StartDate"])
    responses_df["EndDate"] = pd.to_datetime(responses_df["EndDate"])
    wide_df = responses_df.copy()

    question_cols = [
        c for c in responses_df.columns
        if c.startswith(("intro_", "d1_t", "survey"))
    ]
    narrow_df = responses_df[question_cols].copy()
    narrow_df = pd.melt(
        narrow_df,
        id_vars="survey_id",
        var_name="question",
        value_name="value",
    )

    intro_df = narrow_df[narrow_df["question"].str.startswith("intro")].copy()
    relevance_df = narrow_df[narrow_df["question"].str.endswith("relevance")
                             ].copy()
    arms_df = narrow_df[narrow_df["question"].str.endswith(("_cont", "_treat")
                                                           )].copy()
    assert (arms_df.shape[0] + relevance_df.shape[0] +
            intro_df.shape[0]) == narrow_df.shape[0]

    relevance_df["task"] = get_task_id(relevance_df)
    arms_df["task"] = get_task_id(arms_df)
    arms_df["question_num"] = get_question_num(arms_df)
    arms_df["arm"] = arms_df["question"].map(lambda x: x.split("_")[-1]).map({
        "cont":
        "Control",
        "treat":
        "Treatment"
    })

    return property_map, wide_df, (intro_df, relevance_df, arms_df)


def get_likert_ordered():
    likert = [
        "Strongly agree",
        "Agree",
        "Somewhat agree",
        "Neither agree nor disagree",
        "Somewhat disagree",
        "Disagree",
        "Strongly disagree",
    ]
    return likert


def likert_table():
    likert = get_likert_ordered()
    return pd.Series(likert).to_frame(name="value")


def get_fragments_ordered(num_fragments=5):
    fragments = ["Fragment {}".format(i) for i in range(num_fragments)]
    return fragments


def fragments_table(num_fragments=5):
    fragments = get_fragments_ordered()
    return pd.Series(fragments).to_frame(name="value")


def create_full_table(df1, df2, df1_cols, fillna_value=0.0):
    product = df1[df1_cols].assign(key=1).merge(
        df2.assign(key=1), on="key"
    ).drop(
        "key", axis=1
    )
    product = pd.merge(
        product, df1, how="left", on=product.columns.values.tolist()
    )
    if fillna_value is not None:
        product = product.fillna(fillna_value)
    return product


def task_relevance(rel_df):
    cts_df = rel_df.groupby(["task", "value"]).size().to_frame(name="ct")
    cts_df = cts_df.reset_index()
    cts_df = create_full_table(cts_df, likert_table(), ["task"])
    g = sns.FacetGrid(data=cts_df, col="task")
    g.map(sns.barplot, "value", "ct", order=get_likert_ordered())
    g.set_xticklabels(rotation=90)
    g.set_xlabels("Likert Scale")
    g.set_ylabels("Count")
    g.add_legend()
    plt.tight_layout()
    g.tight_layout()
    return cts_df, g


def get_question(arms_df, name):
    name_to_id = {
        "number": 0,
        "rank": 1,
        "access": 2,
    }
    return arms_df[arms_df["question_num"] == name_to_id[name]].copy()


def create_paired_df(df, index=None, columns=None, values=None):
    if index is None:
        index = ["survey_id", "task", "question_num"]
    if columns is None:
        columns = ["arm"]
    if values is None:
        values = "value"
    pdf = pd.pivot_table(df, index=index, columns=columns, values=values)
    pdf = pdf.reset_index()
    return pdf


def stat_results_to_string(results):
    msg = ""
    for task, (res, is_significant) in results.items():
        msg += "Task {}\n".format(task)
        msg += "{}\n".format(res)
        msg += "significant={}\n\n".format(is_significant)
    return msg


def task_level_wilcoxon_test(qdf, alpha=0.05):
    paired = create_paired_df(qdf)
    results = {}
    tasks = qdf.task.unique()
    num_tasks = len(tasks)
    for task in tasks:
        paired_task = paired[paired["task"] == task]
        task_result = scipy.stats.wilcoxon(
            paired_task["Control"].values.astype(float),
            paired_task["Treatment"].values.astype(float)
        )
        is_significant = task_result.pvalue < (alpha / num_tasks)
        results[task] = (task_result, is_significant)
    stat_res = stat_results_to_string(results)
    return stat_res


def num_of_relevant_fragments(arms_df):
    qdf = get_question(arms_df, "number")
    # number
    qdf["value"] = qdf["value"].map(
        lambda x: 0 if pd.isnull(x) else len(x.split(","))
    )
    cts_df = qdf.groupby(["task", "arm", "value"]).size().to_frame(name="ct")
    cts_df = cts_df.reset_index()
    num_frags = len(get_fragments_ordered())
    poss_relevant = list(range(0, num_frags + 1))
    num_table = pd.Series(poss_relevant).to_frame(name="value")
    cts_df = create_full_table(cts_df, num_table, ["task", "arm"])

    g = sns.FacetGrid(data=cts_df, col="task")
    g.map(
        sns.barplot,
        "value",
        "ct",
        "arm",
        dodge=True,
        order=poss_relevant,
        hue_order=["Control", "Treatment"],
        palette={
            "Control": "Orange",
            "Treatment": "Blue"
        },
    )
    g.set_xlabels("Number marked as relevant")
    g.set_ylabels("Count")
    g.add_legend()
    plt.tight_layout()
    g.tight_layout()

    stat_res = task_level_wilcoxon_test(qdf)
    return cts_df, g, stat_res


def rank_best_fragment(arms_df):
    qdf = get_question(arms_df, "rank")
    cts_df = qdf.groupby(["task", "arm", "value"]).size().to_frame(name="ct")
    cts_df = cts_df.reset_index()
    cts_df = create_full_table(cts_df, fragments_table(), ["task", "arm"])
    cts_df["value"] = cts_df["value"].map(lambda x: "F{}".format(x[-1]))
    ordered_labels = ["F{}".format(x[-1]) for x in get_fragments_ordered()]
    g = sns.FacetGrid(data=cts_df, col="task")
    g.map(
        sns.barplot,
        "value",
        "ct",
        "arm",
        dodge=True,
        order=ordered_labels,
        hue_order=["Control", "Treatment"],
        palette={
            "Control": "Orange",
            "Treatment": "Blue"
        },
    )
    g.set_xlabels("Fragment")
    g.set_ylabels("Count")
    g.add_legend()
    plt.tight_layout()
    g.tight_layout()

    qdf = qdf.copy()
    qdf["value"] = qdf["value"].map(lambda x: int(x[-1]))

    stat_res = task_level_wilcoxon_test(qdf)
    return cts_df, g, stat_res


def access_helps(arms_df):
    qdf = get_question(arms_df, "access")
    cts_df = qdf.groupby(["task", "arm", "value"]).size().to_frame(name="ct")
    cts_df = cts_df.reset_index()
    cts_df = create_full_table(cts_df, likert_table(), ["task", "arm"])
    g = sns.FacetGrid(data=cts_df, col="task")
    g.map(
        sns.barplot,
        "value",
        "ct",
        "arm",
        dodge=True,
        order=get_likert_ordered(),
        hue_order=["Control", "Treatment"],
        palette={
            "Control": "Orange",
            "Treatment": "Blue"
        },
    )
    g.set_xticklabels(rotation=90)
    g.set_xlabels("Likert Scale")
    g.set_ylabels("Count")
    g.add_legend()
    plt.tight_layout()
    g.tight_layout()

    qdf = qdf.copy()
    # map likert to ordinal
    likert = get_likert_ordered()
    qdf["value"] = qdf["value"].map(lambda x: likert.index(x))
    stat_res = task_level_wilcoxon_test(qdf)
    return cts_df, g, stat_res


def get_args():
    parser = ArgumentParser(description="Survey analysis")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Survey results in csv from qualtrics",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        print("Creating", args.output_dir)
        os.makedirs(args.output_dir)
    raw_df = pd.read_csv(args.input)

    info, wide, (intro_df, relevance_df, arms_df) = prepare_data(raw_df)

    # is task relevant
    rel_cts, rel_graph = task_relevance(relevance_df)
    rel_cts.to_csv(
        os.path.join(args.output_dir, "task_relevance.csv"), index=False
    )
    rel_graph.savefig(os.path.join(args.output_dir, "task_relevance.pdf"))

    # how many relevant fragments per task
    num_cts, num_graph, num_stat = num_of_relevant_fragments(arms_df)
    num_cts.to_csv(
        os.path.join(args.output_dir, "number_relevant_fragments.csv"),
        index=False
    )
    num_graph.savefig(
        os.path.join(args.output_dir, "number_relevant_fragments.pdf")
    )
    with open(os.path.join(args.output_dir, "number_relevant_fragments.stat"),
              "w") as fout:
        fout.write(num_stat)

    # which fragment is the best
    rank_cts, rank_graph, rank_stat = rank_best_fragment(arms_df)
    rank_cts.to_csv(
        os.path.join(args.output_dir, "best_rank.csv"), index=False
    )
    rank_graph.savefig(os.path.join(args.output_dir, "best_rank.pdf"))
    with open(os.path.join(args.output_dir, "best_rank.stat"), "w") as fout:
        fout.write(rank_stat)

    # does having access to treatment help?
    access_cts, access_graph, access_stat = access_helps(arms_df)
    access_cts.to_csv(
        os.path.join(args.output_dir, "access_helps.csv"), index=False
    )
    access_graph.savefig(os.path.join(args.output_dir, "access_helps.pdf"))
    with open(os.path.join(args.output_dir, "access_helps.stat"), "w") as fout:
        fout.write(access_stat)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
