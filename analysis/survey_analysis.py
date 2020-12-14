#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import Counter
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import statsmodels.stats.descriptivestats


def get_task_id(df):
    return df["question"].map(lambda x: int(x.split("_")[1].replace("t", "")))


def get_question_num(df):
    return df["question"].map(lambda x: int(x.split("_")[2].replace("q", "")))


def pass_validation(df):
    correct_answer = "Scales column named col1"
    ok = df["validation_question"].map(lambda x: x.strip() == correct_answer)
    n = df.shape[0]
    print("{} / {} passed the validation question".format(ok.sum(), n))
    df = df[ok].reset_index(drop=True)
    return df


def prepare_data(df, check_validation=True):
    property_map = dict(df.iloc[0])
    responses_df = df.iloc[2:].copy()
    responses_df["survey_id"] = list(range(0, responses_df.shape[0]))
    responses_df["StartDate"] = pd.to_datetime(responses_df["StartDate"])
    responses_df["EndDate"] = pd.to_datetime(responses_df["EndDate"])

    if check_validation:
        responses_df = pass_validation(responses_df)

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


def get_ordered_year_buckets():
    return ["0 - 1", "1 - 3", "3 - 5", "5+"]


def years_experience_table():
    years = get_ordered_year_buckets()
    return pd.Series(years).to_frame(name="value")


def get_tools_ordered():
    return [
        "Python", "Pandas (Python library)", "SQL", "Julia", "R", "Matlab",
        "SAS", "Stata"
    ]


def tools_table():
    tools = get_tools_ordered()
    return pd.Series(tools).to_frame(name="value")


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


def get_intro_question(intro_df, name):
    name_to_id = {
        "programming_experience": "intro_q0",
        "data_analysis_experience": "intro_q1",
        "tools": "intro_q2",
    }
    return intro_df[intro_df["question"] == name_to_id[name]].copy()


def years_programming_experience(intro_df):
    qdf = get_intro_question(intro_df, "programming_experience")
    cts_df = qdf.groupby(["value"]).size().to_frame(name="ct")
    cts_df = cts_df.reset_index()
    cts_df = pd.merge(years_experience_table(), cts_df, how="left", on="value")
    cts_df["ct"] = cts_df["ct"].fillna(0.0)
    fig, ax = plt.subplots(1)
    sns.barplot(
        data=cts_df,
        x="value",
        y="ct",
        order=get_ordered_year_buckets(),
        ax=ax,
        color="blue",
    )
    ax.set_xlabel("Years of Programming Experience")
    ax.set_ylabel("Participants")
    plt.tight_layout()
    return cts_df, ax


def years_data_analysis_experience(intro_df):
    qdf = get_intro_question(intro_df, "data_analysis_experience")
    cts_df = qdf.groupby(["value"]).size().to_frame(name="ct")
    cts_df = cts_df.reset_index()
    cts_df = pd.merge(years_experience_table(), cts_df, how="left", on="value")
    cts_df["ct"] = cts_df["ct"].fillna(0.0)
    fig, ax = plt.subplots(1)
    sns.barplot(
        data=cts_df,
        x="value",
        y="ct",
        order=get_ordered_year_buckets(),
        ax=ax,
        color="blue",
    )
    ax.set_xlabel("Years of Data Analysis Experience")
    ax.set_ylabel("Participants")
    plt.tight_layout()
    return cts_df, ax


def tool_experience(intro_df):
    qdf = get_intro_question(intro_df, "tools")
    qdf = qdf["value"].map(lambda x: x.split(","))
    answers = [e for grp in qdf.tolist() for e in grp]
    cts_df = pd.DataFrame(
        list(Counter(answers).items()), columns=["value", "ct"]
    )
    cts_df = pd.merge(tools_table(), cts_df, how="left", on="value")
    cts_df["ct"] = cts_df["ct"].fillna(0.0)
    fig, ax = plt.subplots(1)
    sns.barplot(
        data=cts_df,
        x="value",
        y="ct",
        order=get_tools_ordered(),
        ax=ax,
        color="blue",
    )
    ax.set_xlabel("Tools")
    ax.set_ylabel("Participants")
    plt.xticks(rotation="vertical")
    plt.tight_layout()
    return cts_df, ax


def task_relevance(rel_df):
    cts_df = rel_df.groupby(["task", "value"]).size().to_frame(name="ct")
    cts_df = cts_df.reset_index()
    cts_df = create_full_table(cts_df, likert_table(), ["task"])
    cts_df["value"] = cts_df["value"].map(
        lambda x: "L{}".format(get_likert_ordered().index(x))
    )
    possible = ["L{}".format(i) for i in range(0, len(get_likert_ordered()))]
    g = sns.FacetGrid(data=cts_df, col="task")
    g.map(sns.barplot, "value", "ct", order=possible)
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


def single_or_error(vs):
    try:
        n = len(vs)
        # ok on strings...
        if isinstance(vs, str):
            return vs
        if n > 1:
            raise ValueError("Should have single answer, no aggregation")
        else:
            return vs
    except TypeError:
        return vs


def create_paired_df(df, index=None, columns=None, values=None):
    if index is None:
        index = ["survey_id", "task", "question_num"]
    if columns is None:
        columns = ["arm"]
    if values is None:
        values = "value"
    pdf = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=single_or_error,
    )
    pdf = pdf.reset_index()
    return pdf


def stat_results_to_string(results):
    msg = ""
    for task, res_tuple in results.items():
        if len(res_tuple) == 2:
            res, is_significant = res_tuple
            effect_size = np.nan
        elif len(res_tuple) == 3:
            res, is_significant, effect_size = res_tuple
        else:
            raise ValueError("Invalid result len")
        msg += "Task {}\n".format(task)
        msg += "{}\n".format(res)
        msg += "significant={}\n".format(is_significant)
        msg += "effect size={}\n".format(effect_size)
        msg += "\n"
    return msg


def estimated_wilcoxon_effect_size(paired_task, x, y):
    # https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    # see effect size section
    diff = (paired_task[x] - paired_task[y]).values
    absdiff = np.abs(diff)
    is_nz = absdiff != 0
    absdiff = absdiff[is_nz]
    diff = diff[is_nz]
    ranks = scipy.stats.rankdata(absdiff)
    W = np.sum(np.sign(diff) * ranks)
    S = np.sum(np.arange(1, len(ranks) + 1))
    return W / S


def task_level_wilcoxon_test(qdf, x, y, alpha=0.05):
    paired = create_paired_df(qdf)
    results = {}
    tasks = qdf.task.unique()
    num_tasks = len(tasks)
    for task in tasks:
        paired_task = paired[paired["task"] == task]
        task_result = scipy.stats.wilcoxon(
            paired_task[x].values.astype(float),
            paired_task[y].values.astype(float),
            # more appropriate for discrete/ordinal data
            # since zero diff can happen more often
            zero_method="pratt",
        )
        is_significant = task_result.pvalue < (alpha / num_tasks)
        effect_size = estimated_wilcoxon_effect_size(paired_task, x, y)
        results[task] = (task_result, is_significant, effect_size)
    stat_res = stat_results_to_string(results)
    return stat_res


class SignedTest(object):
    def __init__(self, stat, pvalue):
        self.stat = stat
        self.pvalue = pvalue

    def __str__(self):
        return "SignedTestResult(statistic={}, pvalue={})".format(
            self.stat, self.pvalue
        )

    def __repr__(self):
        return str(self)


def task_level_signed_test(qdf, x, y, alpha=0.05):
    paired = create_paired_df(qdf)
    paired["diff"] = paired[x] - paired[y]
    results = {}
    tasks = qdf.task.unique()
    num_tasks = len(tasks)
    for task in tasks:
        paired_task = paired[paired["task"] == task]
        task_result = statsmodels.stats.descriptivestats.sign_test(
            paired_task["diff"],
            mu0=0,
        )
        task_result = SignedTest(*task_result)
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

    stat_res = task_level_wilcoxon_test(qdf, "Treatment", "Control")
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

    stat_res = task_level_wilcoxon_test(qdf, "Control", "Treatment")
    return cts_df, g, stat_res


def access_helps(arms_df):
    qdf = get_question(arms_df, "access")
    cts_df = qdf.groupby(["task", "arm", "value"]).size().to_frame(name="ct")
    cts_df = cts_df.reset_index()
    cts_df = create_full_table(cts_df, likert_table(), ["task", "arm"])

    plot_cts_df = cts_df.copy()
    plot_cts_df["value"] = plot_cts_df["value"].map(
        lambda x: "L{}".format(get_likert_ordered().index(x))
    )
    possible = ["L{}".format(i) for i in range(0, len(get_likert_ordered()))]

    g = sns.FacetGrid(data=plot_cts_df, col="task")
    g.map(
        sns.barplot,
        "value",
        "ct",
        "arm",
        dodge=True,
        order=possible,
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
    n = len(likert)
    # just shift so that best is 7 and worst is 0
    # just for purposes of test...makes more sense so that positive
    # diff is a better rank
    qdf["value"] = qdf["value"].map(lambda x: n - likert.index(x))
    stat_res = "Task Level Wilcoxon Signed Rank Test\n"
    stat_res += task_level_wilcoxon_test(qdf, "Treatment", "Control")
    stat_res += "\n\n Task Level Signed Test\n"
    stat_res += task_level_signed_test(qdf, "Treatment", "Control")
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
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip the validation question (keep all answers)"
    )
    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        print("Creating", args.output_dir)
        os.makedirs(args.output_dir)
    raw_df = pd.read_csv(args.input)[:-1]

    info, wide, (intro_df, relevance_df, arms_df) = prepare_data(
        raw_df,
        check_validation=not args.skip_validation,
    )

    # intro questions
    prog_years_cts, prog_years_graph = years_programming_experience(intro_df)
    prog_years_cts.to_csv(
        os.path.join(args.output_dir, "prog_years.csv"), index=False
    )
    prog_years_graph.get_figure().savefig(
        os.path.join(args.output_dir, "prog_years.pdf")
    )

    data_years_cts, data_years_graph = years_data_analysis_experience(intro_df)
    data_years_cts.to_csv(
        os.path.join(args.output_dir, "data_years.csv"), index=False
    )
    data_years_graph.get_figure().savefig(
        os.path.join(args.output_dir, "data_years.pdf")
    )

    tool_cts, tool_graph = tool_experience(intro_df)
    tool_cts.to_csv(os.path.join(args.output_dir, "tools.csv"), index=False)
    tool_graph.get_figure().savefig(os.path.join(args.output_dir, "tools.pdf"))

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
