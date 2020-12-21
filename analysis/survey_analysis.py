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


def approved_prolific(df, prolific_meta):
    approved_pro = prolific_meta[prolific_meta["status"] == "APPROVED"]
    approved_ids = approved_pro["participant_id"].values
    approved = df["prolific_id"].isin(approved_ids)
    n = df.shape[0]
    print("{} / {} approved prolific ids".format(approved.sum(), n))
    df = df[approved].reset_index(drop=True)
    return df


def prepare_data(df, check_validation=True, prolific_meta=None):
    property_map = dict(df.iloc[0])
    responses_df = df.iloc[2:].copy()
    responses_df["survey_id"] = list(range(0, responses_df.shape[0]))
    responses_df["StartDate"] = pd.to_datetime(responses_df["StartDate"])
    responses_df["EndDate"] = pd.to_datetime(responses_df["EndDate"])

    # remove invalid answers (i.e. previews etc)
    responses_df = responses_df[responses_df["Status"] == "IP Address"]
    # make sure answered everything
    last_qs = ["d1_t3_q2_treat", "d1_t3_q2_cont"]
    completed = ~responses_df[last_qs].isnull().any(axis=1)
    responses_df = responses_df[completed]

    if check_validation:
        responses_df = pass_validation(responses_df)

    if prolific_meta is not None:
        responses_df = approved_prolific(responses_df, prolific_meta)

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


def is_likert_better_or_equal(values, likert_threshold):
    likert = get_likert_ordered()
    likert_ixs = values.map(lambda x: likert.index(x)).values
    threshold_ix = likert.index(likert_threshold)
    return likert_ixs <= threshold_ix


def get_fragments_ordered(num_fragments=5):
    fragments = ["Fragment {}".format(i) for i in range(num_fragments)]
    return fragments


def fragments_table(num_fragments=5, extra=None):
    fragments = get_fragments_ordered()
    if extra is not None:
        fragments += list(extra)
    return pd.Series(fragments).to_frame(name="value")


def get_ordered_year_buckets():
    return ["0 - 1", "1 - 3", "3 - 5", "5+"]


def years_experience_table():
    years = get_ordered_year_buckets()
    return pd.Series(years).to_frame(name="value")


def get_tools_ordered():
    return [
        "Python", "Pandas (Python library)", "SQL", "Julia", "R", "Matlab",
        "SAS", "Stata", "Shell tools (e.g. sed, awk)"
    ]


def tools_table():
    tools = get_tools_ordered()
    return pd.Series(tools).to_frame(name="value")


def get_proficiency_ordered():
    return [
        "No knowledge", "Fundamental awareness", "Novice", "Intermediate",
        "Advanced", "Expert"
    ]


def proficiency_table():
    prof = get_proficiency_ordered()
    return pd.Series(prof).to_frame(name="value")


def create_full_table(df1, df2, df1_cols, fillna_value=0.0):
    df1_unique = df1[df1_cols].drop_duplicates()
    product = df1_unique.assign(key=1).merge(
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
        "pandas": "intro_q3",
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
    # remove nan (no tools)
    qdf = qdf[~pd.isnull(qdf["value"])]
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


def pandas_proficiency(intro_df):
    qdf = get_intro_question(intro_df, "pandas")
    # remove missing (old version)
    qdf = qdf[~pd.isnull(qdf["value"])]
    answers = qdf["value"].values.tolist()
    cts_df = pd.DataFrame(
        list(Counter(answers).items()), columns=["value", "ct"]
    )
    cts_df = pd.merge(proficiency_table(), cts_df, how="left", on="value")
    cts_df["ct"] = cts_df["ct"].fillna(0.0)
    fig, ax = plt.subplots(1)
    sns.barplot(
        data=cts_df,
        x="value",
        y="ct",
        order=get_proficiency_ordered(),
        ax=ax,
        color="blue",
    )
    ax.set_xlabel("Proficiency")
    ax.set_ylabel("Participants")
    plt.xticks(rotation="vertical")
    plt.tight_layout()
    return cts_df, ax


def task_relevance(rel_df):
    # quick summary
    print("Task Relevance")
    rel_df = rel_df.copy()
    rel_df["at_least_agree"] = is_likert_better_or_equal(
        rel_df["value"],
        "Somewhat agree",
    )
    print(rel_df.groupby(["task"])[["at_least_agree"]].mean())

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
        try:
            task_result = scipy.stats.wilcoxon(
                paired_task[x].values.astype(float),
                paired_task[y].values.astype(float),
                # more appropriate for discrete/ordinal data
                # since zero diff can happen more often
                zero_method="pratt",
            )
            is_significant = task_result.pvalue < (alpha / num_tasks)
            effect_size = estimated_wilcoxon_effect_size(paired_task, x, y)
        except Exception as err:
            task_result = None
            is_significant = False
            effect_size = None
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
    print("Number of relevant fragments")
    print(qdf.groupby(["task", "arm"])[["value"]].mean())

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
    cts_df = create_full_table(
        cts_df, fragments_table(extra=["None"]), ["task", "arm"]
    )
    cts_df["value"] = cts_df["value"].map(
        lambda x: "F{}".format(x[-1]) if x != "None" else "None"
    )
    ordered_labels = ["F{}".format(x[-1]) for x in get_fragments_ordered()]
    ordered_labels = ordered_labels + ["None"]
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
    # "None" is mapped to maximum rank + 1
    rank_none = 5
    qdf["value"] = qdf["value"].map(
        lambda x: int(x[-1]) if x != "None" else rank_none
    )
    print("Best rank")
    print(qdf.groupby(["task", "arm"])[["value"]].mean())

    stat_res = task_level_wilcoxon_test(qdf, "Control", "Treatment")
    return cts_df, g, stat_res


def access_helps(arms_df):
    qdf = get_question(arms_df, "access")
    qdf = qdf.copy()
    print("Access helps")
    qdf = qdf.copy()
    qdf["at_least_agree"] = is_likert_better_or_equal(
        qdf["value"],
        "Somewhat agree",
    )
    print(qdf.groupby(["task", "arm"])[["at_least_agree"]].mean())

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


def time_spent(wide):
    time_col = [c for c in wide.columns if c.startswith("Duration")]
    assert len(time_col) == 1
    time_df = wide[time_col]
    time_df = time_df.rename(columns={time_col[0]: "seconds"})
    time_df["seconds"] = time_df["seconds"].astype(float)
    time_df["minutes"] = time_df["seconds"] / 60.
    fig, ax = plt.subplots(1)
    time_df.hist("minutes", ax=ax)
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Participants")
    print("Median completion time: {}".format(time_df["minutes"].median()))
    return time_df, ax


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
    parser.add_argument(
        "--prolific_meta",
        type=str,
        help="Path to prolific population meta csv",
    )
    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        print("Creating", args.output_dir)
        os.makedirs(args.output_dir)
    raw_df = pd.read_csv(args.input)

    prolific_meta = None
    if args.prolific_meta is not None:
        prolific_meta = pd.read_csv(args.prolific_meta)

    info, wide_df, (intro_df, relevance_df, arms_df) = prepare_data(
        raw_df,
        check_validation=not args.skip_validation,
        prolific_meta=prolific_meta
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

    pandas_cts, pandas_graph = pandas_proficiency(intro_df)
    pandas_cts.to_csv(os.path.join(args.output_dir, "pandas.csv"), index=False)
    pandas_graph.get_figure().savefig(
        os.path.join(args.output_dir, "pandas.pdf")
    )

    time_df, time_graph = time_spent(wide_df)
    time_df.to_csv(os.path.join(args.output_dir, "time.csv"), index=False)
    time_graph.get_figure().savefig(os.path.join(args.output_dir, "time.pdf"))

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
