#!/usr/bin/env python3
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
    return [
        "Strongly agree",
        "Agree",
        "Somewhat agree",
        "Neither agree nor disagree",
        "Somewhat disagree",
        "Disagree",
        "Strongly disagree",
    ]


def get_fragment_rank_ordered():
    return [0, 1, 2, 3, 4]


def full_table(df, vals):
    df = df[["task"]].drop_duplicates()
    n_df = df.shape[0]
    rep_df = pd.concat([df] * len(vals), axis=0).reset_index(drop=True)
    rep_df["value"] = vals * n_df
    return rep_df


def full_likert_table(df):
    return full_table(df, get_likert_ordered())


def full_rank_table(df):
    return full_table(df, get_fragment_rank_ordered())


def task_value_counts(full_df, df):
    counts = df.groupby(["task",
                         "value"]).size().to_frame(name="ct").reset_index()
    full_df = pd.merge(full_df, counts, how="left", on=["task", "value"])
    full_df["ct"] = full_df["ct"].fillna(0)
    return full_df


def task_relevance(rel_df):
    cts_df = task_value_counts(full_likert_table(rel_df), rel_df)
    g = sns.FacetGrid(data=cts_df, col="task")
    g.map(sns.barplot, "value", "ct", order=get_likert_ordered())
    g.set_xticklabels(rotation=90)
    g.set_xlabels("Likert Scale")
    g.set_ylabels("Count")
    plt.legend(loc="best")
    plt.tight_layout()
    g.tight_layout()
    return cts_df, g


def num_of_relevant_fragments(arms_df):
    pass


def rank_best_fragment(arms_df):
    question_df = arms_df[arms_df["question_num"] == 1].copy()
    question_df["value"] = question_df["value"].map(
        lambda x: int(x.replace("Fragment", ""))
    )
    results = []
    for arm in question_df.arm.unique():
        arm_df = question_df[question_df["arm"] == arm]
        res = task_value_counts(full_rank_table(arm_df), arm_df)
        res["arm"] = arm
        results.append(res)
    cts_df = pd.concat(results, axis=0)

    g = sns.FacetGrid(data=cts_df, col="task")
    g.map(
        sns.barplot,
        "value",
        "ct",
        "arm",
        dodge=True,
        order=get_fragment_rank_ordered(),
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
    return cts_df, g


def access_helps(arms_df):
    question_df = arms_df[arms_df["question_num"] == 2]
    results = []
    for arm in question_df.arm.unique():
        arm_df = question_df[question_df["arm"] == arm]
        res = task_value_counts(full_likert_table(arm_df), arm_df)
        res["arm"] = arm
        results.append(res)
    cts_df = pd.concat(results, axis=0)

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
    return cts_df, g


## Statistical tests

def num_relevant_test():
    pass


def rank_test():
    pass


def access_test():
    pass
