#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import Counter
import itertools
import inspect
import os
import pickle

import numpy as np
import pandas as pd
import tqdm

from transfer.build_db import (
    FunctionDatabase,
    NodeTypes,
    RelationshipTypes,
)


def start_db(path="sample_db.pkl"):
    with open(path, "rb") as fin:
        db = pickle.load(fin)
    db.startup()
    return db


def get_downsample_tables(tables, n=100, seed=42):
    rng = np.random.RandomState(seed)
    downsampled = []
    for t in tables:
        dt = t.sample(n=min(n, t.shape[0]), replace=False, random_state=rng)
        downsampled.append(dt)
    return downsampled


def create_possible_args(num_args, tables):
    perms = itertools.permutations(tables, num_args)
    return list(perms)


def get_num_args(func):
    return len(inspect.getfullargspec(func).args)


def can_execute(func, downsampled_tables, orig_tables=None):
    func_obj = func.obj
    num_args = get_num_args(func_obj)

    if downsampled_tables is None:
        tables = orig_tables
    else:
        tables = downsampled_tables

    possible_args = create_possible_args(num_args, tables)
    exceptions = []
    for args in possible_args:
        try:
            result = func_obj(*args)
            assert result is not None
            return True, []
        except Exception as err:
            exceptions.append(err)
            continue

    if orig_tables is not None:
        # try again with the original tables
        # rather than downsampled
        return can_execute(func, orig_tables, orig_tables=None)
    else:
        return False, exceptions


def get_wrangling_functions(db):
    function_nodes = db.extracted_functions()
    functions = [db.get_function_from_node(n) for n in function_nodes]
    return functions


def summarize_exceptions(exceptions):
    counter = Counter()
    examples = {}
    for group in exceptions:
        # unique exceptions
        unique_errors = {str(type(err)): err for err in group}
        for err in unique_errors.values():
            key = str(type(err))
            counter[key] += 1
            if key not in examples:
                examples[key] = str(err)

    df = pd.DataFrame(list(counter.items()), columns=["type", "ct"])
    df["pct"] = df["ct"] / df["ct"].sum()
    return df, examples


def compute_executability(db, downsampled_tables=None, orig_tables=None):
    outcome_acc = []
    exceptions_acc = []
    functions = get_wrangling_functions(db)
    for f in tqdm.tqdm(functions):
        success, exceptions = can_execute(
            f, downsampled_tables=downsampled_tables, orig_tables=orig_tables
        )
        outcome_acc.append(success)
        if not success:
            exceptions_acc.append(exceptions)

    msg = "Can execute {}/{}\n".format(np.sum(outcome_acc), len(outcome_acc))
    msg += "{}%".format(np.mean(outcome_acc) * 100)
    df_exceptions, exception_examples = summarize_exceptions(exceptions_acc)
    return msg, df_exceptions, exception_examples


def source_code_functions(db):
    function_nodes = db.extracted_functions()
    msg = ""
    for i, f_node in tqdm.tqdm(enumerate(function_nodes)):
        msg += "# Function {}\n".format(i)
        msg += db.get_code(f_node)
        msg += "\n#=============\n\n"
    return msg


def get_args():
    parser = ArgumentParser(
        description="Compute fraction of functions executable"
    )
    parser.add_argument(
        "--data", type=str, nargs="+", help="Paths to dataset .csv files"
    )
    parser.add_argument(
        "--database",
        type=str,
        help="Path to the wrangling functions database",
        default="sample_db.pkl",
    )
    parser.add_argument(
        "--downsample_n",
        type=int,
        help="Downsample original tables to N",
        default=None
    )
    parser.add_argument("--seed", type=int, help="RNG seed", default=42)
    parser.add_argument("--output_dir", type=str, help="Output directory")
    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    orig_tables = []
    for t in args.data:
        try:
            t = pd.read_csv(t)
        except pd.errors.ParserError:
            # some annoying windows new lines
            t = pd.read_csv(t, lineterminator="\r")
        orig_tables.append(t)

    if args.downsample_n is not None:
        downsampled_tables = get_downsample_tables(
            orig_tables,
            n=args.downsample_n,
            seed=args.seed,
        )
        # slightly larger but still downsampled
        orig_tables = get_downsample_tables(
            orig_tables, n=args.downsample_n * 100, seed=args.seed
        )

    db = start_db(path=args.database)
    outcome, df_exceptions, exception_examples = compute_executability(
        db, downsampled_tables, orig_tables
    )

    outcome_path = os.path.join(args.output_dir, "outcome.txt")
    with open(outcome_path, "w") as fout:
        fout.write(outcome)
        fout.write("\n")

    df_exceptions_path = os.path.join(args.output_dir, "exceptions.csv")
    df_exceptions.to_csv(df_exceptions_path, index=False)

    exception_examples_path = os.path.join(
        args.output_dir, "exception_examples.txt"
    )
    with open(exception_examples_path, "w") as fout:
        for error_type, error_example in exception_examples.items():
            fout.write(error_type)
            fout.write("\n")
            fout.write(error_example)
            fout.write("\n===========\n\n")

    source_path = os.path.join(args.output_dir, "source_code.py")
    with open(source_path, "w") as fout:
        fout.write(source_code_functions(db))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
