#!/usr/bin/env python3
from argparse import ArgumentParser
import ast
from collections import Counter
import importlib

import tqdm


class CollectImports(ast.NodeVisitor):
    def __init__(self):
        self.imports = set([])

    def visit_Import(self, node):
        self.imports.update([a.name for a in node.names])

    def visit_ImportFrom(self, node):
        self.imports.add(node.module)

    def run(self, src):
        if isinstance(src, str):
            src = ast.parse(src)
        self.visit(src)
        return self.imports


def get_requirements(script_path):
    with open(script_path, "r") as fin:
        src = fin.read()
    imps = CollectImports().run(src)
    libs = set([i.split(".")[0] for i in imps])
    return libs


def keep_only_if_fails(libs):
    failed = []
    for l in libs:
        try:
            importlib.import_module(l)
        except ImportError:
            failed.append(l)
    return failed


def rename_packages(pkgs):
    # some packages aren't named the same way via pip
    renamed = {
        "PIL": "pillow",
        "sklearn": "scikit-learn",
    }
    return [renamed.get(p, p) for p in pkgs]


def get_args():
    parser = ArgumentParser(description="Collect imports")
    parser.add_argument("--input", type=str, nargs="+", help="Script paths")
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Filter down to those that fail to import in current env")
    return parser.parse_args()


def main():
    args = get_args()
    cter = Counter()
    for path in tqdm.tqdm(args.input):
        path_libs = get_requirements(path)
        cter.update(path_libs)

    libs = sorted(list(cter.keys()))
    if args.filter:
        libs = keep_only_if_fails(libs)

    libs = rename_packages(libs)
    cter = dict(zip(rename_packages(cter.keys()), cter.values()))

    for l in libs:
        print("# Num scripts:", cter[l])
        print(l)

    return libs


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
