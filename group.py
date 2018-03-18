from argparse import ArgumentParser

from collections import defaultdict
import ast
import abstract
from extract import ExtractedExprs
from abstract import AbstractName, AbstractColumn, AbstractNum, AbstractStr, AbstractTargetType
from astunparse import unparse
import pickle

def exact_grouping(exprs):
    flat_exprs = []
    for _, row in exprs.df.iterrows():
        es = [(e, row['script_id']) for e in row['abstract_exprs']]
        flat_exprs.extend(es)
    
    grouped = defaultdict(lambda: [])
    for expr, _id in flat_exprs:
        grouped[ast.dump(expr)].append((expr, _id))

    return grouped

def filter(grouped):
    # remove any entries with single expr
    grouped = {k:v for k, v in grouped.items() if len(v) > 1}
    # remove entries that are all from same script
    grouped = {k:v for k, v in grouped.items() if len(set([_id for _, _id in v])) > 1}
    return grouped
    
def get_examples(grouped):
    grouped = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
    return [(unparse(abstract.annotate(v[0][0])).strip(), len(v)) for _, v in grouped]
    
def print_examples(example_and_counts):
    for ex, ct in example_and_counts:
        print("====> Representative (count %d)" % ct)
        print(ex)
        print('-' * 10)

def run(extracted):
    grouped = exact_grouping(extracted)
    grouped = filter(grouped)
    examples = get_examples(grouped)
    print_examples(examples)
    
def main(args):
    input_path = args.input_path
    exprs_list = pickle.load(open(input_path, 'rb'))
    for exprs in exprs_list:
        print("Extracting common expressions for %s\n\n" % exprs.csv_path)
        run(exprs)
        print("<<<<<<<<=======================>>>>>>>>>>\n\n\n")

if __name__ == "__main__":
    argparser = ArgumentParser(description='Basic grouping based on abstract exprs')
    argparser.add_argument('input_path', type=str, help='Input of expressions data frame with abstract expressions')
    args = argparser.parse_args()
    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
    

