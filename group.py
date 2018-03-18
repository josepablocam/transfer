from argparse import ArgumentParser

from collections import defaultdict
import ast
import abstract
from extract import ExtractedExprs
from abstract import AbstractName, AbstractColumn, AbstractNum, AbstractStr, AbstractTargetType
from astunparse import unparse
import pickle


class GroupedExprs(object):
    def __init__(self, exprs, have_ids=True):
        if have_ids:
            self.exprs = list(map(lambda x: x[0], exprs))
            self._ids = list(map(lambda x: x[1], exprs))
        else:
            self.exprs = exprs
            self._ids = None
        self.representative = unparse(abstract.annotate(self.exprs[0])).strip()
        abstracts = defaultdict(lambda: set([]))
        for expr in self.exprs:
            for abstract_val in abstract.collect_abstracts(expr):
                concrete_str = unparse(abstract_val.concretize()).strip()
                abstracts[abstract_val.label].add(concrete_str)
        self.abstracts = dict(abstracts)

    def get_number_exprs(self):
        return len(self.exprs)

    def get_number_distinct_programs(self):
        assert self._ids is not None
        return len(set(self._ids))

    def __str__(self):
        _str = "Representative: %s\n" % self.representative
        _str += 'Count Instances: %d\n' %  self.get_number_exprs()
        if self._ids:
            _str +='Count Programs: %d\n' % self.get_number_distinct_programs()
        _str += 'Abstract to Concrete Value Mapping\n'
        _str += '----------------------------------\n'
        for _type, vals in self.abstracts.items():
            vals_str = '(%s)' % ','.join(vals)
            _str += '   %s: %s\n' % (_type, vals_str)
        _str += '\n'
        return _str

    def __len__(self):
        return self.get_number_exprs()


def exact_grouping(exprs):
    flat_exprs = []
    for _, row in exprs.df.iterrows():
        es = [(e, row['script_id']) for e in row['abstract_exprs']]
        flat_exprs.extend(es)

    grouped = defaultdict(lambda: [])
    for expr, _id in flat_exprs:
        grouped[ast.dump(expr)].append((expr, _id))

    results = []
    for _, vals in grouped.items():
        grp = GroupedExprs(vals, have_ids=True)
        results.append(grp)

    return results

def filter(groups):
    # remove any entries with single expr
    groups = [g for g in groups if g.get_number_exprs() > 1]
    # remove entries that are all from same script
    groups = [g for g in groups if g.get_number_distinct_programs() > 1]
    # sort by more common first
    groups = sorted(groups, key=lambda x: x.get_number_exprs(), reverse=True)
    return groups

def run(extracted):
    groups = exact_grouping(extracted)
    groups = filter(groups)
    for group in groups:
        print(group)

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


