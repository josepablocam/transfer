# Extract certain exprs from Kaggle user scripts

from argparse import ArgumentParser
import ast
from difflib import SequenceMatcher
import glob
import inspect
import os
import pickle
import types

from astunparse import unparse
import numpy as np
import pandas as pd
import sqlite3
import tqdm


def get_callable_for_type(_type):
    return set([name for name, _t in inspect.getmembers(_type) if callable(_t)])

def show_progress(ls):
    return tqdm.tqdm(list(ls))

def get_target_column_names_for_type(csv_path, type_pred):
    df = pd.read_csv(csv_path)
    return [col for col, _type in zip(df.columns, df.dtypes) if type_pred(_type)]


class GenericExprExtractor(ast.NodeVisitor):
    """
    Extract expressions based on some generic criteria
    """
    def __init__(self):
        self.strs_to_track = set([])
        self.attributes_to_track = set([])
        self.column_type_pred = None
        self.clear_state()

    def clear_state(self):
        # these need to be refreshed for each program that we traverse
        self.exprs = []
        self.has_target_op = False
        self.names_to_track = set([])

    def visit_Assign(self, node):
        self.has_target_op = False
        self.visit(node.value)
        if self.has_target_op:
            self.exprs.append(node.value)
            for target in node.targets:
              if isinstance(target, ast.Name):
                self.names_to_track.add(target.id)

    def visit_Return(self, node):
        self.has_target_op = False
        if node.value:
            self.visit(node.value)
            if self.has_target_op:
                self.exprs.append(node.value)

    def visit_Attribute(self, node):
        if isinstance(node.attr, str):
            if node.attr in self.strs_to_track:
                self.has_target_op = True
            if node.attr in self.attributes_to_track:
                self.has_target_op = True
                if isinstance(node.value, ast.Name):
                    self.names_to_track.add(node.value.id)
        self.generic_visit(node)

    def visit_Str(self, node):
        if node.s in self.strs_to_track:
            self.has_target_op = True

    def visit_Name(self, node):
        if node.id in self.names_to_track:
            self.has_target_op = True

    def columns_to_track_from_path(self, csv_path):
        if self.column_type_pred is None:
            raise Exception("Must provide a type predication for column string names to track")
        self.strs_to_track = get_target_column_names_for_type(csv_path, self.column_type_pred)

    def _get_exprs_one(self, prog):
        self.clear_state()
        if not isinstance(prog, ast.Module):
            prog = ast.parse(prog)
        self.visit(prog)
        result = []
        for e in self.exprs:
            result.append(e)
        return result

    def get_exprs(self, progs):
        if isinstance(progs, str) or isinstance(progs, ast.Module):
            return self._get_exprs_one(progs)
        results = []
        for p in show_progress(progs):
            results.append(self._get_exprs_one(p))
        return results


class StringExprExtractor(GenericExprExtractor):
    """
    Extract any expression that directly uses a pandas column
    we know to be string
    """
    def __init__(self, str_cols=None):
        super(StringExprExtractor, self).__init__()
        if str_cols:
            self.strs_to_track = set(str_cols)
        self.attributes_to_track = get_callable_for_type(str)
        self.column_type_pred = lambda _type: isinstance(_type, np.object)

class DataFrameExprExtractor(GenericExprExtractor):
    """
    Extract any expression that uses:
        - panda dataframe methods
        - accesses a string of the same name as the columns
    """

    def __init__(self, df_cols=None):
        super(DataFrameExprExtractor, self).__init__()
        if df_cols:
            self.strs_to_track = set(df_cols)
        self.attributes_to_track = get_callable_for_type(pd.DataFrame)
        # all columns
        self.column_type_pred = lambda _type: True


# 25 = https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction/data
# 70 = https://www.kaggle.com/c/santander-customer-satisfaction/data
# 61 = https://www.kaggle.com/c/home-depot-product-search-relevance/data
# 13 = https://www.kaggle.com/c/crowdflower-search-relevance/data
# 66 = https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/data
# 29 = https://www.kaggle.com/c/flavours-of-physics/data
# 24 = https://www.kaggle.com/c/caterpillar-tube-pricing/data
# 12 = https://www.kaggle.com/c/predict-west-nile-virus/data
# 40 = https://www.kaggle.com/c/rossmann-store-sales/data

def csv_path_to_id(path):
    return int(path.split('/')[-3].split('_')[-1])

def can_parse(src):
    try:
        ast.parse(src)
        return True
    except:
        return False
    
    
class ExtractedExprs(object):
    def __init__(self, csv_path, df, extractor):
        self.csv_path = csv_path
        self.df = df
        self.strs_to_track = extractor.strs_to_track
        self.attributes_to_track = extractor.attributes_to_track

    # def __getattribute__(self, attr):
    #     try:
    #         return object.__getattribute__(self, attr)
    #     except AttributeError:
    #         return object.__getattribute__(object.__getattribute__(self, 'df'), attr)


class KaggleExtractor(object):
    def __init__(
            self,
            kaggle_db_path=None,
            max_similarity_ratio=0.5
        ):
        self.kaggle_db_path = kaggle_db_path
        if kaggle_db_path is None:
            self.kaggle_db_path = 'data/database.sqlite'
        self.db_conn = sqlite3.connect(self.kaggle_db_path)
        self.max_similarity_ratio = max_similarity_ratio
        self.programs_cache = {}

    def query_database(self, project_id):
        query = """
         select
           Scripts.AuthorUserId as user_id,
           Scripts.ScriptProjectId AS project_id,
           ScriptVersions.id as script_id,
           Scripts.ForkParentScriptVersionId as parent_id,
           ScriptVersions.ScriptLanguageId as lang_id,
           ScriptVersions.ScriptContent as script
        from ScriptVersions, Scripts
          where
          Scripts.ScriptProjectId = %d AND
          Scripts.CurrentScriptVersionId = ScriptVersions.Id AND
          Scripts.Id IS NOT NULL AND
          ScriptVersions.ScriptLanguageId = (select Id from ScriptLanguages where Name = "Python")
          group by ScriptContent
        """ % project_id
        return pd.read_sql(query, self.db_conn)

    def prune_similar_programs(self, programs):
        # remove very similar scripts
        print("Computing script similarity ratios")
        # standardize the programs a bit by parsing and unparsing
        programs['script'] = programs['script'].map(lambda src: unparse(ast.parse(src)).strip())
        # sort by script so we always just compare forward and prune out some unnecessary comparisons
        programs = programs.sort_values('script').reset_index(drop=True)
        ratios = [np.nan]
        # only take scripts that are likley to be new (not just small variation)
        n_progs = programs.shape[0]
        ignore = set([])
        keep_ixs = []
        for i in show_progress(range(0, n_progs)):
            if i in ignore:
                continue
            src1 = programs.iloc[i]['script']
            for j in show_progress(range(i + 1, n_progs)):
                src2 = programs.iloc[j]['script']
                ratio = SequenceMatcher(lambda x: x.isspace(), src1, src2).ratio()
                if ratio > self.max_similarity_ratio:
                    ignore.add(j)
            keep_ixs.append(i)
        programs = programs.iloc[keep_ixs].reset_index(drop=True)
        return programs

    def get_programs_from_db(self, csv_path):
        _id = csv_path_to_id(csv_path)
        print("Query database for project %d" % _id)
        programs = self.query_database(_id)
        # remove programs that cannot be parsed
        programs['works'] = programs['script'].map(can_parse)
        programs = programs[programs['works']]
        print("%d program that parse" % programs.shape[0])
        return programs

    def get_exprs_df_for_dataset(self, expr_extractor, csv_path):
        if not csv_path in self.programs_cache:
            programs = self.get_programs_from_db(csv_path)
            programs = self.prune_similar_programs(programs)
            self.programs_cache[csv_path] = programs
        programs = self.programs_cache[csv_path].copy()
        print("Extracting expressions for %d scripts" % programs.shape[0])
        expr_extractor.columns_to_track_from_path(csv_path)
        exprs = expr_extractor.get_exprs(programs['script'].values)
        programs['exprs'] = exprs
        programs = programs[programs['exprs'].map(len) > 0]
        return ExtractedExprs(csv_path, programs, expr_extractor)

    def run(self, expr_extractor, input_csv_paths):
        dfs = []
        for csv_path in input_csv_paths:
            try:
                df = self.get_exprs_df_for_dataset(expr_extractor, csv_path)
                dfs.append(df)
            except Exception as err:
                print(err)
                print("Failed extracting for %s" % csv_path)
                pass
        return dfs

def show(df, field='exprs', is_ast=True):
    for _, row in df.iterrows():
        print('-' * 10)
        str_list = [unparse(e).strip() for e in row[field]] if is_ast else row[field]
        print('*>>' + '\n*>> '.join(str_list))
        print('-' * 10)

def main(args):
    kaggle_pattern = 'project_*/input/train.csv'
    kaggle_inputs = glob.glob(os.path.join(args.kaggle_input_dir, kaggle_pattern))
    kaggle_inputs = [os.path.realpath(p) for p in kaggle_inputs]
    if args.projects:
        kaggle_inputs = [csv_path for csv_path in kaggle_inputs if csv_path_to_id(csv_path) in args.projects]
    kaggle_db_path = args.kaggle_db_path
    output_dir = args.output_dir
    max_similarity_ratio = args.max_similarity_ratio
    
    data = KaggleExtractor(kaggle_db_path=kaggle_db_path, max_similarity_ratio=max_similarity_ratio)

    print("Extracting string related expressions")
    str_results = data.run(StringExprExtractor(), kaggle_inputs)
    pickle.dump(str_results, open(os.path.join(output_dir, 'str-exprs.pkl'), 'wb'))

    print("Extracting dataframe related expressions")
    df_results = data.run(DataFrameExprExtractor(), kaggle_inputs)
    pickle.dump(df_results, open(os.path.join(output_dir, 'df-exprs.pkl'), 'wb'))

if __name__ == "__main__":
    argparser = ArgumentParser(description='Extract string and dataframe-based expressions from collection of kaggle scripts')
    argparser.add_argument('kaggle_input_dir', type=str, help='Directory with kaggle data inputs')
    argparser.add_argument('kaggle_db_path', type=str, help='Path to Kaggle sqlite db')
    argparser.add_argument('output_dir', type=str, help='Output directory')
    argparser.add_argument('-p', '--projects', type=int, nargs='*', help='Project ids to execute')
    argparser.add_argument('-s', '--max_similarity_ratio', type=float, help='Maximum similarity ratio for script pruning', default=0.5)
    args = argparser.parse_args()
    try:
        main(args)
    except:
        import pdb
        pdb.post_mortem()

