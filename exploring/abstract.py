# Abstract extracted exprs to remove identifiers/numeric vals etc

from argparse import ArgumentParser
import ast
import _ast
from astunparse import unparse
from copy import deepcopy
import inspect
import pickle
from extract import *

def is_ast_lib_node(e):
    module = inspect.getmodule(e)
    return module == _ast

class AbstractValue(object):
    def __repr__(self):
        return self.label

    def __eq__(self, other):
        return type(self) == type(other)

    def concretize(self):
        return deepcopy(self.concrete_val)

    def annotate(self):
        str_concrete_val = self.concrete_val
        if is_ast_lib_node(str_concrete_val):
            str_concrete_val = unparse(str_concrete_val).strip()
        return ast.Name(id='%s[%s]' % (self.label, str_concrete_val))


class AbstractName(AbstractValue):
    def __init__(self, concrete_val):
        self.concrete_val = concrete_val
        self.label = '<NAME>'

class AbstractColumn(AbstractValue):
    def __init__(self, concrete_val):
        self.concrete_val = concrete_val
        self.label = '<COLUMN>'

class AbstractNum(AbstractValue):
    def __init__(self, concrete_val):
        self.concrete_val = concrete_val
        self.label = '<NUM>'

class AbstractStr(AbstractValue):
    def __init__(self, concrete_val):
        self.concrete_val = concrete_val
        self.label = '<STR>'

class AbstractTargetType(AbstractValue):
    def __init__(self, concrete_val):
        self.concrete_val = concrete_val
        self.label = '<TARGET TYPE>'

ABSTRACT_TYPES = [AbstractName, AbstractColumn, AbstractNum, AbstractStr, AbstractTargetType]


class SimpleExprAbstractor(ast.NodeTransformer):
    def __init__(self,
            strs_to_track,
            attributes_to_track,
        ):
        self.strs_to_track = set(strs_to_track)
        self.attributes_to_track = set(attributes_to_track)

    def visit_Name(self, node):
        copy = deepcopy(node)
        copy.id = AbstractName(node)
        return copy

    def visit_Num(self, node):
        new_node = ast.Num(n=AbstractNum(node))
        return ast.copy_location(new_node, node)

    def visit_Str(self, node):
        copy = deepcopy(node)
        if node.s in self.strs_to_track:
            copy.s = AbstractColumn(node)
        else:
            copy.s = AbstractStr(node)
        return copy

    def visit_Attribute(self, node):
        if node.attr in self.attributes_to_track:
            value = ast.Name(id=AbstractTargetType(node.value))
            value = ast.copy_location(value, node.value)
            new_node = ast.Attribute(value=value, attr=node.attr, ctx=deepcopy(node.ctx))
            return ast.copy_location(new_node, node)
        else:
            return self.generic_visit(node)

    def visit_List(self, node):
        has_tracked_str = False
        for e in node.elts:
            if isinstance(e, ast.Str) and e.s in self.strs_to_track:
                has_tracked_str = True
                break
        if has_tracked_str:
            new_node = ast.Str(s=AbstractColumn(node))
            return ast.copy_location(new_node, node)
        else:
            return self.generic_visit(node)
            
    def visit_UnaryOp(self, node):
        if isinstance(node.op, (ast.UAdd, ast.USub)):
            new_node = ast.Num(n=AbstractNum(node.operand))
            return ast.copy_location(new_node, node)
        return self.generic_visit(node)

    def run(self, expr):
        if not isinstance(expr, ast.Module):
            expr = ast.parse(expr)
        return self.visit(expr)


class TraverseMixedAST(ast.NodeTransformer):
    def __init__(self, fun):
        self.fun = fun

    def visit_Str(self, node):
        try:
            return self.fun(node.s)
        except AttributeError:
            return node

    def visit_Num(self, node):
        try:
            return self.fun(node.n)
        except AttributeError:
            return node

    def visit_Name(self, node):
        try:
            return self.fun(node.id)
        except AttributeError:
            return node

def add_abstract(extracted_exprs):
    abstractor = SimpleExprAbstractor(extracted_exprs.strs_to_track, extracted_exprs.attributes_to_track)
    abstract_exprs = []
    for _, row in show_progress(extracted_exprs.df.iterrows()):
        abstract_exprs.append([abstractor.run(e) for e in row['exprs']])
    extracted_exprs.df['abstract_exprs'] = abstract_exprs
    return extracted_exprs

def concretize(expr):
    assert is_ast_lib_node(expr), 'Can only concretize mixes of _ast/Abstract nodes'
    return TraverseMixedAST(lambda node: node.concretize()).visit(deepcopy(expr))

def annotate(expr):
    assert is_ast_lib_node(expr), 'Can only concretize mixes of _ast/Abstract nodes'
    return TraverseMixedAST(lambda node: node.annotate()).visit(deepcopy(expr))

def collect_abstracts(expr):
    assert is_ast_lib_node(expr), 'Can only concretize mixes of _ast/Abstract nodes'
    abstracts = []
    collect_fun = lambda node: abstracts.append(deepcopy(node))
    TraverseMixedAST(collect_fun).visit(deepcopy(expr))
    return abstracts

def main(args):
    input_path = args.input_path
    output_path = args.output_path
    dfs = pickle.load(open(input_path, 'rb'))
    new_dfs = [add_abstract(df) for df in dfs]
    pickle.dump(new_dfs, open(output_path, 'wb'))

if __name__ == '__main__':
    argparser = ArgumentParser(description='Abstract expressions extracted from Kaggle scripts')
    argparser.add_argument('input_path', type=str, help='Path to pickled list of dataframes with extracted exprs')
    argparser.add_argument('output_path', type=str, help='Path to pickle new results')
    args = argparser.parse_args()
    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()



