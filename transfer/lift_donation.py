import ast
from collections import defaultdict
import textwrap

from astunparse import unparse
import networkx as nx
import pandas as pd
from plpy.analyze.dynamic_trace_events import ExecLine

from . import identify_donation


### Parameters and Return values ###
def comment_out_nodes(graph, predicate):
    for _, node_data in graph.nodes(data=True):
        node_data['treat_as_comment'] = predicate(node_data)
    return graph
        
def is_pandas_read(node_data):
    calls = node_data['calls']
    if calls is None:
        return False
    for call_event in calls:
        if call_event.details['qualname'] == pd.read_csv.__qualname__:
            return True
    return False

# for example: read_*
def annotate_dataframe_uses(graph):
    for _, node_data in graph.nodes(data=True):
        node_data['dataframe_uses'] = set([])
        if node_data['treat_as_comment']:
            continue
        if isinstance(node_data['event'], ExecLine):
            uses = node_data['uses']
            df_uses = [u for u in uses if u.type == pd.DataFrame.__name__]
            node_data['dataframe_uses'].update(df_uses)
    return graph

def annotate_dataframe_defs(graph):
    for _, node_data in graph.nodes(data=True):
        node_data['dataframe_defs'] = set([])
        if node_data['treat_as_comment']:
            continue
        if isinstance(node_data['event'], ExecLine):
            # TODO: this currently isn't here....
            defs = node_data['defs']
            df_defs = [d for d in defs if d.type == pd.DataFrame.__name__]
            node_data['dataframe_defs'].update(df_defs)
    return graph
    
def node_data_in_exec_order(graph):
    return sorted(graph.nodes(data=True), key=lambda x: x[1]['event'].event_id)

def get_free_input_dataframes(graph):
    # => function inputs
    # sort graph by lineno
    global_free = set([])
    defined = set([])
    sorted_by_exec = node_data_in_exec_order(graph)
    for _, node_data in sorted_by_exec:
        free = [var for var in node_data['dataframe_uses'] if not var in defined]
        global_free.update(free)
        defined.update(node_data['dataframe_defs'])
    return free
    
def get_created_dataframes(graph):
    created = set([])
    for _, node_data in graph.nodes(data=True):
        created.update(node_data['dataframe_defs'])
    return created


### Adding user code that we may not get to observe through trace ###
# collects functions and user class definitions
class CollectUserDefinedCallableNames(ast.NodeVisitor):
    def __init__(self):
        self.names = []
        
    def run(self, tree):
        self.visit(tree)
        return self.names
        
    def visit_ClassDef(self, node):
        self.names.append(node.name)
    
    def visit_FunctionDef(self, node):
        # we ignore things that may be private by convention
        if not node.name.startswith('_'):
            self.names.append(node.name)

# collect body (and dependencies) for user callables
class CollectUserDefinedCallableBodies(ast.NodeVisitor):
    def __init__(self, names):
        self.names = names
        self.name_to_body = {}
        self.name_to_dependencies = defaultdict(lambda: set([]))
        self.inside_callable = []
        self.name_to_complete_code = defaultdict(lambda: [])
        
    def run(self, tree):
        self.visit(tree)
        self._populate_user_def_code()
        
    def visit_Name(self, node):
        if self.inside_callable and node.id in self.names:
            enclosing_name = self.inside_callable[-1]
            self.name_to_dependencies[enclosing_name].append(node.id)
    
    def visit_FunctionDef(self, node):
        self.name_to_body[node.name] = unparse(node)
        self.inside_callable.append(node.name)
        self.generic_visit(node)
        self.inside_callable.pop()
        
    def visit_ClassDef(self, node):
        self.name_to_body[node.name] = unparse(node)
        self.inside_callable.append(node.name)
        self.generic_visit(node)
        self.inside_callable.pop()
        
    def _populate_user_def_code0(self, name):
        # this is recursive for now, for clarity, but Python
        # can hit recursion error pretty easily. shouldn't be an issue
        # here since the definitions are likely not to be deeply nested
        if name in self.name_to_complete_code:
            return self.name_to_complete_code[extra]
        else:
            code = []
            code.append(self.name_to_body[name])
    
            depends_on = self.name_to_dependencies[name]
            for dep in depends_on:
                code.extend(self._populate_user_def_code0(dep))
    
            self.name_to_complete_code[name] = code
            
    def _populate_user_def_code(self):
        for name in self.names:
            self._populate_user_def_code0(name)

    def get_code(self, names):
        if isinstance(names, str):
            names = [names]
        user_names = set(names).intersection(self.names)

        code = set([])
        for name in user_names:
            code.update(self.name_to_complete_code[name])
        return list(code)


def build_user_code_map(ast):
    names = CollectUserDefinedCallableNames().run(ast)
    code_map = CollectUserDefinedCallableBodies(names)
    code_map.run(ast)
    return code_map

# TODO: we should make this so that it can also consider class, currently just function
def get_user_defined_functions(graph, user_code_map):
    assert isinstance(user_code_map, CollectUserDefinedCallableBodies)
    names = set([])
    for _, node_data in graph.nodes(data=True):
        names.update([u.name for u in node_data['uses'] if u.type == 'function'])
    return user_code_map.get_code(names)
    
def graph_to_code_block(graph):
    sorted_nodes = node_data_in_exec_order(graph)
    code = []
    for _, node_data in sorted_nodes:
        if not node_data['treat_as_comment']:
            code.append(node_data['src'])
    return '\n'.join(code)

### the donated function class ###
class DonatedFunction(object):
    def __init__(self, name, formal_arg_vars, return_var, cleaning_code, context_code=''):
        self.name = name
        self.formal_arg_vars = formal_arg_vars
        self.return_var = return_var
        # source code
        self.cleaning_code = cleaning_code
        self.context_code = context_code
        self.code = None
        # executable function
        self._obj = None
    
    def get_source(self):
        if self.code is None:
            formal_args_str = ','.join([a.name for a in self.formal_arg_vars])
            # TODO: only add context code if we need it
            # TODO: this currently fucks up because the indent is not going to be correct
            src = f"""
            # cleaning function extracted from user traces
            def {self.name}({formal_args_str}):
                # context code: user definitions for called functions/classes
                {self.context_code}
                # core cleaning code
                {self.cleaning_code}
                return {self.return_var.name}
            """
            src = textwrap.dedent(src)
            self.src = src
        return src

    def _get_code_obj(self):
        if self._obj is None:
            code = self.get_source()
            _globals = {}
            exec(code, _globals)
            self._obj = _globals[self.name]
        return self._obj.__code__
        
    def __call__(self, *args):
        fun_code = self._get_code_obj()
        return fun_code(*args)


def lift_graph_to_functions(graph, script_src):
    # create copy before we annotate/modify
    graph = graph.to_directed()

    # Graph annotation
    # we remove reads and create free (dataframe) variables as a result
    graph = comment_out_nodes(graph, is_pandas_read)
    # annotate with uses/defs for dataframes
    graph = annotate_dataframe_uses(graph)
    graph = annotate_dataframe_defs(graph)
    
    # free dataframe variables become arguments for the function
    formal_args = get_free_input_dataframes(graph)
    # dataframe variables created or the original input
    possible_returns = get_created_dataframes(graph)
    # can always return the original input data frame
    # TODO: this really should only happen if we modify the original input
    possible_returns.update(formal_args)

    # we need to add user code for functions that may get executed but we don't directly trace
    tree = ast.parse(script_src)
    user_code_map = build_user_code_map(tree)
    
    additional_code = get_user_defined_functions(graph, user_code_map)
    additional_code = '\n'.join(additional_code)
    # collect code block and ignore comment nodes
    cleaning_code = graph_to_code_block(graph)

    # since there are multiple possible return types, we lift into multiple functions
    funcs = []
    for _return in possible_returns:
        f = DonatedFunction(name, formal_args, _return, cleaning_code, additional_code)
        funcs.append(f)
    return funcs


