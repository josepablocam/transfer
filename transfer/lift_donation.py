from argparse import ArgumentParser
import ast
from collections import defaultdict
import pickle
import textwrap

from astunparse import unparse
import networkx as nx
import pandas as pd
from plpy.analyze.dynamic_trace_events import ExecLine

from . import identify_donation


### Parameters and Return values ###
def comment_out_nodes(graph, predicate):
    # predicate can check things like if it reads from disk
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

class GetNames(ast.NodeVisitor):
    def __init__(self):
        self.acc = set([])

    def visit_Name(self, node):
        self.acc.add(node.id)

    def run(self, node):
        self.visit(node)
        return self.acc

def can_be_computed(var, defined):
    var_tree = ast.parse(var.name)
    uses_names = GetNames().run(var_tree)
    defined_names = set(d.name for d in defined)
    return uses_names.issubset(defined_names)

def get_free_input_dataframes(graph):
    global_free = set([])
    defined = set([])
    sorted_by_exec = node_data_in_exec_order(graph)
    for _, node_data in sorted_by_exec:
        if node_data['treat_as_comment']:
            continue
        for var in node_data['dataframe_uses']:
            if not var in defined and not can_be_computed(var, defined):
                global_free.add(var)
        # update with all definitions, not just dataframes
        defined.update(node_data['defs'])
    return global_free

def get_created_dataframes(graph):
    # TODO: this should likely also include modified functinos
    created = {}
    sorted_by_exec = node_data_in_exec_order(graph)
    for _, node_data in sorted_by_exec:
        if node_data['treat_as_comment']:
            continue
        # just use names for this, don't care about memory location
        # take last reference to each name
        def_names = {d.name: d for d in node_data['dataframe_defs']}
        created.update(def_names)
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
            return code

    def _populate_user_def_code(self):
        for name in self.names:
            self._populate_user_def_code0(name)

    def run(self, tree):
        self.visit(tree)
        self._populate_user_def_code()

    def get_code(self, names):
        if isinstance(names, str):
            names = [names]
        # portion of names that (may) be user-defined functions and should be added
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

def graph_to_lines(graph):
    sorted_nodes = node_data_in_exec_order(graph)
    code = []
    for _, node_data in sorted_nodes:
        line = '# ' if node_data['treat_as_comment'] else ''
        line += node_data['src']
        code.append(line)
    return code

### the donated function class ###
# TODO: we should add the actual graph (we may want to retrieve this later)
# we should also add some meta information, like what column is the seed for this slice etc
class DonatedFunction(object):
    def __init__(self, graph, name, formal_arg_vars, return_var, cleaning_code, context_code=None):
        # make sure to have a copy of the graph backing thsi function
        self.graph = graph.to_directed()

        # function definition
        self.name = name
        self.formal_arg_vars = list(formal_arg_vars)
        self.return_var = return_var
        # source code
        self.cleaning_code = cleaning_code
        self.context_code = context_code
        self.code = None

        # executable function
        self._obj = None

    # FIXME: this is nasty nasty
    def get_source(self):
        if self.code is None:

            context_code_str = '# no additional context code'
            if self.context_code:
                context_code_str = f'#additional context code from user definitions'
                for _def in self.context_code:
                    context_code_str += textwrap.dedent(_def)
            context_code_str = textwrap.indent(context_code_str, '\t')

            template = """
            # cleaning function extracted from user traces
            def {name}({formal_args_str}):
            {context_code_str}
            \t# core cleaning code
            {core_code_str}
            """
            template = textwrap.dedent(template)
            formal_args_str = ','.join([a.name for a in self.formal_arg_vars])
            core_code = self.cleaning_code + ['return {}'.format(self.return_var.name)]
            core_code_str = '\n'.join(core_code)
            core_code_str = textwrap.indent(core_code_str, '\t')
            # add in return

            code = template.format(
                name=self.name,
                formal_args_str=formal_args_str,
                context_code_str=context_code_str,
                core_code_str=core_code_str
            )

            self.code = code
        return self.code

    def _get_func_obj(self):
        if self._obj is None:
            code = self.get_source()
            _globals = {}
            exec(code, _globals)
            self._obj = _globals[self.name]
        return self._obj

    def __call__(self, *args):
        func = self._get_func_obj()
        return func(*args)


def lift_to_functions(graphs, script_src, name_format=None, name_counter=None):
    # we need to add user code for functions that may get executed but we don't directly trace
    tree = ast.parse(script_src)
    user_code_map = build_user_code_map(tree)

    if isinstance(graphs, nx.DiGraph):
        graphs = [graphs]

    if name_format is None:
        name_format = 'cleaning_function_%d'

    if name_counter is None:
        name_counter = 0

    functions = []

    for graph in graphs:
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
        # FIXME: we also want returns that don't waste compute
        # i.e. if we decide to return X, we should not include computation
        # after that (maybe just comment it out?)
        # can always return the original input data frame
        possible_returns = {d.name:d for d in formal_args}
        possible_returns.update(get_created_dataframes(graph))
        # we can also remove any references to the same object with different names
        possible_returns = {v.id:v for v in possible_returns.values()}
        # and we can then just keep the names
        possible_returns = possible_returns.values()

        additional_code = get_user_defined_functions(graph, user_code_map)
        # collect code block and ignore comment nodes
        cleaning_code = graph_to_lines(graph)

        # since there are multiple possible return types, we lift into multiple functions
        for _return in possible_returns:
            func_name = name_format % name_counter
            f = DonatedFunction(graph, func_name, formal_args, _return, cleaning_code, additional_code)
            functions.append(f)

    return functions

def main(args):
    graph_file = args.graph_file
    src_file = args.src_file
    output_file = args.output_file
    name_format = args.name_format
    name_counter = args.name_counter

    with open(graph_file, 'rb') as f:
        graphs = pickle.load(f)

    with open(src_file, 'r') as f:
        script_src = f.read()

    functions = lift_to_functions(graphs, script_src, name_format=name_format, name_counter=name_counter)
    print('Collected {} functions'.format(len(functions)))

    with open(output_file, 'wb') as f:
        pickle.dump(functions, f)

if __name__ == '__main__':
    parser = ArgumentParser(description='Lift collection of donation graphs to functions')
    parser.add_argument('graph_file', type=str, help='Path to pickled donation graphs')
    parser.add_argument('src_file', type=str, help='Path to original user script to collect additional defs')
    parser.add_argument('output_file', type=str, help='Path to store pickled functions')
    parser.add_argument('-n', '--name_format', type=str, help='String formatting for function name', default='cleaning_func_%d')
    parser.add_argument('-c', '--name_counter', type=int, help='Initialize function counter for naming', default=0)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()


# TODOS:
# 1 - Try this with other scripts
# 2 - Fix so that it works with user defined classes (may need to tweak type annotations during tracing)
# 3 - Try to lift without input args (i.e. don't comment reads out): => basic function executable check
# 4 - Try lifting more
# 5 - Start tests for this project: (convert candidates, identify donations, and lift) we'll live without tests for collect (since browser based)
# 6 - There is additional work that needs to happen for repairing existing code: we may need to change references to columns to potentially new columns
# in the code provided

