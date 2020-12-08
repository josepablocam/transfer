from IPython.core.magic import (
    Magics,
    magics_class,
    line_magic,
    cell_magic,
    line_cell_magic,
)

import pickle


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        from transfer.build_db import (
            FunctionDatabase,
            NodeTypes,
            RelationshipTypes,
        )
        db_classes = {
            "FunctionDatabase": FunctionDatabase,
            "NodeTypes": NodeTypes,
            "RelationshipTypes": RelationshipTypes,
        }
        if name in db_classes:
            return db_classes[name]
        return super().find_class(module, name)


def start(path="sample_db.pkl"):
    with open(path, "rb") as fin:
        db = CustomUnpickler(fin).load()
    db.startup()
    return db


@magics_class
class TransferMagics(Magics):
    def __init__(self, shell, db):
        super().__init__(shell)
        self.db = db

    # @line_magic
    # def tquery(self, line):
    #     print("Full access to the main IPython object:", self.shell)
    #     print(
    #         "Variables in the user namespace:",
    #         list(self.shell.user_ns.keys())
    #     )
    #     query = line.split()
    #     n = 2
    #     query_results = self.db.query(query)[:n]
    #     code_results = [self.db.get_code(r) for r in query_results]
    #     return code_results

    @line_cell_magic
    def tquery(self, line, cell=None):
        if cell is None:
            query = line
        else:
            query = cell
        query = query.strip()
        query = query.split()
        possible_result_ix = query[-1]
        try:
            result_ix = int(possible_result_ix)
        except ValueError:
            result_ix = 1

        # try to see if any strings are actually objects
        clean_query = []
        for term in query:
            try:
                obj = eval(term, self.shell.user_ns)
                clean_query.append(obj)
            except:
                clean_query.append(term)

        try:
            query_results = self.db.query(clean_query)
        except:
            print("Lookup failed...")
            return

        num_avail = len(query_results)
        if num_avail == 0:
            print("No snippets available")
            return

        result_ix = min(result_ix, num_avail)
        query_result = query_results[result_ix - 1]

        code = self.db.get_code(query_result).replace("\t", "  ")
        self.shell.set_next_input(
            '# query={}, Snippet={}/{} \n{}'.format(
                query, result_ix, num_avail, code
            ),
            replace=False
        )


def load_ipython_extension(ipython):
    db = start()
    tdb = TransferMagics(ipython, db)
    ipython.register_magics(tdb)
