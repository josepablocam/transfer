from transfer.build_db import (
    FunctionDatabase,
    NodeTypes,
    RelationshipTypes,
)
import pickle


def start():
    with open("sample_db.pkl", "rb") as fin:
        db = pickle.load(fin)
    db.startup()
    return db
