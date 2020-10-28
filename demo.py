from transfer.build_db import (
    FunctionDatabase,
    NodeTypes,
    RelationshipTypes,
)
import pickle


def start(path="sample_db.pkl"):
    with open(path, "rb") as fin:
        db = pickle.load(fin)
    db.startup()
    return db
