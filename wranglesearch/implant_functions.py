from abc import ABC, abstractmethod


class BaseTransfer(ABC):
    def __init__(self, schema_donor, schema_recipient):
        self.schema_donor = schema_donor
        self.schema_recipient = schema_recipient

    @abstractmethod
    def transfer_candidates(self, function):
        pass


def type_based_candidates(schema_1, schema_2):
    # create a mapping based on type
    # for each column in schema_1
    # -> columns in schema_2 of the same type
    pass


def possible_column_mappings(cols, schema_1, schema_2):
    pass


def add_column_renaming_prefix(mapping, function):
    pass


class TypeTransfer(BaseTransfer):
    def __init__(self, schema_1, schema_2):
        super().__init__(schema_1, schema_2)

    def transfer_candidates(function):
        cols_used = function.columns_used
        mappings = possible_column_mappings(
            cols_used, self.schema_1, self.schema_2
        )
        candidates = []
        for _map in mappings:
            candidate = add_column_renaming_prefix(_map, function)
            candidates.append(candidate)
        return candidates


class DistanceTransfer(object):
    def __init__(self, distance_functions):
        # map from column type to distance function
        self.distance_functions = distance_functions

    def transfer_candidates(function):
        # create mapping such that distance is minimized
        raise NotImplementedError('jose needs to write this')
