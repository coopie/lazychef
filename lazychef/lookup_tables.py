"""LookupTable is an adapter from one adressing space to another.

Typically this means changing from a uri adress space to a integer adress space.
"""

from abc import abstractmethod
import numpy as np
import random
from random import shuffle
from collections import Iterable


class LookupTable(object):
    """Base class for lookuptables"""

    @abstractmethod
    def __getitem__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class TTVLookupTable(LookupTable):
    """Use a ttv split to create a lookup table.
    TTV can have one or more of 'test', 'train', 'validation' keys representing datasets

    Also exposes slices of the lookup table corresponding to different data sets(i.e. test, train, validation sets).

    The shuffle seed needs to be the same for each lookup creation for the same ttv.
    """

    def __init__(self, ttv, shuffle_in_set=False):
        """
        Create a lookup table from a TTV split.

        the shuffle should be deterministic, i.e. shuffling the same ttv should return the same resut.
        """
        self.__indices_for_sets = {}
        lookup_table = []

        index = 0
        data_sets = sorted([x for x in ttv if x in ['train', 'validation', 'test']])
        for data_set in data_sets:

            start_index = index
            uris_for_set = flatten_dict(ttv[data_set])

            index += len(uris_for_set)
            end_index = index
            self.__indices_for_sets[data_set] = (start_index, end_index)

            if shuffle_in_set:
                random.seed(1337)
                shuffle(uris_for_set)
            lookup_table += uris_for_set

        self.uris = np.array(lookup_table)

    def __getitem__(self, key):
        x = self.uris[key]
        if isinstance(x, np.str):
            return str(x)
        else:
            return [str(y) for y in x]

    def __len__(self):
        return len(self.uris)

    def get_set_bounds(self, set_name):
        """Return the slice index slice for a certain data set"""
        return self.__indices_for_sets[set_name]


def flatten_dict(x):
    """
    Flatten dictionary to the lowest level. x can be wither a string or list (leaf nodes), or a dictionary containing
    lists or strings.
    """
    flattened = []

    if type(x) == str:
        flattened += [x]
    elif type(x) == dict:

        keys = sorted([k for k in x])
        for key in keys:
            flattened += flatten_dict(x[key])
    elif is_array_like(x):

        flattened += x
    else:
        raise ValueError('Elements can only be string or dictionaries')

    return flattened


def is_array_like(x):
    return isinstance(x, Iterable) and type(x) != str
