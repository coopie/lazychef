"""Datasources are a way of creating a data pipeline.

They are used as a way of both describing the transformation of data and computing it.
"""

import os
from collections import Iterable
from abc import abstractmethod, ABCMeta
import numpy as np
from numbers import Integral


class Datasource(metaclass=ABCMeta):
    """Base class for all Datasources.

    Note that the default __getitem__ returns a generator (for lazier evaluation).
    """

    def __getitem__(self, key):
        if is_array_like(key):
            return [self._process(uri) for uri in key]
        else:
            return self._process(key)

    @abstractmethod
    def _process(self, ident):
        pass


class AddressAdapter(Datasource):
    """Used to change the naming scheme bewteen two Datasources."""

    def __init__(self, data_source, adapt_ident):
        self.adapt_ident = adapt_ident
        self.data_source = data_source

    def _process(self, ident):
        adapted = self.adapt_ident(ident)
        return self.data_source[adapted]


class FileDatasource(Datasource):
    """Wrapper around file system"""

    def __init__(self, base_dir='', suffix=''):
        self.base_dir = base_dir
        self.suffix = suffix

    def _process(self, ident):
        path_to_file = os.path.join(self.base_dir, ident + self.suffix)
        assert os.path.exists(path_to_file), path_to_file + ' does not exist'
        return path_to_file


class LambdaDatasource(Datasource):
    def __init__(self, data_source, function):
        self.function = function
        self.data_source = data_source

    def _process(self, ident):
        return self.function(self.data_source[ident])


class ArrayDatasource(metaclass=ABCMeta):
    """
    TODO: why this
    Array Datasources must have a fixed size at creation.
    """
    def __getitem__(self, key):
        """
        Key can be one of:
            * An integer
            * A list of integers
            * A list of bools (not supported yet)
            * A slice
        """
        if is_int_like(key):
            return self._process(key)
        elif is_array_like(key):
            return self._process_multiple(key)
        elif type(key) == slice:
            return self._process_multiple(slice_to_range(key, len(self)))
            pass
        else:
            raise RuntimeError('Key: {} is not compatible with this ArrayDatasource'.format(str(key)))

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def _process(self, idx):
        pass

    def _process_multiple(self, indices):
        idices = [neg_index_to_positive(idx, len(self)) for idx in indices]
        return np.array([self._process(idx) for idx in idices])


class LambdaArrayDatasource(ArrayDatasource):
    def __init__(self, data_source, f):
        self.data_source = data_source
        self.f = f

    def __len__(self):
        return len(self.data_source)

    def _process(self, idx):
        return self.f(self.data_source[idx])


def neg_index_to_positive(idx, length):
    if idx < 0:
        return length + idx
    else:
        return idx


def is_array_like(x):
    return isinstance(x, Iterable) and type(x) != str


def is_int_like(x):
    return isinstance(x, Integral) or (isinstance(x, np.ndarray) and np.shape(x) == ())


def slice_to_range(s, max_value):
    start = s.start if s.start is not None else 0
    stop = s.stop if s.stop is not None else max_value
    step = s.step if s.step is not None else 1
    return range(start, stop, step)


def merge_dicts(*dicts):
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged
