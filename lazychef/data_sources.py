"""Datasources are a way of creating a data pipeline.

They are used as a way of both describing the transformation of data and computing it.
"""

import os
from collections import Iterable
from abc import abstractmethod, ABCMeta
import numpy as np
from numbers import Integral
import h5py


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


class CachedDatasource(Datasource):
    FILE_SUFFIX = '.cache.hdf5'

    def __init__(
        self,
        data_source,
        cache_name
    ):
        """
        TODO
        Arguments:
            cache_name: path to the hdf5 file holding the cached data.
                If the name doesnt finish with .cache.hdf5, it's appended to the end of the name.
        """
        self.data_source = data_source
        self.__init_cache(cache_name)

    def __init_cache(self, cache_name):
        if not cache_name.endswith(self.FILE_SUFFIX):
            cache_name += self.FILE_SUFFIX
        self.cache = h5py.File(cache_name, 'a')

    def _process(self, ident):
        if ident in self.cache:
            return self.cache[ident][:]
        else:
            data = self.data_source[ident]
            self.cache[ident] = data
            return data


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
            * A list of bools (TODO)
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
    def __init__(self, data_source, f, size=None):
        self.data_source = data_source
        self.f = f

    def __len__(self):
        return len(self.data_source)

    def _process(self, idx):
        return self.f(self.data_source[idx])


class ArrayLikeAdressAdapter(ArrayDatasource):
    def __init__(self, data_source, index_fn, size=None):
        """
        TODO
        Arguments:
            * size: return value of len(ArrayLikeAdressAdapter).
                Only needed if `data_source` does not have a __len__ method
        """
        self.data_source = data_source
        self.index_fn = index_fn
        if not hasattr(data_source, '__len__') and size is None:
            raise RuntimeError(
                'Datasource: {} does not have a __len__ attribute and size is not given'.format(data_source)
            )
        self.__size = size

    def __len__(self):
        if self.__size is None:
            return len(self.data_source)
        else:
            return self.__size

    def _process(self, idx):
        return self.data_source[self.index_fn(idx)]


class CachedArrayDatasource(ArrayDatasource):
    CACHE_BITARRAY = '_bitarray'
    DATA_NAME = 'data'
    FILE_SUFFIX = '.cache.hdf5'

    def __init__(
        self,
        data_source,
        cache_name,
        size=None
    ):
        """
        TODO
        Arguments:
            size: size declared when `data_source` is a non-fixed-size data_source.
                If None, use len(data_source)
            cache_name: path to the hdf5 file holding the cached data.
                If the name doesnt finish with .cache.hdf5, it's appended to the end of the name.
        """
        self.data_source = data_source
        if size is None:
            size = len(data_source)
        self.size = size

        self.__init_cache(cache_name)

    def __len__(self):
        return self.size

    def __init_cache(self, cache_name):
        if not cache_name.endswith(self.FILE_SUFFIX):
            cache_name += self.FILE_SUFFIX
        cache = h5py.File(cache_name, 'a')

        if self.DATA_NAME not in cache:
            self.__create_cache(cache)

        self.cached_data = cache[self.DATA_NAME]
        self.existence_cache = cache[self.CACHE_BITARRAY]
        self.__update_cache_complete()
        # more here

    def __create_cache(self, cache):
        """
        Create the cache and the existence array
        """
        number_of_samples = len(self)
        example_input = self.data_source[0]
        cache.create_dataset(
            self.DATA_NAME,
            shape=(number_of_samples,) + example_input.shape,
            dtype=example_input.dtype
        )
        # currently not a bit array techincally
        cache.create_dataset(
            self.CACHE_BITARRAY,
            data=np.zeros(number_of_samples, dtype=np.uint8)
        )

    def _process(self, index):
        if self.existence_cache[index]:
            return self.cached_data[index]
        else:
            data = self.data_source[index]
            self.cached_data[index] = data
            self.existence_cache[index] = 1
            return data

    def __update_cache_complete(self):
        self.cache_complete = np.all(self.existence_cache)

    def _process_multiple(self, indices):
        idices = [neg_index_to_positive(idx, len(self)) for idx in indices]
        if self.cache_complete:
            return np.array([self.cached_data[idx] for idx in indices])
        else:
            data = np.array([self._process(idx) for idx in idices])
            self.__update_cache_complete()
            return data


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
