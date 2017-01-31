import unittest
import os
import numpy as np
import h5py

from lazychef.data_sources import Datasource
from lazychef.legacy_data_sources import (
    CachedTTVArrayLikeDatasource, TTVArrayLikeDatasource
)
from test.util import yaml_to_dict

DUMMY_DATA_PATH = os.path.join('test', 'dummy_data')


class DatasourcesTests(unittest.TestCase):

    def test_ttv_array_like_data_source(self):
        dummy_data_source = DummyDatasource()
        subject_info_dir = os.path.join('test', 'dummy_data', 'metadata')
        ttv = yaml_to_dict(os.path.join(subject_info_dir, 'dummy_ttv.yaml'))

        array_ds = TTVArrayLikeDatasource(dummy_data_source, ttv)

        self.assertEqual(len(array_ds), 3)

        all_values = np.fromiter((x for x in array_ds[:]), dtype='int16')

        self.assertTrue(
            np.all(
                np.in1d(
                    all_values,
                    np.array([1, 2, 3])
                )
            )
        )

    def test_subarray_like_data_source(self):
        dummy_data_source = DummyDatasource()
        subject_info_dir = os.path.join('test', 'dummy_data', 'metadata')
        ttv = yaml_to_dict(os.path.join(subject_info_dir, 'dummy_ttv.yaml'))

        array_ds = TTVArrayLikeDatasource(dummy_data_source, ttv)

        def get_all_values_set(ttv, set_name):
            data_set = ttv[set_name]
            uris = []
            for subjectID in data_set:
                uris += data_set[subjectID]
            return uris

        for set_name in ['test', 'train', 'validation']:
            set_ds = array_ds.get_set(set_name)

            self.assertTrue(len(set_ds), 1)

            self.assertEqual(
                [x for x in set_ds[:]],
                [dummy_data_source[x] for x in get_all_values_set(ttv, set_name)]
            )

    def test_cached_ttv_array_like_data_source(self):
        dummy_data_source = DummyDatasource()
        subject_info_dir = os.path.join('test', 'dummy_data', 'metadata')
        ttv = yaml_to_dict(os.path.join(subject_info_dir, 'dummy_ttv.yaml'))

        array_ds = CachedTTVArrayLikeDatasource(dummy_data_source, ttv, data_name='dummy', cache_name='test')

        self.assertEqual(len(array_ds), 3)

        all_values = array_ds[:]

        self.assertTrue(
            np.all(
                np.in1d(
                    all_values,
                    np.array([1, 2, 3])
                )
            )
        )

        f = h5py.File('test.cache.hdf5', 'a')
        self.assertEqual(len(f['dummy']), len(array_ds))

        for in_cache, in_data_source in zip(f['dummy'], array_ds):
            self.assertTrue(
                np.all(
                    in_cache == in_data_source
                )
            )

        # changing a value in the cache now should alter the results returned by the dataset.
        # adressing is to change the value of the test example, which is currently set to 2
        f['dummy'][array_ds.get_set('test').lower] = 322
        all_values = all_values = array_ds[:]
        self.assertTrue(
            np.all(
                np.in1d(
                    all_values,
                    np.array([322, 1, 3])
                )
            )
        )

        # now resetting the cache, we shoud get the original results
        f['dummy' + CachedTTVArrayLikeDatasource.CACHE_BITARRAY_SUFFIX][:] = False
        array_ds._CachedTTVArrayLikeDatasource__init_existence_cache()

        all_values = array_ds[:]
        self.assertTrue(
            np.all(
                np.in1d(
                    all_values,
                    np.array([1, 2, 3])
                )
            )
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('test.cache.hdf5'):
            os.remove('test.cache.hdf5')


def dummy_process_waveforms(path):
    """A way of identifying which file the dummy waveform comes from."""
    filename_split = path.split(os.sep)[-1].split('.')[0].split('_')
    ident = filename_split[0]
    is_happy = filename_split[1] == 'happy'

    frequency = 123

    return (frequency, np.array([int(ident) * 2]) + int(is_happy))


def dummy_process_spectrograms(waveform, *unused):
    times = np.array([1, 2])
    frequencies = np.array([3, 4])
    return (frequencies, times, np.eye(2) * waveform)


class dummyExampleDatasource():
    def __init__(self, arr):
        self.arr = arr

    def get_set(self, set_name):
        set_division = {
            'test': [1],
            'train': [2, 3],
            'validation': [4]
        }
        return dummyExampleDatasource(self.arr[set_division[set_name]])

    def __getitem__(self, key):
        return self.arr[key, 0], self.arr[key, 1]


class DummyDatasource(Datasource):
    def __init__(self):
        self.data = {
            'blorp_2': np.array(1) * 1,
            'blerp_1': np.array(1) * 2,
            'shlerp_322': np.array(1) * 3
        }

    def _process(self, key):
        return self.data[key]


if __name__ == '__main__':
    unittest.main()
