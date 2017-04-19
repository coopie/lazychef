import unittest
import os
import numpy as np
import h5py

from lazychef.data_sources import (
    Datasource,
    LambdaDatasource,
    ArrayDatasource,
    FileDatasource,
    CachedArrayDatasource,
    CachedDatasource
)


DUMMY_DATA_PATH = os.path.join('test', 'dummy_data')
CACHE_ARRAY_PATH = 'test_array.cache.hdf5'
CACHE_PATH = 'test.cache.hdf5'


class DatasourcesTests(unittest.TestCase):

    def test_file_data_source(self):
        ds = FileDatasource(DUMMY_DATA_PATH, suffix='.wav')

        self.assertTrue(
            os.path.isfile(ds['1_sad_kid_1'])
        )
        self.assertEqual(
            ds['1_sad_kid_1'],
            os.path.join(DUMMY_DATA_PATH, '1_sad_kid_1.wav')
        )

        filenames = (x.split('.')[0] for x in os.listdir(DUMMY_DATA_PATH) if x.endswith('wav'))
        self.assertTrue(
            all(
                [os.path.exists(ds[f]) for f in filenames]
            )
        )

    def test_lambda_data_source(self):
        data_source = DummyDatasource()

        lam_ds = LambdaDatasource(data_source, lambda x: x + 1)

        for key in ['blorp_2', 'blerp_1', 'shlerp_322']:
            self.assertEqual(
                lam_ds[key],
                data_source.data[key] + 1
            )

    def test_cached_data_source(self):
        y = {
            str(i): np.repeat(i, i)
            for i in range(10)
        }
        x = CachedDatasource(y, CACHE_PATH)

        data = [x[str(i)] for i in range(10)]
        data2 = [x[str(i)] for i in range(10)]

        for d, d2 in zip(data, data2):
            np.testing.assert_array_equal(d, d2)

        # change underlying data
        for i in range(10):
            y[str(i)] = np.array([123])

        for cached, new in zip(
            [x[str(i)] for i in range(10)],
            [y[str(i)] for i in range(10)]
        ):
            assert not np.array_equal(cached, new)

    def test_array_data_source(self):
        data = np.arange(10)
        data_source = DummyArrayDatasource(data)

        for i in range(len(data_source)):
            self.assertEqual(data_source[i], data[i])

        self.assertTrue(
            np.array_equal(
                data_source[:],
                data
            )
        )
        self.assertTrue(
            np.array_equal(
                data_source[-1],
                data[-1]
            )
        )
        self.assertTrue(
            np.array_equal(
                data_source[1:4:2],
                data[1:4:2]
            )
        )

    def test_cached_array_datasource(self):
        array_ds = np.ones((100, 2)) * np.arange(100).reshape((-1, 1))
        ds = CachedArrayDatasource(array_ds, CACHE_ARRAY_PATH, 100)
        assert not ds.cache_complete

        cache = h5py.File(CACHE_ARRAY_PATH, 'a')

        data = ds[:]
        assert np.array_equal(array_ds, data), 'First values from ds not same as data'
        assert ds.cache_complete, 'datasource should be cache_complete'

        # Now the cache should be instantiated, so changing a value in it should show
        # up in the datasource
        cache[CachedArrayDatasource.DATA_NAME][0, 0] = 123
        assert np.array_equal(ds[0], np.array([123, 0]))

        # test unorderd indexing, as h5py does not support it
        ds[[5, 4, 3, 2, 1]]

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(CACHE_ARRAY_PATH):
            os.remove(CACHE_ARRAY_PATH)
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)


class DummyDatasource(Datasource):
    def __init__(self):
        self.data = {
            'blorp_2': np.array(1) * 1,
            'blerp_1': np.array(1) * 2,
            'shlerp_322': np.array(1) * 3
        }

    def _process(self, key):
        return self.data[key]


class DummyArrayDatasource(ArrayDatasource):
    def __init__(self, a):
        self.a = a

    def _process(self, idx):
        return self.a[idx]

    def __len__(self):
        return len(self.a)


if __name__ == '__main__':
    unittest.main()
