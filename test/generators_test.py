import unittest

import numpy as np
import h5py
import os

from lazychef.generators import Generator, DatasetGenerator, ShuffleDatasetCallback

TEST_FILE = 'generators.data.hdf5'


class TestGenerators(unittest.TestCase):

    def test_dataset_exists(self):
        with h5py.File(TEST_FILE, 'r') as f:
            self.assertTrue('data' in f)

    def test_fitting_batch_size(self):
        f = h5py.File(TEST_FILE, 'r')
        gen = Generator([f['data']], batch_size=1)
        a = []
        for i in range(1, 4):
            a.append(gen.__next__()[0])

        self.assertEqual(
            len(a),
            3
        )

        expected_result = [(np.repeat((i % 3) + 1, 3), np.repeat((i % 3) + 1, 3)) for i in range(3)]
        for actual, expected in zip(a, expected_result):
            self.assertTrue(
                np.all(expected == actual)
            )

    def test_not_well_fitting_batch_size(self):
        f = h5py.File(TEST_FILE, 'r')
        gen = DatasetGenerator([f['data']], batch_size=2)
        a = []
        for i in range(2):
            a.append(next(gen)[0])

        self.assertEqual(
            len(a),
            2
        )

        x = np.array([np.repeat(1, 3), np.repeat(2, 3)])
        expected_result = [x, x]

        for actual, expected in zip(a, expected_result):
            self.assertTrue(
                np.array_equal(
                    expected, actual
                )
            )

    def test_labeled_generator(self):
        f = h5py.File(TEST_FILE, 'r')
        gen = DatasetGenerator([f['data'], f['labels']], batch_size=3)

        X, Y = next(gen)
        np.testing.assert_equal(
            Y,
            np.array([[x] for x in range(1, 4)])
        )
        np.testing.assert_equal(
            X,
            np.array([np.repeat(x, 3) for x in range(1, 4)])
        )

    def test_shuffling(self):
        f = h5py.File(TEST_FILE, 'r')

        shuffle_callback = ShuffleDatasetCallback(seed=1237)
        gen = DatasetGenerator([f['data']], batch_size=1, callbacks=[shuffle_callback])
        a = []
        for i in range(6):
            a.append(next(gen)[0])

        self.assertEqual(
            len(a),
            6
        )
        self.assertFalse(
            np.array_equal(
                np.array(a[:3]),
                np.array(a[3:])
            )
        )

    @classmethod
    def setUpClass(cls):
        f = h5py.File(TEST_FILE, 'w')
        data = np.array([np.repeat(x, 3) for x in range(1, 4)])
        f['data'] = data

        labels = np.array([[x] for x in range(1, 4)])
        f['labels'] = labels

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_FILE):
            os.remove(TEST_FILE)


if __name__ == '__main__':
    unittest.main()
