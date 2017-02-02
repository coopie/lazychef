"""
Classes to wrap Datasource objects around to make generators for mini-batch training.
"""
import numpy as np
import logging


class Generator(object):
    """Base class for a generator.
    """

    def __init__(self, data_sources, batch_size=128):
        """TODO:."""
        self.data_sources = data_sources
        self.chunk_index = 0
        self.batch_size = batch_size

    def _next_batch(self):
        data = [
            ds[self.chunk_index:self.chunk_index + self.batch_size]
            for ds in self.data_sources
        ]
        self.chunk_index += self.batch_size
        return data

    def __next__(self):
        return self._next_batch()

    def __call__(self):
        return next(self)


class DatasetGenerator(Generator):
    """Generator for fixed-size datasets."""

    def __init__(
        self,
        data_sources,
        batch_size,
        callbacks=[]
    ):
        """
        Arguments:
            `data_sources`: These must all have the same length as each other.
        """
        assert len(data_sources) > 0, 'Must have at least one Datasource in generator.'
        assert all([len(data_sources[0]) == len(ds) for ds in data_sources]), \
            'Datasources are not all the same length.'

        super().__init__(data_sources, batch_size=batch_size)
        self.shuffle_indices = np.arange(len(data_sources[0]))
        self.callbacks = callbacks

    def _next_batch(self):
        indices = self.shuffle_indices[self.chunk_index:self.chunk_index + self.batch_size]
        data = [
            ds[list(indices)]  # for compatibility with h5py
            for ds in self.data_sources
        ]
        self.chunk_index += self.batch_size
        return data

    def __next__(self):
        if self.chunk_index == len(self):
            self.chunk_index = 0
            for cb in self.callbacks:
                cb.on_epoch_end(self)

        return self._next_batch()

    def __len__(self):
        """Return the size of the data, to the nearest `batch_size`"""
        return len(self.data_sources[0]) - (len(self.data_sources[0]) % self.batch_size)


class Callback():
    def on_epoch_end(*args):
        pass


class ShuffleDatasetCallback(Callback):
    """Shuffle the dataset every epoch."""
    def __init__(self, seed=None):
        if seed is None:
            seed = np.random.random_integers(0, 2 ** 20, 1)[0]
        self.random = np.random.RandomState(seed)

    def on_epoch_end(self, generator, *args):
        self.random.shuffle(generator.shuffle_indices)


class LogEpochEndCallback(Callback):
    """Shuffle the dataset every epoch."""
    def __init__(self, epochs_completed=0):
        self.epochs_completed = epochs_completed

    def on_epoch_end(self, *args):
        self.epochs_completed += 1
        logging.info('Compeleted epoch {}'.format(self.epochs_completed))
