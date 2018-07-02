from keras.preprocessing.image import ImageDataGenerator, Iterator
from tensorflow.python.keras._impl.keras import backend as K

import numpy as np

class StateGenerator(ImageDataGenerator):
    '''
    Generator for Tak board states
    '''

    def flow(self, X, y=None, batch_size=32, shuffle=False, seed=None, flip_prob=0, rotate_prob=0):
        return StateIterator(X, y, batch_size=batch_size, shuffle=shuffle, seed=seed, flip_prob=flip_prob, rotate_prob=rotate_prob)

class StateIterator(Iterator):

    '''Iterator for StateDataGenerator
    Arguments
        X: Numpy array, the states. Should have rank 3.
        y: Numpy array, the q-values.  Should have rank 1.
        batch_size: int, minibatch size. (default 32)
        shuffle: Boolean, shuffle with each epoch
        seed: random seed.
        flip_prob: float Set[0,1] probability of horizontally flipping generated state
        rotate_prob: float Set[0,1] probability of rotating generated state
    '''

    def __init__(self, X, y, batch_size=32, shuffle=False, seed=None, flip_prob=0, rotate_prob=0):

        self.X = X
        self.y = y
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob

        super(StateIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):

        # The generation of data is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([len(index_array)] + list(self.X.shape)[1:]), dtype=K.floatx())

        for batch_idx, source_idx in enumerate(index_array):
            batch_x[batch_idx] = self.augment(self.X[source_idx])
            
        return batch_x, self.y[index_array]

    def augment(self, x):
        '''
        Augment state with flipping and rotation
        '''

        # random rotate (90, 180, 270)
        if np.random.choice([True, False], p=[self.rotate_prob, 1.-self.rotate_prob]):
            x = np.rot90(x, k=np.random.choice([1, 2, 3]))

        # random horizontal flip
        if np.random.choice([True, False], p=[self.flip_prob, 1.-self.flip_prob]):
            x = self.flip_axis(x, 1)

        return x

    def flip_axis(self, x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

    def on_epoch_end(self):
        pass
