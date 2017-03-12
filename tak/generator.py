from keras.preprocessing.image import ImageDataGenerator, Iterator, flip_axis
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
        flip_prob: float ∈[0,1] probability of horizontally flipping generated state
        rotate_prob: float ∈[0,1] probability of rotating generated state
    '''

    def __init__(self, X, y, batch_size=32, shuffle=False, seed=None, flip_prob=0, rotate_prob=0):

        self.X = X
        self.y = y
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob

        super(StateIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):

        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        # The generation of data is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X[0].shape)))
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
            x = flip_axis(x, 1)

        return x