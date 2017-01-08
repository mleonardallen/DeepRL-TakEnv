from keras.preprocessing.image import ImageDataGenerator, Iterator, flip_axis
from keras.layers.core import Dense, Flatten, Activation, SpatialDropout2D, Lambda, Dropout
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model
from keras.objectives import mse
from sklearn.model_selection import train_test_split
import os.path
import json
import numpy as np

class CnnValueFunction(object):

    def __init__(self):
        self.model = self.create_model();

    def create_model(self):
        model = Sequential([
            # Normalize to keep weight values small with zero mean, improving numerical stability.
            BatchNormalization(axis=1, input_shape=(6, 6, 15)),
            # Conv 2x2
            Convolution2D(30, 2, 2, border_mode = 'same', activation = 'elu'),
            SpatialDropout2D(0.2),
            # Conv 2x2
            Convolution2D(60, 2, 2, border_mode = 'same', activation = 'elu'),
            SpatialDropout2D(0.2),
            # Conv 2x2
            Convolution2D(90, 2, 2, border_mode = 'same', activation = 'elu'),
            SpatialDropout2D(0.2),
            # Flatten
            Flatten(),
            # Fully Connected
            Dense(100, activation = 'elu', W_regularizer = l2(1e-6)),
            Dense(50, activation = 'elu', W_regularizer = l2(1e-6)),
            Dense(10, activation = 'elu', W_regularizer = l2(1e-6)),
            Dense(1)
        ])
        model.summary()

        if os.path.isfile('model.h5'):
            model.load_weights('model.h5')

        model.compile(loss = CnnValueFunction.loss, optimizer = 'adam')
        return model

    def predict(self, X):
        return self.model.predict(X)

    @staticmethod
    def loss(y_true, y_pred):
        return mse(y_true, y_pred)

    def fit(self, X, y):
        self.model.fit(X, y, batch_size=32, nb_epoch=3)
