from keras.preprocessing.image import ImageDataGenerator, Iterator, flip_axis
from keras.layers.core import Dense, Flatten, Activation, SpatialDropout2D, Lambda, Dropout
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model
from keras.objectives import mse
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import os.path
import json
import numpy as np

class CnnValueFunction(object):

    def __init__(self, board_size):
        self.board_size = board_size
        self.filename = 'model-' + str(self.board_size)
        self.model = self.create_model();

    def create_model(self):

        model = Sequential([
            # Normalize to keep weight values small with zero mean, improving numerical stability.
            BatchNormalization(axis=1, input_shape=(self.board_size, self.board_size, 15)),
            # Conv 2x2
            Convolution2D(15, 2, 2, border_mode = 'same', activation = 'elu'),
            SpatialDropout2D(0.1),
            # Conv 2x2
            Convolution2D(15, 2, 2, border_mode = 'same', activation = 'elu'),
            SpatialDropout2D(0.1),
            # Conv 2x2
            Convolution2D(30, 2, 2, border_mode = 'same', activation = 'elu'),
            SpatialDropout2D(0.1),
            # Flatten
            Flatten(),
            # Fully Connected
            Dense(100, activation = 'elu'),
            Dense(20, activation = 'elu'),
            Dense(10, activation = 'elu'),
            Dense(1)
        ])
        model.summary()

        if os.path.isfile(self.filename + '.h5'):
            model.load_weights(self.filename + '.h5')

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss = CnnValueFunction.loss, optimizer = optimizer)
        return model

    def predict(self, X):
        return self.model.predict(X)

    @staticmethod
    def loss(y_true, y_pred):
        return mse(y_true, y_pred)

    def fit(self, X, y):
        self.model.fit(X, y, batch_size=128, nb_epoch=3, shuffle=True)

    def save(self):
        with open(self.filename + '.json' , 'w') as outfile:
            json.dump(self.model.to_json(), outfile)
        self.model.save_weights(self.filename + '.h5')