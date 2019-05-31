from keras import Sequential
from keras.layers import Flatten, Dense, Activation


class TurbodroidModel:

    def __init__(self, window_lenght, obs_space_shape, nb_actions):
        super().__init__()
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(window_lenght,) + obs_space_shape))
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(nb_actions))
        self.model.add(Activation('linear'))

    def get(self):
        return self.model
