from predictions.models.model import Model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

class LongTermShortMemory(Model):
    def __init__(self, intput_shape: tuple) -> None:
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, input_shape=intput_shape))
        self.model.add(LSTM(128))
        self.model.add(Dense(1))