from models.rnn import RNN
from keras.models import Sequential
from keras.layers import Dense, GRU as gru, Dropout
import keras.optimizers as ko
import keras.metrics as km
from keras.regularizers import l2

class GRU(RNN):
    def __init__(self, data, target_col, params: {}, verbose=False):
        super().__init__(data=data, target_col=target_col, params=params, verbose=verbose)
        self.name = 'GRU'

    def build_model(self, X_train, y_train):
        self.model = Sequential()
        self.model.add(
            gru(units=self.units, 
                return_sequences=False, 
                input_shape=(X_train.shape[1], X_train.shape[2]), 
            )
        )
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer=ko.RMSprop(), loss='mean_squared_error', metrics=[km.mean_squared_error])