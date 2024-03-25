from models.rnn import RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM as lstm

class LSTM(RNN):
    def __init__(self, data, target_col, units=10, epochs=100, batch_size=32, steps=5, patience=5):
        super().__init__(data, target_col, units, epochs, batch_size, steps)
        self.name = 'LSTM'

    def build_model(self, X_train, y_train):
        model = Sequential()
        model.add(lstm(units=self.units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(lstm(units=self.units))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
