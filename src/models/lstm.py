from models.rnn import RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM as lstm

class LSTM(RNN):
    def __init__(self, data, target_col, params: {}, verbose=False):
        super().__init__(data=data, target_col=target_col, units=params["units"], 
                        epochs=params["epochs"], batch_size=params["batch_size"], 
                        steps=params["steps"], patience=params["patience"], verbose=verbose)
        self.name = 'LSTM'

    def build_model(self, X_train, y_train):
        model = Sequential()
        model.add(lstm(units=self.units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(lstm(units=self.units))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
