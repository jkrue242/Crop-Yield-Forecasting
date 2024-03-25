from models.rnn import RNN
from keras.models import Sequential
from keras.layers import Input, Conv1D, Dense

class WaveNet(RNN):
    def __init__(self, data, target_col, units=10, epochs=100, batch_size=32, steps=5, filters=32, kernel_size=2, patience=5):
        super().__init__(data, target_col, units, epochs, batch_size, steps)
        self.name = 'WaveNet'
        self.filters = filters
        self.kernel_size = kernel_size

    def build_model(self, X_train, y_train):
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        for rate in (1, 2, 4, 8) * 2:
            model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='causal', dilation_rate=rate, activation='relu'))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mse')
        self.model = model
