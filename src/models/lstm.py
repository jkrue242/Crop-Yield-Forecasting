from math import sqrt, ceil
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM as lstm, Dropout
import pandas as pd
import matplotlib.pyplot as plt

class LSTM:
    def __init__(self, data, target_col, units=10, epochs=100, batch_size=32, steps=5):
        self.data = data
        self.target_col = target_col
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps = steps

        self.train = None
        self.test = None
        self.model = None
        self.sc1 = None
        self.sc2 = None
        self.pred = None
        self.history = None

    def evaluate(self, train_size=0.66, verbose=0):
        size = ceil(self.data.shape[0] * train_size)
        self.train, self.test = self.data[0:size], self.data[size:]
        
        scaled_train_input, scaled_train_output = self.scale_data()
        X_train, y_train = self.get_prev_values(scaled_train_input, scaled_train_output, n_steps=self.steps)

        # reshape
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

        # build model
        self.build_model(X_train, y_train)
        self.history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)

        test_data = self.prepare_test_data(self.train, self.test, n_steps=self.steps)

        # reshape
        test_data = self.reshape_test_data(test_data, n_steps=self.steps)

        # predict
        y_pred = self.model.predict(test_data)
        y_pred_df = pd.DataFrame(y_pred, columns=['Predicted Yield (bu/ac)'])
        final_predictions = self.sc2.inverse_transform(y_pred_df)
        pred_df = pd.DataFrame(final_predictions, columns=['Predicted Yield (bu/ac)'])
        self.pred = pred_df

        rmse = self.history.history['loss'][-1]
        return rmse

    def scale_data(self):
        sc1 = StandardScaler()
        train_scaled_input = sc1.fit_transform(self.train)
        self.sc1 = sc1
        sc2 = StandardScaler()
        train_scaled_output = sc2.fit_transform(self.train[[self.target_col]])
        self.sc2 = sc2
        return train_scaled_input, train_scaled_output

    def get_prev_values(self, x_train_sc, y_train_sc, n_steps):
        X_train, y_train = [], []
        for i in range(n_steps, self.train.shape[0]):
            X_train.append(x_train_sc[i-n_steps:i, :])
            y_train.append(y_train_sc[i, 0])
        return np.array(X_train), np.array(y_train)

    def build_model(self, X_train, y_train):
        model = Sequential()
        model.add(lstm(units=self.units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        # model.add(Dropout(0.2))
        model.add(lstm(units=self.units))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def prepare_test_data(self, train, test, n_steps):
        train_last_steps = train[-n_steps:]
        df = pd.concat((train_last_steps, test), axis=0)

        # rescale
        df = self.sc1.transform(df)
        return df

    def reshape_test_data(self, test_data, n_steps):
        X_pred = []
        for i in range(n_steps, test_data.shape[0]):
            X_pred.append(test_data[i-n_steps:i])
        return np.array(X_pred)


    def plot_results(self):
        test = self.test.reset_index()
        full_data = pd.concat([test, self.pred], axis=1)
        plt.figure(figsize=(16, 8))
        plt.plot(full_data['Year'], full_data[self.target_col], label='Observed Yield (bu/ac)', color='blue')
        plt.plot(full_data['Year'], full_data['Predicted Yield (bu/ac)'], label='Predicted Yield (bu/ac)', color='red')
        plt.legend()
        plt.savefig(f"images/lstm_results_{self.batch_size}batch_{self.epochs}epochs_{self.steps}steps_{self.units}units.png")
        
        plt.figure(figsize=(16, 8))
        plt.plot(self.history.history['loss'], label='Training MSE', color='blue')
        plt.savefig(f"images/lstm_training_mse_{self.batch_size}batch_{self.epochs}epochs_{self.steps}steps_{self.units}units.png")


