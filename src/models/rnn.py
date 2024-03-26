from math import sqrt, ceil
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN as rnn

class RNN:
    def __init__(self, data, target_col, params={}, verbose=False):
        self.data = data
        self.target_col = target_col
        self.units = params["units"]
        self.epochs = params["epochs"]
        self.batch_size = params["batch_size"]
        self.steps = params["steps"]
        self.train = None
        self.test = None
        self.model = None
        self.sc1 = None
        self.sc2 = None
        self.pred = None
        self.history = None
        self.train_size = None
        self.name = 'RNN'
        self.patience = params["patience"]
        self.early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=self.patience,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=0,)        
        self.verbose = verbose
        self.optimizer = params["optimizer"]

    def evaluate(self, train_size=0.66, verbose=0):
        self.train_size = train_size
        size = ceil(self.data.shape[0] * train_size)
        self.train, self.test = self.data[0:size], self.data[size:]

        scaled_train_input, scaled_train_output = self.scale_data()
        X_train, y_train = self.get_prev_values(scaled_train_input, scaled_train_output, n_steps=self.steps)

        # reshape
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        # build model
        self.build_model(X_train, y_train)

        if self.verbose:
            print('\n++++++++++++++++++++++++++++++++++++++++')
            print(f'{self.name} Model Summary:')
            print(self.model.summary())
            print(f'Training {self.name} Model...')
            verbose = 1
        self.history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, 
                            verbose=verbose, callbacks=[self.early_stopping], validation_split=0.1)

        test_data = self.prepare_test_data(self.train, self.test, n_steps=self.steps)

        # reshape
        test_data = self.reshape_test_data(test_data, n_steps=self.steps)

        # predict
        y_pred = self.model.predict(test_data)
        y_pred_df = pd.DataFrame(y_pred, columns=['Predicted Yield (bu/ac)'])

        final_predictions = self.sc2.inverse_transform(y_pred_df)
        pred_df = pd.DataFrame(final_predictions, columns=['Predicted Yield (bu/ac)'])
        self.pred = pred_df
        results = self.model.evaluate(test_data, self.test[self.target_col], batch_size=self.batch_size)
        metrics = f'''
        =========================================\n
        {self.name} Parameters: \n
        epochs: {self.epochs}\n
        batch_size: {self.batch_size}\n
        units: {self.units}\n
        steps: {self.steps}\n
        train_size: {self.train_size}\n
        patience: {self.patience}\n
        ----------------------------------------\n
        {self.name} Metrics: \n 
        mse:{self.history.history["mean_squared_error"][-1]}\n
        loss:{self.history.history["loss"][-1]}\n
        val_mse:{self.history.history["val_mean_squared_error"][-1]}\n
        val_loss:{self.history.history["val_loss"][-1]}
        '''
        print(metrics)

        f = open(f"results/{self.name}/params_acc.txt", "a")
        f.write("\n" + metrics)
        f.close()

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
        model.add(rnn(units=self.units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(rnn(units=self.units, return_sequences=True))
        model.add(rnn(units=self.units))
        model.add(Dense(units=1))
        model.compile(optimizer=self.optimizer, loss='mean_squared_error')
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

        # get first month per year 
        full_data = full_data.reset_index()
        full_data['Year'] = pd.to_datetime(full_data['index']).dt.year
        full_data = full_data.groupby('Year').first().reset_index()

        full_data.to_csv(f"predictions/{self.name}_preds.csv")
        plt.figure(figsize=(12, 10))
        plt.title(f"{self.name} Model Results\n Parameters: batch size: {self.batch_size} | steps: {self.steps} | units: {self.units} | training set size: {self.train_size * 100}% | optimizer: {self.optimizer} | patience: {self.patience}")
        plt.plot(full_data["Year"], full_data[self.target_col], label='Observed Yield (bu/ac)', color='blue')
        plt.plot(full_data["Year"], full_data['Predicted Yield (bu/ac)'], label='Predicted Yield (bu/ac)', color='red')
        plt.xlabel("Year")
        plt.legend()
        plt.savefig(f"images/{self.name}/{self.name}_results_{self.batch_size}batch_{self.steps}steps_{self.units}units_{self.patience}patience.png")
    
        plt.figure(figsize=(12, 10))
        plt.plot(self.history.history['loss'], label='Training loss', color='blue')
        plt.plot(self.history.history['val_loss'], label='Validation loss', color='red')
        plt.title("Training Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(f"images/{self.name}/{self.name}_training_mse_{self.batch_size}batch_{self.steps}steps_{self.units}units_{self.patience}patience.png")