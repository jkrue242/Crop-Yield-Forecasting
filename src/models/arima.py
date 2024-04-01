from statsmodels.tsa.arima.model import ARIMA as arima
from sklearn.metrics import mean_squared_error
from math import sqrt, ceil
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt

"""
Autoregressive integrated moving average (ARIMA) class
"""
class ARIMA:
    """
    ARIMA takes in a dataframe, target column (prediction target), and the following parameters:
    - data: the dataframe containing the data
    - target_col: the column to predict
    - get_order: whether to find the best combination of p, d, and q values
    - p_vals: list of p values to try
    - d_vals: list of d values to try
    - q_vals: list of q values to try
    """
    def __init__(self, data, target_col, get_order=False, p_vals=[0, 1, 2, 3], d_vals=[0, 1, 2, 3], q_vals=[0, 1, 2, 3]):
        self.X = data[target_col].values
        self.order = None
        if get_order:
            self.get_order(p_vals, d_vals, q_vals)
        else:
            self.order = (1, 1, 0)
        self.train = None
        self.test = None
        self.history = None
        self.predictions = None
        self.train_size = None
        self.performance = None

    """
    Evaluate the ARIMA model
    """
    def evaluate(self, train_size=0.8):
        # prepare data
        self.train_size = train_size
        size = ceil(len(self.X) * train_size)
        self.train, self.test = self.X[0:size], self.X[size:]
        self.history = [x for x in self.train]
        self.predictions = list()
        self.performance = []
        print("=========================================")
        print(f'Training ARIMA Model...')
        # walk-forward validation
        for t in range(len(self.test)):
            model = arima(self.history, order=self.order)
            model_fit = model.fit()
            if t == 0:
                print("Model summary: \n", model_fit.summary())
            output = model_fit.forecast()
            yhat = output[0]
            self.predictions.append(yhat)
            obs = self.test[t]
            self.history.append(obs)
            self.performance.append(obs)
        # evaluate forecasts
        rmse = sqrt(mean_squared_error(self.test, self.predictions))
        print(f'RMSE: {rmse}')

    """
    Get the best combination of p, d, and q values
    """
    def get_order(p_values=[0, 1, 2, 3], d_values=[0, 1, 2, 3], q_values=[0, 1, 2, 3]):
        dataset = dataset.astype('float32')
        best_score, best_cfg = float("inf"), None
        # grid search
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    try:
                        mse = evaluate_arima_model(dataset, order)
                        if mse < best_score:
                            best_score, best_cfg = mse, order
                    except:
                        continue
        self.order = best_cfg

    """
    Plot the results of the ARIMA model
    """
    def plot_results(self):
        # plot forecasts against actual outcomes
        fig = plt.figure(figsize=(12, 10))
        years = range(1945+len(self.train), 1945+len(self.train)+len(self.test))
        plt.plot(self.test)
        plt.plot(self.predictions, color='red')
        plt.xticks(range(len(self.test)), labels=years)
        plt.title(f"ARIMA Model Results\n Parameters: order: {self.order} | training set size: {self.train_size * 100}%")
        plt.legend(['Observed', 'Predicted'])
        plt.xlabel("Year")
        plt.savefig('images/arima/arima_results.png')

        # plt.figure(figsize=(16, 8))
        # plt.plot(self.performance, label='Training MSE')
        # plt.savefig(f"images/arima_training_mse.png")
