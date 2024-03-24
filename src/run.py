from utils.preprocess import get_data
from utils.plot import plot_dataset
from models.arima import ARIMA
from models.lstm import LSTM

if __name__ == '__main__':
    ### Import data ###
    data = get_data(start_year=1972, end_year=2021)
    data.to_csv('data/dataset.csv', index=False)
    print('=====================')
    print('Dataset: ')
    print(data.tail())
    print('=====================')

    # plot dataset - images/dataset.png
    plot_dataset(data)

    ### ARIMA Model ###
    arima = ARIMA(data, target_col="Yield (bu/ac)")
    arima.evaluate(train_size=0.66)
    arima.plot_results()

    ### LSTM Model ###

    # grid search for best hyperparameters
    epochs = [10, 50, 100, 200]
    batches = [4, 8, 16, 32]
    units = [10, 20, 50]
    steps = [3, 5, 7, 10]
    train_sizes = [0.6, 0.7, 0.8]
    best_params = {
        'epochs': None,
        'batch_size': None,
        'units': None,
        'steps': None,
        'train_size': None
    }
    best_rmse = float('inf')
    for epoch in epochs:
        for batch in batches:
            for unit in units:
                for step in steps:
                    for size in train_sizes:
                        print(f'Epochs: {epoch} | Batch Size: {batch} | Units: {unit} | Steps: {step} | Train Size: {size}')
                        lstm = LSTM(data, target_col="Yield (bu/ac)", units=unit, epochs=epoch, batch_size=batch, steps=step)
                        rmse = lstm.evaluate(train_size=size)
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params['epochs'] = epoch
                            best_params['batch_size'] = batch
                            best_params['units'] = unit
                            best_params['steps'] = step
                            best_params['train_size'] = size
                            print(f'Best loss: {best_rmse} | Best Params: {best_params}')
    
    lstm = LSTM(data, target_col="Yield (bu/ac)", units=best_params["units"], epochs=best_params["epochs"], batch_size=best_params["batch_size"], steps=best_params["steps"])
    rmse = lstm.evaluate(train_size=size)
    lstm.plot_results()