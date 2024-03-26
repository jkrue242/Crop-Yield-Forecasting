from utils.preprocess import get_data
from utils.plot import plot_dataset
from models.arima import ARIMA
from models.rnn import RNN
from models.lstm import LSTM
from models.gru import GRU
import pandas as pd
import json

def grid_search(model_name:str, params:{}, target_col):
    best_params = {
        'epochs': None,
        'batch_size': None,
        'units': None,
        'steps': None,
        'train_size': None,
        'patience': None
    }
    best_rmse = float('inf')
    for epoch in params['epochs']:
        for batch in params['batches']:
            for unit in params['units']:
                for step in params['steps']:
                    for size in params['train_sizes']:
                        for patience in params['patience']:
                            curr_params = {
                                'epochs': epoch,
                                'batch_size': batch,
                                'units': unit,
                                'steps': step,
                                'train_size': size,
                                'patience': patience
                            }
                            print(f'Model: {model_name} -- Epochs: {epoch} | Batch Size: {batch} | Units: {unit} | Steps: {step} | Train Size: {size} | Patience: {patience}')
                            if model_name == 'lstm':
                                model = LSTM(data, target_col=target_col, params=curr_params)
                            elif model_name == 'gru':
                                model = GRU(data, target_col=target_col, params=curr_params)
                            elif model_name == 'rnn':
                                model = RNN(data, target_col=target_col, params=curr_params)
                            else:
                                raise ValueError('Model name not supported')
                            rmse = model.evaluate(train_size=size)
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_params['epochs'] = epoch
                                best_params['batch_size'] = batch
                                best_params['units'] = unit
                                best_params['steps'] = step
                                best_params['train_size'] = size
                                best_params['patience'] = patience
                                print(f'Best loss: {best_rmse} | Best Params: {best_params}')
    return best_params

def save_dict_to_json(dictionary, filename):
    with open(filename, "w") as outfile: 
        json.dump(distionary, outfile)

def load_params(filename):
    with open(filename) as json_file:
        return json.load(json_file)


if __name__ == '__main__':
    sample_freq = 'M'

    ### Import data ###
    data = get_data(freq=sample_freq)
    data.to_csv('data/dataset.csv', index=False)
    plot_dataset(data)

    target = "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE"

    grid_params = {
        'epochs': [50, 100, 200, 300],
        'batches': [4, 8, 16, 32, 64],
        'units': [10, 20, 50],
        'steps': [10, 20, 50, 100],
        'train_sizes': [0.7],
        'patience': [5]
    }
    # grid search
    # rnn_params = grid_search('rnn', grid_params, target)
    # save_dict_to_json(rnn_params, 'rnn_params.json')
    # lstm_params = grid_search('lstm', grid_params, target)
    # save_dict_to_json(lstm_params, 'lstm_params.json')
    # gru_params = grid_search('gru', grid_params, target)
    # save_dict_to_json(gru_params, 'gru_params.json')

    rnn_params = load_params(filename='rnn_params.json')
    lstm_params = load_params(filename='lstm_params.json')
    gru_params = load_params(filename='gru_params.json')

    # ### ARIMA Model ###
    arima = ARIMA(data, target_col=target)
    arima.evaluate(train_size=0.7)
    arima.plot_results()

    ### RNN Model ###
    rnn = RNN(data, target_col=target, params=lstm_params, verbose=True)
    rmse = rnn.evaluate(train_size=0.7)
    rnn.plot_results()

    ### LSTM Model ###
    lstm = LSTM(data, target_col=target, params=lstm_params, verbose=True)
    rmse = lstm.evaluate(train_size=0.7)
    lstm.plot_results()

    ### GRU Model ###
    gru = GRU(data, target_col=target, params=gru_params, verbose=True)
    rmse = gru.evaluate(train_size=0.7)
    gru.plot_results()
