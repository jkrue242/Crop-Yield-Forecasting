from utils.preprocess import get_data
from utils.plot import plot_dataset
from models.arima import ARIMA
from models.rnn import RNN
from models.lstm import LSTM
from models.gru import GRU
import pandas as pd
import json
from keras.optimizers import Adam, RMSprop, SGD

"""
Basic grid search function to find the best parameters for a given model
"""
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
    save_dict_to_json(params=best_params, filename=f'{model_name}_params.json')
    return best_params

"""
Save model parameters to a json file
"""
def save_dict_to_json(dictionary, filename):
    with open(filename, "w") as outfile: 
        json.dump(distionary, outfile)

"""
Load model parameters from a json file
"""
def load_params(filename):
    with open(filename) as json_file:
        return json.load(json_file)

# driver code
if __name__ == '__main__':
    # set upsample frequency to monthly
    sample_freq = 'M'

    # import data
    data = get_data(freq=sample_freq)

    # save dataset to .csv
    data.to_csv('data/dataset.csv', index=False)
    target = "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE"

    params = {
        'epochs': 200, 
        'batch_size': 64, 
        'units': 60, 
        'steps': 110, 
        'loss': 'mean_squared_error',
        'train_size': 0.80, 
        'patience': 20, 
        }

    ### RNN Model ###
    rnn = RNN(data, target_col=target, params=params, verbose=True)
    rmse = rnn.evaluate(train_size=params['train_size'])
    rnn.plot_results()

    ### LSTM Model ###
    lstm = LSTM(data, target_col=target, params=params, verbose=True)
    rmse = lstm.evaluate(train_size=params['train_size'])
    lstm.plot_results()

    # ### GRU Model ###
    gru = GRU(data, target_col=target, params=params, verbose=True)
    gru.evaluate(train_size=params['train_size'])
    gru.plot_results()