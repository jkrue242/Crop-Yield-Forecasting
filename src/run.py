from utils.preprocess import get_data
from utils.plot import plot_dataset
from models.arima import ARIMA
from models.lstm import LSTM
from models.gru import GRU
from models.wavenet import WaveNet

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


if __name__ == '__main__':

    ### Import data ###
    data = get_data()
    data.to_csv('data/dataset.csv', index=False)
    plot_dataset(data)
    target = "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE"

    grid_params = {
        'epochs': [50, 100, 200],
        'batches': [4, 8, 16, 32],
        'units': [10, 20, 50],
        'steps': [5, 10, 15, 20],
        'train_sizes': [0.7],
        'patience': [5]
    }

    # lstm_params = grid_search(model_name='lstm', params=grid_params, target_col=target)
    # gru_params = grid_search(model_name='gru', params=grid_params, target_col=target)
    # params = [lstm_params, gru_params]
    # params = {
    #     'epochs': 100,
    #     'batch_size': 4,
    #     'units': 50,
    #     'steps': 20,
    #     'train_size': 0.8,
    #     'patience': 5
    # }
    # params = [params, params]

    lstm_params = {
        'epochs': 100, 
        'batch_size': 4, 
        'units': 10, 
        'steps': 10, 
        'train_size': 0.7, 
        'patience': 3}

    gru_params = {
        'epochs': 200,
        'batch_size': 32,
        'units': 50,
        'steps': 5, 
        'train_size': 0.7,
        'patience': 5
    }
    # ### ARIMA Model ###
    arima = ARIMA(data, target_col=target)
    arima.evaluate(train_size=0.7)
    arima.plot_results()

    # ### LSTM Model ###
    lstm = LSTM(data, target_col=target, params=lstm_params, verbose=True)
    rmse = lstm.evaluate(train_size=0.7)
    lstm.plot_results()

    ### GRU Model ###
    gru = GRU(data, target_col=target, params=gru_params, verbose=True)
    rmse = gru.evaluate(train_size=0.7)
    # gru.plot_results()

    ### WaveNet Model ###
    # wavenet = WaveNet(data, target_col="CORN, GRAIN - YIELD, MEASURED IN BU / ACRE", units=50, epochs=100, batch_size=4, steps=20, filters=32, kernel_size=2)
    # rmse = wavenet.evaluate(train_size=0.8)
    # wavenet.plot_results()