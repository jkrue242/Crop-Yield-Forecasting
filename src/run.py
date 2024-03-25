from utils.preprocess import get_data
from utils.plot import plot_dataset
from models.arima import ARIMA
from models.lstm import LSTM
from models.gru import GRU
from models.wavenet import WaveNet

def grid_search(model_name:str, params:{}):
    best_params = {
        'epochs': None,
        'batch_size': None,
        'units': None,
        'steps': None,
        'train_size': None
    }
    best_rmse = float('inf')
    for epoch in params['epochs']:
        for batch in params['batches']:
            for unit in params['units']:
                for step in params['steps']:
                    for size in params['train_sizes']:
                        print(f'Epochs: {epoch} | Batch Size: {batch} | Units: {unit} | Steps: {step} | Train Size: {size}')
                        if model_name == 'lstm':
                            model = LSTM(data, target_col="Yield (bu/ac)", units=unit, epochs=epoch, batch_size=batch, steps=step)
                        elif model_name == 'gru':
                            model = GRU(data, target_col="Yield (bu/ac)", units=unit, epochs=epoch, batch_size=batch, steps=step)
                        else:
                            raise ValueError('Model name not supported')
                        rmse = lstm.evaluate(train_size=size)
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params['epochs'] = epoch
                            best_params['batch_size'] = batch
                            best_params['units'] = unit
                            best_params['steps'] = step
                            best_params['train_size'] = size
                            print(f'Best loss: {best_rmse} | Best Params: {best_params}')
    return best_params


if __name__ == '__main__':

    ### Import data ###
    data = get_data()
    data.to_csv('data/dataset.csv', index=False)
    plot_dataset(data)

    params = {
        'epochs': [50, 100, 200],
        'batches': [4, 8, 16, 32],
        'units': [10, 20, 50],
        'steps': [5, 10, 15, 20],
        'train_sizes': [0.6, 0.7, 0.8]
    }
    # best_params = grid_search('lstm', params)
 
    # ### ARIMA Model ###
    arima = ARIMA(data, target_col="CORN, GRAIN - YIELD, MEASURED IN BU / ACRE")
    arima.evaluate(train_size=0.8)
    arima.plot_results()

    # ### LSTM Model ###
    lstm = LSTM(data, target_col="CORN, GRAIN - YIELD, MEASURED IN BU / ACRE", units=50, epochs=200, batch_size=4, steps=20)
    rmse = lstm.evaluate(train_size=0.8)
    lstm.plot_results()

    ### GRU Model ###
    gru = GRU(data, target_col="CORN, GRAIN - YIELD, MEASURED IN BU / ACRE", units=50, epochs=100, batch_size=4, steps=20)
    rmse = gru.evaluate(train_size=0.8)
    gru.plot_results()

    ### WaveNet Model ###
    wavenet = WaveNet(data, target_col="CORN, GRAIN - YIELD, MEASURED IN BU / ACRE", units=50, epochs=100, batch_size=4, steps=20, filters=32, kernel_size=2)
    rmse = wavenet.evaluate(train_size=0.8)
    wavenet.plot_results()