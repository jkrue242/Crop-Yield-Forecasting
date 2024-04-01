from models.rnn import RNN
from keras.models import Sequential
from keras.layers import Dense, GRU as gru, Dropout
import keras.optimizers as ko
import keras.metrics as km
from keras.regularizers import l2

"""
Gated Recurrent Unit (GRU) class, inherits from the RNN class
"""
class GRU(RNN):
    """
    GRU takes in a dataframe, target column (prediction target), and the following parameters:
    - units: number of units in the GRU layer
    - epochs: number of epochs to train the model
    - batch_size: size of the batch to train the model
    - steps: number of steps (months) to look back in the data
    - lossfn: loss function to use for training
    - patience: number of epochs to wait before early stopping
    - verbose: whether to print out model summary and training progress
    stored in a dictionary.
    """
    def __init__(self, data, target_col, params: {}, verbose=False):
        super().__init__(data=data, target_col=target_col, params=params, verbose=verbose)
        self.name = 'GRU'

    """
    Build the GRU model
    """
    def build_model(self, X_train, y_train):
        # build the sequential model
        self.model = Sequential()
        self.model.add(
            gru(units=self.units, 
                return_sequences=False, 
                input_shape=(X_train.shape[1], X_train.shape[2]), 
            )
        )
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=1))

        # compile
        self.model.compile(optimizer=ko.RMSprop(), loss=self.lossfn, metrics=[km.mean_squared_error])