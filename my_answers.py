import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [series[k:k+window_size] for k in range(len(series)-window_size)]
    y = [series[k+window_size] for k in range(len(series)-window_size)]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential([
                LSTM(5, input_shape=(window_size,1)),
                Dense(1)
        ])
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    import string

    return ''.join([c.lower() if c.lower() in punctuation or c.lower() in string.ascii_lowercase else " " for c in text])

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs

    # Initially I assumed step_size output at a time.
    # my_range = range(0, len(text)-window_size-step_size, step_size) 
    # outputs = [text[k+window_size:k+window_size+step_size] for k in my_range]
    # However I realized that we still output a single character
    my_range = range(0, len(text)-window_size, step_size) 
    outputs = [text[k+window_size] for k in my_range]
    inputs = [text[k:k+window_size] for k in my_range]
    return inputs,outputs



# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    """ 
    - layer 1 should be an LSTM module with 200 hidden units --> note this should have input_shape = (window_size, num_chars)
    - layer 2 should be a linear module, fully connected, with num_chars hidden units
    - layer 3 should be a softmax activation (since we are solving a multiclass classification) Use the categorical_crossentropy loss
    """
    model = Sequential([
                LSTM(200, input_shape=(window_size,num_chars)),
                Dense(num_chars, activation='softmax')
        ])

    return model
