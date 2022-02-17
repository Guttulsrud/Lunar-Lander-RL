import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

options = {
    'network': {
        'layers': [
            {'nodes': 50, 'activation': 'relu'},
            {'nodes': 50, 'activation': 'relu'},
            {'nodes': 4, 'activation': 'linear'},
        ],
        'loss': 'mse',
        "learning_rate": 1e-3,
    }

}


def get_q_network(options):
    layers = options['layers']
    network = Sequential()

    network.add(keras.Input(shape=(8,)))

    for layer in layers:
        network.add(Dense(layer['nodes'], activation=layer['activation']))

    return network


get_q_network(options)
