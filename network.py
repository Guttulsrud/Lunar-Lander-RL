from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def get_q_network(options):
    layers = options['layers']
    learning_rate = options['learning_rate']
    loss_function = options['loss_function']
    model = Sequential()

    model.add(Input(shape=(8,)))

    for layer in layers:
        model.add(Dense(layer['nodes'], activation=layer['activation']))

    model.compile(loss=loss_function, optimizer=Adam(learning_rate))
    return model




