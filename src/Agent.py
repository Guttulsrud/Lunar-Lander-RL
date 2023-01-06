import os

import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime

from ReplayBuffer import ReplayBuffer
from utils import MODEL_TYPE


class Agent:

    def __init__(self, config, hparams, pre_trained_model=None):
        self.config = config
        self.hparams = hparams
        self.exploration_rate = config['training']['epsilon']['max']
        self.exploration_rate_min = config['training']['epsilon']['min']
        self.exploration_rate_decrement = config['training']['epsilon']['change']
        self.batch_size = self.hparams['batch_size']
        self.memory = ReplayBuffer(config)
        self.action_space = np.arange(self.config['number_of_actions'])

        self.batch_index = np.arange(self.batch_size, dtype=np.int32)

        if not pre_trained_model:
            if self.config['general']['model_type'] == MODEL_TYPE.SINGLE:
                self.build_naive_model()
            elif self.config['general']['model_type'] == MODEL_TYPE.MULTI:
                self.build_multi_timestep_model()
            else:
                raise Exception('Invalid model type!')
        else:
            self.model = pre_trained_model

    def build_multi_timestep_model(self):
        layers = self.config['network']['layers']
        learning_rate = self.config['network']['learning_rate']
        loss_function = self.config['network']['loss_function']
        self.model = Sequential()

        self.model.add(Input(shape=(self.config['training']['input_shape'],)))

        for layer in layers:
            self.model.add(Dense(layer['nodes'], activation=layer['activation']))

        self.model.compile(loss=loss_function, optimizer=Adam(learning_rate))

    def build_naive_model(self):

        loss_function = self.config['network']['loss_function']

        learning_rate = self.hparams['learning_rate']
        hidden_size = self.hparams['hidden_size']
        layers = self.hparams['layers']
        activation = self.hparams['activation']

        self.model = Sequential()
        self.model.add(Input(shape=(8,)))

        for _ in range(layers):
            self.model.add(Dense(hidden_size, activation=activation))

        self.model.add(Dense(4, activation='linear'))

        self.model.compile(loss=loss_function, optimizer=Adam(learning_rate))

    def remember(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def choose_action(self, current, policy='explore'):
        current = current[np.newaxis, :]
        random_chance = np.random.uniform(0, 1)
        if random_chance > self.exploration_rate or policy == 'exploit':
            prediction = self.model.predict(current, verbose=0)
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            # There is not enough training data yet to fill a batch
            return

        state, action, reward, next_state, done = self.memory.sample_batch(self.batch_size)

        q_eval = self.model.predict(state, verbose=0)
        q_next = self.model.predict(next_state, verbose=0)

        gradients = reward + self.config['gamma'] * np.max(q_next, axis=1) * (1 - done)
        q_eval[self.batch_index, action] = gradients

        self.model.fit(state, q_eval, verbose=0)

        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate *= self.exploration_rate_decrement

    def save_model(self, score):
        # Get the current date and time
        now = datetime.now()
        model_name = now.strftime("%y-%m-%d_%H-%M")

        # Get the score as a string and remove the decimal part
        score = str(score).split(".")[0]
        model_name = f'{model_name}_SCORE_{score}'

        # Check if the save location directory exists
        is_dir = os.path.isdir(self.config['save_location'])
        if not is_dir:
            os.mkdir(self.config['save_location'])

        # Save the model using tf.saved_model.save
        tf.saved_model.save(self.model, self.config['save_location'] + model_name)
