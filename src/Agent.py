import os

import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from datetime import datetime

from ReplayBuffer import ReplayBuffer


class Agent:
    def __init__(self, config, pre_trained_model):
        self.config = config
        self.exploration_rate = config['training']['epsilon']['max']
        self.exploration_rate_min = config['training']['epsilon']['min']
        self.exploration_rate_decrement = config['training']['epsilon']['change']
        self.batch_size = self.config['training']['batch_size']
        self.memory = ReplayBuffer(config)
        self.action_space = np.arange(self.config['number_of_actions'])
        if not pre_trained_model:
            self.build_naive_model()
        else:
            self.model = pre_trained_model

    def build_naive_model(self):
        layers = self.config['network']['layers']
        learning_rate = self.config['network']['learning_rate']
        loss_function = self.config['network']['loss_function']
        self.model = Sequential()

        self.model.add(Input(shape=(self.config['input_dimensions'],)))

        for layer in layers:
            self.model.add(Dense(layer['nodes'], activation=layer['activation']))

        self.model.compile(loss=loss_function, optimizer=Adam(learning_rate))

    def remember(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def choose_action(self, state, policy='explore'):
        state = state[np.newaxis, :]
        random_chance = np.random.uniform(0, 1)
        if random_chance > self.exploration_rate or policy == 'exploit':
            prediction = self.model.predict(state)
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            # There is not enough training data yet to fill a batch
            return

        state, action, reward, next_state, done = self.memory.sample_batch(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)
        action_indices = [int(x) for x in action_indices]

        q_eval = self.model.predict(state)
        q_next = self.model.predict(next_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        gradients = reward + self.config['gamma'] * np.max(q_next, axis=1) * (1 - done)
        q_target[batch_index, action_indices] = gradients

        self.model.fit(state, q_target, verbose=0)

        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate *= self.exploration_rate_decrement

    def save_model(self, score):
        now = datetime.now()
        model_name = now.strftime("%y-%m-%d_%H-%M")

        score = str(score).split(".")[0]
        model_name = f'{model_name}_SCORE_{score}'
        is_dir = os.path.isdir(self.config['save_location'])

        if not is_dir:
            os.mkdir(self.config['save_location'])

        self.model.save(self.config['save_location'] + model_name)




