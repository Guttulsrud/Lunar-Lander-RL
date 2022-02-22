import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from datetime import datetime

from ReplayBuffer import ReplayBuffer


class Agent:
    def __init__(self, config):
        self.config = config
        self.exploration_rate = config['training']['epsilon']['max']
        self.exploration_rate_min = config['training']['epsilon']['min']
        self.exploration_rate_decrement = config['training']['epsilon']['change']
        self.batch_size = self.config['training']['batch_size']
        self.memory = ReplayBuffer(config)
        self.action_space = np.arange(self.config['number_of_actions'])
        self.model = None
        self.build_model()

    def build_model(self):
        layers = self.config['network']['layers']
        learning_rate = self.config['network']['learning_rate']
        loss_function = self.config['network']['loss_function']
        self.model = Sequential()

        self.model.add(Input(shape=(8,)))

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

        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)
        action_indices = [int(x) for x in action_indices]

        q_next = self.model.predict(next_state)
        q_eval = self.model.predict(state)
        q_target = q_eval.copy()

        max_actions = np.argmax(q_eval, axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        calculated_rewards = reward + self.config['gamma'] * q_next[batch_index, max_actions.astype(int)] * (1 - done)
        q_target[batch_index, action_indices] = calculated_rewards

        self.model.fit(state, q_target, verbose=0)

        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate *= self.exploration_rate_decrement

    def save_model(self):
        model_name = datetime.now().isoformat()
        model_name = model_name[:-7]
        model_name = model_name.replace('T', '_')
        model_name = model_name.replace(':', '-')

        self.model.save(self.config['save_location'] + model_name)
