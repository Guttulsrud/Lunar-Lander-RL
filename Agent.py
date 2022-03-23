import numpy as np
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Lambda, Add
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from datetime import datetime

from ReplayBuffer import ReplayBuffer



#TODO: Import cv2 og håndter bilder som i tutorial

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
        self.build_cnn_model()

    def build_cnn_model(self, dueling=False):
        input_shape = (400, 600)
        action_space = self.config['network']['layers'][2]['nodes']

        x_input = Input(input_shape)
        x = x_input
        x = Conv2D(64, 5, strides=(3, 3), padding="valid", input_shape=input_shape, activation="relu",
                   data_format="channels_first")(x)
        x = Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", data_format="channels_first")(x)
        x = Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", data_format="channels_first")(x)
        x = Flatten()(x)
        x = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(x)
        x = Dense(256, activation="relu", kernel_initializer='he_uniform')(x)
        x = Dense(64, activation="relu", kernel_initializer='he_uniform')(x)

        if dueling:
            state_value = Dense(1, kernel_initializer='he_uniform')(x)
            state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

            action_advantage = Dense(s, kernel_initializer='he_uniform')(x)
            action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(
                action_advantage)

            x = Add()([state_value, action_advantage])
        else:
            # Output Layer with # of actions: 2 nodes (left, right)
            x = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(x)

        model = Model(inputs=x_input, outputs=x, name='CartPole PER D3QN CNN model')
        model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
                      metrics=["accuracy"])

        model.summary()
        self.model = model

    def build_naive_model(self):
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

    def save_model(self):
        model_name = datetime.now().isoformat()
        model_name = model_name[:-7]
        model_name = model_name.replace('T', '_')
        model_name = model_name.replace(':', '-')

        self.model.save(self.config['save_location'] + model_name)
