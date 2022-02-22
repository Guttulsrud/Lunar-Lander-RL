import numpy as np

from utils import reverse_one_hot


class ReplayBuffer:
    def __init__(self, config):
        self.memory_size = config['memory']['size']
        self.input_dimensions = config['input_dimensions']
        self.number_of_actions = config['number_of_actions']
        self.memory_counter = 0

        self.state_memory = np.zeros((self.memory_size, self.input_dimensions))
        self.next_state_memory = np.zeros((self.memory_size, self.input_dimensions))
        self.action_memory = np.zeros((self.memory_size, self.number_of_actions))
        self.reward_memory = np.zeros(self.memory_size)
        self.done_memory = np.zeros(self.memory_size)

    def store(self, state, action, reward, next_state, done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index] = done

        action = reverse_one_hot(action, self.number_of_actions)
        self.action_memory[index] = action
        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)
        batch_indices = np.random.choice(max_memory, batch_size)

        state_batch = self.state_memory[batch_indices]
        action_batch = self.action_memory[batch_indices]
        reward_batch = self.reward_memory[batch_indices]
        next_state_batch = self.next_state_memory[batch_indices]
        done_batch = self.done_memory[batch_indices]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
