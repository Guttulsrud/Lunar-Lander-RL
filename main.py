from network import get_q_network
from collections import deque
import gym

from utils import get_config, format_observation

config = get_config()

num_episodes = config['num_episodes']
num_env_steps = config['num_env_steps']
q_network = get_q_network(config['network'])
memory = deque(maxlen=1_000_000)

environment = gym.make('LunarLander-v2')
environment.reset()

print('State shape: ', environment.observation_space.shape)
print('Number of actions: ', environment.action_space.n)

for _ in range(num_episodes):

    state = environment.reset()

    for _ in range(200):
        environment.render()
        action = environment.action_space.sample()
        observation, reward, done, unsure = environment.step(action)
        if done:
            break

        observation = format_observation(observation)
        print(observation)
