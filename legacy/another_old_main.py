from network import get_q_network
from collections import deque
import gym
import numpy as np

from old_utils import get_config, format_observation

config = get_config()

num_episodes = config['num_episodes']
num_env_steps = config['num_env_steps']
batch_size = config['training']['batch_size']
model = get_q_network(config['network'])
replay_buffer = deque(maxlen=1_000_000)
gamma = .99

environment = gym.make('LunarLander-v2')
environment.reset()

max_exploration_rate = exploration_rate = config['training']['epsilon']['max']
min_exploration_rate  = config['training']['epsilon']['min']
exploration_decay_rate = config['training']['epsilon']['change']

print('State shape: ', environment.observation_space.shape)
print('Number of actions: ', environment.action_space.n)

scores = []

for episode in range(num_episodes):
    score = 0

    # state = environment.reset()
    # for _ in range(200):
    #     state = np.reshape(state, (1, 8))
    #     action_values = model.predict(state)
    #     action = np.argmax(action_values[0])
    #     state, reward, done, metadata = environment.step(action)
    #     environment.render()
    #     score += reward
    #     if done:
    #         break
    # scores.append(score)
    # print(f"Episode = {episode}, Score = {score}, Avg_Score = {np.mean(scores[-10:])}, Epsilon = {epsilon}")

    state = environment.reset()
    for _ in range(200):

        state = np.reshape(state, (1, 8))

        exploration_rate_threshold = np.random.uniform(0, 1)
        if exploration_rate_threshold <= exploration_rate:
            action = np.random.choice(4)
        else:
            action_values = model.predict(state)
            action = np.argmax(action_values[0])

        next_state, reward, done, metadata = environment.step(action)
        print(reward)
        environment.render()
        input("Press Enter to continue...")

        next_state = np.reshape(next_state, (1, 8))
        replay_buffer.append((state, action, next_state, reward, done))

        if len(replay_buffer) >= batch_size:
            sample_choices = np.array(replay_buffer)
            training_data = np.random.choice(len(sample_choices), batch_size)

            states = []
            actions = []
            next_states = []
            rewards = []
            finishes = []
            for index in training_data:
                states.append(replay_buffer[index][0])
                actions.append(replay_buffer[index][1])
                next_states.append(replay_buffer[index][2])
                rewards.append(replay_buffer[index][3])
                finishes.append(replay_buffer[index][4])
            states = np.array(states)
            actions = np.array(actions)
            next_states = np.array(next_states)
            rewards = np.array(rewards)
            finishes = np.array(finishes)
            states = np.squeeze(states)
            next_states = np.squeeze(next_states)

            predictions_next_state = model.predict_on_batch(next_states)
            target_q_value = model.predict_on_batch(states)
            max_predictions_next_state = np.amax(predictions_next_state, axis=1)

            target_q_value[np.arange(batch_size), actions] = \
                rewards + gamma * max_predictions_next_state * (1 - finishes)
            model.fit(states, target_q_value, verbose=0)

        state = next_state
        if done:
            break

    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    # observation = format_observation(observation)
    # print(observation)
