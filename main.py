from Agent import Agent
from utils import get_config
import gym
import numpy as np

if __name__ == '__main__':

    config = get_config()
    agent = Agent(config)
    env = gym.make('LunarLander-v2')

    score_history = []
    exploration_rate_history = []

    for episode in range(config['number_of_episodes']):

        # collect and train
        observation = env.reset()
        for step in range(config['max_steps']):
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, next_observation, done)
            observation = next_observation
            agent.learn()
            if done:
                continue

        # evaluate
        score = 0
        observation = env.reset()
        for step in range(config['max_steps']):
            env.render()
            action = agent.choose_action(observation, policy='exploit')
            next_observation, reward, done, info = env.step(action)
            observation = next_observation
            score += reward
            if done:
                continue

        exploration_rate_history.append(agent.exploration_rate)
        score_history.append(score)

        average_score = np.mean(score_history[max(0, episode - 10):episode + 1])
        print(f'Episode: {episode}, Score: {score}, Average Score: {average_score}')

    # Plot results here :D
