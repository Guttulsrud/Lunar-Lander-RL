import json

import numpy as np
import yaml
from yaml.loader import SafeLoader
import tensorflow as tf


def save_results_to_file(episode, avg, scores, config, file_name):
    with open(f'../results/{file_name}.json', 'r') as f:
        results = json.load(f)
        results['results'].append(
            {'episode': episode,
             'average_return': avg,
             'episode_scores': scores,
             'uncertainty': config['uncertainty'],
             })
    with open(f'../results/{file_name}.json', 'w') as f:
        json.dump(results, f)


def reverse_one_hot(value, length):
    output = np.zeros(length)
    output[value] = 1.0
    return output


def load_model(path):
    path = f'../models/{path}'

    return tf.keras.models.load_model(path)


class MODEL_TYPE:
    SINGLE = 'single'
    DOUBLE = 'double'


def get_config():
    with open('../config.yml') as f:
        config = yaml.load(f, Loader=SafeLoader)

    if config['general']['model_type'] == MODEL_TYPE.SINGLE:
        config['input_dimensions'] = 8
    elif config['general']['model_type'] == MODEL_TYPE.DOUBLE:
        config['input_dimensions'] = 16

    return config


def evaluate_double_agent(environment, agent, config, episodes, render=False):
    episode_scores = []

    for episode in range(episodes):

        current_observation = environment.reset()
        previous_observation = current_observation

        score = 0.0

        for step in range(config['max_steps']):

            previous_and_current_observation = np.append(previous_observation, current_observation)
            action = agent.choose_action(previous_and_current_observation, policy='exploit')

            next_observation, reward, done, info = environment.step(action)

            previous_observation = current_observation
            current_observation = next_observation

            score += reward
            if render:
                environment.render()

            if done:
                break

        episode_scores.append(score)

    average_return = sum(episode_scores) / config['evaluation_episodes']

    return average_return, episode_scores


def evaluate_agent(environment, agent, config, episodes):
    if config['general']['model_type'] == MODEL_TYPE.SINGLE:
        return evaluate_single_agent(environment=environment,
                                     agent=agent,
                                     config=config,
                                     episodes=episodes)
    elif config['general']['model_type'] == MODEL_TYPE.DOUBLE:
        return evaluate_double_agent(environment=environment,
                                     agent=agent,
                                     config=config,
                                     episodes=episodes)


def evaluate_single_agent(environment, agent, config, episodes):
    episode_scores = []

    for episode in range(episodes):

        observation = environment.reset()
        score = 0.0

        for step in range(config['max_steps']):
            action = agent.choose_action(observation, policy='exploit')
            next_observation, reward, done, info = environment.step(action)
            score += reward
            observation = next_observation
            if config['general']['render_evaluation']:
                environment.render()

            if done:
                break
        episode_scores.append(score)

    average_return = sum(episode_scores) / episodes

    return average_return, episode_scores
