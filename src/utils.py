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
    MULTI = 'multi'


def get_config():
    with open('../config.yml') as f:
        config = yaml.load(f, Loader=SafeLoader)

    if config['general']['model_type'] == MODEL_TYPE.SINGLE:
        config['input_dimensions'] = 8
    elif config['general']['model_type'] == MODEL_TYPE.DOUBLE:
        config['input_dimensions'] = 17
    elif config['general']['model_type'] == MODEL_TYPE.MULTI:
        config['input_dimensions'] = 35

    return config


def evaluate_multi_agent(environment, agent, config, episodes, timesteps=4):
    episode_scores = []

    for episode in range(episodes):

        current_observation = environment.reset()

        observations = [current_observation for _ in range(timesteps)]
        actions = [0 for _ in range(timesteps - 1)]

        next_observations_and_actions = np.append(observations, actions)

        score = 0.0

        for step in range(config['max_steps']):

            observations_and_actions = next_observations_and_actions

            action = agent.choose_action(observations_and_actions, policy='exploit')

            next_observation, reward, done, info = environment.step(action)

            observations = np.roll(observations, shift=-1)
            actions = np.roll(actions, shift=-1)
            observations[-1] = next_observation
            actions[-1] = action
            next_observations_and_actions = np.append(observations, actions)

            score += reward
            if config['general']['render_evaluation']:
                environment.render()

            if done:
                break

        episode_scores.append(score)

    average_return = sum(episode_scores) / config['evaluation_episodes']

    return average_return, episode_scores


def evaluate_double_agent(environment, agent, config, episodes, render=False):
    episode_scores = []

    for episode in range(episodes):

        current_observation = environment.reset()
        previous_observation = current_observation
        previous_action = 0

        score = 0.0

        for step in range(config['max_steps']):
            # gravity = config['uncertainty']['gravity']['value']
            # wind = config['uncertainty']['wind']['value']
            previous_and_current = np.append(previous_observation, current_observation)
            previous_and_current_observation = np.append(previous_and_current, previous_action)
            # previous_and_current_observation = np.append(previous_and_current_observation, wind)
            # previous_and_current_observation = np.append(previous_and_current_observation, gravity)

            action = agent.choose_action(previous_and_current_observation, policy='exploit')

            next_observation, reward, done, info = environment.step(action)

            previous_observation = current_observation
            current_observation = next_observation
            previous_action = action

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
    elif config['general']['model_type'] == MODEL_TYPE.MULTI:
        return evaluate_multi_agent(environment=environment,
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
