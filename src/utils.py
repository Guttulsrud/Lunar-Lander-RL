import json
import random

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
    path = f'../saved_models/{path}'

    return tf.keras.models.load_model(path)


class MODEL_TYPE:
    SINGLE = 'single'
    MULTI = 'multi'


def get_config(testing=False):
    with open('../config.yml') as f:
        config = yaml.load(f, Loader=SafeLoader)

    if config['general']['model_type'] == 'multi':
        timesteps = config['training']['timesteps']
        config['training']['input_shape'] = timesteps * 8 + (timesteps - 1) * 4
    else:
        config['training']['input_shape'] = 8

    if testing:
        config['general']['verbose'] = True
        config['general']['render_evaluation'] = True

    return config


def one_hot(value, classes=4):
    output = np.zeros(classes)
    output[value] = 1
    return output


def memoize(f):
    memo = {}

    def helper(self, x):
        if x not in memo:
            memo[x] = f(self, x)
        return memo[x]

    return helper


def evaluate_multi_agent(environment, agent, config):
    current_observation = environment.reset()

    timesteps = config['training']['timesteps']

    observations = [current_observation for _ in range(timesteps)]
    actions = [[0, 0, 0, 0] for _ in range(timesteps - 1)]

    next_observations_and_actions = np.append(observations, actions)

    score = 0.0

    for step in range(config['max_steps']):
        observations_and_actions = next_observations_and_actions
        action = agent.choose_action(observations_and_actions, policy='exploit')
        next_observation, reward, done, info = environment.step(action)

        observations = np.roll(observations, shift=1, axis=0)
        observations[0] = next_observation

        actions = np.roll(actions, shift=1, axis=0)
        actions[0] = one_hot(action)
        next_observations_and_actions = np.append(observations, actions)

        score += reward

        if config['general']['render_evaluation']:
            environment.render()

        if done:
            break

    return score


def evaluate_agent(environment, agent, config):
    if config['general']['model_type'] == MODEL_TYPE.SINGLE:
        return evaluate_single_agent(environment=environment,
                                     agent=agent,
                                     config=config)
    elif config['general']['model_type'] == MODEL_TYPE.MULTI:
        return evaluate_multi_agent(environment=environment,
                                    agent=agent,
                                    config=config)


def evaluate_single_agent(environment, agent, config):
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

    return score


def determine_uncertainties(config):
    # Determine random start position for lander
    if not config['uncertainty'].get('start_position'):
        x = random.randrange(config['uncertainty']['start_positions_x_range'][0],
                             config['uncertainty']['start_positions_x_range'][1])
        y = random.randrange(config['uncertainty']['start_positions_y_range'][0],
                             config['uncertainty']['start_positions_y_range'][1])
        config['uncertainty']['start_position'] = x, y

    # Determine random start gravity of planet
    if not config['uncertainty'].get('gravity'):
        config['uncertainty']['gravity'] = random.randrange(config['uncertainty']['gravity_range'][1],
                                                            config['uncertainty']['gravity_range'][0])

    if not config['uncertainty'].get('wind'):
        config['uncertainty']['wind'] = random.randrange(config['uncertainty']['gravity_range'][1],
                                                         config['uncertainty']['gravity_range'][0])

    return config
