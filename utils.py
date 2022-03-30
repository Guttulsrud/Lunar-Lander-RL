import json

import numpy as np
import yaml
from yaml.loader import SafeLoader


def save_results_to_file(episode, avg, scores, config, file_name):

    with open(f'results/{file_name}.json', 'r') as f:
        results = json.load(f)
        results['results'].append(
            {'episode': episode,
             'average_return': avg,
             'episode_scores': scores,
             'uncertainty': config['uncertainty'],
             })
    with open(f'results/{file_name}.json', 'w') as f:
        json.dump(results, f)


def reverse_one_hot(value, length):
    output = np.zeros(length)
    output[value] = 1.0
    return output


def get_config():
    with open('config.yml') as f:
        config = yaml.load(f, Loader=SafeLoader)

    return config


def evaluate_agent(environment, agent, config, render=False, verbose=True, ):
    episode_scores = []
    if verbose:
        print('Evaluating ...')
    for episode in range(config['evaluation_episodes']):

        observation = environment.reset()
        score = 0.0

        for step in range(config['max_steps']):
            action = agent.choose_action(observation, policy='exploit')
            next_observation, reward, done, info = environment.step(action)
            score += reward
            observation = next_observation
            if render:
                environment.render()

            if done or observation[1] > 2.0:
                break

        if verbose:
            print('Reward: ', score)

        episode_scores.append(score)

    average_return = sum(episode_scores) / config['evaluation_episodes']

    if verbose:
        print(f'Average return: {average_return}')

    # environment.close()

    return average_return, episode_scores


def format_observation(observation):
    return {
        'position': {
            'x': observation[0],
            'y': observation[1],
            'angle': observation[4],
        },
        'velocity': {
            'x': observation[2],
            'y': observation[3],
            'angular_velocity': observation[5],
        },

        'left_leg_contact': observation[6],
        'right_leg_contact': observation[7]
    }
