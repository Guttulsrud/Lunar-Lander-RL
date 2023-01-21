import itertools
import json
import os
from random import randrange
from datetime import datetime
import numpy as np
import optuna

from Agent import Agent
from utils import get_config, MODEL_TYPE, evaluate_agent, one_hot, memoize
from custom_lunar_lander_environment import LunarLander


def run_episode(wind, gravity, start_position, agent, config):
    environment = LunarLander(wind_power=wind, gravity=gravity, start_position=start_position)

    score = evaluate_agent(environment=environment,
                           agent=agent,
                           config=config)

    return score


class TrainingHandler(object):

    def __init__(self, thread, trial, hparams, dev_note='', pre_trained_model=None, testing=False):
        self.testing = testing
        self.config = get_config(testing)
        self.hparams = hparams
        self.config['training']['input_shape'] = self.hparams['timesteps'] * 8 + (self.hparams['timesteps'] - 1) * 4
        self.dev_note = dev_note
        # self.agent = Agent(config=self.config, pre_trained_model=pre_trained_model)
        self.agent = None
        self.created_at = None
        self.environment = None
        self.thread = thread
        self.trial = trial
        self.best_score = self.config['best_score']

        # if self.config['general']['save_results']:
        #     self.create_result_file()

    def run(self, hparams):
        self.created_at = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.created_at += f'_{randrange(1000000000000)}'
        if self.config['general']['save_results']:
            self.create_result_file()

        self.agent = Agent(config=self.config, hparams=hparams)

        for episode in range(self.config['number_of_episodes']):
            print(
                f'\nTHREAD {self.thread + 1} // TRIAL {self.trial + 1} ----- EPISODE {episode + 1}/{self.config["number_of_episodes"]}')
            if self.config['general']['model_type'] == MODEL_TYPE.SINGLE:
                score = self.run_single_episode(episode)
            elif self.config['general']['model_type'] == MODEL_TYPE.MULTI:
                score = self.run_multi_episode(episode)

        return score

    def reload_config(self):
        self.config = get_config()
        self.config['training']['timesteps'] = self.hparams['timesteps']

    def determine_uncertainties(self):
        gravity = self.config['uncertainty']['gravity']
        start_position = self.config['uncertainty']['start_position']
        wind = self.config['uncertainty']['wind']

        # Determine random start position for lander
        if start_position['enabled']:
            x_low = start_position['x_range'][0]
            x_high = start_position['x_range'][1]

            y_low = start_position['y_range'][0]
            y_high = start_position['y_range'][1]

            self.config['uncertainty']['start_position']['value'] = randrange(x_low, x_high), randrange(y_low,
                                                                                                        y_high)
        else:
            self.config['uncertainty']['start_position']['value'] = start_position['default']
        # Determine random start gravity of planet
        if gravity['enabled']:
            gravity_low = gravity['range'][0]
            gravity_high = gravity['range'][1]

            self.config['uncertainty']['gravity']['value'] = randrange(gravity_low, gravity_high)
        else:
            self.config['uncertainty']['gravity']['value'] = gravity['default']

        if wind['enabled']:
            wind_low = wind['range'][0]
            wind_high = wind['range'][1]

            self.config['uncertainty']['wind']['value'] = randrange(wind_low, wind_high)
        else:
            self.config['uncertainty']['wind']['value'] = wind['default']

    @memoize
    def determine_evaluation_uncertainties(self, mode):

        wind_range = self.config['uncertainty']['wind']['range']
        gravity_range = self.config['uncertainty']['gravity']['range']
        start_position_range = self.config['uncertainty']['start_position']['x_range']

        if mode == 'simple':
            step = 1
            samples = 2
        elif mode == 'robust':
            step = 2
            samples = 5
        else:
            raise 'Missing Evaluation Mode (simple/robust)'

        if self.config['uncertainty']['wind']['enabled']:
            wind = np.round(np.linspace(wind_range[0], wind_range[1], samples + 2)[1:-1:step])
        else:
            wind = [self.config['uncertainty']['wind']['default']]

        if self.config['uncertainty']['gravity']['enabled']:
            gravity = np.round(np.linspace(gravity_range[0], gravity_range[1], samples + 2)[1:-1:step])
        else:
            gravity = [self.config['uncertainty']['gravity']['default']]

        if self.config['uncertainty']['start_position']['enabled']:
            start_position = np.round(
                np.linspace(start_position_range[0], start_position_range[1], samples + 2)[1:-1:step])
            start_position = [(x, 400) for x in start_position]
        else:
            start_position = [self.config['uncertainty']['start_position']['default']]

        wind = [int(x) for x in wind]
        gravity = [int(x) for x in gravity]
        start_position = [(int(x), y) for x, y in start_position]

        return list(itertools.product(wind, gravity, start_position))

    def evaluate(self, mode):

        scores = []
        uncertainty_combinations = self.determine_evaluation_uncertainties(mode)

        for wind, gravity, start_position in uncertainty_combinations:
            score = run_episode(wind, gravity, start_position, self.agent, self.config)
            scores.append(score)

        # print(scores)

        return scores

    def evaluation(self, episode):
        # print('Evaluating...')

        simple_eval_scores = self.evaluate(mode='simple')
        score = np.average(simple_eval_scores)

        # print(f'Received average score of {score} on simple evaluation')

        robust_eval_scores = [0]  # ensures that we can do avg

        if self.config['robust_test_threshold'] < score:
            robust_eval_scores = self.evaluate(mode='robust')
            robust_eval_scores += simple_eval_scores
            score = np.average(robust_eval_scores)

            # print(f'Received average score of {score} on robust evaluation')

        if self.config['general']['save_results']:
            # self.agent.save_model(score=score)
            self.save_results_to_file(episode=episode,
                                      simple_eval_scores=simple_eval_scores,
                                      robust_eval_scores=robust_eval_scores)

        return score

    def run_multi_episode(self, episode):
        self.reload_config()
        self.determine_uncertainties()

        wind = self.config['uncertainty']['wind']['value']
        gravity = self.config['uncertainty']['gravity']['value']
        start_position = self.config['uncertainty']['start_position']['value']

        self.environment = LunarLander(wind_power=wind, gravity=gravity, start_position=start_position)
        timesteps = self.config['training']['timesteps']

        current_observation = self.environment.reset()

        observations = [current_observation for _ in range(timesteps)]
        actions = [[0, 0, 0, 0] for _ in range(timesteps - 1)]

        next_observations_and_actions = np.append(observations, actions)

        score = 0.0

        for _ in range(self.config['max_steps']):
            if self.testing:
                break

            if self.config['general']['render_training']:
                self.environment.render()

            observations_and_actions = next_observations_and_actions

            action = self.agent.choose_action(observations_and_actions)
            next_observation, reward, done, info = self.environment.step(action)

            score += reward

            observations = np.roll(observations, shift=1, axis=0)
            observations[0] = next_observation

            actions = np.roll(actions, shift=1, axis=0)
            actions[0] = one_hot(action)

            next_observations_and_actions = np.append(observations, actions)

            self.agent.remember(observations_and_actions, action, reward, next_observations_and_actions, done)
            self.agent.learn()

            if done:
                break

        self.evaluation(episode)

    def run_single_episode(self, episode):
        self.reload_config()
        self.determine_uncertainties()
        wind = self.config['uncertainty']['wind']['value']
        gravity = self.config['uncertainty']['gravity']['value']
        start_position = self.config['uncertainty']['start_position']['value']

        self.environment = LunarLander(wind_power=wind, gravity=gravity, start_position=start_position)

        observation = self.environment.reset()

        for step in range(self.config['max_steps']):
            if self.testing:
                break
            if self.config['general']['render_training']:
                self.environment.render()

            action = self.agent.choose_action(observation)
            next_observation, reward, done, info = self.environment.step(action)

            self.agent.remember(observation, action, reward, next_observation, done)
            self.agent.learn()

            observation = next_observation

            if done:
                break

        return self.evaluation(episode)

    def save_results_to_file(self, episode, simple_eval_scores, robust_eval_scores):
        with open(f'../results/{self.created_at}_THREAD{self.thread + 1}_TRIAL{self.trial + 1}.json', 'r') as f:
            results = json.load(f)
            results['results'].append(
                {
                    'episode': episode,
                    'simple_eval_scores': simple_eval_scores,
                    'robust_eval_scores': robust_eval_scores,
                    'uncertainty': self.config['uncertainty'],
                })

        with open(f'../results/{self.created_at}_THREAD{self.thread + 1}_TRIAL{self.trial + 1}.json', 'w') as f:
            json.dump(results, f)

    def create_result_file(self):
        is_dir = os.path.isdir('../results')

        if not is_dir:
            os.mkdir('../results')

        with open(f'../results/{self.created_at}_THREAD{self.thread + 1}_TRIAL{self.trial + 1}.json', 'w') as f:
            json.dump({'note': self.dev_note, 'results': [], 'config': self.config, 'hparams': self.hparams}, f)
