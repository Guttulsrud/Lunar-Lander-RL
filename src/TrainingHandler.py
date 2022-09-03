import itertools
import json
import os
from random import randrange
from datetime import datetime
import numpy as np

from Agent import Agent
from utils import get_config, MODEL_TYPE, evaluate_agent, one_hot
from custom_lunar_lander_environment import LunarLander


class TrainingHandler:
    def __init__(self, dev_note='', pre_trained_model=None, testing=False):
        self.testing = testing
        self.config = get_config(testing)
        self.dev_note = dev_note
        self.agent = Agent(config=self.config, pre_trained_model=pre_trained_model)

        self.created_at = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.environment = None
        self.best_score = self.config['best_score']

        if self.config['general']['save_results']:
            self.create_result_file()

    def run(self):
        for episode in range(self.config['number_of_episodes']):
            print(f'\n----- EPISODE {episode}/{self.config["number_of_episodes"]} -----\n')
            if self.config['general']['model_type'] == MODEL_TYPE.SINGLE:
                self.run_single_episode(episode)
            elif self.config['general']['model_type'] == MODEL_TYPE.MULTI:
                self.run_multi_episode(episode)

    def reload_config(self):
        self.config = get_config()

    def determine_uncertainties(self):
        gravity = self.config['uncertainty']['gravity']
        random_start_position = self.config['uncertainty']['random_start_position']
        wind = self.config['uncertainty']['wind']

        # Determine random start position for lander
        if random_start_position['enabled']:
            x_low = random_start_position['x_range'][0]
            x_high = random_start_position['x_range'][1]

            y_low = random_start_position['y_range'][0]
            y_high = random_start_position['y_range'][1]

            self.config['uncertainty']['random_start_position']['value'] = randrange(x_low, x_high), randrange(y_low,
                                                                                                               y_high)
        else:
            self.config['uncertainty']['random_start_position']['value'] = random_start_position['default']
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

    def determine_evaluation_uncertainties(self, mode):

        wind_range = self.config['uncertainty']['wind']['range']
        gravity_range = self.config['uncertainty']['gravity']['range']
        random_start_position_range = self.config['uncertainty']['random_start_position']['x_range']

        if mode == 'simple':
            step = 1
            samples = 2

        elif mode == 'robust':
            step = 2
            samples = 5

        else:
            raise 'Missing Evaluation Mode (simple/robust)'

        wind = np.round(np.linspace(wind_range[0], wind_range[1], samples + 2)[1:-1:step]).astype(int)
        gravity = np.round(np.linspace(gravity_range[0], gravity_range[1], samples + 2)[1:-1:step]).astype(int)
        start_position = np.round(
            np.linspace(random_start_position_range[0], random_start_position_range[1], samples + 2)[1:-1:step]).astype(
            int)

        start_position = [(x, 400) for x in start_position]

        return list(itertools.product(wind, gravity, start_position))

    def evaluate(self, mode):

        scores = []
        uncertainty_combinations = self.determine_evaluation_uncertainties(mode)
        print(uncertainty_combinations)

        for wind, gravity, start_position in uncertainty_combinations:

            if self.config['general']['verbose']:
                print(
                    f'Constructing LunarLander environment with gravity: {gravity}, position: {start_position}, wind: {wind}')

            self.environment = LunarLander(wind_power=wind, gravity=gravity, start_position=start_position)

            score = evaluate_agent(environment=self.environment,
                                   agent=self.agent,
                                   config=self.config)

            scores.append(score)

        return scores

    def evaluation(self, episode):
        print('Evaluating...')

        simple_eval_scores = self.evaluate(mode='simple')
        score = np.average(simple_eval_scores)

        robust_eval_scores = [0]  # ensures that we can do avg

        if self.config['robust_test_threshold'] < score:
            robust_eval_scores = self.evaluate(mode='robust')
            robust_eval_scores.append(*simple_eval_scores)
            score = np.average(robust_eval_scores)

        if self.config['general']['save_results']:
            self.agent.save_model(score=score)
            self.save_results_to_file(episode=episode,
                                      simple_eval_scores=simple_eval_scores,
                                      robust_eval_scores=robust_eval_scores)

    def run_multi_episode(self, episode):
        self.reload_config()
        self.determine_uncertainties()

        wind = self.config['uncertainty']['wind']['value']
        gravity = self.config['uncertainty']['gravity']['value']
        random_start_position = self.config['uncertainty']['random_start_position']['value']

        self.environment = LunarLander(wind_power=wind, gravity=gravity, start_position=random_start_position)
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
        self.environment = LunarLander(self.config)

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

        self.evaluation(episode)

    def save_results_to_file(self, episode, simple_eval_scores, robust_eval_scores):
        with open(f'../results/{self.created_at}.json', 'r') as f:
            results = json.load(f)
            results['results'].append(
                {
                    'episode': episode,
                    'simple_eval_scores': simple_eval_scores,
                    'robust_eval_scores': robust_eval_scores,
                    'uncertainty': self.config['uncertainty'],
                })

        with open(f'../results/{self.created_at}.json', 'w') as f:
            json.dump(results, f)

    def create_result_file(self):
        is_dir = os.path.isdir('../results')

        if not is_dir:
            os.mkdir('../results')

        with open(f'../results/{self.created_at}.json', 'w') as f:
            json.dump({'note': self.dev_note, 'results': [], 'config': self.config}, f)
