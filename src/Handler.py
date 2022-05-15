import json
import os
from random import randrange
from datetime import datetime
import requests
import numpy as np

from Agent import Agent
from utils import get_config, evaluate_double_agent, evaluate_single_agent, MODEL_TYPE, evaluate_agent
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Disables GPU
from new_custom_lunar_lander import LunarLander


# from custom_lunar_lander import LunarLander


# todo: Get better name for this
class Handler:
    def __init__(self, dev_note, pre_trained_model=None):
        self.config = get_config()
        self.dev_note = dev_note

        self.agent = Agent(config=self.config, pre_trained_model=pre_trained_model)

        self.created_at = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.environment = None
        self.best_score = self.config['best_score']

        for x in os.listdir('../models'):
            if int(x.split('SCORE_')[1]) > self.best_score:
                self.best_score = int(x.split('SCORE_')[1])

        if self.config['general']['save_results']:
            self.create_result_file()

    def run(self):
        for episode in range(self.config['number_of_episodes']):
            print(f'\n----- EPISODE {episode}/{self.config["number_of_episodes"]} -----\n')
            if self.config['general']['model_type'] == MODEL_TYPE.SINGLE:
                self.run_single_episode(episode)
            elif self.config['general']['model_type'] == MODEL_TYPE.DOUBLE:
                self.run_double_episode(episode)

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

    def simulate(self, robust=False):

        avg_total = []
        scores_total = []

        evaluations_per_configuration = 3 if robust else 1
        number_of_configurations = 11 if robust else 3

        if self.config['uncertainty']['gravity']['enabled']:
            gravity_config = list(
                np.linspace(-15, -5, number_of_configurations, dtype=int)) * evaluations_per_configuration
        else:
            default_gravity = self.config['uncertainty']['gravity']['default']
            gravity_config = np.full(number_of_configurations * evaluations_per_configuration, default_gravity)

        if self.config['uncertainty']['wind']['enabled']:
            wind_config = list(
                np.linspace(1, 19, number_of_configurations, dtype=int)) * evaluations_per_configuration
        else:
            default_wind = self.config['uncertainty']['wind']['default']
            wind_config = np.full(number_of_configurations * evaluations_per_configuration, default_wind)

        if self.config['uncertainty']['random_start_position']['enabled']:
            x = np.linspace(0, 551, number_of_configurations, dtype=int)
            y = [400 for _ in range(number_of_configurations)]
            position_config = list(zip(x, y)) * evaluations_per_configuration
        else:
            default_start_position = self.config['uncertainty']['random_start_position']['default']
            position_config = np.full((number_of_configurations * evaluations_per_configuration, 2),
                                      default_start_position)

        np.random.shuffle(position_config)
        np.random.shuffle(gravity_config)
        np.random.shuffle(wind_config)

        for position, gravity, wind in zip(position_config, gravity_config, wind_config):
            self.config['uncertainty']['gravity']['value'] = int(gravity)
            self.config['uncertainty']['wind']['value'] = int(wind)
            self.config['uncertainty']['random_start_position']['value'] = int(position[0]), int(position[1])
            self.environment = LunarLander(self.config)

            avg, scores = evaluate_agent(environment=self.environment,
                                         agent=self.agent,
                                         config=self.config,
                                         episodes=1)

            avg_total.append(avg)
            scores_total += scores

        avg = np.mean(avg_total)
        return avg, scores_total

    def evaluate(self, episode):
        print('Evaluating..')
        self.reload_config()
        self.determine_uncertainties()

        avg, scores = self.simulate()
        print('Agent got average return on simple evaluation: ', avg)

        if avg > self.config['robust_test_threshold']:
            print('Agent received score above threshold')
            avg, scores = self.simulate(robust=True)
            print(f'Agent received score of {avg} on robust test')

            if avg > self.best_score:
                print('Agent is new best. Saving model')
                self.best_score = avg
                self.agent.save_model(score=avg)

        if self.config['general']['save_results']:
            self.save_results_to_file(episode=episode, avg=avg, scores=scores)

    def run_double_episode(self, episode):
        self.reload_config()
        self.determine_uncertainties()
        self.environment = LunarLander(self.config)

        current_observation = self.environment.reset()
        previous_observation = current_observation

        for step in range(self.config['max_steps']):
            if self.config['general']['render_training']:
                self.environment.render()

            previous_and_current_observation = np.append(previous_observation, current_observation)
            action = self.agent.choose_action(previous_and_current_observation)

            next_observation, reward, done, info = self.environment.step(action)
            current_and_next_observation = np.append(current_observation, next_observation)

            self.agent.remember(previous_and_current_observation, action, reward, current_and_next_observation, done)
            previous_observation = current_observation
            current_observation = next_observation

            self.agent.learn()

            if done:
                break

        self.evaluate(episode)

    def run_single_episode(self, episode):
        self.reload_config()
        self.determine_uncertainties()
        self.environment = LunarLander(self.config)

        observation = self.environment.reset()

        print(f'Training..')
        for step in range(self.config['max_steps']):
            if self.config['general']['render_training']:
                self.environment.render()

            action = self.agent.choose_action(observation)
            next_observation, reward, done, info = self.environment.step(action)

            self.agent.remember(observation, action, reward, next_observation, done)
            observation = next_observation
            self.agent.learn()

            if done:
                break

        self.evaluate(episode)

    def save_results_to_file(self, episode, avg, scores):
        with open(f'../results/{self.created_at}.json', 'r') as f:
            results = json.load(f)
            results['results'].append(
                {
                    'episode': episode,
                    'average_return': avg,
                    'episode_scores': scores,
                    'uncertainty': self.config['uncertainty'],
                })

        print(results)
        with open(f'../results/{self.created_at}.json', 'w') as f:
            json.dump(results, f)

        # requests.post('http://localhost:5000/send', json={"data": avg, "title": self.dev_note, "training": False})

    def create_result_file(self):

        is_dir = os.path.isdir('../results')

        if not is_dir:
            os.mkdir('../results')

        with open(f'../results/{self.created_at}.json', 'w') as f:
            json.dump({'note': self.dev_note, 'results': [], 'config': self.config}, f)


def terminate_episode(observation, done):
    x = observation[0]
    y = observation[1]
    if done:
        return True
