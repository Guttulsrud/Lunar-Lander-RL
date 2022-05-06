import json
import os
from random import randrange
from datetime import datetime
import requests
import numpy as np

from Agent import Agent
from utils import get_config, evaluate_agent
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from custom_lunar_lander import LunarLander


# todo: Get better name for this
class Handler:
    def __init__(self, dev_note, pre_trained_model=None):
        self.config = get_config()
        self.dev_note = dev_note + str(self.config['number_of_episodes'])

        self.agent = Agent(config=self.config, pre_trained_model=pre_trained_model)

        self.created_at = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.environment = None
        self.best_score_threshold = self.config['best_score_threshold']
        self.best_score = -30_000
        if self.config['general']['save_results']:
            self.create_result_file()

    def run(self):
        for episode in range(self.config['number_of_episodes']):
            self.run_episode(episode)

    def reload_config(self):
        self.config = get_config()

    def determine_uncertainties(self):
        # Determine random start position for lander
        if not self.config['uncertainty'].get('start_position'):
            x_low = self.config['uncertainty']['start_positions_x_range'][0]
            x_high = self.config['uncertainty']['start_positions_x_range'][1]

            y_low = self.config['uncertainty']['start_positions_y_range'][0]
            y_high = self.config['uncertainty']['start_positions_y_range'][1]

            self.config['uncertainty']['start_position'] = randrange(x_low, x_high), randrange(y_low, y_high)

        # Determine random start gravity of planet
        if not self.config['uncertainty'].get('gravity'):
            gravity_low = self.config['uncertainty']['gravity_range'][1]
            gravity_high = self.config['uncertainty']['gravity_range'][0]

            self.config['uncertainty']['gravity'] = randrange(gravity_low, gravity_high)

    def simulate(self, gravity_range=-5):
        print(gravity_range)
        if gravity_range == 1:
            gravity_max = -15
        else:
            gravity_max = -20
        avg_total = []
        scores_total = []
        gravity = range(-5, gravity_max, gravity_range)
        length = len(gravity)
        for progress, gravity in enumerate(gravity):
            self.config['uncertainty']['gravity'] = gravity

            self.environment = LunarLander(self.config)
            avg, scores = evaluate_agent(self.environment, self.agent, self.config, progress + 1, length, render=True)
            avg_total.append(avg)
            scores_total += scores

        avg = np.mean(avg_total)
        scores = np.mean(scores_total)
        return avg, scores

    def evaluate(self, episode):

        self.reload_config()
        # self.determine_uncertainties()
        avg, scores = self.simulate()
        print('Average: ', avg)

        if avg > self.best_score:
            print('New best model!')
            self.best_score = avg
            self.agent.save_model(score=avg)
            self.simulate(gravity_range=-1)

        if self.config['general']['save_results']:
            self.save_results_to_file(episode=episode, avg=avg, scores=scores)

    def run_episode(self, episode):
        self.reload_config()
        self.determine_uncertainties()

        self.environment = LunarLander(self.config)

        print(f'Episode: {episode}.')
        current_observation = self.environment.reset()
        previous_observation = current_observation

        for step in range(self.config['max_steps']):
            # if self.config['general']['render_env']:
            #     self.environment.render()

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
        with open(f'../results/{self.created_at}.json', 'w') as f:
            json.dump(results, f)

        requests.post('http://localhost:5000/send', json={"data": avg, "title": self.dev_note})

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
