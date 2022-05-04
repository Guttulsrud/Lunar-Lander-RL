import json
import os
from random import randrange
from datetime import datetime
from Agent import Agent
from utils import get_config, evaluate_agent
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from custom_lunar_lander import LunarLander


class Handler:
    def __init__(self, dev_note, pre_trained_model=None):
        self.config = get_config()
        self.dev_note = dev_note + str(self.config['number_of_episodes'])

        self.agent = Agent(config=self.config, pre_trained_model=pre_trained_model)

        self.created_at = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.environment = None
        self.best_score_threshold = self.config['best_score_threshold']

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

    def evaluate(self, episode):
        self.reload_config()
        # self.determine_uncertainties()
        self.config['uncertainty']['gravity'] = -5
        self.environment = LunarLander(self.config)
        avg1, scores1 = evaluate_agent(self.environment, self.agent, self.config)

        self.reload_config()
        self.config['uncertainty']['gravity'] = -15
        self.environment = LunarLander(self.config)
        avg2, scores2 = evaluate_agent(self.environment, self.agent, self.config)

        avg = (avg1 + avg2) / 2
        scores = scores1 + scores2

        if avg > self.best_score_threshold:
            self.best_score_threshold = avg
            self.agent.save_model(score=avg)

        if self.config['general']['save_results']:
            self.save_results_to_file(episode=episode, avg=avg, scores=scores)

    def run_episode(self, episode):
        self.reload_config()
        self.determine_uncertainties()

        self.environment = LunarLander(self.config)

        print(f'Episode: {episode}.')
        observation = self.environment.reset()

        for step in range(self.config['max_steps']):
            if self.config['general']['render_env']:
                self.environment.render()

            action = self.agent.choose_action(observation)
            next_observation, reward, done, info = self.environment.step(action)

            self.agent.remember(observation, action, reward, next_observation, done)
            observation = next_observation
            self.agent.learn()

            if done:
                break
            # if terminate_episode(observation, done):
            #     break

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
