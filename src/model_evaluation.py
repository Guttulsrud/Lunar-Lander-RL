from utils import get_config, load_model
from custom_lunar_lander import LunarLander
from random import randrange
import numpy as np


def evaluate_agent(agent, render=True, verbose=True):
    config = get_config()


    episode_scores = []
    for episode in range(100):
        #gravity_low = config['uncertainty']['gravity_range'][1]
        #gravity_high = config['uncertainty']['gravity_range'][0]

 #       config['uncertainty']['gravity'] = randrange(-7, -5)
        # config['uncertainty']['gravity'] = randrange(gravity_low, gravity_high)
        #print('Gravity: ', config['uncertainty']['gravity'])
        pos = [randrange(0, 550), 0]
        config['uncertainty']['random_start_position']['value'] = pos
        print('Position: ', pos)

        environment = LunarLander(config)

        observation = environment.reset()
        score = 0.0

        for step in range(config['max_steps']):
            prediction = agent.predict(observation[np.newaxis, :])
            action = np.argmax(prediction)
            next_observation, reward, done, info = environment.step(action)
            score += reward
            observation = next_observation
            if render:
                environment.render()

            if done:
                break

        if verbose:
            print('Reward: ', score)

        episode_scores.append(score)

    average_return = sum(episode_scores) / config['evaluation_episodes']

    if verbose:
        print(f'Average return: {average_return}')


if __name__ == '__main__':
    model = load_model(path='22-05-12_12-27_SCORE_281')
    evaluate_agent(agent=model)
