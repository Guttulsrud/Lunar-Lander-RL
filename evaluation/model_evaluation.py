from utils import get_config, load_model, one_hot
from new_custom_lunar_lander import LunarLander
from random import randrange
import numpy as np


def evaluate_agent(agent, render=True, verbose=True):
    config = get_config()


    episode_scores = []
    for episode in range(100):
        pos = [randrange(0, 550), 400]
        grav = randrange(-15, -5)
        wind = randrange(1, 19)
        config['uncertainty']['random_start_position']['value'] = pos
        config['uncertainty']['gravity']['value'] = grav
        config['uncertainty']['wind']['value'] = wind

        environment = LunarLander(config)

        observation = environment.reset()
        timesteps = 4

        observations = [observation for _ in range(timesteps)]
        actions = [[0, 0, 0, 0] for _ in range(timesteps - 1)]

        next_observations_and_actions = np.append(observations, actions)
        score = 0.0

        for step in range(config['max_steps']):

            observations_and_actions = next_observations_and_actions

            prediction = agent.predict(observations_and_actions[np.newaxis, :])
            action = np.argmax(prediction)

            next_observation, reward, done, info = environment.step(action)

            observations = np.roll(observations, shift=1, axis=0)
            observations[0] = next_observation

            actions = np.roll(actions, shift=1, axis=0)
            actions[0] = one_hot(action)

            next_observations_and_actions = np.append(observations, actions)

            score += reward
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
    model = load_model(path='multi4x512 score_217')
    evaluate_agent(agent=model)
