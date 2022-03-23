import random

from Agent import Agent
from utils import get_config, evaluate_agent
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from custom_lunar_lander import LunarLander
import json

if __name__ == '__main__':

    config = get_config()
    agent = Agent(config)

    for episode in range(config['number_of_episodes']):
        config['environment']['start_position'] = random.choice(config['environment']['start_positions'])
        env = LunarLander(config)

        print(f'Episode: {episode}. \nCollecting data ...')
        env.reset()
        observation = env.render(mode="rgb_array")

        for step in range(config['max_steps']):
            observation = observation[:, :, 0]
            print(observation.shape)

            action = agent.choose_action(observation)
            _, reward, done, info = env.step(action)
            next_observation = env.render(mode="rgb_array")

            agent.remember(observation, action, reward, next_observation, done)
            observation = next_observation
            agent.learn()

            if done:
                break

        avg, scores = evaluate_agent(env, agent, config)

        # write results to file
        with open('results.json', 'r') as f:
            results = json.load(f)
            results['results'].append(
                {'episode': episode,
                 'average_return': avg,
                 'episode_scores': scores,
                 'spawn': config['environment']['start_positions']
                 })
        with open('results.json', 'w') as f:
            json.dump(results, f)
