from Agent import Agent
from utils import get_config, evaluate_agent
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from custom_lunar_lander import LunarLander

if __name__ == '__main__':

    config = get_config()
    agent = Agent(config)
    env = LunarLander(config)

    for episode in range(config['number_of_episodes']):
        print(f'Episode: {episode}. \nCollecting data ...')

        observation = env.reset()
        for step in range(config['max_steps']):
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
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
                {'episode': episode, 'average_return': avg, 'episode_scores': scores, })
        with open('results.json', 'w') as f:
            json.dump(results, f)
