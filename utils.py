def format_observation(observation):
    observation = observation['observation']
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


def compute_avg_return(environment, policy, num_episodes=10, render=False):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward

            if render:
                environment.render()

        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]
