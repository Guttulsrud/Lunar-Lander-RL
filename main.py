import gym

env = gym.make('LunarLander-v2')


def handle_observation(observation):
    # Observation Space
    # There are 8 states: the coordinates of the lander in `x` & `y`, its linear
    # velocities in `x` & `y`, its angle, its angular velocity, and two boleans
    # showing if each leg is in contact with the ground or not.

    x_coordinate = observation[0]
    y_coordinate = observation[1]
    x_linear_velocity = observation[2]
    y_linear_velocity = observation[3]
    angle = observation[4]
    angular_velocity = observation[5]
    left_leg_contact = observation[6]
    right_leg_contact = observation[7]

    return 'nje'


def get_action(env):
    # Action Space
    # There are four discrete actions available: do nothing, fire left
    # orientation engine, fire main engine, fire right orientation engine.
    action = env.action_space.sample()
    return action


for episode in range(100):
    observation = env.reset()
    for step in range(50):
        env.render()
        action = get_action(env)

        observation, reward, done, info = env.step(action)

        handle_observation(observation)

# Rewards
# Reward for moving from the top of the screen to the landing pad and zero
# speed is about 100..140 points.
# If the lander moves away from the landing pad it loses reward.
# If the lander crashes, it receives an additional -100 points. If it comes
# to rest, it receives an additional +100 points. Each leg with ground
# contact is +10 points.
# Firing the main engine is -0.3 points each frame. Firing the side engine
# is -0.03 points each frame. Solved is 200 points.
