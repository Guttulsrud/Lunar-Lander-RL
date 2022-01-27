import gym

env = gym.make('LunarLander-v2')


def format_observation(observation):
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

    # Observation Space
    # There are 8 states: the coordinates of the lander in `x` & `y`, its linear
    # velocities in `x` & `y`, its angle, its angular velocity, and two boleans
    # showing if each leg is in contact with the ground or not.


def get_action(env):
    # Action Space
    # There are four discrete actions available: do nothing, fire left
    # orientation engine, fire main engine, fire right orientation engine.
    action = env.action_space.sample()
    return action


# Starting State
# The lander starts at the top center of the viewport with a random initial
# force applied to its center of mass.
#
# ### Episode Termination
# The episode finishes if:
# 1) the lander crashes (the lander body gets in contact with the moon);
# 2) the lander gets outside of the viewport (`x` coordinate is greater than 1);
# 3) the lander is not awake.
#       From the [Box2D docs](https://box2d.org/documentation/md__d_1__git_hub_box2d_docs_dynamics.html#autotoc_md61),
#     a body which is not awake is a body which doesn't move and doesn't
#     collide with any other body:
# > When Box2D determines that a body (or group of bodies) has come to rest,
# > the body enters a sleep state which has very little CPU overhead. If a
# > body is awake and collides with a sleeping body, then the sleeping body
# > wakes up. Bodies will also wake up if a joint or contact attached to
# > them is destroyed.

class Agent:

    def __init__(self):
        print('nje')

    def run(self, env, observation):
        action = env.action_space.sample()
        return action


def run():
    agent = Agent()

    for episode in range(100):
        observation = format_observation(env.reset())

        for step in range(50):
            env.render()
            action = agent.run(env, observation)

            observation, reward, done, info = env.step(action)

            observation = format_observation(observation)


run()
# Rewards
# Reward for moving from the top of the screen to the landing pad and zero
# speed is about 100..140 points.
# If the lander moves away from the landing pad it loses reward.
# If the lander crashes, it receives an additional -100 points. If it comes
# to rest, it receives an additional +100 points. Each leg with ground
# contact is +10 points.
# Firing the main engine is -0.3 points each frame. Firing the side engine
# is -0.03 points each frame. Solved is 200 points.
