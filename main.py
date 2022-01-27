import gym
env = gym.make('LunarLander-v2')

# env is created, now we can use it:
for episode in range(100):
    observation = env.reset()
    for step in range(50):
        env.render()
        action = env.action_space.sample()  # or given a custom model, action = policy(observation)
        observation, reward, done, info = env.step(action)

