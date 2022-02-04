from __future__ import absolute_import, division, print_function
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from AgentController import AgentController
from utils import compute_avg_return

env_name = 'LunarLander-v2'

num_iterations = 20000
initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000
batch_size = 64
log_interval = 200
num_eval_episodes = 10
eval_interval = 1000
render_environment = True

def get_action(env):
    action = env.action_space.sample()
    return action


def run():
    train_py_env = suite_gym.load(env_name)
    eval_py_env = suite_gym.load(env_name)
    env = suite_gym.load(env_name)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    options = {
        'network_shape': [8]
    }
    AG = AgentController(options, train_env, env)
    agent = AG.get_agent()

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    # agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes, render_environment)
    returns = [avg_return]

    # Reset the environment.
    time_step = train_py_env.reset()

    for _ in range(num_iterations):
        step = agent.train_step_counter.numpy()
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes, render_environment)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)


run()
