from __future__ import absolute_import, division, print_function
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from AgentController import AgentController
from utils import compute_avg_return
from tf_agents import specs
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step, trajectory
from tf_agents.utils import common

env_name = 'LunarLander-v2'

num_iterations = 20000
initial_collect_steps = 100
collect_steps_per_iteration = 10
replay_buffer_max_length = 100000
batch_size = 64
log_interval = 200
num_eval_episodes = 10
eval_interval = 1000
render_environment = False
max_length = 1000
replay_buffer_capacity = 1000
num_train_steps = 10
n_step_update = 2  # @param {type:"integer"}


def get_action(env):
    action = env.action_space.sample()
    return action


def run():
    train_py_env = suite_gym.load(env_name)
    eval_py_env = suite_gym.load(env_name)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # tf_env = suite_gym.load(env_name)
    # environment = tf_py_environment.TFPyEnvironment(tf_env)

    options = {
        'network_shape': [8]
    }

    AG = AgentController(options, train_env)  # Can be function
    agent = AG.get_agent()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

    # Add an observer that adds to the replay buffer:
    replay_observer = [replay_buffer.add_batch]

    collect_op = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers=replay_observer,
        num_steps=collect_steps_per_iteration).run()

    def collect_step(environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]


    for _ in range(num_iterations):
        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience)

        step = agent.train_step_counter.numpy()
        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        train_env.render()
        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes, True)
            print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
            returns.append(avg_return)

        # trajectories, _ = next(iterator)
        # loss = agent.train(experience=trajectories)
        #
        # # replay_buffer.clear()
        # step = agent.train_step_counter.numpy()
        # avg_return = compute_avg_return(environment, agent.policy, num_eval_episodes, False)
        # print('step = {0}: Average Return = {1}'.format(step, avg_return))
        # returns.append(avg_return)
    train_env.close()
    eval_env.close()
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    # agent.train = common.function(agent.train)


#     for _ in range(num_iterations):
#         replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
#             agent.collect_data_spec,
#             batch_size=batch_size,
#             max_length=max_length)
#         print(agent.collect_data_spec)
#
#         step = agent.train_step_counter.numpy()
#         avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes, render_environment)
#         print('step = {0}: Average Return = {1}'.format(step, avg_return))
#         # returns.append(avg_return)
#
#
run()
