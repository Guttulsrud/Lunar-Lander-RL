from __future__ import absolute_import, division, print_function

import numpy as np
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


env_name = 'LunarLander-v2'

num_iterations = 20000
initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000
batch_size = 64
log_interval = 200
num_eval_episodes = 10
eval_interval = 1000


def get_action(env):
    action = env.action_space.sample()
    return action


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def run():
    pass
    #train_py_env = suite_gym.load(env_name)
    # eval_py_env = suite_gym.load(env_name)
    # env = suite_gym.load(env_name)
    # train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    # eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    #
    # options = {
    #     'network_shape': [8]
    # }
    # AG = AgentController(options, train_env, env)
    # agent = AG.get_agent()
    #
    # table_name = 'uniform_table'
    # replay_buffer_signature = tensor_spec.from_spec(
    #     agent.collect_data_spec)
    # replay_buffer_signature = tensor_spec.add_outer_dim(
    #     replay_buffer_signature)
    #
    # table = reverb.Table(
    #     table_name,
    #     max_size=replay_buffer_max_length,
    #     sampler=reverb.selectors.Uniform(),
    #     remover=reverb.selectors.Fifo(),
    #     rate_limiter=reverb.rate_limiters.MinSize(1),
    #     signature=replay_buffer_signature)
    #
    # reverb_server = reverb.Server([table])
    #
    # replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    #     agent.collect_data_spec,
    #     table_name=table_name,
    #     sequence_length=2,
    #     local_server=reverb_server)
    #
    # rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    #     replay_buffer.py_client,
    #     table_name,
    #     sequence_length=2)
    #
    # py_driver.PyDriver(
    #     train_py_env,
    #     py_tf_eager_policy.PyTFEagerPolicy(
    #         AG.get_policy(), use_tf_function=True),
    #     [rb_observer],
    #     max_steps=initial_collect_steps).get_policy(train_py_env.reset())
    #
    # dataset = replay_buffer.as_dataset(
    #     num_parallel_calls=3,
    #     sample_batch_size=batch_size,
    #     num_steps=2).prefetch(3)
    # iterator = iter(dataset)
    #
    # # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    # # agent.train = common.function(agent.train)
    #
    # # Reset the train step.
    # agent.train_step_counter.assign(0)
    #
    # # Evaluate the agent's policy once before training.
    # avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    # returns = [avg_return]
    #
    # # Reset the environment.
    # time_step = train_py_env.reset()
    #
    # # Create a driver to collect experience.
    # collect_driver = py_driver.PyDriver(
    #     env,
    #     py_tf_eager_policy.PyTFEagerPolicy(
    #         agent.collect_policy, use_tf_function=True),
    #     [rb_observer],
    #     max_steps=collect_steps_per_iteration)
    #
    # for _ in range(num_iterations):
    #
    #     # Collect a few steps and save to the replay buffer.
    #     time_step, _ = collect_driver.run(time_step)
    #
    #     # Sample a batch of data from the buffer and update the agent's network.
    #     experience, unused_info = next(iterator)
    #     train_loss = agent.train(experience).loss
    #
    #     step = agent.train_step_counter.numpy()
    #
    #     if step % log_interval == 0:
    #         print('step = {0}: loss = {1}'.format(step, train_loss))
    #
    #     if step % eval_interval == 0:
    #         avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    #         print('step = {0}: Average Return = {1}'.format(step, avg_return))
    #         returns.append(avg_return)


run()
