from __future__ import absolute_import, division, print_function

from tf_agents.policies import random_tf_policy

from AgentController import AgentController
from utils import compute_avg_return
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import matplotlib.pyplot as plt

env_name = 'LunarLander-v2'

num_iterations = 8000
collect_steps_per_iteration = 10
num_eval_episodes = 1
batch_size = 32
log_interval = 250
eval_interval = 250
replay_buffer_capacity = 150_000
config = {

    "render_eval_env": True,
    "render_training_env": False,
    "agent": {
        "layers": [64, 64],
        "learning_rate": 1e-3,
        "n_step_update": 2,
        "epsilon_greedy": 0.2
    },
    "initial_collect_steps": 5000,
}


def run():
    train_py_env = suite_gym.load(env_name)
    eval_py_env = suite_gym.load(env_name)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    render_training_env = config['render_training_env']
    render_eval_env = config['render_eval_env']
    n_step_update = config['agent']['n_step_update']

    AG = AgentController(config['agent'], train_env)  # Can be function
    agent = AG.get_agent()
    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)



    def collect_step(environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())
    for _ in range(config['initial_collect_steps']):
        collect_step(train_env, random_policy)

    # This loop is so common in RL, that we provide standard implementations of
    # these. For more details see the drivers module.

    # Dataset generates trajectories with shape [BxTx...] where
    # T = n_step_update + 1.
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size,
        num_steps=n_step_update + 1).prefetch(3)

    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    # agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):
        # Add an observer that adds to the replay buffer:
        replay_observer = [replay_buffer.add_batch]

        dynamic_step_driver.DynamicStepDriver(
            train_env,
            agent.collect_policy,
            observers=replay_observer,
            num_steps=collect_steps_per_iteration).run()
        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience)

        step = agent.train_step_counter.numpy()
        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))
            print('_____________________________________________________')

        if render_training_env:
            train_env.render()

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes, render_eval_env)
            print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
            returns.append(avg_return)


    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=250)
    plt.show()

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
