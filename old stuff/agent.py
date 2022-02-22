import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from q_network import get_q_network


def get_agent(options, environment):
    learning_rate = options['learning_rate']
    n_step_update = options['n_step_update']
    epsilon_greedy = options['epsilon_greedy']

    q_network = get_q_network(options, environment)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        environment.time_step_spec(),
        environment.action_spec(),
        epsilon_greedy=epsilon_greedy,
        q_network=q_network,
        n_step_update=n_step_update,
        optimizer=optimizer,
        gamma=0.99,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=counter)

    return agent
