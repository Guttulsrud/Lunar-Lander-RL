from tensorflow.python.framework.indexed_slices import tensor_spec
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


def dense_layer(num_units):
    return tf.keras.layers.Dense(num_units, activation=tf.keras.activations.relu)


class AgentController:

    def __init__(self, options, train_env):
        layers = options['layers']
        learning_rate = options['learning_rate']
        n_step_update = options['n_step_update']
        epsilon_greedy = options['epsilon_greedy']

        action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # its output.
        dense_layers = [dense_layer(num_units) for num_units in layers]
        q_values_layer = tf.keras.layers.Dense(num_actions, activation=tf.keras.activations.linear, )
        q_net = sequential.Sequential(dense_layers + [q_values_layer])

        counter = tf.Variable(0)
        self.agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            epsilon_greedy=epsilon_greedy,
            q_network=q_net,
            n_step_update=n_step_update,
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=counter)

    def get_agent(self):
        return self.agent

    # def get_policy(self, data_collection=False):
    #     policy = self.agent.collect_policy if data_collection else self.agent.policy
    #     return policy
