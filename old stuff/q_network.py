from tensorflow.python.framework.indexed_slices import tensor_spec
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

def dense_layer(num_units):
    return tf.keras.layers.Dense(num_units, activation=tf.keras.activations.relu)


def get_q_network(options, environment):
    action_tensor_spec = tensor_spec.from_spec(environment.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    dense_layers = [dense_layer(num_units) for num_units in options['layers']]
    q_values_layer = tf.keras.layers.Dense(num_actions, activation=tf.keras.activations.linear)
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    return q_net
