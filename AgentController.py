
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))


class AgentController:

    def __init__(self, options, train_env, env):
        fc_layer_params = (100, 50)
        action_tensor_spec = tensor_spec.from_spec(env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        # Define a helper function to create Dense layers configured with the right
        # activation and kernel initializer.
        def dense_layer(num_units):
            return tf.keras.layers.Dense(
                num_units,
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=2.0, mode='fan_in', distribution='truncated_normal'))

        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # its output.
        dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
        q_net = sequential.Sequential(dense_layers + [q_values_layer])

        # q_net = Sequential()
        # q_net.add(Dense(8, input_shape=(8,)))
        # q_net.add(Dense(4))
        # q_net.compile(optimizer='adam', loss='mse')

        self.agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=Adam(),
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=tf.Variable(0))

        self.agent.initialize()

    def get_agent(self):
        return self.agent

    def get_policy(self, data_collection=False):
        policy = self.agent.collect_policy if data_collection else self.agent.policy
        return policy
