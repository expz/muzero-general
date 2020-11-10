import math
from abc import ABC, abstractmethod

import tensorflow as tf


class MuZeroNetwork:
    def __new__(cls, config):
        if config.network == "fullyconnected":
            return MuZeroFullyConnectedNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
            )
        elif config.network == "resnet":
            return MuZeroResidualNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.blocks,
                config.channels,
                config.reduced_channels_reward,
                config.reduced_channels_value,
                config.reduced_channels_policy,
                config.resnet_fc_reward_layers,
                config.resnet_fc_value_layers,
                config.resnet_fc_policy_layers,
                config.support_size,
                config.downsample,
            )
        else:
            raise NotImplementedError(
                'The network parameter should be "fullyconnected" or "resnet".'
            )


class AbstractNetwork(ABC, tf.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation, training=False):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action, training=False):
        pass

    def get_weights(self):
        return {
            var.name: tf.convert_to_tensor(var).numpy()
            for var in self.variables
        }

    def set_weights(self, weights):
        vars = {var.name: var for var in self.variables}
        for name in weights:
            if name in vars:
                vars[name].assign(weights[name])


##################################
######## Fully Connected #########


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        # TODO: Replace removed torch.nn.DataParallel wrapper.
        self.representation_network = (
            mlp(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size,
                name='representation',
            )
        )

        self.dynamics_encoded_state_network = (
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
                name='dynamics',
            )
        )
        self.dynamics_reward_network = (
            mlp(encoding_size, fc_reward_layers, self.full_support_size, name='reward')
        )

        self.prediction_policy_network = (
            mlp(encoding_size, fc_policy_layers, self.action_space_size, name='policy')
        )
        self.prediction_value_network = (
            mlp(encoding_size, fc_value_layers, self.full_support_size, name='value')
        )

    def prediction(self, encoded_state, training=False):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation, training=False):
        encoded_state = self.representation_network(
            tf.reshape(observation, (observation.shape[0], -1))
        )
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = tf.keras.backend.min(encoded_state, axis=1, keepdims=True)[0]
        max_encoded_state = tf.keras.backend.max(encoded_state, axis=1, keepdims=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state = tf.clip_by_value(scale_encoded_state, 1e-5, float('inf'))
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action, training=False):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = tf.one_hot(tf.squeeze(action, axis=-1), self.action_space_size)
        x = tf.concat((encoded_state, action_one_hot), axis=-1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = tf.keras.backend.min(next_encoded_state, axis=1, keepdims=True)[0]
        max_next_encoded_state = tf.keras.backend.max(next_encoded_state, axis=1, keepdims=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state = tf.clip_by_value(scale_next_encoded_state, 1e-5, float('inf'))
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation, training=False):
        encoded_state = self.representation(observation, training)
        policy_logits, value = self.prediction(encoded_state, training)
        # reward equal to 0 for consistency
        reward = tf.math.log(
            tf.one_hot(
                tf.fill(observation.shape[0], self.full_support_size // 2),
                self.full_support_size
            )
        )

        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action, training=False):
        next_encoded_state, reward = self.dynamics(encoded_state, action, training)
        policy_logits, value = self.prediction(next_encoded_state, training)
        return value, reward, policy_logits, next_encoded_state


###### End Fully Connected #######
##################################


##################################
############# ResNet #############


def conv3x3(in_channels, out_channels, stride=1):
    return tf.keras.layers.Conv2D(
        out_channels, kernel_size=3, strides=stride, padding=1, use_bias=False
    )


# Residual block
class ResidualBlock(tf.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = tf.keras.layers.BatchNormalization()

    def __call__(self, t, training=False):
        x = self.conv1(t)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x += t
        x = tf.nn.relu(x)
        return x

class ResidualBlocks(tf.Module):
    def __init__(self, num_blocks, num_channels, stride=1):
        super().__init__()
        self.blocks = [ResidualBlock(num_channels, stride) for _ in range(num_blocks)]
    
    def __call__(self, t, training=False):
        x = t
        for block in self.blocks:
            x = block(x, training=training)
        return x


# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(tf.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            out_channels // 2,
            kernel_size=3,
            strides=2,
            padding=1,
            use_bias=False,
        )
        self.resblocks1 = ResidualBlocks(2, out_channels // 2)
        self.conv2 = tf.keras.layers.Conv2D(
            out_channels,
            kernel_size=3,
            strides=2,
            padding=1,
            use_bias=False,
        )
        self.resblocks2 = ResidualBlocks(3, out_channels)
        self.pooling1 = tf.keras.layers.AveragePooling2D(pool_size=3, strides=2, padding=1)
        self.resblocks3 = ResidualBlocks(3, out_channels)
        self.pooling2 = tf.keras.layers.AveragePooling2D(pool_size=3, strides=2, padding=1)

    def __call__(self, x, training=False):
        x = self.conv1(x)
        x = self.resblocks1(x, training=training)
        x = self.conv2(x)
        x = self.resblocks2(x, training=training)
        x = self.pooling1(x)
        x = self.resblocks3(x, training=training)
        x = self.pooling2(x)
        return x


class DownsampleCNN(tf.Module):
    def __init__(self, in_channels, out_channels, h_w):
        """
        Warning: this only works with h_w which evenly divides size
        of the output of self.features, which must have the same height
        and width. This is because the final
        average pooling layer is a stand-in for adaptive pooling.
        """
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.features = tf.keras.Sequential(
            tf.keras.layers.Conv2D(
                mid_channels, kernel_size=h_w[0] * 2, strides=4, padding=2
            ),
            tf.keras.layers.ReLU(inplace=True),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            tf.keras.layers.Conv2D(out_channels, kernel_size=5, padding=2),
            tf.keras.layers.ReLU(inplace=True),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        )
        input_size = self.features.output_shape[-2]
        output_size = h_w
        stride = input_size // output_size
        pool_size = input_size - (output_size - 1) * stride
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=stride, padding=0)

    def __call__(self, x, training=False):
        x = self.features(x, training=training)
        x = self.avgpool(x)
        return x


class RepresentationNetwork(tf.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        num_blocks,
        num_channels,
        downsample,
    ):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            if self.downsample == "resnet":
                self.downsample_net = DownSample(
                    observation_shape[2] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                )
            elif self.downsample == "CNN":
                self.downsample_net = DownsampleCNN(
                    observation_shape[2] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                    (
                        math.ceil(observation_shape[0] / 16),
                        math.ceil(observation_shape[1] / 16),
                    ),
                )
            else:
                raise NotImplementedError('downsample should be "resnet" or "CNN".')
        self.conv = conv3x3(
            observation_shape[2] * (stacked_observations + 1) + stacked_observations,
            num_channels,
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.resblocks = ResidualBlocks(num_blocks, num_channels)

    def __call__(self, x, training=False):
        if self.downsample:
            x = self.downsample_net(x, training=training)
        else:
            x = self.conv(x)
            x = self.bn(x, training=training)
            x = tf.nn.relu(x)

        x = self.resblocks(x, training=training)
        return x


class DynamicsNetwork(tf.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
    ):
        super().__init__()
        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = tf.keras.layers.BatchNormalization()
        self.resblocks = ResidualBlocks(num_blocks, num_channels - 1)

        self.conv1x1_reward = tf.keras.layers.Conv2D(
            reduced_channels_reward, 1
        )
        self.block_output_size_reward = block_output_size_reward
        self.fc = mlp(
            self.block_output_size_reward, fc_reward_layers, full_support_size,
        )

    def __call__(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.resblocks(x, training=training)
        state = x
        x = self.conv1x1_reward(x)
        x = tf.reshape(x, (-1, self.block_output_size_reward))
        reward = self.fc(x)
        return state, reward


class PredictionNetwork(tf.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
    ):
        super().__init__()
        self.resblocks = ResidualBlocks(num_blocks, num_channels)

        self.conv1x1_value = tf.keras.layers.Conv2D(reduced_channels_value, 1)
        self.conv1x1_policy = tf.keras.layers.Conv2D(reduced_channels_policy, 1)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(
            self.block_output_size_value, fc_value_layers, full_support_size
        )
        self.fc_policy = mlp(
            self.block_output_size_policy, fc_policy_layers, action_space_size,
        )

    def __call__(self, x, training=False):
        x = self.resblocks(x, training=training)
        value = self.conv1x1_value(x)
        policy = self.conv1x1_policy(x)
        value = tf.reshape(value, (-1, self.block_output_size_value))
        policy = tf.reshape(policy, (-1, self.block_output_size_policy))
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class MuZeroResidualNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        reduced_channels_value,
        reduced_channels_policy,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        support_size,
        downsample,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        block_output_size_reward = (
            (
                reduced_channels_reward
                * math.ceil(observation_shape[0] / 16)
                * math.ceil(observation_shape[1] / 16)
            )
            if downsample
            else (reduced_channels_reward * observation_shape[0] * observation_shape[0])
        )

        block_output_size_value = (
            (
                reduced_channels_value
                * math.ceil(observation_shape[0] / 16)
                * math.ceil(observation_shape[1] / 16)
            )
            if downsample
            else (reduced_channels_value * observation_shape[0] * observation_shape[1])
        )

        block_output_size_policy = (
            (
                reduced_channels_policy
                * math.ceil(observation_shape[0] / 16)
                * math.ceil(observation_shape[1] / 16)
            )
            if downsample
            else (reduced_channels_policy * observation_shape[0] * observation_shape[1])
        )

        self.representation_network = (
            RepresentationNetwork(
                observation_shape,
                stacked_observations,
                num_blocks,
                num_channels,
                downsample,
            )
        )

        self.dynamics_network = (
            DynamicsNetwork(
                num_blocks,
                num_channels + 1,
                reduced_channels_reward,
                fc_reward_layers,
                self.full_support_size,
                block_output_size_reward,
            )
        )

        self.prediction_network = (
            PredictionNetwork(
                action_space_size,
                num_blocks,
                num_channels,
                reduced_channels_value,
                reduced_channels_policy,
                fc_value_layers,
                fc_policy_layers,
                self.full_support_size,
                block_output_size_value,
                block_output_size_policy,
            )
        )

    def prediction(self, encoded_state, training=False):
        policy, value = self.prediction_network(encoded_state, training)
        return policy, value

    def representation(self, observation, training=False):
        encoded_state = self.representation_network(observation, training)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = (
            tf.expand_dims(
                tf.keras.backend.min(
                    tf.reshape(encoded_state, (
                        -1,
                        encoded_state.shape[1] * encoded_state.shape[2],
                        encoded_state.shape[3],
                    )),
                    axis=1,
                    keepdims=True
                )[0],
                axis=-1
            )
        )
        max_encoded_state = (
            tf.expand_dims(
                tf.keras.backend.min(
                    tf.reshape(encoded_state, (
                        -1,
                        encoded_state.shape[1] * encoded_state.shape[2],
                        encoded_state.shape[3],
                    )),
                    axis=1,
                    keepdims=True
                )[0],
                axis=-1
            )
        )
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state = tf.clip_by_value(scale_encoded_state, 1e-5, float('inf'))
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action, training=False):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            tf.identity(tf.ones(
                (
                    encoded_state.shape[0],
                    encoded_state.shape[1],
                    encoded_state.shape[2],
                    1,
                )
            ))
        )
        action_one_hot = (
            action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        x = tf.concat((encoded_state, action_one_hot), axis=-1)
        next_encoded_state, reward = self.dynamics_network(x, training)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = (
            tf.expand_dims(
                tf.keras.backend.min(
                    tf.reshape(next_encoded_state, (
                        -1,
                        next_encoded_state.shape[1] * next_encoded_state.shape[2],
                        next_encoded_state.shape[3],
                    )),
                    axis=1,
                    keepdims=True
                )[0],
                axis=-1
            )
        )
        max_next_encoded_state = (
            tf.expand_dims(
                tf.keras.backend.min(
                    tf.reshape(next_encoded_state, (
                        -1,
                        next_encoded_state.shape[1] * next_encoded_state.shape[2],
                        next_encoded_state.shape[3],
                    )),
                    axis=1,
                    keepdims=True
                )[0],
                axis=-1
            )
        )
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state = tf.clip_by_value(scale_next_encoded_state, 1e-5, float('inf'))
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation, training=False):
        encoded_state = self.representation(observation, training)
        policy_logits, value = self.prediction(encoded_state, training)
        # reward equal to 0 for consistency
        reward = tf.identity(tf.math.log(
            tf.one_hot(
                tf.fill(observation.shape[0], self.full_support_size // 2),
                self.full_support_size
            )
        ))
        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action, training=False):
        next_encoded_state, reward = self.dynamics(encoded_state, action, training)
        policy_logits, value = self.prediction(next_encoded_state, training)
        return value, reward, policy_logits, next_encoded_state


########### End ResNet ###########
##################################


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=tf.keras.activations.linear,
    activation=tf.keras.activations.elu,
    name=None,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        dense_name = name + f'_dense_{i}' if name is not None else None
        act_name = name + f'_act_{i}' if name is not None else None
        layers += [
            tf.keras.layers.Dense(sizes[i + 1], name=dense_name),
            tf.keras.layers.Activation(act, name=act_name)
        ]
    return tf.keras.Sequential(layers)


def support_to_scalar(logits, support_size):
    probabilities = tf.nn.softmax(logits, axis=1)
    basis = tf.range(-support_size, support_size + 1, 1.0)
    x = tf.tensordot(probabilities, basis, axes=[[-1], [0]])
    x = tf.math.sign(x) * (
        ((tf.math.sqrt(1 + 4 * 0.001 * (tf.math.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x

# def support_to_scalar(logits, support_size):
#     """
#     Transform a categorical representation to a scalar
#     See paper appendix Network Architecture
#     """
#     # Decode to a scalar
#     probabilities = tf.nn.softmax(logits, axis=1)
#     support = tf.tile(
#         tf.convert_to_tensor([[x for x in range(-support_size, support_size + 1)]], dtype=tf.float32),
#         [probabilities.shape[0], 1]
#     )
#     x = tf.reduce_sum(support * probabilities, axis=1, keepdims=True)
#     # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
#     x = tf.math.sign(x) * (
#         ((tf.math.sqrt(1 + 4 * 0.001 * (tf.math.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
#         ** 2
#         - 1
#     )
#     return x


def scalar_to_support(t, bound):
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    t = tf.math.sign(t) * (tf.math.sqrt(tf.math.abs(t) + 1) - 1) + 0.001 * t
    t_clipped = tf.clip_by_value(t, -bound, bound)
    shape = tf.concat([tf.shape(t), tf.constant([2 * bound + 1])], 0)
    dtype = t_clipped.dtype

    # Negative numbers round toward zero (up). Make them non-negative to fix.
    indices_l = tf.cast(t_clipped + tf.cast(tf.identity(bound), dtype=dtype), tf.int32) - tf.identity(bound)
    indices_u = indices_l + 1

    # TODO: precompute tile and repeat
    left = tf.reshape(tf.cast(indices_u, dtype) - t_clipped, (-1,))
    right = tf.reshape(t_clipped - tf.cast(indices_l, dtype), (-1,))

    def zip_with_indices(u, x, y):
        return tf.transpose(tf.stack([
            tf.repeat(tf.range(x), tf.reshape(y, (1,))),
            tf.tile(tf.range(y), tf.reshape(x, (1,))),
            tf.reshape(u, (-1,))
        ]))

    indices_l = zip_with_indices(indices_l + bound, tf.shape(t)[0], tf.shape(t)[1])
    indices_u = zip_with_indices(indices_u + bound, tf.shape(t)[0], tf.shape(t)[1])
    return tf.scatter_nd(indices_l, left, shape) + tf.scatter_nd(indices_u, right, shape)

# def scalar_to_support(x, support_size):
#     """
#     Transform a scalar to a categorical representation with (2 * support_size + 1) categories
#     See paper appendix Network Architecture
#     """
#     # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
#     x = tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1) - 1) + 0.001 * x

#     # Encode on a vector
#     x = tf.clip_by_value(x, -support_size, support_size)
#     floor = tf.math.floor(x)
#     prob = x - floor
#     logits = tf.identity(tf.zeros(x.shape[0], x.shape[1], 2 * support_size + 1))

#     def zip_with_indices(u, x, y):
#         return tf.transpose(tf.stack([
#             tf.tile(tf.range(x), tf.reshape(y, (1,))),
#             tf.repeat(tf.range(y), tf.reshape(x, (1,))),
#             tf.reshape(u, (-1,))
#         ]))
    
#     tf.tensor_scatter_nd_update(
#         logits,
#         2,
#         tf.expand_dims(tf.cast(floor + support_size, tf.int32), axis=-1),
#         tf.expand_dims(1 - prob, axis=-1)
#     )
#     indexes = floor + support_size + 1
#     prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
#     indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
#     logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
#    return logits
