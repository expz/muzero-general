import copy
import time

import numpy
import ray
import tensorflow as tf

import muzero.models as models


def scale_gradient(t, scale):
    """Retain the value of t while reducing its gradient"""
    return tf.cast(scale, dtype=tf.float32) * t + tf.cast(1 - scale, dtype=tf.float32) * tf.stop_gradient(t)


@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        tf.random.set_seed(self.config.seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))

        self.training_step = initial_checkpoint["training_step"]

        if not tf.config.experimental.list_physical_devices('GPU'):
            print("You are not training on GPU.\n")

        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.config.lr_init,
                momentum=self.config.momentum
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.lr_init
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )

        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.set_weights(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            index_batch, batch = ray.get(replay_buffer.get_batch.remote())
            self.update_lr()
            (
                priorities,
                total_loss,
                value_loss,
                reward_loss,
                policy_loss,
            ) = self.update_weights(batch)

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(self.optimizer.get_weights()),
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.learning_rate.numpy(),
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                }
            )

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio:
                while (
                    self.training_step
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    > self.config.ratio
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

    def update_weights(self, batch):
        """
        Perform one training step.
        """
        result = []
        loss = self.batch_loss(batch, result)

        # Optimize
        self.optimizer.minimize(loss, self.model.trainable_variables)
        self.training_step += 1

        return result


    def batch_loss(self, batch, result):
        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            gradient_scale_batch,
        ) = batch

        def compute_loss():
            nonlocal observation_batch
            nonlocal action_batch
            nonlocal target_value
            nonlocal target_reward
            nonlocal target_policy
            nonlocal weight_batch
            nonlocal gradient_scale_batch

            # Keep values as scalars for calculating the priorities for the prioritized replay
            target_value_scalar = numpy.array(target_value, dtype="float32")
            priorities = numpy.zeros_like(target_value_scalar, dtype="float32")

            if self.config.PER:
                weight_batch = tf.identity(tf.cast(weight_batch, dtype=tf.float32))
            observation_batch = tf.identity(tf.cast(observation_batch, dtype=tf.float32))
            action_batch = tf.expand_dims(tf.identity(action_batch), axis=-1)
            target_value = tf.identity(tf.cast(target_value, dtype=tf.float32))
            target_reward = tf.identity(tf.cast(target_reward, dtype=tf.float32))
            target_policy = tf.identity(tf.cast(target_policy, dtype=tf.float32))
            gradient_scale_batch = tf.identity(tf.cast(gradient_scale_batch, dtype=tf.float32))
            # observation_batch: batch, channels, height, width
            # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
            # target_value: batch, num_unroll_steps+1
            # target_reward: batch, num_unroll_steps+1
            # target_policy: batch, num_unroll_steps+1, len(action_space)
            # gradient_scale_batch: batch, num_unroll_steps+1

            target_value = models.scalar_to_support(target_value, self.config.support_size)
            target_reward = models.scalar_to_support(
                target_reward, self.config.support_size
            )
            # target_value: batch, num_unroll_steps+1, 2*support_size+1
            # target_reward: batch, num_unroll_steps+1, 2*support_size+1

            # obs batch
            # B x H x W x C
            # 128 x 1 x 1 x 4 (cartpole)
            # value/reward
            # B x N
            # 128 x 21 (cartpole)
            # policy
            # B x A
            # 128 x 2 (cartpole)
            # hidden state
            # B x X
            # 128 x 8 (cartpole)

            ## Generate predictions
            value, reward, policy_logits, hidden_state = self.model.initial_inference(
                observation_batch, training=True
            )
            predictions = [(value, reward, policy_logits)]
            for i in range(1, action_batch.shape[1]):
                value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                    hidden_state, action_batch[:, i], training=True
                )
                # Scale the gradient at the start of the dynamics function (See paper appendix Training)
                hidden_state = scale_gradient(hidden_state, 0.5)
                predictions.append((value, reward, policy_logits))
            # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)

            ## Compute losses
            value_loss, reward_loss, policy_loss = (0, 0, 0)
            value, reward, policy_logits = predictions[0]
            value_sq = tf.squeeze(value, axis=-1) if value.shape[-1] == 1 else value
            reward_sq = tf.squeeze(reward, axis=-1) if reward.shape[-1] == 1 else reward
            # Ignore reward loss for the first batch step
            current_value_loss, _, current_policy_loss = self.loss_function(
                value_sq,
                reward_sq,
                policy_logits,
                target_value[:, 0],
                target_reward[:, 0],
                target_policy[:, 0],
            )
            value_loss += current_value_loss
            policy_loss += current_policy_loss
            # Compute priorities for the prioritized replay (See paper appendix Training)
            pred_value_scalar = (
                models.support_to_scalar(value, self.config.support_size)
                .numpy()
                .squeeze()
            )
            priorities[:, 0] = (
                numpy.abs(pred_value_scalar - target_value_scalar[:, 0])
                ** self.config.PER_alpha
            )

            for i in range(1, len(predictions)):
                value, reward, policy_logits = predictions[i]
                value_sq = tf.squeeze(value, axis=-1) if value.shape[-1] == 1 else value
                reward_sq = tf.squeeze(reward, axis=-1) if reward.shape[-1] == 1 else reward
                (
                    current_value_loss,
                    current_reward_loss,
                    current_policy_loss,
                ) = self.loss_function(
                    value_sq,
                    reward_sq,
                    policy_logits,
                    target_value[:, i],
                    target_reward[:, i],
                    target_policy[:, i],
                )
                # Scale gradient by the number of unroll steps (See paper appendix Training)
                current_value_loss = scale_gradient(
                    current_value_loss, gradient_scale_batch[:, i]
                )
                current_reward_loss = scale_gradient(
                    current_reward_loss, gradient_scale_batch[:, i]
                )
                current_policy_loss = scale_gradient(
                    current_policy_loss, gradient_scale_batch[:, i]
                )

                value_loss += current_value_loss
                reward_loss += current_reward_loss
                policy_loss += current_policy_loss

                # Compute priorities for the prioritized replay (See paper appendix Training)
                pred_value_scalar = (
                    models.support_to_scalar(value, self.config.support_size)
                    .numpy()
                    .squeeze()
                )
                priorities[:, i] = (
                    numpy.abs(pred_value_scalar - target_value_scalar[:, i])
                    ** self.config.PER_alpha
                )

            l2_loss = 0
            for t in self.model.trainable_variables:
                l2_loss += self.config.weight_decay * tf.nn.l2_loss(t).numpy()

            # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
            loss = value_loss * self.config.value_loss_weight + reward_loss + policy_loss
            if self.config.PER:
                # Correct PER bias by using importance-sampling (IS) weights
                loss *= weight_batch
            # Mean over batch dimension (pseudocode do a sum)
            loss = tf.math.reduce_mean(loss) + l2_loss

            result.append(priorities)
            # For log purpose
            result.append(loss.numpy())
            result.append(tf.math.reduce_mean(value_loss).numpy())
            result.append(tf.math.reduce_mean(reward_loss).numpy())
            result.append(tf.math.reduce_mean(policy_loss).numpy())

            return loss
        
        return compute_loss


    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        tf.keras.backend.set_value(self.optimizer.learning_rate, lr)

    @staticmethod
    def loss_function(
        value,
        reward,
        policy_logits,
        target_value,
        target_reward,
        target_policy,
    ):
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = tf.reduce_sum(-target_value * tf.nn.log_softmax(value, axis=1), axis=1)
        reward_loss = tf.reduce_sum(-target_reward * tf.nn.log_softmax(reward, axis=1), axis=1)
        policy_loss = tf.reduce_sum(-target_policy * tf.nn.log_softmax(policy_logits), axis=1)
        return value_loss, reward_loss, policy_loss
