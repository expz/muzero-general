import copy
import datetime
import os

import tensorflow as tf


class MuZeroConfig:
    pass


DEFAULT_CONFIG = {
    'seed': 0,  # Seed for numpy, tensorflow and the game
    'save_model': True,  # Save the checkpoint in results_path as model.checkpoint
    'train_on_gpu': True if len(tf.config.experimental.list_physical_devices('GPU')) > 0 else False,  # Train on GPU if available
    'optimizer': "SGD",  # "Adam" or "SGD". Paper uses SGD
    'momentum': 0.9,  # Used only if optimizer is SGD
    'weight_decay': 1e-4,  # L2 weights regularization

    'num_unroll_steps': 10,  # Number of game moves to keep for every batch element
    'td_steps': 5,  # Number of steps in the future to take into account for calculating the target value

    # Reanalyze (See paper appendix Reanalyse)
    'use_last_model_value': True,  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
    'reanalyse_on_gpu': False,

    'ratio': None,  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    # Temperature
    'visit_softmax_temperature_fn': lambda trained_steps: 1,
}


def update_config(config, new_config):
    updated_config = copy.deepcopy(config)
    updated_config.update(new_config)
    return updated_config


def with_defaults(config):
    DEFAULT_CONFIG['results_path'] = os.path.join(
        os.getcwd(),
        'results',
        config['name'],
        datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    )  # Path to store the model weights and TensorBoard logs
    return update_config(DEFAULT_CONFIG, config)
