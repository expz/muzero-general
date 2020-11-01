import gym
import numpy

from .abstract_game import AbstractGame
from .atari import CONFIG as ATARI_CONFIG
from .defaults import update_config

try:
    import cv2
except ModuleNotFoundError:
    raise ModuleNotFoundError('Please run "pip install gym[atari]"')


CONFIG = update_config(ATARI_CONFIG, {
    # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

    ### Game
    'name': 'breakout',
    'stacked_observations': 0,  # Number of previous observations and previous actions to add to the current observation

    ### Self-Play
    'num_workers': 1,  # Number of simultaneous threads/workers self-playing to feed the replay buffer
    'max_moves': 2500,  # Maximum number of moves if game is not finished before
    'num_simulations': 30,  # Number of future moves self-simulated

    ### Network
    'network': "resnet",  # "resnet" / "fullyconnected"
    'support_size': 10,  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

    # Residual Network
    'downsample': "resnet",  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
    'blocks': 2,  # Number of blocks in the ResNet
    'channels': 16,  # Number of channels in the ResNet
    'reduced_channels_reward': 4,  # Number of channels in reward head
    'reduced_channels_value': 4,  # Number of channels in value head
    'reduced_channels_policy': 4,  # Number of channels in policy head
    'resnet_fc_reward_layers': [16],  # Define the hidden layers in the reward head of the dynamic network
    'resnet_fc_value_layers': [16],  # Define the hidden layers in the value head of the prediction network
    'resnet_fc_policy_layers': [16],  # Define the hidden layers in the policy head of the prediction network

    # Fully Connected Network
    'encoding_size': 10,
    'fc_representation_layers': [],  # Define the hidden layers in the representation network
    'fc_dynamics_layers': [16],  # Define the hidden layers in the dynamics network
    'fc_reward_layers': [16],  # Define the hidden layers in the reward network
    'fc_value_layers': [],  # Define the hidden layers in the value network
    'fc_policy_layers': [],  # Define the hidden layers in the policy network

    ### Training
    'training_steps': int(1000e3),  # Total number of training steps (ie weights update according to a batch)
    'batch_size': 16,  # Number of parts of games to train on at each training step
    'checkpoint_interval': 500,  # Number of training steps before using the model for self-playing

    'optimizer': "Adam",  # "Adam" or "SGD". Paper uses SGD

    # Exponential learning rate schedule
    'lr_init': 0.005,  # Initial learning rate
    'lr_decay_rate': 1,  # Set it to 1 to use a constant learning rate
    'lr_decay_steps': 350e3,

    ### Replay Buffer
    'replay_buffer_size': int(1e6),  # Number of self-play games to keep in the replay buffer

    # Reanalyze (See paper appendix Reanalyse)
    'use_last_model_value': False,  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
})


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = gym.make("Breakout-v4")
        if seed is not None:
            self.env.seed(seed)

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, _ = self.env.step(action)
        observation = cv2.resize(observation, (96, 96), interpolation=cv2.INTER_AREA)
        observation = numpy.asarray(observation, dtype="float32") / 255.0
        observation = numpy.moveaxis(observation, -1, 0)
        return observation, reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.        

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(4))

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        observation = self.env.reset()
        observation = cv2.resize(observation, (96, 96), interpolation=cv2.INTER_AREA)
        observation = numpy.asarray(observation, dtype="float32") / 255.0
        observation = numpy.moveaxis(observation, -1, 0)
        return observation

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

