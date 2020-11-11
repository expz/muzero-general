import gym
import numpy

from .abstract_game import AbstractGame
from .defaults import with_defaults

try:
    import cv2
except ModuleNotFoundError:
    raise ModuleNotFoundError('Please run "pip install gym[atari]"')


def visit_softmax_temperature_fn(trained_steps):
    """
    Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
    The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

    Returns:
        Positive float.
    """
    if trained_steps < 500e3:
        return 1.0
    elif trained_steps < 750e3:
        return 0.5
    else:
        return 0.25


CONFIG = with_defaults({
    # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
    'max_num_gpus': None,  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

    ### Game
    'name': 'atari',
    'observation_shape': (3, 96, 96),  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
    'action_space': list(range(4)),  # Fixed list of all possible actions. You should only edit the length
    'players': list(range(1)),  # List of players. You should only edit the length
    'stacked_observations': 32,  # Number of previous observations and previous actions to add to the current observation

    # Evaluate
    'muzero_player': 0,  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
    'opponent': None,  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

    ### Self-Play
    'num_workers': 350,  # Number of simultaneous threads/workers self-playing to feed the replay buffer
    'selfplay_on_gpu': False,
    'max_moves': 27000,  # Maximum number of moves if game is not finished before
    'num_simulations': 50,  # Number of future moves self-simulated
    'discount': 0.997,  # Chronological discount of the reward
    'temperature_threshold': None,  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

    # Root prior exploration noise
    'root_dirichlet_alpha': 0.25,
    'root_exploration_fraction': 0.25,

    # UCB formula
    'pb_c_base': 19652,
    'pb_c_init': 1.25,

    ### Network
    'network': "resnet",  # "resnet" / "fullyconnected"
    'support_size': 300,  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

    # Residual Network
    'downsample': "resnet",  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
    'blocks': 16,  # Number of blocks in the ResNet
    'channels': 256,  # Number of channels in the ResNet
    'reduced_channels_reward': 256,  # Number of channels in reward head
    'reduced_channels_value': 256,  # Number of channels in value head
    'reduced_channels_policy': 256,  # Number of channels in policy head
    'resnet_fc_reward_layers': [256, 256],  # Define the hidden layers in the reward head of the dynamic network
    'resnet_fc_value_layers': [256, 256],  # Define the hidden layers in the value head of the prediction network
    'resnet_fc_policy_layers': [256, 256],  # Define the hidden layers in the policy head of the prediction network

    # Fully Connected Network
    'encoding_size': 10,
    'fc_representation_layers': [],  # Define the hidden layers in the representation network
    'fc_dynamics_layers': [16],  # Define the hidden layers in the dynamics network
    'fc_reward_layers': [16],  # Define the hidden layers in the reward network
    'fc_value_layers': [],  # Define the hidden layers in the value network
    'fc_policy_layers': [],  # Define the hidden layers in the policy network

    ### Training
    'training_steps': int(1000e3),  # Total number of training steps (ie weights update according to a batch)
    'batch_size': 1024,  # Number of parts of games to train on at each training step
    'checkpoint_interval': int(1e3),  # Number of training steps before using the model for self-playing
    'value_loss_weight': 0.25,  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)

    # Exponential learning rate schedule
    'lr_init': 0.05,  # Initial learning rate
    'lr_decay_rate': 0.1,  # Set it to 1 to use a constant learning rate
    'lr_decay_steps': 350e3,

    ### Replay Buffer
    'replay_buffer_size': int(1e6),  # Number of self-play games to keep in the replay buffer
    'PER': True,  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
    'PER_alpha': 1,  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

    ### Adjust the self play / training ratio to avoid over/underfitting
    'self_play_delay': 0,  # Number of seconds to wait after each played game
    'training_delay': 0,  # Number of seconds to wait after each training step

    # Temperature
    'visit_softmax_temperature_fn': visit_softmax_temperature_fn,
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

