import numpy

from .abstract_game import AbstractGame
from .defaults import with_defaults


CONFIG = with_defaults({
    # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
    'max_num_gpus': None,  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

    ### Game
    'name': 'simple_grid',
    'observation_shape': (1, 1, 9),  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
    'action_space': list(range(2)),  # Fixed list of all possible actions. You should only edit the length
    'players': list(range(1)),  # List of players. You should only edit the length
    'stacked_observations': 0,  # Number of previous observations and previous actions to add to the current observation

    # Evaluate
    'muzero_player': 0,  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
    'opponent': None,  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

    ### Self-Play
    'num_workers': 1,  # Number of simultaneous threads/workers self-playing to feed the replay buffer
    'selfplay_on_gpu': False,
    'max_moves': 6,  # Maximum number of moves if game is not finished before
    'num_simulations': 10,  # Number of future moves self-simulated
    'discount': 0.978,  # Chronological discount of the reward
    'temperature_threshold': None,  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

    # Root prior exploration noise
    'root_dirichlet_alpha': 0.25,
    'root_exploration_fraction': 0.25,

    # UCB formula
    'pb_c_base': 19652,
    'pb_c_init': 1.25,

    ### Network
    'network': "fullyconnected",  # "resnet" / "fullyconnected"
    'support_size': 10,  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
    
    # Residual Network
    'downsample': False,  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
    'blocks': 1,  # Number of blocks in the ResNet
    'channels': 2,  # Number of channels in the ResNet
    'reduced_channels_reward': 2,  # Number of channels in reward head
    'reduced_channels_value': 2,  # Number of channels in value head
    'reduced_channels_policy': 2,  # Number of channels in policy head
    'resnet_fc_reward_layers': [],  # Define the hidden layers in the reward head of the dynamic network
    'resnet_fc_value_layers': [],  # Define the hidden layers in the value head of the prediction network
    'resnet_fc_policy_layers': [],  # Define the hidden layers in the policy head of the prediction network

    # Fully Connected Network
    'encoding_size': 5,
    'fc_representation_layers': [16],  # Define the hidden layers in the representation network
    'fc_dynamics_layers': [16],  # Define the hidden layers in the dynamics network
    'fc_reward_layers': [16],  # Define the hidden layers in the reward network
    'fc_value_layers': [16],  # Define the hidden layers in the value network
    'fc_policy_layers': [16],  # Define the hidden layers in the policy network

    ### Training
    'training_steps': 30000,  # Total number of training steps (ie weights update according to a batch)
    'batch_size': 32,  # Number of parts of games to train on at each training step
    'checkpoint_interval': 10,  # Number of training steps before using the model for self-playing
    'value_loss_weight': 1,  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
    'optimizer': "Adam",  # "Adam" or "SGD". Paper uses SGD

    # Exponential learning rate schedule
    'lr_init': 0.0064,  # Initial learning rate
    'lr_decay_rate': 1,  # Set it to 1 to use a constant learning rate
    'lr_decay_steps': 1000,

    ### Replay Buffer
    'replay_buffer_size': 5000,  # Number of self-play games to keep in the replay buffer
    'num_unroll_steps': 7,  # Number of game moves to keep for every batch element
    'td_steps': 7,  # Number of steps in the future to take into account for calculating the target value
    'PER': True,  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
    'PER_alpha': 0.5,  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

    ### Adjust the self play / training ratio to avoid over/underfitting
    'self_play_delay': 0.2,  # Number of seconds to wait after each played game
    'training_delay': 0,  # Number of seconds to wait after each training step
})


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = GridEnv()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return [[observation]], reward*10, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.        

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(2))

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return [[self.env.reset()]]

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Down",
            1: "Right",
        }
        return f"{action_number}. {actions[action_number]}"

class GridEnv:
    def __init__(self, size=3):
        self.size = size
        self.position = [0, 0]

    def legal_actions(self):
        legal_actions = list(range(2))
        if self.position[0] == (self.size - 1):
            legal_actions.remove(0)
        if self.position[1] == (self.size - 1):
            legal_actions.remove(1)
        return legal_actions

    def step(self, action):
        if action not in self.legal_actions():
            pass
        elif action == 0:
            self.position[0] += 1
        elif action == 1:
            self.position[1] +=1
        
        reward = 1 if self.position == [self.size - 1]*2 else 0
        return self.get_observation(), reward, bool(reward)

    def reset(self):
        self.position = [0, 0]
        return self.get_observation()

    def render(self):
        im = numpy.full((self.size, self.size), "-")
        im[self.size -1, self.size -1] = "1"
        im[self.position[0], self.position[1]] = "x"
        print(im)

    def get_observation(self):
        observation = numpy.zeros((self.size, self.size))
        observation[self.position[0]][self.position[1]] = 1
        return observation.flatten()
