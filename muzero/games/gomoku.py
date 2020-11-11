import math

import numpy

from .abstract_game import AbstractGame
from .defaults import with_defaults


training_steps = 10000


def visit_softmax_temperature_fn(trained_steps):
    """
    Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
    The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

    Returns:
    Positive float.
    """
    if trained_steps < 0.5 * training_steps:
        return 1.0
    elif trained_steps < 0.75 * training_steps:
        return 0.5
    else:
        return 0.25


CONFIG = with_defaults({
    # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
    'max_num_gpus': None,  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

    ### Game
    'name': 'gomoku',
    'observation_shape': (3, 11, 11),  # Dimensions of the game observation, must be 3 (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
    'action_space': list(range(11 * 11)),  # Fixed list of all possible actions. You should only edit the length
    'players': list(range(2)),  # List of players. You should only edit the length
    'stacked_observations': 0,  # Number of previous observations and previous actions to add to the current observation

    # Evaluate
    'muzero_player': 0,  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
    'opponent': "random",  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

    ### Self-Play
    'num_workers': 2,  # Number of simultaneous threads/workers self-playing to feed the replay buffer
    'selfplay_on_gpu': False,
    'max_moves': 121,  # Maximum number of moves if game is not finished before
    'num_simulations': 400,  # Number of future moves self-simulated
    'discount': 1,  # Chronological discount of the reward
    'temperature_threshold': None,  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

    # Root prior exploration noise
    'root_dirichlet_alpha': 0.3,
    'root_exploration_fraction': 0.25,

    # UCB formula
    'pb_c_base': 19652,
    'pb_c_init': 1.25,

    ### Network
    'network': "resnet",  # "resnet" / "fullyconnected"
    'support_size': 10,  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
    
    # Residual Network
    'downsample': False,  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
    'blocks': 6,  # Number of blocks in the ResNet
    'channels': 128,  # Number of channels in the ResNet
    'reduced_channels_reward': 2,  # Number of channels in reward head
    'reduced_channels_value': 2,  # Number of channels in value head
    'reduced_channels_policy': 4,  # Number of channels in policy head
    'resnet_fc_reward_layers': [64],  # Define the hidden layers in the reward head of the dynamic network
    'resnet_fc_value_layers': [64],  # Define the hidden layers in the value head of the prediction network
    'resnet_fc_policy_layers': [64],  # Define the hidden layers in the policy head of the prediction network
    
    # Fully Connected Network
    'encoding_size': 32,
    'fc_representation_layers': [],  # Define the hidden layers in the representation network
    'fc_dynamics_layers': [64],  # Define the hidden layers in the dynamics network
    'fc_reward_layers': [64],  # Define the hidden layers in the reward network
    'fc_value_layers': [],  # Define the hidden layers in the value network
    'fc_policy_layers': [],  # Define the hidden layers in the policy network

    ### Training
    'training_steps': training_steps,  # Total number of training steps (ie weights update according to a batch)
    'batch_size': 512,  # Number of parts of games to train on at each training step
    'checkpoint_interval': 50,  # Number of training steps before using the model for self-playing
    'value_loss_weight': 1,  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)

    'optimizer': "Adam",  # "Adam" or "SGD". Paper uses SGD

    # Exponential learning rate schedule
    'lr_init': 0.002,  # Initial learning rate
    'lr_decay_rate': 0.9,  # Set it to 1 to use a constant learning rate
    'lr_decay_steps': 10000,

    ### Replay Buffer
    'replay_buffer_size': 10000,  # Number of self-play games to keep in the replay buffer
    'num_unroll_steps': 121,  # Number of game moves to keep for every batch element
    'td_steps': 121,  # Number of steps in the future to take into account for calculating the target value
    'PER': True,  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
    'PER_alpha': 0.5,  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

    # Reanalyze (See paper appendix Reanalyse)
    'use_last_model_value': False,  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
    'reanalyse_device': "cpu",  # "cpu" / "cuda"
    'reanalyse_num_gpus': 0,  # Number of GPUs to use for the reanalyse, it can be fractional, don't fortget to take the train worker and the selfplay workers into account

    ### Adjust the self play / training ratio to avoid over/underfitting
    'self_play_delay': 0,  # Number of seconds to wait after each played game
    'training_delay': 0,  # Number of seconds to wait after each training step
    'ratio': 1,  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    'visit_softmax_temperature_fn': visit_softmax_temperature_fn,
})


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Gomoku()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def close(self):
        """
        Properly close the game.
        """
        pass

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        valid = False
        action = None
        while not valid:
            valid, action = self.env.human_input_to_action()
        return action

    def action_to_string(self, action):
        """
        Convert an action number to a string representing the action.
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        return self.env.action_to_human_input(action)


class Gomoku:
    def __init__(self):
        self.board_size = 11
        self.board = numpy.zeros((self.board_size, self.board_size), dtype="int32")
        self.player = 1
        self.board_markers = [
            chr(x) for x in range(ord("A"), ord("A") + self.board_size)
        ]

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros((self.board_size, self.board_size), dtype="int32")
        self.player = 1
        return self.get_observation()

    def step(self, action):
        x = math.floor(action / self.board_size)
        y = action % self.board_size
        self.board[x][y] = self.player

        done = self.is_finished()

        reward = 1 if done else 0

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        board_to_play = numpy.full((11, 11), self.player, dtype="int32")
        return numpy.array([board_player1, board_player2, board_to_play])

    def legal_actions(self):
        legal = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    legal.append(i * self.board_size + j)
        return legal

    def is_finished(self):
        has_legal_actions = False
        directions = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in range(self.board_size):
            for j in range(self.board_size):
                # if no stone is on the position, don't need to consider this position
                if self.board[i][j] == 0:
                    has_legal_actions = True
                    continue
                # value-value at a coord, i-row, j-col
                player = self.board[i][j]
                # check if there exist 5 in a line
                for d in directions:
                    x, y = i, j
                    count = 0
                    for _ in range(5):
                        if (x not in range(self.board_size)) or (
                            y not in range(self.board_size)
                        ):
                            break
                        if self.board[x][y] != player:
                            break
                        x += d[0]
                        y += d[1]
                        count += 1
                        # if 5 in a line, store positions of all stones, return value
                        if count == 5:
                            return True
        return not has_legal_actions

    def render(self):
        marker = "  "
        for i in range(self.board_size):
            marker = marker + self.board_markers[i] + " "
        print(marker)
        for row in range(self.board_size):
            print(chr(ord("A") + row), end=" ")
            for col in range(self.board_size):
                ch = self.board[row][col]
                if ch == 0:
                    print(".", end=" ")
                elif ch == 1:
                    print("X", end=" ")
                elif ch == -1:
                    print("O", end=" ")
            print()

    def human_input_to_action(self):
        human_input = input("Enter an action: ")
        if (
            len(human_input) == 2
            and human_input[0] in self.board_markers
            and human_input[1] in self.board_markers
        ):
            x = ord(human_input[0]) - 65
            y = ord(human_input[1]) - 65
            if self.board[x][y] == 0:
                return True, x * self.board_size + y
        return False, -1

    def action_to_human_input(self, action):
        x = math.floor(action / self.board_size)
        y = action % self.board_size
        x = chr(x + 65)
        y = chr(y + 65)
        return x + y
