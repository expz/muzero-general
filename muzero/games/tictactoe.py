import numpy

from .abstract_game import AbstractGame
from .defaults import with_defaults


CONFIG = with_defaults({
    # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
    'max_num_gpus': 1,  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

    ### Game
    'name': 'tictactoe',
    'observation_shape': (3, 3, 3),  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
    'action_space': list(range(9)),  # Fixed list of all possible actions. You should only edit the length
    'players': list(range(2)),  # List of players. You should only edit the length
    'stacked_observations': 0,  # Number of previous observations and previous actions to add to the current observation

    # Evaluate
    'muzero_player': 0,  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
    'opponent': "expert",  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

    ### Self-Play
    'num_workers': 1,  # Number of simultaneous threads/workers self-playing to feed the replay buffer
    'selfplay_on_gpu': False,
    'max_moves': 9,  # Maximum number of moves if game is not finished before
    'num_simulations': 25,  # Number of future moves self-simulated
    'discount': 1,  # Chronological discount of the reward
    'temperature_threshold': None,  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

    # Root prior exploration noise
    'root_dirichlet_alpha': 0.1,
    'root_exploration_fraction': 0.25,

    # UCB formula
    'pb_c_base': 19652,
    'pb_c_init': 1.25,

    ### Network
    'network': "resnet",  # "resnet" / "fullyconnected"
    'support_size': 10,  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

    # Residual Network
    'downsample': False,  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
    'blocks': 1,  # Number of blocks in the ResNet
    'channels': 16,  # Number of channels in the ResNet
    'reduced_channels_reward': 16,  # Number of channels in reward head
    'reduced_channels_value': 16,  # Number of channels in value head
    'reduced_channels_policy': 16,  # Number of channels in policy head
    'resnet_fc_reward_layers': [8],  # Define the hidden layers in the reward head of the dynamic network
    'resnet_fc_value_layers': [8],  # Define the hidden layers in the value head of the prediction network
    'resnet_fc_policy_layers': [8],  # Define the hidden layers in the policy head of the prediction network

    # Fully Connected Network
    'encoding_size': 32,
    'fc_representation_layers': [],  # Define the hidden layers in the representation network
    'fc_dynamics_layers': [16],  # Define the hidden layers in the dynamics network
    'fc_reward_layers': [16],  # Define the hidden layers in the reward network
    'fc_value_layers': [],  # Define the hidden layers in the value network
    'fc_policy_layers': [],  # Define the hidden layers in the policy network

    ### Training
    'training_steps': 1000000,  # Total number of training steps (ie weights update according to a batch)
    'batch_size': 64,  # Number of parts of games to train on at each training step
    'checkpoint_interval': 10,  # Number of training steps before using the model for self-playing
    'value_loss_weight': 0.25,  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
    'optimizer': "Adam",  # "Adam" or "SGD". Paper uses SGD

    # Exponential learning rate schedule
    'lr_init': 0.003,  # Initial learning rate
    'lr_decay_rate': 1,  # Set it to 1 to use a constant learning rate
    'lr_decay_steps': 10000,

    ### Replay Buffer
    'replay_buffer_size': 3000,  # Number of self-play games to keep in the replay buffer
    'num_unroll_steps': 20,  # Number of game moves to keep for every batch element
    'td_steps': 20,  # Number of steps in the future to take into account for calculating the target value
    'PER': True,  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
    'PER_alpha': 0.5,  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

    ### Adjust the self play / training ratio to avoid over/underfitting
    'self_play_delay': 0,  # Number of seconds to wait after each played game
    'training_delay': 0,  # Number of seconds to wait after each training step
})


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = TicTacToe()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

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
        while True:
            try:
                row = int(
                    input(
                        f"Enter the row (1, 2 or 3) to play for the player {self.to_play()}: "
                    )
                )
                col = int(
                    input(
                        f"Enter the column (1, 2 or 3) to play for the player {self.to_play()}: "
                    )
                )
                choice = (row - 1) * 3 + (col - 1)
                if (
                    choice in self.legal_actions()
                    and 1 <= row
                    and 1 <= col
                    and row <= 3
                    and col <= 3
                ):
                    break
            except:
                pass
            print("Wrong input, try again")
        return choice

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.
        
        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        row = action_number // 3 + 1
        col = action_number % 3 + 1
        return f"Play row {row}, column {col}"


class TicTacToe:
    def __init__(self):
        self.board = numpy.zeros((3, 3), dtype="int32")
        self.player = 1

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros((3, 3), dtype="int32")
        self.player = 1
        return self.get_observation()

    def step(self, action):
        row = action // 3
        col = action % 3
        self.board[row, col] = self.player

        done = self.have_winner() or len(self.legal_actions()) == 0

        reward = 1 if self.have_winner() else 0

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1, 0)
        board_player2 = numpy.where(self.board == -1, 1, 0)
        board_to_play = numpy.full((3, 3), self.player)
        return numpy.array([board_player1, board_player2, board_to_play], dtype="int32")

    def legal_actions(self):
        legal = []
        for i in range(9):
            row = i // 3
            col = i % 3
            if self.board[row, col] == 0:
                legal.append(i)
        return legal

    def have_winner(self):
        # Horizontal and vertical checks
        for i in range(3):
            if (self.board[i, :] == self.player * numpy.ones(3, dtype="int32")).all():
                return True
            if (self.board[:, i] == self.player * numpy.ones(3, dtype="int32")).all():
                return True

        # Diagonal checks
        if (
            self.board[0, 0] == self.player
            and self.board[1, 1] == self.player
            and self.board[2, 2] == self.player
        ):
            return True
        if (
            self.board[2, 0] == self.player
            and self.board[1, 1] == self.player
            and self.board[0, 2] == self.player
        ):
            return True

        return False

    def expert_action(self):
        board = self.board
        action = numpy.random.choice(self.legal_actions())
        # Horizontal and vertical checks
        for i in range(3):
            if abs(sum(board[i, :])) == 2:
                ind = numpy.where(board[i, :] == 0)[0][0]
                action = numpy.ravel_multi_index(
                    (numpy.array([i]), numpy.array([ind])), (3, 3)
                )[0]
                if self.player * sum(board[i, :]) > 0:
                    return action

            if abs(sum(board[:, i])) == 2:
                ind = numpy.where(board[:, i] == 0)[0][0]
                action = numpy.ravel_multi_index(
                    (numpy.array([ind]), numpy.array([i])), (3, 3)
                )[0]
                if self.player * sum(board[:, i]) > 0:
                    return action

        # Diagonal checks
        diag = board.diagonal()
        anti_diag = numpy.fliplr(board).diagonal()
        if abs(sum(diag)) == 2:
            ind = numpy.where(diag == 0)[0][0]
            action = numpy.ravel_multi_index(
                (numpy.array([ind]), numpy.array([ind])), (3, 3)
            )[0]
            if self.player * sum(diag) > 0:
                return action

        if abs(sum(anti_diag)) == 2:
            ind = numpy.where(anti_diag == 0)[0][0]
            action = numpy.ravel_multi_index(
                (numpy.array([ind]), numpy.array([2 - ind])), (3, 3)
            )[0]
            if self.player * sum(anti_diag) > 0:
                return action

        return action

    def render(self):
        print(self.board[::-1])
