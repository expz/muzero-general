"""
This is a very simple form of twenty one. Ace only counts as value 1 not 1 or
11 for simplicity. This means that there is no such thing as a natural or two
card 21. This is a good example of showing how it can provide a good solution
to even luck based games.
"""

import numpy

from .abstract_game import AbstractGame
from .defaults import with_defaults


CONFIG = with_defaults({
    # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
    'max_num_gpus': None,  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

    ### Game
    'name': 'twentyone',
    'observation_shape': (3,3,3),  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
    'action_space': list(range(2)),  # Fixed list of all possible actions. You should only edit the length
    'players': list(range(1)),  # List of players. You should only edit the length
    'stacked_observations': 0,  # Number of previous observations and previous actions to add to the current observation

    # Evaluate
    'muzero_player': 0,  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
    'opponent': None,  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

    ### Self-Play
    'num_workers': 4,  # Number of simultaneous threads/workers self-playing to feed the replay buffer
    'selfplay_on_gpu': False,
    'max_moves': 21,  # Maximum number of moves if game is not finished before
    'num_simulations': 21,  # Number of future moves self-simulated
    'discount': 1,  # Chronological discount of the reward
    'temperature_threshold': None ,  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

    # Root prior exploration noise
    'root_dirichlet_alpha': 0.25,
    'root_exploration_fraction': 0.25,

    # UCB formula
    'pb_c_base': 19652,
    'pb_c_init': 1.25,

    ### Network
    'network': "resnet",  # "resnet" / "fullyconnected"
    'support_size': 10,  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

    # Residual Network
    'downsample': False,  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
    'blocks': 2,  # Number of blocks in the ResNet
    'channels': 32,  # Number of channels in the ResNet
    'reduced_channels_reward': 32,  # Number of channels in reward head
    'reduced_channels_value': 32,  # Number of channels in value head
    'reduced_channels_policy': 32,  # Number of channels in policy head
    'resnet_fc_reward_layers': [16],  # Define the hidden layers in the reward head of the dynamic network
    'resnet_fc_value_layers': [16],  # Define the hidden layers in the value head of the prediction network
    'resnet_fc_policy_layers': [16],  # Define the hidden layers in the policy head of the prediction network

    # Fully Connected Network
    'encoding_size': 32,
    'fc_representation_layers': [16],  # Define the hidden layers in the representation network
    'fc_dynamics_layers': [16],  # Define the hidden layers in the dynamics network
    'fc_reward_layers': [16],  # Define the hidden layers in the reward network
    'fc_value_layers': [16],  # Define the hidden layers in the value network
    'fc_policy_layers': [16],  # Define the hidden layers in the policy network



    ### Training
    'training_steps': 15000,  # Total number of training steps (ie weights update according to a batch)
    'batch_size': 64,  # Number of parts of games to train on at each training step
    'checkpoint_interval': 10,  # Number of training steps before using the model for self-playing
    'value_loss_weight': 0.25,  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)

    # Exponential learning rate schedule
    'lr_init': 0.03,  # Initial learning rate
    'lr_decay_rate': 0.75,  # Set it to 1 to use a constant learning rate
    'lr_decay_steps': 150000,



    ### Replay Buffer
    'replay_buffer_size': 10000,  # Number of self-play games to keep in the replay buffer
    'num_unroll_steps': 20,  # Number of game moves to keep for every batch element
    'td_steps': 50,  # Number of steps in the future to take into account for calculating the target value
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
        self.env = TwentyOne(seed)

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 10, done

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
        choice = input(
            f"Enter the action (0) Hit, or (1) Stand for the player {self.to_play()}: "
        )
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter either (0) Hit or (1) Stand : ")
        return int(choice)

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Hit",
            1: "Stand",
        }
        return f"{action_number}. {actions[action_number]}"


class TwentyOne:
    def __init__(self, seed):
        self.random = numpy.random.RandomState(seed)

        self.player_hand = self.deal_card_value()
        self.dealer_hand = self.deal_card_value()

        self.player = 1

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.player_hand = self.deal_card_value()
        self.dealer_hand = self.deal_card_value()
        self.player = 1
        return self.get_observation()

    """
    Action: 0 = Hit
    Action: 1 = Stand
    """

    def step(self, action):

        if action == 0:
            self.player_hand += self.deal_card_value()

        done = self.is_busted() or action == 1 or self.player_hand == 21

        if done:
            self.dealer_plays()

        return self.get_observation(), self.get_reward(done), done

    def get_observation(self):
        return [
            numpy.full((3, 3), self.player_hand, dtype="float32"),
            numpy.full((3, 3), self.dealer_hand, dtype="float32"),
            numpy.full((3, 3), 0),
        ]

    def legal_actions(self):
        # 0 = hit
        # 1 = stand
        return [0, 1]

    def get_reward(self, done):
        if not done:
            return 0
        if self.player_hand <= 21 and self.dealer_hand < self.player_hand:
            return 1
        if self.player_hand <= 21 and self.dealer_hand > 21:
            return 1
        if self.player_hand > 21:
            return -1
        if self.player_hand == self.dealer_hand:
            return 0
        return -1

    def deal_card_value(self):
        card = self.random.randint(1, 13)
        if card >= 10:
            value = 10
        else:
            value = card
        return value

    def dealer_plays(self):
        if self.player_hand > 21:
            return
        while self.dealer_hand <= 16:
            self.dealer_hand += self.deal_card_value()

    def is_busted(self):
        if self.player_hand > 21:
            return True

    def render(self):
        print("Dealer hand: " + str(self.dealer_hand))
        print("Player hand: " + str(self.player_hand))
