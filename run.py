import os
import sys

import nevergrad
import ray

from muzero.muzero import MuZero, hyperparameter_search


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Train directly with "python muzero.py cartpole"
        muzero = MuZero(sys.argv[1])
        muzero.train()
    else:
        print("\nWelcome to MuZero! Here's a list of games:")
        # Let user pick a game
        games = [
            filename[:-3]
            for filename in sorted(
                os.listdir(os.path.dirname(os.path.realpath(__file__)) + "/muzero/games")
            )
            if filename.endswith(".py") and filename != "abstract_game.py"
        ]
        for i in range(len(games)):
            print(f"{i}. {games[i]}")
        choice = input("Enter a number to choose the game: ")
        valid_inputs = [str(i) for i in range(len(games))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")

        # Initialize MuZero
        choice = int(choice)
        game_name = games[choice]
        muzero = MuZero(game_name)

        while True:
            # Configure running options
            options = [
                "Train",
                "Load pretrained model",
                "Diagnose model",
                "Render some self play games",
                "Play against MuZero",
                "Test the game manually",
                "Hyperparameter search",
                "Exit",
            ]
            print()
            for i in range(len(options)):
                print(f"{i}. {options[i]}")

            choice = input("Enter a number to choose an action: ")
            valid_inputs = [str(i) for i in range(len(options))]
            while choice not in valid_inputs:
                choice = input("Invalid input, enter a number listed above: ")
            choice = int(choice)
            if choice == 0:
                muzero.train()
            elif choice == 1:
                checkpoint_path = input(
                    "Enter a path to the model.checkpoint, or ENTER if none: "
                )
                while checkpoint_path and not os.path.isfile(checkpoint_path):
                    checkpoint_path = input("Invalid checkpoint path. Try again: ")
                replay_buffer_path = input(
                    "Enter a path to the replay_buffer.pkl, or ENTER if none: "
                )
                while replay_buffer_path and not os.path.isfile(replay_buffer_path):
                    replay_buffer_path = input(
                        "Invalid replay buffer path. Try again: "
                    )
                muzero.load_model(
                    checkpoint_path=checkpoint_path,
                    replay_buffer_path=replay_buffer_path,
                )
            elif choice == 2:
                muzero.diagnose_model(30)
            elif choice == 3:
                muzero.test(render=True, opponent="self", muzero_player=None)
            elif choice == 4:
                muzero.test(render=True, opponent="human", muzero_player=0)
            elif choice == 5:
                env = muzero.Game()
                env.reset()
                env.render()

                done = False
                while not done:
                    action = env.human_to_action()
                    observation, reward, done = env.step(action)
                    print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
                    env.render()
            elif choice == 6:
                # Define here the parameters to tune
                # Parametrization documentation: https://facebookresearch.github.io/nevergrad/parametrization.html
                muzero.terminate_workers()
                del muzero
                budget = 20
                parallel_experiments = 2
                lr_init = nevergrad.p.Log(a_min=0.0001, a_max=0.1)
                discount = nevergrad.p.Log(lower=0.95, upper=0.9999)
                parametrization = nevergrad.p.Dict(lr_init=lr_init, discount=discount)
                best_hyperparameters = hyperparameter_search(
                    game_name, parametrization, budget, parallel_experiments, 20
                )
                muzero = MuZero(game_name, best_hyperparameters)
            else:
                break
            print("\nDone")

    ray.shutdown()
