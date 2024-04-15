import numpy as np
import random
from Hex import HexBoard  # Assuming your HexBoard class is in hex_board.py
from MCTS import MCTS, ReplayBuffer
from ANET import ANet

import json

def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

config = load_config('config.json')


def play_hex_with_mcts():
    board_size = config['hex_game']['board_size']
    total_games = config['hex_game']['total_games']

    iteration_limit = config['mcts']['iteration_limit']

    save_interval = config['training']['save_interval']
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs'] # Number of epochs to train ANET for each batch.
    game_counter = 0
    player1_wins = 0
    player2_wins = 0



    anet = ANet(
        board_size=config['hex_game']['board_size'], 
        learning_rate=config['anet']['learning_rate'], 
        hidden_layers=config['anet']['hidden_layers'], 
        activation=config['anet']['activation'], 
        optimizer_name=config['anet']['optimizer'], 
        num_cached_nets=config['anet']['num_cached_nets']
    )
    #anet.load_net("anet_params_100_50.h5") # Load the parameters from the previous training session


    epsilon_start = config["mcts"]["epsilon_start"] # Initial epsilon value for exploration
    epsilon_end = config["mcts"]["epsilon_end"]   # Minimum epsilon value
    epsilon_decay = config["mcts"]["epsilon_decay"]  # Decay rate per game

    epsilon = epsilon_start  # Current epsilon value
    starting_player = 2  # Player 1 starts the game
    for game_index in range(total_games):
        Replay_buffer = ReplayBuffer()
        starting_player = 3-starting_player 
        game = HexBoard(board_size, starting_player)
        mcts = MCTS(iteration_limit=iteration_limit, anet=anet)
        print("Current board:")
        print(game.get_nn_input(starting_player))
        batch_size = config.get('training').get('batch_size')
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        current_player = starting_player

        while not game.is_game_over():
            input_varaible = game.get_nn_input(current_player)
            game.render()

            #print(f"Player {current_player}'s turn.")
            
            move, move_probabilities = mcts.MCTS_search(game, epsilon=epsilon)
            game.make_move(*move, current_player)

            #print(f"MCTS calculates the following prob distribtution: {move_probabilities}")
            print(f"MCTS played move: {move}")
            Replay_buffer.push(input_varaible, move_probabilities)
            # Store the input variables and the target variables
            if game.check_win(current_player):
                winner = "1" if game.check_win(1) else "2"
                print(f"Game Over! Player {winner} wins!")
                if winner == "1":
                    player1_wins += 1
                else:
                    player2_wins += 1
                #print(Replay_buffer) #For visualization
                game.render()
                print(f"Game between player 1 (o) and player 2(x), the winner is player {winner}.")
                break
            
            current_player = 2 if current_player == 1 else 1
        game_counter+=1
        states, target_probs_dicts = Replay_buffer.sample(batch_size)
        target_probs = np.array([Replay_buffer.convert_probs_to_array(prob_dict, board_size = board_size) for prob_dict in target_probs_dicts])
        print(states)
        anet.train(states, target_probs, epochs=num_epochs)
        print("Player 2 win ratio: ", player2_wins/(game_index+1))
        if game_counter % save_interval == 0: 
            anet.save_net(f"anet_params_DEMO{game_counter}.h5")
            print(f"Saved ANET parameters after game {game_counter+1}.")

if __name__ == "__main__":
    play_hex_with_mcts()
    