import numpy as np
import random
from Hex import HexBoard  # Assuming your HexBoard class is in hex_board.py
from MCTS import MCTS, ReplayBuffer
from ANET import ANet


def play_hex_with_mcts(board_size=4, iteration_limit=100):
    game_counter = 0
    save_interval = 50
    total_games = 200
    iteration_limit=iteration_limit
    anet = ANet(board_size=4, learning_rate=0.001, hidden_layers=[64, 64], activation='relu', optimizer_name='adam', num_cached_nets=10)
    batch_size = 32  # This can be adjusted based on your needs and dataset size.
    num_epochs = 10  # Number of epochs to train ANET for each batch.
    player1_wins = 0
    player2_wins = 0
    Replay_buffer = ReplayBuffer()


    for game_index in range(total_games):
        game = HexBoard(board_size)
        mcts = MCTS(iteration_limit=iteration_limit, anet=anet)
        current_player = game.current_player # This is redunant as for now
        print("Current board:")
        print(game.get_nn_input(current_player))
        input_varaible = game.get_nn_input(current_player)

        while not game.is_game_over():
            game.render()

            #print(f"Player {current_player}'s turn.")
            if current_player == 1:
                move, move_probabilities = mcts.MCTS_search(game)
                game.make_move(*move, current_player)

                #print(f"MCTS calculates the following prob distribtution: {move_probabilities}")
                #print(f"MCTS played move: {move}")
            else:  # Player 2 uses MCTS
                #move = mcts.UCT(game)
                move, move_probabilities = mcts.MCTS_search(game)
                game.make_move(*move, current_player)

                #print(f"MCTS calculates the following prob distribtution: {move_probabilities}")
                #print(f"MCTS played move: {move}")
                Replay_buffer.push(input_varaible, move_probabilities) # Store the input variables and the target variables
            if game.check_win(current_player):
                winner = "1" if game.check_win(1) else "2"
                print(f"Game Over! Player {winner} wins!")
                if winner == "1":
                    player1_wins += 1
                else:
                    player2_wins += 1
                #print(Replay_buffer) #For visualization
                break
            
            current_player = 2 if current_player == 1 else 1
        print("Player 2 win ratio: ", player2_wins/(game_index+1))
        if game_counter % save_interval == 0:
            anet.save_net(f"anet_params_{game_counter}.h5")
            print(f"Saved ANET parameters after game {game_counter+1}.")

if __name__ == "__main__":
    play_hex_with_mcts(board_size=4, iteration_limit=100)
    