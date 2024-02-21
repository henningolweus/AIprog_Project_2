import numpy as np
import random
from Hex import HexBoard  # Assuming your HexBoard class is in hex_board.py
from MCTS import MCTS, ReplayBuffer
from ANET import ANet


def play_hex_with_mcts(board_size=4, iteration_limit=150):
    game = HexBoard(board_size)
    mcts = MCTS(iteration_limit=iteration_limit)
    anet = ANet(board_size=4, learning_rate=0.001, hidden_layers=[64, 64], activation='relu', optimizer_name='adam', num_cached_nets=10)
    Replay_buffer = ReplayBuffer()

    current_player = game.current_player # This is redunant as for now

    while True:
        print("Current board:")
        print(game.get_nn_input(current_player))
        input_varaible = game.get_nn_input(current_player)
        
        game.render()
        if game.check_win(current_player):
            winner = "1" if game.check_win(1) else "2"
            print(f"Game Over! Player {winner} wins!")
            print(Replay_buffer) #For visualization
            break

        print(f"Player {current_player}'s turn.")
        if current_player == 1:
            row, col = map(int, input("Enter row and column separated by space: ").strip().split())
            game.make_move(row, col, current_player)
        else:  # Player 2 uses MCTS
            #move = mcts.UCT(game)
            move, move_probabilities = mcts.MCTS_search(game)
            game.make_move(*move, current_player)

            print(f"MCTS calculates the following prob distribtution: {move_probabilities}")
            print(f"MCTS played move: {move}")
            Replay_buffer.push(input_varaible, move_probabilities) # Store the input variables and the target variables

        current_player = 2 if current_player == 1 else 1

if __name__ == "__main__":
    play_hex_with_mcts(board_size=4, iteration_limit=30)
    