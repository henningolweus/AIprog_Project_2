import numpy as np
import random
from Hex import HexBoard  # Assuming your HexBoard class is in hex_board.py
from MCTS import MCTS


def play_hex_with_mcts(board_size=4, iteration_limit=1000):
    game = HexBoard(board_size)
    mcts = MCTS(iteration_limit=iteration_limit)
    current_player = game.current_player

    while True:
        print("Current board:")
        game.render()
        if game.check_win(current_player):
            winner = "1" if game.check_win(1) else "2"
            print(f"Game Over! Player {winner} wins!")
            break

        print(f"Player {current_player}'s turn.")
        if current_player == 1:
            row, col = map(int, input("Enter row and column separated by space: ").strip().split())
            game.make_move(row, col, current_player)
        else:  # Player 2 uses MCTS
            move = mcts.UCT(game)
            game.make_move(*move, current_player)
            print(game.get_nn_input(current_player))

            print(f"MCTS played move: {move}")

        current_player = 2 if current_player == 1 else 1

if __name__ == "__main__":
    play_hex_with_mcts(board_size=4, iteration_limit=1000)