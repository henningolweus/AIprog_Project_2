import unittest
from MCTS import MCTS
from Hex import HexBoard

class TestMCTS(unittest.TestCase):
    def test_mcts_search(self):
        # Create a Hex board
        board_size = 5
        hex_board = HexBoard(board_size)

        # Create an instance of MCTS
        mcts = MCTS()

        # Perform MCTS search on the initial board state
        root_state = hex_board.get_state()
        mcts.MCTS_search(root_state)

        # Assert that the search has been performed successfully
        # Add your assertions here

if __name__ == '__main__':
    unittest.main()
