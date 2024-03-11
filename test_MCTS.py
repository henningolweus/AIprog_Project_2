import unittest
from MCTS import MCTS
from Hex import HexBoard
from ANET import ANet

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

#if __name__ == '__main__':
#    unittest.main()

anet = ANet(board_size=4, learning_rate=0.001, hidden_layers=[64, 64], activation='relu', optimizer_name='adam', num_cached_nets=10)
anet.save_net(f"anet_params_0.h5")