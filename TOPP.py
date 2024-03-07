import numpy as np
from ANET import ANet
from Hex import HexBoard
from MCTS import MCTS

class TOPP:
    def __init__(self, anet_paths, board_size, G=25):
        self.anet_paths = anet_paths
        self.board_size = board_size
        self.G = G  # Number of games per match
        self.anets = self.load_anets()

    def load_anets(self):
        anets = []
        for path in self.anet_paths:
            anet = ANet(board_size=self.board_size)
            anet.load_net(path)
            anets.append(anet)
        return anets

    def play_game_between_policies(self, policy1, policy2):
        game = HexBoard(self.board_size)
        current_player = 1
        while not game.is_game_over():
            anet = policy1 if current_player == 1 else policy2
            mcts = MCTS(iteration_limit=50, anet=anet)
            move, _ = mcts.MCTS_search(game)
            game.make_move(*move, current_player)
            current_player = 3 - current_player
        winner = "1" if game.check_win(1) else "2"
        return winner

    def round_robin_tournament(self):
        results = np.zeros((len(self.anets), len(self.anets)), dtype=int)
        for i, policy1 in enumerate(self.anets):
            for j, policy2 in enumerate(self.anets):
                if i == j:
                    continue  # Skip playing against itself
                policy1_wins = 0
                for _ in range(self.G):
                    winner = self.play_game_between_policies(policy1, policy2)
                    if winner == "1":
                        policy1_wins += 1
                results[i, j] = policy1_wins
                results[j, i] = self.G - policy1_wins  # Assuming binary outcome
        return results

    def analyze_results(self, results):
        # Here you can implement more sophisticated analysis
        print("Tournament Results Matrix (Rows: Players, Columns: Opponents, Values: Wins):")
        print(results)
        # Additional analysis can be added here


if __name__ == "__main__":
    saved_models_paths = [
        "anet_params_20.h5", 
        "anet_params_40.h5", 
        "anet_params_60.h5",
        "anet_params_80.h5", 
        "anet_params_100.h5"
    ]
    topp = TOPP(anet_paths=saved_models_paths, board_size=4, G=25)
    results = topp.round_robin_tournament()
    topp.analyze_results(results)