import numpy as np
from ANET import ANet
from Hex import HexBoard
import matplotlib.pyplot as plt
import seaborn as sns

import json
def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)
config = load_config('config.json')


class TOPP:
    def __init__(self, anet_paths, board_size, G=25, visualise=False):
        self.anet_paths = anet_paths
        self.board_size = board_size
        self.G = G  # Number of games per match
        self.anets = self.load_anets()
        self.visualise = visualise

    def load_anets(self):
        anets = []
        for path in self.anet_paths:
            anet = ANet(board_size=self.board_size)
            anet.load_net(path)
            anets.append(anet)
        return anets
    
    def select_move_based_on_probabilities(self, legal_moves, move_probabilities, board_size):
        move_probabilities = move_probabilities.flatten()
        # Convert legal moves to indices in the probability array
        legal_indices = [row * board_size + col for row, col in legal_moves]
        # Filter and normalize probabilities for legal moves
        filtered_probs = np.zeros(move_probabilities.shape)
        filtered_probs[legal_indices] = move_probabilities[legal_indices]
        total_probs = np.sum(filtered_probs)
        if total_probs == 0:
            print("Warning: Sum of probabilities is zero. Fallback to uniform distribution among legal moves.")
            normalized_probs = np.zeros_like(filtered_probs)
            normalized_probs[legal_indices] = 1.0 / len(legal_indices) if len(legal_indices) > 0 else 0
        else:
            normalized_probs = filtered_probs / total_probs

        # Ensure no NaN values remain (sanity check)
        if np.any(np.isnan(normalized_probs)) or np.sum(normalized_probs) == 0:
            print("Error: Normalized probabilities still contain NaN or zero sum after adjustment.")
            normalized_probs = np.nan_to_num(normalized_probs)  # Convert NaN to zero
            if np.sum(normalized_probs) == 0 and len(legal_indices) > 0:
                # Assign equal probability to all legal moves as a last resort
                normalized_probs[legal_indices] = 1.0 / len(legal_indices)

        # Select a move based on the normalized probabilities
        selected_index = np.random.choice(len(normalized_probs), p=normalized_probs)
        selected_move = divmod(selected_index, board_size)  # Convert index back to row, col format
        
        return selected_move
    

    def play_game_between_policies(self, policy1, policy2, current_player):
        current_player = current_player
        game = HexBoard(self.board_size, current_player)
        while not game.is_game_over():
            anet = policy1 if current_player == 1 else policy2
            legal_moves = game.get_legal_moves()
            nn_input = game.get_nn_input_translated(current_player)
            move_probabilities = anet.predict(nn_input)
            move = self.select_move_based_on_probabilities(legal_moves,move_probabilities, self.board_size)
            game.make_move(*move, current_player)
            current_player = 3 - current_player
        if game.check_win(1):
            winner = "1" 
        elif game.check_win(2):
            winner = "2"
        else:
            winner = "0"
        
        if self.visualise:
            game.render() #Visualize the game
        return winner

    def round_robin_tournament(self):
        results = np.zeros((len(self.anets), len(self.anets)), dtype=int)
        for i, policy1 in enumerate(self.anets):
            for j, policy2 in enumerate(self.anets):
                if i == j:
                    continue  # Skip playing against itself
                policy1_wins = 0
                policy2_wins = 0
                current_player = 1
                for _ in range(self.G):
                    winner = self.play_game_between_policies(policy1, policy2, current_player)
                    if winner == "1":
                        policy1_wins += 1
                    elif winner == "2":
                        policy2_wins += 1
                    current_player = 3 - current_player  # Alternate starting player
                    if self.visualise:
                        print(f"Game between policy {i+1} (o) and policy {j+1} (x)")
                results[i, j] = policy1_wins
                results[j, i] = policy2_wins  # Assuming binary outcome
        return results


    def analyze_results(self, results):
    
        self.rank_policies(results)
        self.statistical_summary(results)
        self.specific_opponent_performance(results)


    def rank_policies(self, results):
        print("\nRanking of Policies:")
        wins = [(i+1, sum(row)) for i, row in enumerate(results)]
        ranked = sorted(wins, key=lambda x: x[1], reverse=True)
        for rank, (policy, win_count) in enumerate(ranked, start=1):
            print(f"{rank}. Policy {policy} with {win_count} wins")

    def statistical_summary(self, results):
        print("\nStatistical Summary:")
        wins = [sum(row) for row in results]
        average = np.mean(wins)
        median = np.median(wins)
        std_dev = np.std(wins)
        print(f"Average Wins: {average:.2f}, Median Wins: {median}, Standard Deviation: {std_dev:.2f}")

    def specific_opponent_performance(self, results):
        print("\nPerformance Against Specific Opponents:")
        for i, row in enumerate(results):
            best_opponent = np.argmax(row)
            worst_opponent = np.argmin(row)
            print(f"Policy {i+1} performed best against Policy {best_opponent+1} and worst against Policy {worst_opponent+1}")

if __name__ == "__main__":
    saved_models_paths = config['saved_models_paths']
    #topp = TOPP(anet_paths=saved_models_paths, board_size=4, G=25, visualise = config['visualization']['show_board'])
    topp = TOPP(
        anet_paths=saved_models_paths,  # Assuming this needs to be dynamically generated during runtime
        board_size=config['hex_game']['board_size'],
        G=config['topp']['games_per_match'],
        visualise=config['visualization']['show_board']
)
    results = topp.round_robin_tournament()
    topp.analyze_results(results)
