import numpy as np
import random
import math
from Hex import HexBoard  # Assuming your HexBoard class is in hex_board.py
from ANET import ANet

class Node:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.c = 1
        self.visits = 0
        self.untried_moves = game_state.get_legal_moves()
        self.current_player = game_state.current_player
        self.opponent_player = 3 - game_state.current_player  # Assuming current_player is 1 or 2
        self.predicted_move_probabilities = None



    def UCTSelectChild(self):
        """
        Select a child node using the UCT (Upper Confidence bounds applied to Trees) metric.
        """
        log_parent_visits = math.log(self.visits)
        if self.current_player == 2:
            return min(self.children, key=lambda c: c.wins / (c.visits + 1) - self.c * math.sqrt(log_parent_visits / (1 + c.visits)))
        else:
            return max(self.children, key=lambda c: c.wins / (c.visits + 1) + self.c * math.sqrt(log_parent_visits / (1 + c.visits)))

    def AddChild(self, move, new_game_state):
        """
        Remove the move from untried_moves and add a new child node for this move.
        Return the added child node.
        """
        child = Node(game_state=new_game_state, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def Update(self, result, leaf_node_player):
        """
        Update this node's data from the result of a simulation.
        """
        self.visits += 1
        self.wins+= result

class MCTS:
   
    def __init__(self, iteration_limit, anet=None, player=2):
        self.iteration_limit = iteration_limit
        self.root_node = None
        self.anet = anet
        self.player = 2

    def tree_policy(self, node):
        while node.children != [] and (not node.game_state.check_win(1) ) and (not node.game_state.check_win(2) ):
            node = node.UCTSelectChild()
        return node
    
    """
    def expand(self, node):
        legal_moves = node.game_state.get_legal_moves()  # Refresh the list of legal moves if necessary
        while node.untried_moves:
            move = random.choice(node.untried_moves)  # Randomly select a move
            if move in legal_moves:  # Ensure the move is still legal 
                new_game_state = node.game_state.clone()
                new_game_state.make_move(*move, node.opponent_player)  # Make the move for the opponent (opponent_player is the player who just moved
                node.AddChild(move, new_game_state)
            else:
                # This else block is optional and can be used for debugging
                print(f"Attempted to expand an illegal or already tried move: {move}")
                node.untried_moves.remove(move)  # Remove the move to prevent future attempts

        if not node.children:
            print("No legal moves were available for expansion or all moves have been tried.")


        """
    def expand(self, node):
        if node.untried_moves != []:
            while node.untried_moves != []:
                move = random.choice(node.untried_moves)
                # print(move)
                # print("MOVE JUST EXPANDED")
                new_game_state = node.game_state.clone()
                new_game_state.make_move(*move, node.current_player) #Changed this to current
                node.AddChild(move, new_game_state)
        else :
            #print("NO UNTRIED MOVES")'
            pass

    def select_move_based_on_probabilities(self, legal_moves, move_probabilities, board_size):
        """
        Select a move based on normalized probabilities, excluding illegal moves.
        
        Parameters:
        - legal_moves: List of tuples representing legal moves (e.g., [(row1, col1), (row2, col2), ...]).
        - move_probabilities: Numpy array of probabilities predicted by ANet for all moves.
        
        Returns:
        - A tuple representing the selected move.
        """
        move_probabilities = move_probabilities.flatten()
        
        # Convert legal moves to indices in the probability array
        legal_indices = [row * board_size + col for row, col in legal_moves]
        
        # Filter and normalize probabilities for legal moves
        filtered_probs = np.zeros(move_probabilities.shape)
        filtered_probs[legal_indices] = move_probabilities[legal_indices]
        normalized_probs = filtered_probs / np.sum(filtered_probs)
        
        # Select a move based on the normalized probabilities
        selected_index = np.random.choice(len(normalized_probs), p=normalized_probs)
        selected_move = divmod(selected_index, board_size)  # Convert index back to row, col format
        
        return selected_move

    def rollout(self, node,epsilon=0.1,  randomChoice = False):
        """
        Simulate the game from the current node's state until a terminal state is reached.
        A simpler policy (e.g., random moves) is used for the simulation.
        The result of the simulation is returned.
        """
        current_game_state = node.game_state.clone()
        current_player = node.current_player

        while not current_game_state.is_game_over():
            # Get all legal moves for the current state
            legal_moves = current_game_state.get_legal_moves()
            if not legal_moves:
                break  # If there are no legal moves, exit the loop
            if randomChoice:
                # Select a random move from the legal moves
                move = random.choice(legal_moves)
            else:
                if random.random() < epsilon:
                    move = random.choice(legal_moves)
                else:
                    nn_input = current_game_state.get_nn_input_translated(current_player)
                    move_probabilities = self.anet.predict(nn_input)
                    board_size = current_game_state.get_board_size()
                    move = self.select_move_based_on_probabilities(legal_moves,move_probabilities, board_size)
            # Make the move on a cloned state to simulate the game without affecting the actual game tree
            current_game_state.make_move(*move, current_player)
            # Switch to the other player
            current_player = 3 - current_player

        # Return the result of the game from the perspective of the node's player
        # If the current game state's winner is the node's player, return 1, else return -1
        if current_game_state.check_win(1):
            return 1  # Player 1 wins
        else:
            return -1  # Player 2 wins
    
    def backpropagate(self, node, result):
        leaf_node_player = node.current_player
        while node is not None:  # backpropagate from the expanded node and work back to the root node
            node.Update(result, leaf_node_player)  # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parent



    def MCTS_search(self, root_state, epsilon):
        """
        Conduct a MCTS search for iteration_limit iterations starting from root_state.
        Return the best move from the root_state.
        """
        root_state = root_state.clone()
        found_node = False

        if self.root_node is not None:
            if root_state.__eq__(self.root_node.game_state):
                #print("ROOT NODE FOUND. WAS INITIAL NODE")
                found_node = True
            else:
                for child in self.root_node.children:
                    if root_state.__eq__(child.game_state):
                        found_node = True
                        self.root_node = child
                        #print("ROOT NODE FOUND. WAS CHILD NODE")
                        break
        else:
            self.root_node = Node(game_state=root_state)


        if not found_node:
            print("PLAYED NODE NOT FOUND IN TREE")

        for _ in range(self.iteration_limit):

            # Select
            leaf_node = self.tree_policy(self.root_node)

            # Expand
            self.expand(leaf_node)
            #New Rollout. It also handels the case where there are no children
            if leaf_node.children:
                if random.random() < epsilon:
                    node_for_rollout = random.choice(leaf_node.children)
                else:
                    legal_moves = leaf_node.game_state.get_legal_moves()
                    nn_input = leaf_node.game_state.get_nn_input_translated(leaf_node.current_player)
                    move_probabilities = self.anet.predict(nn_input)
                    board_size = leaf_node.game_state.get_board_size()
                    move = self.select_move_based_on_probabilities(legal_moves,move_probabilities, board_size)
                    node_for_rollout = random.choice(leaf_node.children)
                    for child in leaf_node.children:
                        if child.move == move:
                            node_for_rollout = child
                result = self.rollout(node_for_rollout, epsilon, randomChoice=False)
            else:
                # If no children, perform rollout from the leaf node itself
                result = self.rollout(leaf_node, epsilon)

            # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
            #node_for_rollout = random.choice(leaf_node.children)
            #result = self.rollout(node_for_rollout)
            # print("Iteration:" + str(_) + "/" + str(self.iteration_limit)+ "The result of the rollout is: ", result)

            # Backpropagate
            self.backpropagate(leaf_node, result)
        move_probabilities = self.calculateMoveProbabilities(self.root_node)

        # Sort children by visits to get a list from most visited to least
        sorted_children = sorted(self.root_node.children, key=lambda c: c.visits, reverse=True)
        
        # Find the first legal move among the sorted children
        legal_moves = root_state.get_legal_moves()
        for child in sorted_children:
            if child.move in legal_moves:
                best_move = child.move
                # Since this move is selected, we update the root_node to this child for future searches
                self.root_node = child
                break
        else:
            print("No legal moves found in MCTS search. This should not happen if the game is not over.")
            best_move = None  # Fallback case, should not happen if there are legal moves available

        return best_move, move_probabilities
        

    def calculateMoveProbabilities(self, root_node):
        """
        Generate a probability distribution over moves based on visit counts of child nodes of the root.
        Adjusted to handle the case where total_visits is zero by returning a uniform distribution or other logic.
        """
        total_visits = sum(child.visits for child in root_node.children)
        if total_visits == 0:
            # Handle the case where there are no visits to any children.
            # Option 1: Return a uniform distribution among all children.
            num_children = len(root_node.children)
            if num_children == 0:
                return {}  # Handle the case with no children.
            uniform_probability = 1 / num_children
            move_probabilities = {child.move: uniform_probability for child in root_node.children}
        else:
            # Normal case, where we calculate probabilities based on visits.
            move_probabilities = {child.move: child.visits / total_visits for child in root_node.children}
        return move_probabilities
    






class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, prob_dist):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # Store state and MCTS probability distribution
        self.buffer[self.position] = (state, prob_dist)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, prob_dist = map(np.stack, zip(*batch))
        return state, prob_dist

    def __len__(self):
        return len(self.buffer)
    
    def convert_probs_to_array(self, probs_dict, board_size=4):
        # Initialize an array of zeros for each move's probability
        probs_array = np.zeros(board_size**2)
        
        # Iterate over each move-probability pair in the dictionary
        for move, prob in probs_dict.items():
            # Calculate the index in the array corresponding to the move
            index = move[0] * board_size + move[1]
            # Set the probability in the corresponding array position
            probs_array[index] = prob
        
        return probs_array
    
    def __str__(self):
        buffer_contents = f'Replay Buffer Size: {len(self.buffer)}/{self.capacity}\n'
        buffer_contents += 'Contents:\n'
        for i, (state, prob_dist) in enumerate(self.buffer):
            buffer_contents += f'  Item {i+1}: State = {state}, Prob. Dist. = {prob_dist}\n'
        return buffer_contents

