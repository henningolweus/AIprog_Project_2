import numpy as np
import random
import math
from Hex import HexBoard  # Assuming your HexBoard class is in hex_board.py

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
        self.player_just_moved = 3 - game_state.current_player  # Assuming current_player is 1 or 2



    def UCTSelectChild(self):
        """
        Select a child node using the UCT (Upper Confidence bounds applied to Trees) metric.
        """
        log_parent_visits = math.log(self.visits)
        if self.player_just_moved == 1:
            return max(self.children, key=lambda c: c.wins / (c.visits+1) + self.c*math.sqrt( log_parent_visits /(1+ c.visits)))
        else:
            return min(self.children, key=lambda c: c.wins / (c.visits+1) - self.c*math.sqrt( log_parent_visits /(1+ c.visits)))

    def AddChild(self, move, new_game_state):
        """
        Remove the move from untried_moves and add a new child node for this move.
        Return the added child node.
        """
        child = Node(game_state=new_game_state, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def Update(self, result):
        """
        Update this node's data from the result of a simulation.
        """
        self.visits += 1
        self.wins += result

class MCTS:
   
    def __init__(self, iteration_limit):
        self.iteration_limit = iteration_limit
        self.root_node = None

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
                new_game_state.make_move(*move, node.player_just_moved)
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
                print(move)
                print("MOVE JUST EXPANDED")
                new_game_state = node.game_state.clone()
                new_game_state.make_move(*move, node.current_player) #Changed this to current
                node.AddChild(move, new_game_state)
        else :
            print("NO UNTRIED MOVES")

        

    def rollout(self, node):
        """
        Simulate the game from the current node's state until a terminal state is reached.
        A simpler policy (e.g., random moves) is used for the simulation.
        The result of the simulation is returned.
        """
        current_game_state = node.game_state.clone()
        current_player = node.player_just_moved

        while not current_game_state.is_game_over():
            # Get all legal moves for the current state
            legal_moves = current_game_state.get_legal_moves()
            if not legal_moves:
                break  # If there are no legal moves, exit the loop
            
            # Select a random move from the legal moves
            move = random.choice(legal_moves)
            # Make the move on a cloned state to simulate the game without affecting the actual game tree
            current_game_state.make_move(*move, current_player)
            # Switch to the other player
            current_player = 3 - current_player

        # Return the result of the game from the perspective of the node's player
        # If the current game state's winner is the node's player, return 1, else return -1
        if current_game_state.check_win(node.player_just_moved):
            return 1  # Node's player wins
        elif current_game_state.check_win(3 - node.player_just_moved):
            return -1  # Opponent wins
        else:
            return 0  # Draw or incomplete game
    
    def backpropagate(self, node, result):
        while node is not None:  # backpropagate from the expanded node and work back to the root node
            node.Update(result)  # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parent



    def MCTS_search(self, root_state):
        """
        Conduct a MCTS search for iteration_limit iterations starting from root_state.
        Return the best move from the root_state.
        """
        root_state = root_state.clone()
        changed_node = False

        if self.root_node is not None:
            for child in self.root_node.children:
                if root_state.__eq__(child.game_state):
                    changed_node = True
                    self.root_node = child
                    print("ROOT NODE CHANGED")
                    break
        else:
            self.root_node = Node(game_state=root_state)


        if not changed_node:
            print("PLAYED NODE NOT FOUND IN TREE")

        for _ in range(self.iteration_limit):

            # Select
            leaf_node = self.tree_policy(self.root_node)

            # Expand
            self.expand(leaf_node)
            #New Rollout. It also handels the case where there are no children
            if leaf_node.children:
                node_for_rollout = random.choice(leaf_node.children)
                result = self.rollout(node_for_rollout)
            else:
                # If no children, perform rollout from the leaf node itself
                result = self.rollout(leaf_node)
            self.backpropagate(leaf_node, result)

            # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
            #node_for_rollout = random.choice(leaf_node.children)
            #result = self.rollout(node_for_rollout)
            print("The result of the rollout is: ", result)

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
    
    def __str__(self):
        buffer_contents = f'Replay Buffer Size: {len(self.buffer)}/{self.capacity}\n'
        buffer_contents += 'Contents:\n'
        for i, (state, prob_dist) in enumerate(self.buffer):
            buffer_contents += f'  Item {i+1}: State = {state}, Prob. Dist. = {prob_dist}\n'
        return buffer_contents

