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
        self.player_just_moved = 3 - game_state.current_player  # Assuming current_player is 1 or 2

    def UCTSelectChild(self):
        """
        Select a child node using the UCT (Upper Confidence bounds applied to Trees) metric.
        """
        log_parent_visits = math.log(self.visits)
        return max(self.children, key=lambda c: c.wins / (c.visits+1) + self.c*math.sqrt( log_parent_visits /(1+ c.visits)))

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
   
    def __init__(self, iteration_limit=30):
        self.iteration_limit = iteration_limit
        self.root_node = None

    def tree_policy(self, node):
        while node.children != []:
            node = node.UCTSelectChild()
        return node

    def expand(self, node):
        if node.untried_moves != []:
            move = random.choice(node.untried_moves)
            new_game_state = node.game_state.clone()
            new_game_state.make_move(*move, node.player_just_moved)
            node.AddChild(random.choice(node.untried_moves), new_game_state)
        else :
            print("NO UNTRIED MOVES")

    def rollout(self, node):
        print(node.children)
        node = node.children[0]
        while node.untried_moves != [] and node.game_state.check_win(node.player_just_moved):
            move = random.choice(node.untried_moves)
            new_game_state = node.game_state.clone()
            new_game_state.make_move(*move, node.player_just_moved)
            node.AddChild(random.choice(node.untried_moves), new_game_state)
            node = node.children[0]
        return node
    
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
                    break
        else:
            self.root_node = Node(game_state=root_state)


        if not changed_node:
            print("NODE NOT FOUND")

        for _ in range(self.iteration_limit):

            # Select
            leaf_node = self.tree_policy(self.root_node)

            # Expand
            self.expand(leaf_node)

            # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
            final_node = self.rollout(leaf_node)
            result = final_node.game_state.get_result(final_node.player_just_moved)

            # Backpropagate
            self.backpropagate(final_node, result)
            move_probabilities = self.calculateMoveProbabilities(self.root_node)
            best_node = max(self.root_node.children, key=lambda c: c.visits)
            best_move = best_node.move

            # Remove the child node of the leaf node
            leaf_node.children[0].children = []

        self.root_node = best_node
        # Return the move that was most visited
        return best_move, move_probabilities
    
    
    def calculateMoveProbabilities(self, root_node):
        """
        Generate a probability distribution over moves based on visit counts of child nodes of the root.
        """
        total_visits = sum(child.visits for child in root_node.children)
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

