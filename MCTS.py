import numpy as np
import random
from Hex import HexBoard  # Assuming your HexBoard class is in hex_board.py

class Node:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = game_state.get_legal_moves()
        self.player_just_moved = 3 - game_state.current_player  # Assuming current_player is 1 or 2

    def UCTSelectChild(self):
        """
        Select a child node using the UCT (Upper Confidence bounds applied to Trees) metric.
        """
        import math
        log_parent_visits = math.log(self.visits)
        return max(self.children, key=lambda c: c.wins / c.visits + math.sqrt(2 * log_parent_visits / c.visits))

    def AddChild(self, move, game_state):
        """
        Remove the move from untried_moves and add a new child node for this move.
        Return the added child node.
        """
        child = Node(game_state=game_state, parent=self, move=move)
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
    def __init__(self, iteration_limit=1000):
        self.iteration_limit = iteration_limit

    def UCT(self, root_state):
        """
        Conduct a UCT search for iteration_limit iterations starting from root_state.
        Return the best move from the root_state.
        """
        root_node = Node(game_state=root_state)

        for _ in range(self.iteration_limit):
            node = root_node
            state = root_state.clone()  # Ensure you have a
            # method to clone the game state in your HexBoard class

            # Select
            while node.untried_moves == [] and node.children != []:  # node is fully expanded and non-terminal
                node = node.UCTSelectChild()
                state.make_move(*node.move, state.current_player)

            # Expand
            if node.untried_moves:  # if we can expand (i.e. state/node is non-terminal)
                move = random.choice(node.untried_moves) 
                next_state = state.clone()
                next_state.make_move(*move, next_state.current_player)
                node = node.AddChild(move, next_state)  # add child and descend tree

            # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
            while state.get_legal_moves():  # while state is non-terminal
                state.make_move(*random.choice(state.get_legal_moves()), state.current_player)

            # Backpropagate
            while node is not None:  # backpropagate from the expanded node and work back to the root node
                node.Update(state.get_result(node.player_just_moved))  # state is terminal. Update node with result from POV of node.playerJustMoved
                node = node.parent

        # Return the move that was most visited
        return max(root_node.children, key=lambda c: c.visits).move


