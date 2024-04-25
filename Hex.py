####Main Man¨
import numpy as np
import random
#################Functional logic with correct vizualization################
class HexBoard:
    def __init__(self, size, starting_player=1):
        self.size = size
        self.board = np.zeros((size, size, 2), dtype=int)
        self.current_player = starting_player  # brukes 

    """
    def __eq__(self, other):
        if not isinstance(other, HexBoard):
            return NotImplemented
        return (np.array_equal(self.board, other.board) and 
                self.current_player == other.current_player)
    """
    def __eq__(self, other):
        if not isinstance(other, HexBoard):
            print("Comparison failed: The other object is not an instance of HexBoard.")
            return False
        
        boards_equal = np.array_equal(self.board, other.board)
        players_equal = (self.current_player == other.current_player)
        
        if not boards_equal or not players_equal:
            print("Comparison details:")
            print(f"Boards equal: {boards_equal}")
            if not boards_equal:
                print("Self board:")
                print(self.board)
                print("Other board:")
                print(other.board)
                
            print(f"Current players equal: {players_equal}")
            print(f"Self current player: {self.current_player}, Other current player: {other.current_player}")
        
        return boards_equal and players_equal
    
    def get_board_size(self):
        return self.size

    def render(self):
        """Render the board to the console in a more visually intuitive diamond shape."""

        def cell_symbol(cell):
            if np.array_equal(cell, [1, 0]):
                return 'o'  # Player 1
            elif np.array_equal(cell, [0, 1]):
                return 'x'  # Player 2
            return '.'

        print(" " * (self.size * 2 - 1) + "N")  # North
        # Top half of the diamond
        for r in range(self.size):
            print(" " * ((self.size - r - 1) * 2), end='')
            for c in range(r + 1):
                symbol = cell_symbol(self.board[r - c, c])  
                print(symbol, end=' ')
                if c < r:
                    print("-", end=' ')
            print()

        # Bottom half of the diamond
        for r in range(self.size - 2, -1, -1):
            print(" " * ((self.size - r - 1) * 2), end='')
            for c in range(r + 1):
                symbol = cell_symbol(self.board[self.size - c - 1, self.size - r + c - 1])  
                print(symbol, end=' ')
                if c < r:
                    print("-", end=' ')
            print()
        print(" " * (self.size * 2 - 1) + "S")  # South

        # Correcting East and West markers
        print(" " * 2 + "W" + " " * (self.size * 4 - 5) + "E")  

    
    def get_nn_input_translated(self, current_player):
        # Flatten the board to create a single array
        flat_board = np.zeros((self.board.shape[0], self.board.shape[1]), dtype=int)


        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if np.array_equal(self.board[i, j], [1, 0]):
                    flat_board[i, j] = 1
                elif np.array_equal(self.board[i, j], [0, 1]):
                    flat_board[i, j] = -1
                elif np.array_equal(self.board[i, j], [0, 0]):
                    flat_board[i, j] = 0
        
        flat_board = flat_board.flatten()
        # Add the current player indicator at the end of the flattened board
        current_player_indicator = 1 if current_player == 1 else -1
        nn_input = np.append(flat_board, current_player_indicator)
        return nn_input  # Return as NumPy array

    
    def clone(self):
        """
        Create a deep copy of the board.
        """
        new_board = HexBoard(self.size)
        new_board.board = np.copy(self.board)
        new_board.current_player = self.current_player
        return new_board
    
    def get_result(self, player_just_moved):
        """
        Determine the game result from the perspective of 'player_just_moved'.
        :param player_just_moved: The player who made the last move.
        :return: 1 if 'player_just_moved' has won, -1 otherwise.
        """
        if self.check_win(player_just_moved) and player_just_moved == 1:
            return 1  # The player who just moved has won the game.
        elif self.check_win(player_just_moved) and player_just_moved == 2:
            return -1  # The player who just moved has won the game.
        else:
            return 0  # Noone won

    # Remember to update or add other necessary methods.

    def get_legal_moves(self):
        """
        Returns a list of all legal moves on the board.
        Legal moves are those where the cell is empty (represented as [0, 0]).
        """
        return [(row, col) for row in range(self.size) for col in range(self.size) if np.all(self.board[row, col] == [0, 0])]

    def make_move(self, row, col, player):
        """
        Attempts to place a piece on the board for the given player.
        - row, col: The coordinates where the player wants to place their piece.
        - player: The player making the move, 1 for Player 1 and 2 for Player 2.
        """
        if (row, col) not in self.get_legal_moves():
            return False  # Move is illegal if it's not in the list of legal moves.
        self.board[row, col] = [1, 0] if player == 1 else [0, 1]
        self.current_player = 2 if player == 1 else 1 # Moved the logic inside this function
        return True
    def is_game_over(self):
        """
        Checks if the game is a draw.
        A draw occurs if the board is full and no player has won.
        Returns True if the game is a draw, False otherwise.
        """
        return all(np.all(cell != [0, 0]) for cell in self.board.flatten()) or self.check_win(1) or self.check_win(2)

    def check_win(self, player):
        """
        Checks if the specified player has won.
        - player: The player to check for a win condition, 1 for Player 1 and 2 for Player 2.
        Uses a depth-first search (dfs) starting from each piece belonging to the player on their starting edge.
        Returns True if a connecting path from one side to the opposite side is found, False otherwise.
        """
        visited = set()  # Tracks visited cells to prevent infinite loops during DFS.
        player_id = [1, 0] if player == 1 else [0, 1]  # Identifies the player's pieces on the board.
        for row_or_col in range(self.size):
            # For Player 1, check for any pieces on the top row. For Player 2, check the leftmost column.
            if player == 1 and np.all(self.board[0, row_or_col] == player_id):
                if self.dfs(0, row_or_col, visited, player_id):
                    return True  # Player 1 wins if a path to the bottom row is found.
            if player == 2 and np.all(self.board[row_or_col, 0] == player_id):
                if self.dfs(row_or_col, 0, visited, player_id):
                    return True  # Player 2 wins if a path to the rightmost column is found.
        return False

    def dfs(self, row, col, visited, player_id):
        """
        Depth-first search to find a path from one side to the opposite side for the current player.
        - row, col: The current cell being visited.
        - visited: A set of already visited cells to avoid revisiting.
        - player_id: Identifies the current player's pieces on the board.
        Returns True if a path to the opposite side is found, False otherwise.
        """
        if (row, col) in visited:
            return False  # Current cell already visited, avoid cycles.
        if player_id == [1, 0] and row == self.size - 1 or player_id == [0, 1] and col == self.size - 1:
            return True  # A path is found to the target edge.
        visited.add((row, col))  # Mark the current cell as visited.
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]  # All possible move directions.
        for d_row, d_col in directions:
            n_row, n_col = row + d_row, col + d_col
            # Check for valid next cells that belong to the player and haven't been visited.
            if 0 <= n_row < self.size and 0 <= n_col < self.size and np.all(self.board[n_row, n_col] == player_id):
                if self.dfs(n_row, n_col, visited, player_id):
                    return True  # Found a path through this branch.
        return False  # No path found through this cell.

if __name__ == "__main__":
    size = 4
    hex_board = HexBoard(size, 1)
    current_player = hex_board.current_player

    while True:
        hex_board.render()
        print(f"Player {current_player}'s turn.")
        row, col = map(int, input("Enter row and column separated by space: ").strip().split())
        nn_input = hex_board.get_nn_input(current_player)
        print("Neural network input:", nn_input)
        
        if hex_board.make_move(row, col, current_player):
            if hex_board.check_win(current_player):
                hex_board.render()
                print(f"Game Over! Player {current_player} wins!")
                break
            current_player = 2 if current_player == 1 else 1
        else:
            print("Invalid move. Try again.")
