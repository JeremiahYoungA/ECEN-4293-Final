from src.hex_engine.utils.coordinates import DIRECTIONS

class HexBoard:
    def __init__(self, pieces=None, turn=0, last_move=None, streaks=None):
        """
        Initializes the board using a sparse dictionary for O(n) memory.
        pieces: Dict[tuple, int] mapping (a, b, c) to player_id.
        """
        self.pieces = pieces if pieces is not None else {}

        # 0  is first move of game (Player 1)
        # 1 is first piece of player 2
        # 2 is second piece of player 2
        # 3 is first piece of player 1
        # 4 is second piece of player 1
        self.turn = turn 
        self.last_move = last_move

        # Data structure for O(1) win detection. 
        # Format: {player_id: {axis_index: {coordinate: streak_length}}}
        # We track the length of a streak ONLY at its endpoints.
        self._streaks = streaks if streaks is not None else {
            1: {0: {}, 1: {}, 2: {}}, 
            2: {0: {}, 1: {}, 2: {}}
        }

    def get_current_player(self):
        """
        Calculates the current player based on Connect6 rules.
        Sequence: P1, P2, P2, P1, P1, P2, P2...
        """
        return 1 if (self.turn % 4) in (0, 3) else 2

    def is_occupied(self, coord):
        """Returns True if a piece exists at the given cube coordinate."""
        return coord in self.pieces

    def get_piece(self, coord):
        """Returns the player_id at coord, or None if empty."""
        return self.pieces.get(coord, None)

    def get_pieces(self, player):
        """Returns a list of (a,b,c) tuples for given player."""
        return [coord for coord, pid in self.pieces.items() if pid == player]

    def place_piece(self, coord):
        """
        Places a piece functionally. Returns a NEW HexBoard instance and a boolean 
        indicating if this move resulted in a win.
        """
        if coord in self.pieces:
            raise ValueError(f"Coordinate {coord} is already occupied.")

        player_id = self.get_current_player()
        new_board = self.copy()
        
        new_board.pieces[coord] = player_id
        new_board.last_move = coord
        new_board.turn += 1

        # Perform O(1) streak updates and check for win
        is_win = new_board._update_streaks(coord, player_id)
        
        return new_board, is_win

    def _update_streaks(self, coord, player_id):
        """
        Updates the streak endpoints and returns True if a 6-in-a-row is formed.
        Executes in strictly O(1) time complexity.
        """
        win_detected = False
        
        # There are 6 directions, which form 3 axes.
        # Axis 0: DIRECTIONS[0] & DIRECTIONS[3]
        # Axis 1: DIRECTIONS[1] & DIRECTIONS[4]
        # Axis 2: DIRECTIONS[2] & DIRECTIONS[5]
        for axis in range(3):
            dir_vec = DIRECTIONS[axis]
            opp_vec = DIRECTIONS[axis + 3]
            
            # Find neighbors along this axis
            n1 = (coord[0] + dir_vec[0], coord[1] + dir_vec[1], coord[2] + dir_vec[2])
            n2 = (coord[0] + opp_vec[0], coord[1] + opp_vec[1], coord[2] + opp_vec[2])
            
            # Get existing streak lengths connected to these neighbors
            len1 = self._streaks[player_id][axis].get(n1, 0)
            len2 = self._streaks[player_id][axis].get(n2, 0)
            
            new_len = 1 + len1 + len2
            
            if new_len >= 6:
                win_detected = True
            
            # Clean up the old internal endpoints
            if len1 > 0 and n1 in self._streaks[player_id][axis]:
                del self._streaks[player_id][axis][n1]
            if len2 > 0 and n2 in self._streaks[player_id][axis]:
                del self._streaks[player_id][axis][n2]
                    
            # Calculate the new far endpoints of this combined streak
            end1 = (coord[0] + dir_vec[0] * len1, coord[1] + dir_vec[1] * len1, coord[2] + dir_vec[2] * len1)
            end2 = (coord[0] + opp_vec[0] * len2, coord[1] + opp_vec[1] * len2, coord[2] + opp_vec[2] * len2)
            
            # Store the new total length at the far endpoints
            self._streaks[player_id][axis][end1] = new_len
            self._streaks[player_id][axis][end2] = new_len
        
        return win_detected

    def get_occupied_coordinates(self):
        """Returns a list of all coordinates currently on the board."""
        return list(self.pieces.keys())

    def copy(self):
        """
        Creates a deep copy of the board state. 
        Essential for MCTS simulations to maintain immutability.
        """
        # We manually copy the nested streaks dictionary to ensure isolated state
        new_streaks = {
            pid: {axis: dict(coords) for axis, coords in axes.items()} 
            for pid, axes in self._streaks.items()
        }
        
        return HexBoard(
            pieces=self.pieces.copy(),
            turn=self.turn,
            last_move=self.last_move,
            streaks=new_streaks
        )

    def check_win(self):
        """
        Returns true if the board is currently in a win state based on the last move.
        This simply checks the existing length at the last move's position.
        """
        if not self.last_move:
            return False
            
        player_id = self.pieces[self.last_move]
        for axis in range(3):
            if self._streaks[player_id][axis].get(self.last_move, 0) >= 6:
                return True
        return False
    
    def delete(self):
        """
        Explicitly clears internal data structures to aid memory management 
        and avoid relying solely on the garbage collector.
        """
        self.pieces.clear()
        for player_id in self._streaks:
            for axis in self._streaks[player_id]:
                self._streaks[player_id][axis].clear()
        self._streaks.clear()