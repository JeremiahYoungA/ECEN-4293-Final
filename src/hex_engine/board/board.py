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
        # History for mutable in-place moves (used by do_move/undo_move)
        self.history = []

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
        # Keep functional API for callers that expect immutability.
        new_board = self.copy()
        is_win = new_board.do_move(coord)
        return new_board, is_win

    def do_move(self, coord):
        """Mutate the board in-place and record history for O(1) undo.

        Returns True if the move produced a win.
        """
        if coord in self.pieces:
            raise ValueError(f"Coordinate {coord} is already occupied.")

        player_id = self.get_current_player()

        # Create compact history record storing neighbor streak lengths
        rec = {
            'coord': coord,
            'player_id': player_id,
            'has_old_last_move': self.last_move is not None,
            'old_last_move': self.last_move,
            'len1': [0, 0, 0],
            'len2': [0, 0, 0]
        }

        # Place piece
        self.pieces[coord] = player_id
        self.last_move = coord
        self.turn += 1

        win = self._update_streaks(coord, player_id, rec, is_undo=False)

        # Record history for perfect undo
        self.history.append(rec)
        return win

    def undo_move(self):
        """Undo the last in-place move using recorded history."""
        if not self.history:
            return

        rec = self.history.pop()
        coord = rec['coord']
        player_id = rec['player_id']

        # Remove placed piece and restore meta
        if coord in self.pieces:
            del self.pieces[coord]
        self.turn -= 1
        if rec['has_old_last_move']:
            self.last_move = rec['old_last_move']
        else:
            self.last_move = None

        # Revert streak endpoints
        self._update_streaks(coord, player_id, rec, is_undo=True)

    def _update_streaks(self, coord, player_id, rec=None, is_undo=False):
        """Unified streak updater that supports both do_move and undo_move.

        If `is_undo` is False, `rec` must be a dict to record previous neighbor
        lengths. When `is_undo` is True, `rec` provides the saved lengths to restore.
        Returns True if a win was detected when applying the move.
        """
        win_detected = False

        for axis in range(3):
            dir_vec = DIRECTIONS[axis]
            opp_vec = DIRECTIONS[axis + 3]

            n1 = (coord[0] + dir_vec[0], coord[1] + dir_vec[1], coord[2] + dir_vec[2])
            n2 = (coord[0] + opp_vec[0], coord[1] + opp_vec[1], coord[2] + opp_vec[2])

            if not is_undo:
                l1 = self._streaks[player_id][axis].get(n1, 0)
                l2 = self._streaks[player_id][axis].get(n2, 0)
                # record for potential undo
                if rec is not None:
                    rec['len1'][axis] = l1
                    rec['len2'][axis] = l2

                new_len = 1 + l1 + l2
                if new_len >= 6:
                    win_detected = True

                if l1 > 0 and n1 in self._streaks[player_id][axis]:
                    del self._streaks[player_id][axis][n1]
                if l2 > 0 and n2 in self._streaks[player_id][axis]:
                    del self._streaks[player_id][axis][n2]

                end1 = (coord[0] + dir_vec[0] * l1, coord[1] + dir_vec[1] * l1, coord[2] + dir_vec[2] * l1)
                end2 = (coord[0] + opp_vec[0] * l2, coord[1] + opp_vec[1] * l2, coord[2] + opp_vec[2] * l2)

                self._streaks[player_id][axis][end1] = new_len
                self._streaks[player_id][axis][end2] = new_len
            else:
                # undo: remove endpoints created by the move and restore previous endpoints
                l1 = rec['len1'][axis]
                l2 = rec['len2'][axis]

                end1 = (coord[0] + dir_vec[0] * l1, coord[1] + dir_vec[1] * l1, coord[2] + dir_vec[2] * l1)
                end2 = (coord[0] + opp_vec[0] * l2, coord[1] + opp_vec[1] * l2, coord[2] + opp_vec[2] * l2)

                # Remove the endpoints that the move added (if present)
                if end1 in self._streaks[player_id][axis]:
                    del self._streaks[player_id][axis][end1]
                if end2 in self._streaks[player_id][axis]:
                    del self._streaks[player_id][axis][end2]

                # Restore previous endpoints and neighbor entries
                if l1 > 0:
                    self._streaks[player_id][axis][end1] = l1
                    self._streaks[player_id][axis][n1] = l1
                if l2 > 0:
                    self._streaks[player_id][axis][end2] = l2
                    self._streaks[player_id][axis][n2] = l2

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
        
        nb = HexBoard(
            pieces=self.pieces.copy(),
            turn=self.turn,
            last_move=self.last_move,
            streaks=new_streaks
        )
        # copy history for functional semantics
        nb.history = list(self.history)
        return nb

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
        self.history.clear()