# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libcpp.vector cimport vector
import numpy as np
cimport numpy as cnp

# History struct for perfect O(1) undo
cdef struct MoveRecord:
    long long coord_p
    int player_id
    bint has_old_last_move
    long long old_last_move_p
    int len1[3]
    int len2[3]

# Bit-packing logic
cdef inline long long pack_coord(int a, int b, int c) noexcept:
    return ((<long long>a + 1000000) << 42) | ((<long long>b + 1000000) << 21) | (<long long>c + 1000000)

cdef inline (int, int, int) unpack_coord(long long packed) noexcept:
    cdef int c = <int>(packed & 0x1FFFFF) - 1000000
    cdef int b = <int>((packed >> 21) & 0x1FFFFF) - 1000000
    cdef int a = <int>((packed >> 42) & 0x1FFFFF) - 1000000
    return (a, b, c)

cdef int[6][3] C_DIRECTIONS = [
    [1, -1, 0], [1, 0, -1], [0, 1, -1],
    [-1, 1, 0], [-1, 0, 1], [0, -1, 1]
]

cdef class HexBoard:
    def __init__(self, dict pieces_dict=None, int turn=0, object last_move=None):
        self.turn = turn
        self.last_move = last_move
        
        # Initialize the 6 streak maps (2 players * 3 axes)
        if self._streaks.empty():
            self._streaks.resize(6)
            
        if pieces_dict:
            for coord, pid in pieces_dict.items():
                self.pieces[pack_coord(coord[0], coord[1], coord[2])] = pid

    cpdef int get_current_player(self):
        """Calculates current player based on Connect6 rules."""
        return 1 if (self.turn % 4) in (0, 3) else 2

    cpdef cnp.ndarray get_pieces(self, int player_id):
        cdef int count = 0
        cdef pair[long long, int] item
        for item in self.pieces:
            if item.second == player_id:
                count += 1
                
        cdef cnp.ndarray[int, ndim=2] coords = np.empty((count, 3), dtype=np.int32)
        cdef int i = 0
        for item in self.pieces:
            if item.second == player_id:
                coords[i, 2] = <int>(item.first & 0x1FFFFF) - 1000000
                coords[i, 1] = <int>((item.first >> 21) & 0x1FFFFF) - 1000000
                coords[i, 0] = <int>((item.first >> 42) & 0x1FFFFF) - 1000000
                i += 1
        return coords

    cpdef tuple place_piece(self, tuple coord):
        """Standard functional move generation using the in-place logic."""
        cdef HexBoard new_board = self.copy()
        cdef bint is_win = new_board.do_move(coord)
        return new_board, is_win

    cpdef bint do_move(self, tuple coord):
        """Mutates the board in place and records destructive streak updates for undo."""
        cdef int a = coord[0], b = coord[1], c = coord[2]
        cdef long long p = pack_coord(a, b, c)
        
        if self.pieces.count(p):
            raise ValueError(f"Coordinate {coord} is already occupied.")

        cdef int player_id = self.get_current_player()
        
        # Setup the history record
        cdef MoveRecord rec
        rec.coord_p = p
        rec.player_id = player_id
        rec.has_old_last_move = self.last_move is not None
        if rec.has_old_last_move:
            rec.old_last_move_p = pack_coord(self.last_move[0], self.last_move[1], self.last_move[2])
        else:
            rec.old_last_move_p = 0
            
        self.pieces[p] = player_id
        self.last_move = coord
        self.turn += 1

        cdef bint win_detected = self._update_streaks(a, b, c, player_id, &rec, False)
        
        self.history.push_back(rec)
        return win_detected

    cpdef void undo_move(self):
        """Reverses the last do_move() perfectly in O(1) time using shared streak logic."""
        if self.history.empty():
            return
            
        cdef MoveRecord rec = self.history.back()
        self.history.pop_back()
        
        cdef int a, b, c
        a, b, c = unpack_coord(rec.coord_p)
        
        # 1. Restore piece map and turn
        self.pieces.erase(rec.coord_p)
        self.turn -= 1
        if rec.has_old_last_move:
            self.last_move = unpack_coord(rec.old_last_move_p)
        else:
            self.last_move = None
            
        # 2. Revert streaks using unified function
        self._update_streaks(a, b, c, rec.player_id, &rec, True)

    cdef bint _update_streaks(self, int a, int b, int c, int player_id, MoveRecord* rec, bint is_undo):
        cdef int p_idx = player_id - 1
        cdef bint win_detected = False
        cdef int axis, l1, l2, new_len, s_idx
        cdef long long n1_p, n2_p, end1_p, end2_p
        
        for axis in range(3):
            s_idx = p_idx * 3 + axis
            n1_p = pack_coord(a + C_DIRECTIONS[axis][0], b + C_DIRECTIONS[axis][1], c + C_DIRECTIONS[axis][2])
            n2_p = pack_coord(a + C_DIRECTIONS[axis+3][0], b + C_DIRECTIONS[axis+3][1], c + C_DIRECTIONS[axis+3][2])
            
            if not is_undo:
                l1 = self._streaks[s_idx][n1_p] if self._streaks[s_idx].count(n1_p) else 0
                l2 = self._streaks[s_idx][n2_p] if self._streaks[s_idx].count(n2_p) else 0
                rec.len1[axis] = l1
                rec.len2[axis] = l2
            else:
                l1 = rec.len1[axis]
                l2 = rec.len2[axis]
            
            end1_p = pack_coord(a + C_DIRECTIONS[axis][0] * l1, b + C_DIRECTIONS[axis][1] * l1, c + C_DIRECTIONS[axis][2] * l1)
            end2_p = pack_coord(a + C_DIRECTIONS[axis+3][0] * l2, b + C_DIRECTIONS[axis+3][1] * l2, c + C_DIRECTIONS[axis+3][2] * l2)
            
            if not is_undo:
                new_len = 1 + l1 + l2
                if new_len >= 6:
                    win_detected = True
                
                if l1 > 0: self._streaks[s_idx].erase(n1_p)
                if l2 > 0: self._streaks[s_idx].erase(n2_p)
                
                self._streaks[s_idx][end1_p] = new_len
                self._streaks[s_idx][end2_p] = new_len
            else:
                if self._streaks[s_idx].count(end1_p):
                    self._streaks[s_idx].erase(end1_p)
                if self._streaks[s_idx].count(end2_p):
                    self._streaks[s_idx].erase(end2_p)
                    
                if l1 > 0:
                    self._streaks[s_idx][end1_p] = l1
                    self._streaks[s_idx][n1_p] = l1
                if l2 > 0:
                    self._streaks[s_idx][end2_p] = l2
                    self._streaks[s_idx][n2_p] = l2
                    
        return win_detected

    cpdef HexBoard copy(self):
        cdef HexBoard nb = HexBoard(turn=self.turn, last_move=self.last_move)
        nb.pieces = self.pieces 
        
        # Vector assignment replaces the slow nested loops entirely!
        nb._streaks = self._streaks 
        nb.history = self.history
        return nb

    cpdef bint check_win(self):
        if self.last_move is None: 
            return False
        cdef long long p = pack_coord(self.last_move[0], self.last_move[1], self.last_move[2])
        cdef int player_id = self.pieces[p]
        cdef int p_idx = player_id - 1
        cdef int s_idx
        for axis in range(3):
            s_idx = p_idx * 3 + axis
            if self._streaks[s_idx].count(p) and self._streaks[s_idx][p] >= 6:
                return True
        return False