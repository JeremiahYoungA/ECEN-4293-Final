# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
import numpy as np
cimport numpy as cnp

# We pack the (a, b, c) coordinates into a single 64-bit integer 
# for maximum performance in C++ maps.
cdef inline long long pack_coord(int a, int b, int c) noexcept:
    return ((<long long>a + 1000000) << 42) | ((<long long>b + 1000000) << 21) | (<long long>c + 1000000)

cdef inline (int, int, int) unpack_coord(long long packed) noexcept:
    cdef int c = <int>(packed & 0x1FFFFF) - 1000000
    cdef int b = <int>((packed >> 21) & 0x1FFFFF) - 1000000
    cdef int a = <int>((packed >> 42) & 0x1FFFFF) - 1000000
    return (a, b, c)

# Directions for O(1) win detection math
cdef int[6][3] C_DIRECTIONS = [
    [1, -1, 0], [1, 0, -1], [0, 1, -1],
    [-1, 1, 0], [-1, 0, 1], [0, -1, 1]
]

cdef class HexBoard:
    # C++ data structures for speed
    cdef unordered_map[long long, int] pieces
    cdef unordered_map[long long, int] _streaks[2][3] # [player_index][axis_index]
    
    # Public attributes
    cdef public int turn
    cdef public object last_move 
    
    def __init__(self, dict pieces_dict=None, int turn=0, object last_move=None):
        self.turn = turn
        self.last_move = last_move
        
        # Populate pieces from Python dict if provided
        if pieces_dict:
            for coord, pid in pieces_dict.items():
                self.pieces[pack_coord(coord[0], coord[1], coord[2])] = pid

    cpdef int get_current_player(self):
        """Calculates current player: P1, P2, P2, P1, P1..."""
        return 1 if (self.turn % 4) in (0, 3) else 2

    cpdef bint is_occupied(self, tuple coord):
        return self.pieces.count(pack_coord(coord[0], coord[1], coord[2])) > 0

    cpdef object get_piece(self, tuple coord):
        cdef long long p = pack_coord(coord[0], coord[1], coord[2])
        if self.pieces.count(p):
            return self.pieces[p]
        return None

    cpdef cnp.ndarray get_pieces(self, int player_id):
        """
        CRITICAL FIX: Returns a dense (N, 3) numpy array directly from C++.
        Bypasses Python tuples entirely so Numba can consume it instantly.
        """
        cdef int count = 0
        cdef pair[long long, int] item
        
        # 1. Count matching pieces to avoid dynamic array resizing
        for item in self.pieces:
            if item.second == player_id:
                count += 1
                
        # 2. Allocate exact numpy memory
        cdef cnp.ndarray[int, ndim=2] coords = np.empty((count, 3), dtype=np.int32)
        
        # 3. Fill array directly using bitwise unpacking
        cdef int i = 0
        for item in self.pieces:
            if item.second == player_id:
                coords[i, 2] = <int>(item.first & 0x1FFFFF) - 1000000
                coords[i, 1] = <int>((item.first >> 21) & 0x1FFFFF) - 1000000
                coords[i, 0] = <int>((item.first >> 42) & 0x1FFFFF) - 1000000
                i += 1
                
        return coords

    cpdef tuple place_piece(self, tuple coord):
        """Returns (new_board, is_win)"""
        cdef int a = coord[0], b = coord[1], c = coord[2]
        cdef long long p = pack_coord(a, b, c)
        
        if self.pieces.count(p):
            raise ValueError(f"Coordinate {coord} is already occupied.")

        cdef int player_id = self.get_current_player()
        cdef HexBoard new_board = self.copy()
        
        new_board.pieces[p] = player_id
        new_board.last_move = coord
        new_board.turn += 1

        cdef bint is_win = new_board._update_streaks(a, b, c, player_id)
        return new_board, is_win

    cdef bint _update_streaks(self, int a, int b, int c, int player_id):
        cdef int p_idx = player_id - 1
        cdef bint win_detected = False
        cdef int axis, len1, len2, new_len
        cdef long long n1_p, n2_p, end1_p, end2_p
        
        for axis in range(3):
            n1_p = pack_coord(a + C_DIRECTIONS[axis][0], 
                              b + C_DIRECTIONS[axis][1], 
                              c + C_DIRECTIONS[axis][2])
            n2_p = pack_coord(a + C_DIRECTIONS[axis+3][0], 
                              b + C_DIRECTIONS[axis+3][1], 
                              c + C_DIRECTIONS[axis+3][2])
            
            len1 = self._streaks[p_idx][axis][n1_p] if self._streaks[p_idx][axis].count(n1_p) else 0
            len2 = self._streaks[p_idx][axis][n2_p] if self._streaks[p_idx][axis].count(n2_p) else 0
            
            new_len = 1 + len1 + len2
            if new_len >= 6:
                win_detected = True
            
            if len1 > 0: self._streaks[p_idx][axis].erase(n1_p)
            if len2 > 0: self._streaks[p_idx][axis].erase(n2_p)
            
            end1_p = pack_coord(a + C_DIRECTIONS[axis][0] * len1,
                                b + C_DIRECTIONS[axis][1] * len1,
                                c + C_DIRECTIONS[axis][2] * len1)
            end2_p = pack_coord(a + C_DIRECTIONS[axis+3][0] * len2,
                                b + C_DIRECTIONS[axis+3][1] * len2,
                                c + C_DIRECTIONS[axis+3][2] * len2)
            
            self._streaks[p_idx][axis][end1_p] = new_len
            self._streaks[p_idx][axis][end2_p] = new_len
            
        return win_detected

    cpdef HexBoard copy(self):
        cdef HexBoard nb = HexBoard.__new__(HexBoard)
        nb.turn = self.turn
        nb.last_move = self.last_move
        nb.pieces = self.pieces # C++ unordered_map assignment is a lightning-fast deep copy
        
        cdef int p, ax
        for p in range(2):
            for ax in range(3):
                nb._streaks[p][ax] = self._streaks[p][ax]
        return nb

    cpdef bint check_win(self):
        if self.last_move is None: 
            return False
        cdef long long p = pack_coord(self.last_move[0], self.last_move[1], self.last_move[2])
        cdef int player_id = self.pieces[p]
        cdef int p_idx = player_id - 1
        for axis in range(3):
            if self._streaks[p_idx][axis].count(p) and self._streaks[p_idx][axis][p] >= 6:
                return True
        return False

    def delete(self):
        """Explicitly clear C++ structures to prevent memory leaks during MCTS."""
        self.pieces.clear()
        cdef int p, ax
        for p in range(2):
            for ax in range(3):
                self._streaks[p][ax].clear()