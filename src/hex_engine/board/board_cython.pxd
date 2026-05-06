# cython: language_level=3

from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
cimport numpy as cnp

# We must expose the struct so MCTS knows how big the history vector is
cdef struct MoveRecord:
    long long coord_p
    int player_id
    bint has_old_last_move
    long long old_last_move_p
    int len1[3]
    int len2[3]

# Declare the C-level memory layout and public methods of HexBoard
cdef class HexBoard:
    cdef unordered_map[long long, int] pieces
    cdef vector[unordered_map[long long, int]] _streaks
    cdef vector[MoveRecord] history
    cdef public int turn
    cdef public object last_move
    
    cpdef int get_current_player(self)
    cpdef cnp.ndarray get_pieces(self, int player_id)
    cpdef tuple place_piece(self, tuple coord)
    cpdef bint do_move(self, tuple coord)
    cpdef void undo_move(self)
    cdef bint _update_streaks(self, int a, int b, int c, int player_id, MoveRecord* rec, bint is_undo)
    cpdef HexBoard copy(self)
    cpdef bint check_win(self)