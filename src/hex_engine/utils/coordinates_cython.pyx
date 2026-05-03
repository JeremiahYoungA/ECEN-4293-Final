# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport abs as cabs

# C-level directions for internal Cython use (stack allocated)
# These are used for the fast O(1) neighbor/win logic
cdef int[6][3] C_DIRECTIONS = [
    [1, -1, 0], [1, 0, -1], [0, 1, -1],
    [-1, 1, 0], [-1, 0, 1], [0, -1, 1]
]

# Standard Python directions for UI/External use
# This is our public-facing "constant"
DIRECTIONS = np.array([
    (1, -1, 0), (1, 0, -1), (0, 1, -1),
    (-1, 1, 0), (-1, 0, 1), (0, -1, 1)
], dtype=np.int32)

# --- 1. INTERNAL C-LEVEL API (Zero Overhead) ---

cdef inline void _validate_c(int a, int b, int c) except *:
    if (a + b + c) != 0:
        raise ValueError(f"Invalid cube coordinates: ({a}, {b}, {c}) sum to {a+b+c}, not 0.")

cdef inline int _get_distance_c(int a1, int b1, int c1, int a2, int b2, int c2) noexcept:
    """Raw C distance calculation used by internal engine."""
    return (cabs(a1 - a2) + cabs(b1 - b2) + cabs(c1 - c2)) // 2

# --- 2. PUBLIC PYTHON/NUMBA INTERFACE (Bridge) ---

cpdef void validate(int a, int b, int c):
    _validate_c(a, b, c)

cpdef tuple two_to_three(int a, int b):
    cdef int c = -a - b
    _validate_c(a, b, c)
    return (a, b, c)

cpdef tuple three_to_two(int a, int b, int c):
    _validate_c(a, b, c)
    return (a, b)

def get_all_neighbors(int a, int b, int c):
    """
    Returns a dense numpy array of all 6 neighbors. 
    Ideal for feeding move generation to the Evaluation layer.
    """
    cdef cnp.ndarray[int, ndim=2] neighbors = np.empty((6, 3), dtype=np.int32)
    cdef int i
    for i in range(6):
        neighbors[i, 0] = a + C_DIRECTIONS[i][0]
        neighbors[i, 1] = b + C_DIRECTIONS[i][1]
        neighbors[i, 2] = c + C_DIRECTIONS[i][2]
    return neighbors

cpdef tuple get_neighbor(int a, int b, int c, int direction_index):
    """Returns a specific neighbor as a tuple for board keying."""
    if direction_index < 0 or direction_index >= 6:
        raise ValueError("Direction index must be between 0 and 5.")
    return (a + C_DIRECTIONS[direction_index][0], 
            b + C_DIRECTIONS[direction_index][1], 
            c + C_DIRECTIONS[direction_index][2])

cpdef int get_distance(int a1, int b1, int c1, int a2, int b2, int c2) noexcept:
    return _get_distance_c(a1, b1, c1, a2, b2, c2)

def get_batch_distance(int ta, int tb, int tc, int[:, :] sources):
    """
    Bridge function for the Numba evaluation layer.
    Calculates distances from one target to an array of sources using memoryviews.
    """
    cdef int n = sources.shape[0]
    cdef cnp.ndarray[int, ndim=1] distances = np.empty(n, dtype=np.int32)
    cdef int i
    for i in range(n):
        distances[i] = (cabs(ta - sources[i, 0]) + 
                        cabs(tb - sources[i, 1]) + 
                        cabs(tc - sources[i, 2])) // 2
    return distances