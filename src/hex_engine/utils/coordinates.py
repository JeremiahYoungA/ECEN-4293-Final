# Claude AI assisted with: Code auto complete based on /docs
import numpy as np

DIRECTIONS = np.array([
    (1, -1, 0),   # right
    (1, 0, -1),   # down-right
    (0, 1, -1),   # down-left
    (-1, 1, 0),   # left
    (-1, 0, 1),   # up-left
    (0, -1, 1)    # up-right
], dtype=int)

def validate(a, b, c):
    if (a + b + c) != 0:
        raise ValueError("Invalid coordinates: a + b + c must equal 0")

def two_to_three(a, b):
    c = -a - b
    validate(a, b, c)
    return a, b, c

def three_to_two(a, b, c):
    validate(a, b, c)
    return a, b

def get_all_neighbors(coord):
    return np.array(coord) + DIRECTIONS

def get_neighbor(coord, direction_index):
    if direction_index < 0 or direction_index >= len(DIRECTIONS):
        raise ValueError("Invalid direction index")
    d_arr = np.add(coord, DIRECTIONS[direction_index])
    return tuple(d_arr)

def get_distance(coord1, coord2):
    abs_diff = np.abs(np.array(coord1) - np.array(coord2))
    return np.sum(abs_diff) / 2

def get_batch_distance(target_coord, sources_array):
    abs_diff = np.abs(sources_array - np.array(target_coord))
    return np.sum(abs_diff, axis=1) / 2