import numpy as np
import numba as nb

@nb.njit(fastmath=True, nogil=True)
def _calculate_influence_fields(white_pieces, black_pieces, base_chunk, center_a, center_b, center_c, constant=1.0):
    """
    Calculates raw influence arrays for both players.
    Applies the center offset dynamically to avoid array allocation overhead.
    """
    nw, nb_pcs, m = white_pieces.shape[0], black_pieces.shape[0], base_chunk.shape[0]
    w_inf = np.zeros(m, dtype=np.float64)
    b_inf = np.zeros(m, dtype=np.float64)
    
    for j in range(m):
        ca = base_chunk[j, 0] + center_a
        cb = base_chunk[j, 1] + center_b
        cc = base_chunk[j, 2] + center_c
        is_occupied = False
        
        # White influence
        for i in range(nw):
            dist = (abs(white_pieces[i, 0] - ca) + abs(white_pieces[i, 1] - cb) + abs(white_pieces[i, 2] - cc)) // 2
            if dist == 0:
                is_occupied = True
                break
            w_inf[j] += constant / (dist * dist)
            
        if is_occupied:
            w_inf[j] = 0.0
            continue

        # Black influence
        for i in range(nb_pcs):
            dist = (abs(black_pieces[i, 0] - ca) + abs(black_pieces[i, 1] - cb) + abs(black_pieces[i, 2] - cc)) // 2
            if dist == 0:
                is_occupied = True
                break
            b_inf[j] += constant / (dist * dist)
            
        if is_occupied:
            w_inf[j] = 0.0
            b_inf[j] = 0.0

    return w_inf, b_inf

@nb.njit(fastmath=True, nogil=True)
def _calculate_advantage(w_inf, b_inf):
    """Computes the scalar advantage score from influence fields."""
    return np.sum(w_inf) - np.sum(b_inf)

@nb.njit(fastmath=True, nogil=True)
def _calculate_candidate_scores(w_inf, b_inf):
    """Combines fields to find areas of high contention/interest."""
    return np.abs(w_inf) + np.abs(b_inf)

@nb.njit(fastmath=True, nogil=True)
def _get_top_candidates(w_inf, b_inf, base_chunk, center_a, center_b, center_c, top_n):
    """
    Extracts the top N moves using a lightning-fast O(N*K) inline insertion sort.
    This entirely skips the O(N log N) overhead and memory allocation of np.argsort.
    """
    scores = np.abs(w_inf) + np.abs(b_inf)
    
    out_moves = np.zeros((top_n, 3), dtype=np.int32)
    out_scores = np.zeros(top_n, dtype=np.float64)
    count = 0
    
    for i in range(scores.size):
        s = scores[i]
        if s > 0:
            # If the candidate qualifies for the top_n list
            if count < top_n or s > out_scores[top_n - 1]:
                # Find the insertion index
                insert_idx = count if count < top_n else top_n - 1
                while insert_idx > 0 and s > out_scores[insert_idx - 1]:
                    insert_idx -= 1
                
                # Shift elements down to make room
                shift_end = count if count < top_n else top_n - 1
                for k in range(shift_end, insert_idx, -1):
                    out_scores[k] = out_scores[k - 1]
                    out_moves[k, 0] = out_moves[k - 1, 0]
                    out_moves[k, 1] = out_moves[k - 1, 1]
                    out_moves[k, 2] = out_moves[k - 1, 2]
                    
                # Insert the new top candidate and apply the coordinate offset
                out_scores[insert_idx] = s
                out_moves[insert_idx, 0] = base_chunk[i, 0] + center_a
                out_moves[insert_idx, 1] = base_chunk[i, 1] + center_b
                out_moves[insert_idx, 2] = base_chunk[i, 2] + center_c
                
                if count < top_n:
                    count += 1
                    
    # Fallback for the very first move of the game
    if count == 0:
        return np.zeros((1, 3), dtype=np.int32)
        
    return out_moves[:count]

class Evaluator:
    def __init__(self, search_radius=15, influence_constant=1.0):
        self.search_radius = search_radius
        self.influence_constant = influence_constant
        self.base_chunk = self._generate_chunk_base((0, 0, 0), search_radius)

    def _generate_chunk_base(self, center, radius):
        """Generates a perfect hexagon of cube coordinates. (Now only runs ONCE)"""
        n_elements = 3 * radius * (radius + 1) + 1
        coords = np.empty((n_elements, 3), dtype=np.int32)
        idx = 0
        
        for q in range(-radius, radius + 1):
            r1 = max(-radius, -q - radius)
            r2 = min(radius, -q + radius)
            for r in range(r1, r2 + 1):
                coords[idx, 0] = center[0] + q
                coords[idx, 1] = center[1] + r
                coords[idx, 2] = center[2] - q - r
                idx += 1
                
        return coords

    def _get_influence_fast(self, board):
        """Internal ultra-fast data fetch for MCTS (No array allocations)."""
        w_pts = board.get_pieces(player_id=1)
        b_pts = board.get_pieces(player_id=2)
        if board.last_move:
            return w_pts, b_pts, board.last_move[0], board.last_move[1], board.last_move[2]
        return w_pts, b_pts, 0, 0, 0

    def get_influence(self, board):
        """
        Public API: Returns w_inf, b_inf, and the fully resolved chunk.
        Kept intact for backward compatibility with visualize_board.py and tests.
        """
        w_pts, b_pts, ca, cb, cc = self._get_influence_fast(board)
        w_inf, b_inf = _calculate_influence_fields(w_pts, b_pts, self.base_chunk, ca, cb, cc, self.influence_constant)
        # Allocate the chunk dynamically for the visualizer
        chunk = self.base_chunk + np.array([ca, cb, cc], dtype=np.int32)
        return w_inf, b_inf, chunk

    def evaluate(self, board):
        """Calculates the scalar advantage score for the position."""
        w_pts, b_pts, ca, cb, cc = self._get_influence_fast(board)
        w_inf, b_inf = _calculate_influence_fields(w_pts, b_pts, self.base_chunk, ca, cb, cc, self.influence_constant)
        return _calculate_advantage(w_inf, b_inf)

    def get_candidate_moves(self, board, top_n=10):
        """Returns top N moves prioritized by total influence density."""
        w_pts, b_pts, ca, cb, cc = self._get_influence_fast(board)
        w_inf, b_inf = _calculate_influence_fields(w_pts, b_pts, self.base_chunk, ca, cb, cc, self.influence_constant)
        return _get_top_candidates(w_inf, b_inf, self.base_chunk, ca, cb, cc, top_n)

    def warmup(self):
        """Triggers JIT compilation for all hotpath functions."""
        dummy_p = np.zeros((1, 3), dtype=np.int32)
        dummy_c = np.ones((1, 3), dtype=np.int32)
        w, b = _calculate_influence_fields(dummy_p, dummy_p, dummy_c, 0, 0, 0, 1.0)
        _calculate_advantage(w, b)
        _calculate_candidate_scores(w, b)
        _get_top_candidates(w, b, dummy_c, 0, 0, 0, 10)