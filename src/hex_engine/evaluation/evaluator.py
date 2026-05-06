import numpy as np
import numba as nb

@nb.njit(fastmath=True, nogil=True)
def _calculate_influence_fields(white_pieces, black_pieces, chunk_coords, constant=1.0):
    """
    Calculates raw influence arrays for both players.
    Returns (white_influence, black_influence). 
    Occupied cells are zeroed out.
    """
    nw, nb_pcs, m = white_pieces.shape[0], black_pieces.shape[0], chunk_coords.shape[0]
    w_inf = np.zeros(m, dtype=np.float64)
    b_inf = np.zeros(m, dtype=np.float64)
    
    for j in range(m):
        ca, cb, cc = chunk_coords[j, 0], chunk_coords[j, 1], chunk_coords[j, 2]
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

class Evaluator:
    def __init__(self, search_radius=15, influence_constant=1.0):
        self.search_radius = search_radius
        self.influence_constant = influence_constant

    def _generate_chunk(self, center, radius):
        """Generates a perfect hexagon of cube coordinates centered at the target."""
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

    def get_influence(self, board):
        """Returns the raw white and black influence arrays for the current state."""
        w_pts = board.get_pieces(player_id=1)
        b_pts = board.get_pieces(player_id=2)
        center = board.last_move if board.last_move else (0, 0, 0)
        chunk = self._generate_chunk(center, self.search_radius)
        
        w_inf, b_inf = _calculate_influence_fields(w_pts, b_pts, chunk, self.influence_constant)
        return w_inf, b_inf, chunk

    def evaluate(self, board):
        """Calculates the scalar advantage score for the position."""
        w_inf, b_inf, _ = self.get_influence(board)
        return _calculate_advantage(w_inf, b_inf)

    def get_candidate_moves(self, board, top_n=10):
        """Returns top N moves prioritized by total influence density."""
        w_inf, b_inf, chunk = self.get_influence(board)
        scores = _calculate_candidate_scores(w_inf, b_inf)
        
        if top_n >= scores.size:
            top_indices = np.arange(scores.size)
        else:
            top_indices = np.argpartition(scores, -top_n)[-top_n:]
        
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        moves = chunk[top_indices[scores[top_indices] > 0]]
        
        # Fallback for the very first move of the game
        if moves.shape[0] == 0:
            return np.array([[0, 0, 0]], dtype=np.int32)
            
        return moves

    def warmup(self):
        """Triggers JIT compilation for all hotpath functions."""
        dummy_p = np.zeros((1, 3), dtype=np.int32)
        dummy_c = np.ones((1, 3), dtype=np.int32)
        w, b = _calculate_influence_fields(dummy_p, dummy_p, dummy_c, 1.0)
        _calculate_advantage(w, b)
        _calculate_candidate_scores(w, b)