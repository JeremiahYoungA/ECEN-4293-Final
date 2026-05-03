import numpy as np
import numba as nb

@nb.njit(fastmath=True, nogil=True)
def _compute_influence_field(pieces, chunk_coords, constant):
    """Accumulates influence (constant / d^2) for a set of pieces over a grid."""
    n, m = pieces.shape[0], chunk_coords.shape[0]
    influence = np.zeros(m, dtype=np.float64)
    
    for i in range(n):
        pa, pb, pc = pieces[i, 0], pieces[i, 1], pieces[i, 2]
        for j in range(m):
            ca, cb, cc = chunk_coords[j, 0], chunk_coords[j, 1], chunk_coords[j, 2]
            # Manhattan distance in cube coords
            dist = (abs(pa - ca) + abs(pb - cb) + abs(pc - cc)) // 2
            if dist > 0:
                influence[j] += constant / (dist * dist)
            else:
                influence[j] += constant * 1e4 # Piece location spike
    return influence

@nb.njit(fastmath=True, nogil=True)
def _compute_advantage(white_pieces, black_pieces, chunk_coords):
    """Returns difference between white and black influence totals."""
    w_inf = _compute_influence_field(white_pieces, chunk_coords, 1.0)
    b_inf = _compute_influence_field(black_pieces, chunk_coords, 1.0)
    return np.sum(w_inf) - np.sum(b_inf)

class Evaluator:
    def __init__(self, chunk_size=30):
        self.chunk_size = chunk_size

    def _generate_chunk(self, center, size):
        """Generates a local (size x size) grid of cube coordinates."""
        coords = np.empty((size * size, 3), dtype=np.int32)
        half = size // 2
        idx = 0
        for a in range(center[0] - half, center[0] + half):
            for b in range(center[1] - half, center[1] + half):
                coords[idx, 0] = a
                coords[idx, 1] = b
                coords[idx, 2] = -a - b
                idx += 1
        return coords

    def evaluate(self, board):
        """Scores position using influence fields over the active board area."""
        w_pts = board.get_pieces(player_id=1)
        b_pts = board.get_pieces(player_id=2)

        if w_pts.shape[0] == 0 and b_pts.shape[0] == 0:
            return 0.0

        # Center evaluation on the most recent move
        center = board.last_move if board.last_move else (0, 0, 0)
        chunk = self._generate_chunk(center, self.chunk_size)

        return _compute_advantage(w_pts, b_pts, chunk)

    def warmup(self):
        """Triggers JIT compilation with dummy data to prevent first-move lag."""
        dummy_p = np.array([[0,0,0]], dtype=np.int32)
        dummy_c = np.array([[1,1,-2]], dtype=np.int32)
        _compute_advantage(dummy_p, dummy_p, dummy_c)