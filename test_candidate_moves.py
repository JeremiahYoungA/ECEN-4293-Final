# Claude AI assisted with: Test case design
"""
Test suite for Evaluator.get_candidate_moves() to verify it returns at most top_n moves.
"""
import numpy as np
from src.hex_engine.board.board_cython import HexBoard
from src.hex_engine.evaluation.evaluator import Evaluator


def test_candidate_moves_respects_top_n():
    """Verify that get_candidate_moves returns at most top_n moves."""
    print("\n=== Testing get_candidate_moves returns top_n moves ===")
    
    board = HexBoard()
    evaluator = Evaluator(search_radius=15)
    evaluator.warmup()
    
    test_cases = [1, 5, 10, 20, 50]
    
    for top_n in test_cases:
        candidates = evaluator.get_candidate_moves(board, top_n=top_n)
        num_moves = candidates.shape[0]
        
        # The number of returned moves should not exceed top_n
        assert num_moves <= top_n, \
            f"Expected at most {top_n} moves, but got {num_moves}"
        
        print(f"   [OK] top_n={top_n}: returned {num_moves} moves (≤ {top_n})")


def test_candidate_moves_with_populated_board():
    """Verify top_n constraint when board has multiple pieces."""
    print("\n=== Testing get_candidate_moves with populated board ===")
    
    board = HexBoard()
    evaluator = Evaluator(search_radius=15)
    evaluator.warmup()
    
    # Place some pieces on the board to create non-uniform influence
    moves_to_play = [
        (0, 0, 0),
        (1, -1, 0),
        (2, -2, 0),
        (-1, 1, 0),
        (-2, 2, 0),
    ]
    
    for move in moves_to_play:
        board.do_move(move)
    
    test_cases = [3, 7, 15, 30]
    
    for top_n in test_cases:
        candidates = evaluator.get_candidate_moves(board, top_n=top_n)
        num_moves = candidates.shape[0]
        
        assert num_moves <= top_n, \
            f"Expected at most {top_n} moves, but got {num_moves}"
        
        print(f"   [OK] top_n={top_n}: returned {num_moves} moves (≤ {top_n})")


def test_candidate_moves_are_sorted_by_score():
    """Verify that returned moves are sorted by score (descending)."""
    print("\n=== Testing get_candidate_moves scores are descending ===")
    
    board = HexBoard()
    evaluator = Evaluator(search_radius=15)
    evaluator.warmup()
    
    # Place pieces to create influence
    board.do_move((0, 0, 0))
    board.do_move((1, -1, 0))
    board.do_move((2, -2, 0))
    
    top_n = 10
    candidates = evaluator.get_candidate_moves(board, top_n=top_n)
    
    # Get influence to verify ordering
    w_inf, b_inf, chunk = evaluator.get_influence(board)
    scores = np.abs(w_inf) + np.abs(b_inf)
    
    # Map returned candidates back to their scores
    returned_scores = []
    for move in candidates:
        for idx, coord in enumerate(chunk):
            if np.array_equal(coord, move):
                returned_scores.append(scores[idx])
                break
    
    # Verify descending order
    for i in range(len(returned_scores) - 1):
        assert returned_scores[i] >= returned_scores[i+1], \
            f"Scores not in descending order: {returned_scores[i]} < {returned_scores[i+1]}"
    
    print(f"   [OK] Returned {len(candidates)} moves in descending score order")


def test_candidate_moves_all_have_positive_score():
    """Verify that all returned moves have positive scores."""
    print("\n=== Testing all candidate moves have positive scores ===")
    
    board = HexBoard()
    evaluator = Evaluator(search_radius=15)
    evaluator.warmup()
    
    # Place pieces to create influence
    board.do_move((0, 0, 0))
    board.do_move((1, -1, 0))
    board.do_move((-1, 1, 0))
    
    candidates = evaluator.get_candidate_moves(board, top_n=10)
    
    # Get influence to verify scores are positive
    w_inf, b_inf, chunk = evaluator.get_influence(board)
    scores = np.abs(w_inf) + np.abs(b_inf)
    
    # Check all returned candidates have positive scores
    for move in candidates:
        for idx, coord in enumerate(chunk):
            if np.array_equal(coord, move):
                assert scores[idx] > 0, \
                    f"Move {move} has non-positive score: {scores[idx]}"
                break
    
    print(f"   [OK] All {len(candidates)} returned moves have positive scores")


if __name__ == "__main__":
    test_candidate_moves_respects_top_n()
    test_candidate_moves_with_populated_board()
    test_candidate_moves_are_sorted_by_score()
    test_candidate_moves_all_have_positive_score()
    print("\n=== All tests passed! ===\n")
