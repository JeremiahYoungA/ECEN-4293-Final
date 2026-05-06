"""
Standalone profiling script for MCTS to analyze time breakdown across phases.
"""
from src.hex_engine.board.board_cython import HexBoard
from src.hex_engine.evaluation.evaluator import Evaluator
from src.hex_engine.search.mcts import MCTS

def profile_mcts_from_scratch():
    """Profile MCTS on an empty board (best case for fast iterations)."""
    print("\n=== PROFILING MCTS FROM SCRATCH ===")
    board = HexBoard()
    evaluator = Evaluator(search_radius=15)
    evaluator.warmup()
    mcts = MCTS(evaluator=evaluator, exploration_constant=1.414)
    
    # Run with profiling enabled
    best_move = mcts.search(board, iterations=500, profile=True)
    print(f"Best move found: {best_move}\n")


def profile_mcts_with_pieces():
    """Profile MCTS with some pieces already on the board."""
    print("\n=== PROFILING MCTS WITH POPULATED BOARD ===")
    board = HexBoard()
    evaluator = Evaluator(search_radius=15)
    evaluator.warmup()
    mcts = MCTS(evaluator=evaluator, exploration_constant=1.414)
    
    # Place some pieces
    board.do_move((0, 0, 0))
    board.do_move((1, -1, 0))
    board.do_move((2, -2, 0))
    board.do_move((-1, 1, 0))
    board.do_move((-2, 2, 0))
    
    best_move = mcts.search(board, iterations=500, profile=True)
    print(f"Best move found: {best_move}\n")


def profile_comparison():
    """Compare profiling across different iteration counts."""
    print("\n=== PROFILING WITH DIFFERENT ITERATION COUNTS ===")
    board = HexBoard()
    evaluator = Evaluator(search_radius=15)
    evaluator.warmup()
    
    for iterations in [100, 250, 500, 1000]:
        print(f"\n--- Running {iterations} iterations ---")
        mcts = MCTS(evaluator=evaluator, exploration_constant=1.414)
        mcts.search(board, iterations=iterations, profile=True)


if __name__ == "__main__":
    profile_mcts_from_scratch()
    profile_mcts_with_pieces()
    profile_comparison()
