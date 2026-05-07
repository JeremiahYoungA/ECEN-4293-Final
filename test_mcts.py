# Claude AI assisted with: Test design for tree building, UCB1 selection, simulation, backpropagation
import time
from src.hex_engine.board.board_cython import HexBoard
from src.hex_engine.evaluation.evaluator import Evaluator
from src.hex_engine.search.mcts import MCTS

def run_mcts_tests():
    print("=== Testing MCTS Engine ===")
    
    # 1. Initialize dependencies
    print("1. Initializing Board and Evaluator...")
    board = HexBoard()
    evaluator = Evaluator(search_radius=15)
    print("   Warming up Numba JIT (this takes a moment)...")
    evaluator.warmup() 
    print("   [OK] Evaluator ready.")
    
    # 2. Initialize MCTS
    print("\n2. Initializing MCTS...")
    mcts = MCTS(evaluator=evaluator, exploration_constant=1.414)
    print("   [OK] MCTS ready.")
    
    # 3. Run Search on an empty board
    print("\n3. Running MCTS Search (1000 iterations) on an Empty Board...")
    start_time = time.perf_counter()
    best_move = mcts.search(board, iterations=1000)
    end_time = time.perf_counter()
    
    print(f"   -> Search completed in {end_time - start_time:.4f} seconds.")
    print(f"   -> Best Move Suggested: {best_move}")
    
    assert best_move is not None, "MCTS failed to return a move!"
    assert len(best_move) == 3, "Move is not a 3D coordinate!"
    assert sum(best_move) == 0, "Move does not satisfy cube coordinate constraint (a+b+c=0)!"
    print("   [OK] Empty board search successful.")
    
    # 4. Run Search on a board with an active skirmish
    print("\n4. Placing pieces and running MCTS in a skirmish...")
    board.do_move((0, 0, 0))   # P1 plays center
    board.do_move((1, -1, 0))  # P2 plays adjacent
    board.do_move((2, -2, 0))  # P2 extends line
    
    start_time = time.perf_counter()
    best_move_2 = mcts.search(board, iterations=1000)
    end_time = time.perf_counter()
    
    print(f"   -> Search completed in {end_time - start_time:.4f} seconds.")
    print(f"   -> Best Move Suggested: {best_move_2}")
    
    assert best_move_2 is not None, "MCTS failed to return a move!"
    assert best_move_2 not in [(0, 0, 0), (1, -1, 0), (2, -2, 0)], "MCTS suggested an already occupied space!"
    assert sum(best_move_2) == 0, "Move does not satisfy cube coordinate constraint!"
    print("   [OK] Skirmish board search successful.")

    print("\n========================================")
    print(" ✅ ALL MCTS TESTS PASSED!")
    print("========================================")

if __name__ == "__main__":
    run_mcts_tests()