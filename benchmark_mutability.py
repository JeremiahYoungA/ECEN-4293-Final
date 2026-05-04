import time
import faulthandler
faulthandler.enable()
from src.hex_engine.board.board_cython import HexBoard

def run_mutation_benchmark():
    # Setup a game with 100 random moves to simulate a deep MCTS rollout
    moves = [(i, -i, 0) for i in range(100)] 

    ITERATIONS = 30000
    print(f"--- MCTS Rollout Benchmark ({ITERATIONS} Rollouts of 100 moves) ---")
    

    # 1. The Copy Strategy (Functional)
    start_copy = time.perf_counter()
    for _ in range(ITERATIONS):
        board = HexBoard()
        for move in moves:
            # Uses place_piece which internally calls copy() and do_move()
            board, _ = board.place_piece(move)
            
    copy_time = time.perf_counter() - start_copy
    print(f"Copy Strategy Time:    {copy_time:.4f} seconds")

    # 2. The Do/Undo Strategy (In-Place Mutation)
    start_mutate = time.perf_counter()
    for _ in range(ITERATIONS):
        board = HexBoard()
        for move in moves:
            # Go down the tree
            board.do_move(move)
            
            # (In a real MCTS, evaluate children here)
            
            # Go back up the tree
            board.undo_move()
            
    mutate_time = time.perf_counter() - start_mutate
    print(f"Do/Undo Strategy Time: {mutate_time:.4f} seconds")

    print(f"\nResult: Do/Undo is {copy_time / mutate_time:.2f}x faster for deep rollouts.")

if __name__ == "__main__":
    run_mutation_benchmark()