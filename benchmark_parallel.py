# Claude AI assisted with: Parallel scaling benchmark design
import time
import multiprocessing
from src.hex_engine.board.board_cython import HexBoard
from src.hex_engine.evaluation.evaluator import Evaluator
from src.hex_engine.search.mcts import MCTS
from src.hex_engine.analysis.analysis import Analyzer

def run_parallel_benchmark():
    TOTAL_ITERATIONS = 100000
    thread_counts = [1, 2, 4, 8, 16]
    
    print("==================================================")
    print("   MCTS PARALLELIZATION BENCHMARK")
    print("==================================================")
    print(f"Target Iterations: {TOTAL_ITERATIONS:,}")
    print("--------------------------------------------------")

    # Setup a standard mid-game skirmish to evaluate
    board = HexBoard()
    board.do_move((0, 0, 0))   
    board.do_move((1, -1, 0))  
    board.do_move((2, -2, 0))  
    board.do_move((-1, 1, 0))

    # ---------------------------------------------------------
    # 1. Single-Threaded Baseline
    # ---------------------------------------------------------
    print("\n[1/2] Running Single-Threaded Baseline...")
    print("      (Warming up Numba, please wait...)")
    
    # Warmup Numba BEFORE starting the timer!
    single_evaluator = Evaluator(search_radius=15)
    single_evaluator.warmup()
    mcts_single = MCTS(evaluator=single_evaluator, exploration_constant=1.414)
    
    print("      (Running search...)")
    start_single = time.perf_counter()
    best_single = mcts_single.search(board, iterations=TOTAL_ITERATIONS, profile=False)
    end_single = time.perf_counter()
    
    single_time = end_single - start_single
    single_nps = TOTAL_ITERATIONS / single_time
    print(f"  -> Time: {single_time:.3f} seconds")
    print(f"  -> Speed: {single_nps:.0f} nodes/sec")
    print(f"  -> Best Move: {best_single}")

    # ---------------------------------------------------------
    # 2. Multi-Threaded Benchmark (Scaling)
    # ---------------------------------------------------------
    print("\n[2/2] Running Parallel Scaling Benchmark...")
    print(f"{'Cores':<7} | {'Time (s)':<10} | {'Speed (n/s)':<13} | {'Speedup':<9} | {'Efficiency'}")
    print("-" * 60)
    
    for cores in thread_counts:
        # Spin up an analyzer restricted to the specific core count
        analyzer = Analyzer(num_workers=cores, search_radius=15)
        
        # EXPLICIT WARMUP: Forces all workers to compile Numba functions
        # BEFORE the timer starts, completely removing the compilation penalty!
        analyzer.warmup(verbose=False)
        
        start_parallel = time.perf_counter()
        # Suppress verbose output to keep our loop summary table clean
        best_parallel = analyzer.analyze_move(board, total_iterations=TOTAL_ITERATIONS, verbose=False)
        end_parallel = time.perf_counter()
        
        parallel_time = end_parallel - start_parallel
        parallel_nps = TOTAL_ITERATIONS / parallel_time
        
        speedup = single_time / parallel_time
        efficiency = (speedup / cores) * 100
        
        print(f"{cores:<7d} | {parallel_time:<10.3f} | {parallel_nps:<13.0f} | {speedup:<8.2f}x | {efficiency:.1f}%")
        
        # Clean up the background processes for the next loop iteration
        analyzer.shutdown()
        
    print("==================================================\n")

if __name__ == "__main__":
    run_parallel_benchmark()