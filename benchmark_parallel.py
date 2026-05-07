# Claude AI assisted with: Parallel scaling benchmark design
import time
import multiprocessing
import matplotlib.pyplot as plt
from src.hex_engine.board.board_cython import HexBoard
from src.hex_engine.evaluation.evaluator import Evaluator
from src.hex_engine.search.mcts import MCTS
from src.hex_engine.analysis.analysis import Analyzer

def run_parallel_benchmark():
    TOTAL_ITERATIONS = 1000000
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
    
    # Collect results for plotting
    cores_list = []
    times_list = []
    speedups_list = []
    
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
        
        # Store for plotting
        cores_list.append(cores)
        times_list.append(parallel_time)
        speedups_list.append(speedup)
        
        print(f"{cores:<7d} | {parallel_time:<10.3f} | {parallel_nps:<13.0f} | {speedup:<8.2f}x | {efficiency:.1f}%")
        
        # Clean up the background processes for the next loop iteration
        analyzer.shutdown()
        
    print("==================================================\n")
    
    # ---------------------------------------------------------
    # 3. Plot Results
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cores vs Time
    ax1.plot(cores_list, times_list, 'b-o', linewidth=2, markersize=8, label='Actual Time')
    ax1.axhline(y=single_time, color='r', linestyle='--', label=f'Single-threaded: {single_time:.3f}s')
    ax1.set_xlabel('Number of Cores', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('Parallel MCTS: Cores vs Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(cores_list)
    
    # Plot 2: Cores vs Speedup (with linear scaling line for reference)
    linear_speedup = cores_list
    ax2.plot(cores_list, speedups_list, 'g-o', linewidth=2, markersize=8, label='Actual Speedup')
    ax2.plot(cores_list, linear_speedup, 'k--', linewidth=2, label='Perfect Linear Speedup')
    ax2.set_xlabel('Number of Cores', fontsize=12)
    ax2.set_ylabel('Speedup Factor', fontsize=12)
    ax2.set_title('Parallel MCTS: Speedup Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(cores_list)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_parallel_benchmark()