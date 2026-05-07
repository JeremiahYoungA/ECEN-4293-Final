# Claude AI assisted with: Benchmark design, performance comparison infrastructure
import time
import numpy as np
from src.hex_engine.evaluation.evaluator import Evaluator, _calculate_influence_fields

def pure_python_baseline(white_pieces, black_pieces, base_chunk, center_a, center_b, center_c):
    """A standard Python for-loop implementation of your influence field."""
    w_inf = 0.0
    b_inf = 0.0
    
    # Standard Python iteration
    for chunk_idx in range(len(base_chunk)):
        ca = base_chunk[chunk_idx, 0] + center_a
        cb = base_chunk[chunk_idx, 1] + center_b
        cc = base_chunk[chunk_idx, 2] + center_c
        
        for p in white_pieces:
            dist = (abs(p[0] - ca) + abs(p[1] - cb) + abs(p[2] - cc)) // 2
            if dist > 0: w_inf += 1.0 / (dist * dist)
                
        for p in black_pieces:
            dist = (abs(p[0] - ca) + abs(p[1] - cb) + abs(p[2] - cc)) // 2
            if dist > 0: b_inf += 1.0 / (dist * dist)
            
    return w_inf - b_inf

def run_benchmark():
    # 1. Setup evaluator with search_radius=25
    evaluator = Evaluator(search_radius=25)
    base_chunk = evaluator.base_chunk
    
    # 2. Simulate 50 pieces on the board for each player
    # Using random coordinates within the chunk
    np.random.seed(42)
    indices_w = np.random.choice(len(base_chunk), 50, replace=False)
    indices_b = np.random.choice(len(base_chunk), 50, replace=False)
    white_pieces = base_chunk[indices_w]
    black_pieces = base_chunk[indices_b]
    
    # Use center offset
    center_a, center_b, center_c = 0, 0, 0

    # 3. Warmup Numba (compile the C code)
    print("Warming up Numba compiler...")
    _calculate_influence_fields(white_pieces, black_pieces, base_chunk, center_a, center_b, center_c, 1.0)
    
    ITERATIONS = 100

    # 4. Benchmark Pure Python
    print(f"\nRunning Pure Python Baseline ({ITERATIONS} iterations)...")
    start_py = time.perf_counter()
    for _ in range(ITERATIONS):
        pure_python_baseline(white_pieces, black_pieces, base_chunk, center_a, center_b, center_c)
    py_time = time.perf_counter() - start_py

    # 5. Benchmark Vectorized Numba
    print(f"Running Numba Vectorized Engine ({ITERATIONS} iterations)...")
    start_nb = time.perf_counter()
    for _ in range(ITERATIONS):
        _calculate_influence_fields(white_pieces, black_pieces, base_chunk, center_a, center_b, center_c, 1.0)
    nb_time = time.perf_counter() - start_nb

    # 6. Results
    print("\n--- RESULTS ---")
    print(f"Python Time: {py_time:.4f} seconds")
    print(f"Numba Time:  {nb_time:.4f} seconds")
    print(f"Speedup:     {py_time / nb_time:.2f}x faster")

if __name__ == "__main__":
    run_benchmark()