import time
import numpy as np
from src.hex_engine.evaluation.evaluator import Evaluator, _calculate_influence_fields

def pure_python_baseline(white_pieces, black_pieces, chunk_coords):
    """A standard Python for-loop implementation of your influence field."""
    w_inf = 0.0
    b_inf = 0.0
    
    # Standard Python iteration
    for c in chunk_coords:
        ca, cb, cc = c[0], c[1], c[2]
        
        for p in white_pieces:
            dist = (abs(p[0] - ca) + abs(p[1] - cb) + abs(p[2] - cc)) // 2
            if dist > 0: w_inf += 1.0 / (dist * dist)
                
        for p in black_pieces:
            dist = (abs(p[0] - ca) + abs(p[1] - cb) + abs(p[2] - cc)) // 2
            if dist > 0: b_inf += 1.0 / (dist * dist)
            
    return w_inf - b_inf

def run_benchmark():
    # 1. Setup a "50x50" equivalent grid (radius 25)
    # This generates roughly 1,875 hex coordinates
    evaluator = Evaluator(search_radius=25)
    chunk = evaluator._generate_chunk((0,0,0), 25)
    
    # 2. Simulate 50 pieces on the board for each player
    # Using random coordinates within the chunk
    np.random.seed(42)
    indices_w = np.random.choice(len(chunk), 50, replace=False)
    indices_b = np.random.choice(len(chunk), 50, replace=False)
    white_pieces = chunk[indices_w]
    black_pieces = chunk[indices_b]

    # 3. Warmup Numba (compile the C code)
    print("Warming up Numba compiler...")
    _calculate_influence_fields(white_pieces, black_pieces, chunk, 1.0)
    
    ITERATIONS = 100

    # 4. Benchmark Pure Python
    print(f"\nRunning Pure Python Baseline ({ITERATIONS} iterations)...")
    start_py = time.perf_counter()
    for _ in range(ITERATIONS):
        pure_python_baseline(white_pieces, black_pieces, chunk)
    py_time = time.perf_counter() - start_py

    # 5. Benchmark Vectorized Numba
    print(f"Running Numba Vectorized Engine ({ITERATIONS} iterations)...")
    start_nb = time.perf_counter()
    for _ in range(ITERATIONS):
        _calculate_influence_fields(white_pieces, black_pieces, chunk, 1.0)
    nb_time = time.perf_counter() - start_nb

    # 6. Results
    print("\n--- RESULTS ---")
    print(f"Python Time: {py_time:.4f} seconds")
    print(f"Numba Time:  {nb_time:.4f} seconds")
    print(f"Speedup:     {py_time / nb_time:.2f}x faster")

if __name__ == "__main__":
    run_benchmark()