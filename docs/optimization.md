# Optimization Plan

To achieve the targets in `goals.md`, different optimization methods were considered. The chosen optimization was the implementation of **Numba** and **Cython**.

## Evaluation

The sparse array will be converted to a dense numpy array for array-based mathematical computations utilizing NumPy. NumPy already utilizes a C library, and for this reason, Numba will be utilized, as it compliments the existing C code and compiles large loops of NumPy operations into a single block of machine code.

### Implementation of Numba
- **Decorator Utilization:** `@njit(fastmath=True, nogil=True)` will be applied to cause influence calculation loops to compile into native machine code and bypass the typical interpeter.
- **Objects:** Coordinates will be in raw numpy arrays to ensure Numba is compatible with the numerical data.
- **Zero-Allocation Math:** Profiling the time spent by MCTS revealed overhead caused by garbage-collection of intermediate NumPy arrays inside `evaluator.py`, such as `np.zeros`. Several steps (loops) were pushed into a single loop. Instead of storing individual influence fields, a running sum is kept, avoiding storage of the extra hex values. This eliminated the intermediate arrays.

## Search

Monte Carlo Tree Search (MCTS) requires simulations of many different boards, generating large trees. Numba cannot optimize the nested dictionaries used to describe these trees, so Cython will be used to compile them ahead of time into native C.

### Implementation of Cython

To implement Cython, board.py will be converted to board.pyx. This changes the following
- **cdef Classes:** `HexBoard` will be redefined as a `cdef class` to remove object overhead.
- **C++ Memory Structures:** Python dictionaries will be replaced with C++ `std::unordered_map`
- **Lightweight Nodes:** MCTS nodes will be C structs with direct C-pointers to child nodes
- **Explicit Memory Management:** Manual memory management utilizing `malloc()` and similar functions allow bypassing Python's garbage collector.

### State Mutation & History Stack

While originally planned to have immutability, restricted mutations were found to be ~20-25x faster. Restricted mutations were implemented as follows.

- C++ `std::vector` acts as a LIFO History Stack.
- `do_move` caches `MoveRecord` to the stack, storing the change to the win-detection streaks
- `undo_move` pops `MoveRecord` and reconstructs the past state

## Parallelization

To accelerate the simulation, parallelization across multiple CPU cores is used.

- **Root Node Division:** The root has some number of nodes chosen by the heuristic. These nodes are divided between the arrays to counter overlap of simulated trees.
- **Scaling Efficiency:** Effeciency can be seen with benchmark_parallel.py

# Build & Implementation

There are a couple required steps to run things smoothly
1. **Pre-compiling Cython:** A `setup.py` script will be needed that utilizes `setuptools` and `Cython.Build.cythonize`. This compiles the `.pyx` files into native C shared libraries before execution.
2. **Numba Startup:** Forcing Numba's Just-In-Time (JIT) compiler to translate loops into machine code before the first move will avoid an initial stutter.