# Optimization Plan

To achieve the targets in `goals.md`, different optimization methods were considered. The chosen optimization was the implementation of **Numba** and **Cython**.

## Evaluation

The sparse array will be converted to a dense numpy array for array-based mathematical computations utilizing NumPy. NumPy already utilizes a C library, and for this reason, Numba will be utilized, as it compliments the existing C code and compiles large loops of NumPy operations into a single block of machine code.

### Implementation of Numba
- **Decorator Utilization:** `@njit(fastmath=True, nogil=True)` will be applied to cause influence calculation loops to compile into native machine code and bypass the typical interpeter.
- **Objects:** Coordinates will be in raw numpy arrays to ensure Numba is compatible with the numerical data.

## Search

Monte Carlo Tree Search (MCTS) requires simulations of many different boards, generating large trees. Numba cannot optimize the nested dictionaries used to describe these trees, so Cython will be used to compile them ahead of time into native C.

### Implementation of Cython

To implement Cython, board.py will be converted to board.pyx. This changes the following
- **cdef Classes:** `HexBoard` will be redefined as a `cdef class` to remove object overhead.
- **C++ Memory Structures:** Python dictionaries will be replaced with C++ `std::unordered_map`
- **Lightweight Nodes:** MCTS nodes will be C structs with direct C-pointers to child nodes
- **Explicit Memory Management:** an explicit `delete()` method mapped to memory deallocation to replace garbage collection

# Build & Implementation

There are a couple required steps to run things smoothly
1. **Pre-compiling Cython:** A `setup.py` script will be needed that utilizes `setuptools` and `Cython.Build.cythonize`. This compiles the `.pyx` files into native C shared libraries before execution.
2. **Numba Startup:** Forcing Numba's Just-In-Time (JIT) compiler to translate loops into machine code before the first move will avoid an initial stutter.