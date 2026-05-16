# Hex Game Engine

A high-performance Monte Carlo Tree Search (MCTS) engine for the hexagonal board game Hex, with Cython and Numba optimizations.

## ⚠️ Status

**Note:** The heuristics are not yet tuned, and the AI does not play well. This project is a performance-focused implementation demonstrating MCTS architecture and optimization techniques rather than strong gameplay.

## Quick Start

**Setup:**
```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

**Play against the AI:**
```bash
python visualize_board.py
```

## Documentation

- [Architecture](docs/architecture.md) — System design and module layout
- [Data Structures](docs/data_structures.md) — Cube coordinate system
- [Goals](docs/goals.md) — Performance targets
- [Optimization](docs/optimization.md) — Cython and Numba implementation
- [Tutorial](docs/tutorial.md) — Detailed setup

## Project Structure

```
src/hex_engine/
├── analysis/       # Analysis orchestration
├── board/          # Board state management
├── evaluation/     # Heuristic scoring
├── search/         # MCTS implementation
├── utils/          # Utilities and constants
└── ui/             # Visualization

tests/              # Unit tests
benchmarks/         # Performance benchmarks
```

## Development

- **Tests:** `pytest tests/`
- **Benchmarks:** `python benchmark_parallel.py`, `python profile_mcts.py`
