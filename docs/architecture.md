# Architecture
## Inspiration from matklad's post

This document describes the structure of the hex engine project. 
Use this to:
- Understand how the codebase is organized
- Find where to make changes
- Learn design decisions behind the code

## Table of Contents
1. [High Level Design](#high-level-design)
2. [Code Map](#code-map)
3. [Data Structures](#data-structures)
4. [Cross-cutting Concerns](#cross-cutting-concerns)

See [goals.md](goals.md) for performance targets and design constraints.

---

## High Level Design

At the highest level, this system will take the input of a gameboard and player turn, and
provide an output of a datapacket containing the following:
- An *optimal* next move for the corresponding player
- A numerical advantage score, indicating who has advantage
- A 'Mate in N' value, indicating number of moves before the closest forced win.

The system is responsible **understanding** the board, **evaluating** advantage, **suggesting** *optimal* moves, and **detecting** forced wins.

## Code Map

### src/hex_engine/
Main package that contains the core functionality

#### analysis/

Orchestrates search, evaluation, and win detection for game analysis
**Inputs**
- current board
- current player

**Outputs**
- Move suggestion
- Numerical advantage score
- Forced win (Mate in N)

**Note:** This is the simplest version due to deadline, inputs may expand to include future board position of specific branches.

#### board/

Manages board state
**Responsibilities:**
- Store piece positions using sparse representation
- Support O(1) piece ownership queries
- Check neighbor ownership
- Support board copying

#### evaluation/

Calculates numerical advantage using influence field heuristic.
**Responsibilities:**
- Compute influence field across board using piecewise evaluation strategy
- Chunk infinite board into local coordinate meshes (dynamically calculated)
- Evaluate positions without traversing entire unbounded space

**Inputs**
- Current board

**Outputs**
- Numerical score

#### search/

Implements Monte Carlo Tree Search (MCTS) algorithm for optimal move discovery and forced-win detection.

**Responsibilities:**
- Execute MCTS simulations on game tree
- Maintain tree statistics (visit counts, win rates)
- Select most promising moves via UCB (Upper Confidence Bound)
- Utilize pruning guidance from evaluation module to cut low-value branches
- Detect forced wins (Mate in N) with O(1) complexity
- Support parallel simulation with shared tree state

**Inputs**
- Current board state
- Current player
- Search depth/iteration limits

**Outputs**
- Move suggestion
- Forced win (Mate in N)

**Algorithm Overview:**
Uses 4-phase MCTS: Selection (UCB), Expansion, Simulation (random playouts), and Backpropagation. Periodically queries evaluation module for heuristic scores to prune low-value branches and accelerate convergence.

#### ui/

#### utils/

Provides core utility functions and constants for the hex engine.
**Responsibilities:**
- Cube coordinate operations (neighbor finding, distance calculations)
- Win detection along directional sequences
- Game constants (direction vectors, piece types, win threshold)
- Coordinate validation and move legality checking

## Architectural Invariants

- **Sparse Board Representation**: Only occupied positions stored; no finite/bounded grid assumption. Enables infinite hexagonal boards.
- **Functional Search**: Search does not mutate board state during simulation. Board remains immutable across all operations.
- **Bounded Evaluation**: Piecewise evaluation strategy prevents unbounded computation on infinite board.

## Architectural Absences

The following design choices are **NOT** implemented:

- **No Global Optimality** - MCTS produces statistically favorable moves, not provably optimal decisions
- **No Standalone Win Detection** - Forced-win detection integrates with search tree (not static analysis)

## Layers and Boundaries

### Layer Structure

**Application Layer** (`analysis/`)
- Sole entry point for all analysis operations
- Orchestrates board, search, and evaluation modules
- Coordinates results into output datapacket

**Core Logic Layer** (`board/`, `search/`, `evaluation/`)
- `board/`: Manages immutable game state
- `search/`: Performs MCTS simulations with periodic pruning guidance from evaluation
- `evaluation/`: Computes heuristic scores; used by search to prune low-value branches
- `search/` and `evaluation/` are tightly coupled for heuristic pruning efficiency

**Primitives Layer** (`utils/`)
- Provides utility functions to all layers
- Zero external dependencies
- Coordinate operations, win detection, game constants, validation

### Data Flow

- **Input**: Application receives (board state, current player)
- **Within Core Logic**: Search queries board (read-only), periodically calls evaluation for pruning guidance
- **Output**: Search and evaluation results flow back to application as final answers
- **Dependencies**: All layers depend on utils/ for shared operations

### Immutability Guarantee

Both search and evaluation operate functionally: neither modifies board state or persisted game data. All operations are deterministic transformations of immutable inputs.

## Data Structures

### Cube Coordinates
Three-axis hexagonal coordinate system (a, b, c) where a + b + c = 0. Enables efficient hex positioning, distance calculations, and neighbor queries. See [data_structures.md](data_structures.md) for detailed specification.

## Cross Cutting Concerns

### Parallel Simulation Coordination

The search module runs multiple MCTS simulations in parallel to improve convergence speed and statistical accuracy. Parallelization coordinates:
- Multiple threads/processes sharing a common tree state
- Carefully chosen random seeds to ensure simulation diversity
- Merging of tree statistics across parallel workers
- Lock-free or synchronized access to shared MCTS tree