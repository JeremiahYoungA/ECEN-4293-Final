# Project Goals and Performance Targets

This document outlines the measurable objectives and performance targets for the hex engine project.

## Primary Objectives

1. **Strategic Move Suggestion** - Recommend the most statistically favorable coordinate
2. **Dynamic Position Evaluation** - Display real-time numerical advantage scores  
3. **6-in-a-Row Detection** - Identify forced wins ("Mate in N") where a player can achieve 6 consecutive pieces within reachable depth
4. **Asynchronous Computation** - Maintain responsive interaction during heavy simulations

## Performance Targets

### 6-in-a-Row Detection Complexity
Detecting 6 consecutive pieces along any direction must demonstrate **O(1) time complexity** relative to the total number of pieces on board. This enables fast evaluation without traversing all board positions.

### Search Response Time
Engine must identify winning moves in **~2 seconds** when presented with known geometric traps in 100% of test cases.

### Move Suggestion Accuracy
Influence-based pruning must exclude at least **90% of legal moves** in a local evaluation chunk while maintaining **100% retention** of moves that contribute to detected forced-win sequences.

### Statistical Convergence
Running **N parallel simulations** must reduce standard error of evaluation score by factor approaching **1/√N**, demonstrating proper convergence properties.

### Vectorization Performance
Field calculation using vectorized operations must perform at least **20× faster** than equivalent Python for-loop over a 50×50 grid.

## Design Constraints

- **Unbounded Board** - System must handle infinite hex grid without storing entire board state
- **Sparse State** - Store only occupied positions to maintain O(n) memory where n = pieces
- **Real-time Responsiveness** - UI remains responsive during background MCTS computation
