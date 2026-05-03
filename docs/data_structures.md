# Data Structures

## Cube Coordinates

### Overview
Cube coordinates represent positions on a hexagonal grid using three axes (a, b, c) positioned at 120° angles to each other. This system is optimal for hex grids because it enables simple neighbor calculations, distance metrics, and validation.

### Coordinate System

```
        +a (up)
        /\
       /  \
      /    \
  +c /______\ +b
    (diagonal up-left) (diagonal up-right)
```

Each hex is indexed by (a, b, c) representing distance traveled along each axis.

### Constraint

**All valid coordinates must satisfy: a + b + c = 0**

This constraint means:
- Only 2 degrees of freedom are needed to specify a position
- The third coordinate is determined by the first two: `c = -a - b`
- Invalid coordinates can be detected by checking the sum

### Neighbor Directions

The six neighbors of any hex at (a, b, c) are obtained by adding one of these direction vectors:

| Direction   | Vector      | Example from (0,0,0) |
|-------------|-------------|----------------------|
| Up          | (1, -1, 0)  | (1, -1, 0)           |
| Up-Right    | (1, 0, -1)  | (1, 0, -1)           |
| Down-Right  | (0, 1, -1)  | (0, 1, -1)           |
| Down        | (-1, 1, 0)  | (-1, 1, 0)           |
| Down-Left   | (-1, 0, 1)  | (-1, 0, 1)           |
| Up-Left     | (0, -1, 1)  | (0, -1, 1)           |

All direction vectors sum to 0, preserving the constraint.

### Distance Metric

Distance between any two coordinates is:
$$d = \frac{|a_1 - a_2| + |b_1 - b_2| + |c_1 - c_2|}{2}$$

This is the minimum number of hex steps required to travel from one position to another.

### Applications in Hex Engine

1. **Win Detection** - Check neighbors along 6 directions for consecutive pieces
2. **Influence Field** - Calculate distance for $E = \sum \frac{c}{d^2}$ heuristic
3. **Move Generation** - Efficiently enumerate valid moves around current pieces
4. **Search Tree** - Natural representation of board positions in MCTS

### Implementation Notes

- Store coordinates as tuples or named data structures: `Coord = (a: int, b: int, c: int)`
- Always validate constraint during board state creation
- Use direction vectors as immutable constants for neighbor iteration

## Board Representation

### Sparse Storage

The board stores only occupied positions using a sparse data structure keyed by cube coordinates.

**Storage Structure:**
- `Set[Coord]` or `Dict[Coord, Piece]` maintains occupied hexagonal positions
- Each entry maps a coordinate to piece ownership state

**Operations:**
- Add piece: `occupied.add((a, b, c))`
- Check occupancy: `(a, b, c) in occupied`
- Retrieve all pieces: `for coord in occupied: ...`
- Remove piece: `occupied.discard((a, b, c))`

