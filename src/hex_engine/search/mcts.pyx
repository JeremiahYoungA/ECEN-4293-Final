# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, log
import numpy as np
cimport numpy as cnp
import time

from src.hex_engine.board.board_cython cimport HexBoard

# C-level Node for the MCTS tree
cdef struct Node:
    int a, b, c           # Move that led to this node
    double wins           # Win count (from POV of player who made the move)
    int visits            # Total simulations
    int player            # Player who made the move
    Node* parent
    Node** children
    int num_children
    int capacity
    bint is_fully_expanded

# Added `noexcept` to prevent Cython from checking for Python exceptions without the GIL
cdef Node* create_node(int a, int b, int c, int player, Node* parent) noexcept nogil:
    cdef Node* n = <Node*>malloc(sizeof(Node))
    n.a = a
    n.b = b
    n.c = c
    n.wins = 0.0
    n.visits = 0
    n.player = player
    n.parent = parent
    n.num_children = 0
    n.capacity = 10
    n.children = <Node**>malloc(n.capacity * sizeof(Node*))
    n.is_fully_expanded = False
    return n

cdef void free_tree(Node* n) noexcept nogil:
    if n == NULL: return
    for i in range(n.num_children):
        free_tree(n.children[i])
    free(n.children)
    free(n)

cdef class MCTS:
    cdef double exploration_constant
    cdef object evaluator
    cdef public dict timing_stats
    
    def __init__(self, evaluator, double exploration_constant=1.414):
        self.evaluator = evaluator
        self.exploration_constant = exploration_constant
        self.timing_stats = {
            'total': 0.0,
            'selection': 0.0,
            'expansion': 0.0,
            'simulation': 0.0,
            'backprop': 0.0,
            'rollback': 0.0,
            'iterations': 0
        }

    def search(self, board, int iterations=1000, bint profile=False):
        """
        Executes MCTS iterations and returns the best move.
        If profile=True, also returns timing statistics.
        """
        # Reset timing stats
        self.timing_stats = {
            'total': 0.0,
            'selection': 0.0,
            'expansion': 0.0,
            'simulation': 0.0,
            'backprop': 0.0,
            'rollback': 0.0,
            'iterations': iterations
        }
        
        cdef int current_p = board.get_current_player()
        cdef Node* root = create_node(0, 0, 0, 0, NULL) # Dummy root
        
        start_total = time.perf_counter()
        
        for i in range(iterations):
            # Pass the board directly! No copying!
            self._iterate(root, board)
            
        end_total = time.perf_counter()
        self.timing_stats['total'] = end_total - start_total
        
        # Select best child of root
        cdef Node* best_child = NULL
        cdef int max_visits = -1
        for i in range(root.num_children):
            if root.children[i].visits > max_visits:
                max_visits = root.children[i].visits
                best_child = root.children[i]
        
        best_move = (best_child.a, best_child.b, best_child.c) if best_child else None
        
        # Cleanup
        free_tree(root)
        
        if profile:
            self.print_profile()
        
        return best_move

    cdef void _iterate(self, Node* root, object board):
        """A single MCTS iteration: Selection, Expansion, Simulation, Backprop."""
        cdef Node* node = root
        cdef bint is_win = False
        cdef int moves_made = 0
        
        # 1. Selection (Selection uses UCB1)
        t_sel_start = time.perf_counter()
        while node.num_children > 0 and node.is_fully_expanded:
            node = self._select_child(node)
            is_win = board.do_move((node.a, node.b, node.c))
            moves_made += 1
            if is_win: break
        self.timing_stats['selection'] += time.perf_counter() - t_sel_start
        
        # 2. Expansion
        cdef Node* expanded_node = NULL
        t_exp_start = time.perf_counter()
        if not is_win:
            expanded_node = self._expand(node, board)
            if expanded_node != node:
                node = expanded_node
                is_win = board.do_move((node.a, node.b, node.c))
                moves_made += 1
        self.timing_stats['expansion'] += time.perf_counter() - t_exp_start

        # 3. Simulation (Rollout)
        t_sim_start = time.perf_counter()
        cdef double result = self._simulate(board, is_win)
        self.timing_stats['simulation'] += time.perf_counter() - t_sim_start

        # 4. Backpropagation
        t_bp_start = time.perf_counter()
        self._backpropagate(node, result)
        self.timing_stats['backprop'] += time.perf_counter() - t_bp_start

        # 5. History Stack Rollback
        # Instantly rewind the board state back to the root for the next iteration
        t_rb_start = time.perf_counter()
        for _ in range(moves_made):
            board.undo_move()
        self.timing_stats['rollback'] += time.perf_counter() - t_rb_start

    cdef Node* _select_child(self, Node* parent) noexcept nogil:
        cdef double best_val = -1.0
        cdef Node* best_node = NULL
        cdef double uct_val
        
        for i in range(parent.num_children):
            # UCB1 Formula
            if parent.children[i].visits == 0:
                return parent.children[i]
                
            uct_val = (parent.children[i].wins / parent.children[i].visits) + \
                      self.exploration_constant * sqrt(log(<double>parent.visits) / parent.children[i].visits)
            
            if uct_val > best_val:
                best_val = uct_val
                best_node = parent.children[i]
        
        return best_node

    cdef Node* _expand(self, Node* node, object board):
        # Get candidate moves from evaluator (Heuristic Pruning)
        candidates = self.evaluator.get_candidate_moves(board, top_n=10)
        
        # In a real impl, we filter out moves already in children
        # For simplicity in this MVP version, we expand one move at a time
        if node.num_children < len(candidates):
            m = candidates[node.num_children]
            new_node = create_node(m[0], m[1], m[2], board.get_current_player(), node)
            
            # Resize children array if needed
            if node.num_children == node.capacity:
                # Omitted for brevity: node.children = realloc(...)
                pass
                
            node.children[node.num_children] = new_node
            node.num_children += 1
            
            if node.num_children == len(candidates):
                node.is_fully_expanded = True
                
            return new_node
        return node

    cdef double _simulate(self, object board, bint already_won):
        """Simplified simulation using the evaluator's heuristic score."""
        if already_won:
            # Result is 1.0 if the player who just moved won
            return 1.0
        
        # Use the heuristic field as a proxy for rollout result to save time
        # Normalize score between 0 and 1
        score = self.evaluator.evaluate(board)
        return 1.0 / (1.0 + np.exp(-score / 10.0))

    cdef void _backpropagate(self, Node* node, double result) noexcept nogil:
        while node != NULL:
            node.visits += 1
            node.wins += result
            result = 1.0 - result # Opposing player's perspective
            node = node.parent

    def print_profile(self):
        """Print a formatted timing report of MCTS phases."""
        total_time = self.timing_stats['total']
        print("\n" + "="*60)
        print("MCTS PROFILING REPORT")
        print("="*60)
        print(f"Total iterations:     {self.timing_stats['iterations']}")
        print(f"Total time:           {total_time:.4f}s ({total_time*1000:.2f}ms)")
        print(f"Time per iteration:   {total_time/self.timing_stats['iterations']*1000:.2f}ms")
        print("-"*60)
        
        phases = ['selection', 'expansion', 'simulation', 'backprop', 'rollback']
        for phase in phases:
            phase_time = self.timing_stats[phase]
            pct = (phase_time / total_time * 100) if total_time > 0 else 0
            avg_per_iter = phase_time / self.timing_stats['iterations'] * 1000
            print(f"{phase.ljust(15)} {phase_time:8.4f}s  ({pct:5.1f}%)  avg: {avg_per_iter:6.2f}ms/iter")
        
        print("="*60 + "\n")