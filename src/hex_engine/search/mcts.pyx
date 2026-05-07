# Claude AI assisted with: Cython syntax, performance optimization
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.stdlib cimport malloc, free, realloc, rand, srand
from libc.math cimport sqrt, log, exp
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

# Seed the C random number generator once on module load
srand(<unsigned int>time.time())

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
    cdef int i
    for i in range(n.num_children):
        free_tree(n.children[i])
    free(n.children)
    free(n)

cdef class MCTS:
    cdef double exploration_constant
    cdef object evaluator
    cdef public dict timing_stats
    cdef public list move_stats
    
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
        self.move_stats = []

    def search(self, board, int iterations=1000, bint profile=False):
        """
        Executes MCTS iterations and returns the best move.
        If profile=True, prints timing and move statistics.
        """
        self.timing_stats = {k: 0.0 for k in self.timing_stats}
        self.timing_stats['iterations'] = iterations
        self.move_stats = []
        
        cdef Node* root = create_node(0, 0, 0, 0, NULL)
        
        start_total = time.perf_counter()
        cdef int i
        for i in range(iterations):
            self._iterate(root, board)
            
        end_total = time.perf_counter()
        self.timing_stats['total'] = end_total - start_total
        
        self._extract_root_children_stats(root)
        
        cdef Node* best_child = NULL
        cdef int max_visits = -1
        for i in range(root.num_children):
            if root.children[i].visits > max_visits:
                max_visits = root.children[i].visits
                best_child = root.children[i]
        
        best_move = (best_child.a, best_child.b, best_child.c) if best_child else None
        free_tree(root)
        
        if profile:
            self.print_profile()
            self.print_move_statistics()
        
        return best_move

    cdef void _iterate(self, Node* root, object board):
        cdef Node* node = root
        cdef bint is_win = False
        cdef int moves_made = 0
        cdef int i
        
        # 1. Selection
        t_sel_start = time.perf_counter()
        while node.num_children > 0 and node.is_fully_expanded:
            node = self._select_child(node)
            is_win = board.do_move((node.a, node.b, node.c))
            moves_made += 1
            if is_win: break
        self.timing_stats['selection'] += time.perf_counter() - t_sel_start
        
        # 2. Expansion
        t_exp_start = time.perf_counter()
        if not is_win:
            expanded_node = self._expand(node, board)
            if expanded_node != node:
                node = expanded_node
                is_win = board.do_move((node.a, node.b, node.c))
                moves_made += 1
        self.timing_stats['expansion'] += time.perf_counter() - t_exp_start

        # 3. Simulation (Optimized Leaf Evaluation)
        t_sim_start = time.perf_counter()
        cdef double result = self._simulate(board, is_win)
        self.timing_stats['simulation'] += time.perf_counter() - t_sim_start

        # 4. Backpropagation
        t_bp_start = time.perf_counter()
        self._backpropagate(node, result)
        self.timing_stats['backprop'] += time.perf_counter() - t_bp_start

        # 5. Rollback
        t_rb_start = time.perf_counter()
        for i in range(moves_made):
            board.undo_move()
        self.timing_stats['rollback'] += time.perf_counter() - t_rb_start

    cdef Node* _select_child(self, Node* parent) noexcept nogil:
        cdef double best_val = -1.0
        cdef Node* best_node = NULL
        cdef double uct_val
        cdef int i
        for i in range(parent.num_children):
            if parent.children[i].visits == 0:
                return parent.children[i]
            uct_val = (parent.children[i].wins / parent.children[i].visits) + \
                      self.exploration_constant * sqrt(log(<double>parent.visits) / parent.children[i].visits)
            if uct_val > best_val:
                best_val = uct_val
                best_node = parent.children[i]
        return best_node

    cdef Node* _expand(self, Node* node, object board):
        candidates = self.evaluator.get_candidate_moves(board, top_n=10)
        if node.num_children < len(candidates):
            m = candidates[node.num_children]
            new_node = create_node(m[0], m[1], m[2], board.get_current_player(), node)
            if node.num_children == node.capacity:
                node.capacity *= 2
                node.children = <Node**>realloc(node.children, node.capacity * sizeof(Node*))
            node.children[node.num_children] = new_node
            node.num_children += 1
            if node.num_children == len(candidates):
                node.is_fully_expanded = True
            return new_node
        return node

    cdef double _simulate(self, object board, bint already_won):
        """
        OPTIMIZED: Heuristic Leaf Evaluation.
        Instead of a heavy 50-move rollout with expensive Numba calls, we evaluate the 
        leaf node directly. This provides a high-quality value estimate at 1/50th the cost.
        """
        if already_won:
            return 1.0 # Current player won
            
        # Get the scalar influence advantage from the Evaluator
        cdef double score = self.evaluator.evaluate(board)
        
        # Connect6 Score Normalization (Sigmoid squash)
        # We squash the advantage score into a [0, 1] probability of winning.
        # 10.0 is the temperature; adjust if the AI is too confident or too timid.
        return 1.0 / (1.0 + exp(-score / 10.0))

    cdef void _backpropagate(self, Node* node, double result) noexcept nogil:
        while node != NULL:
            node.visits += 1
            node.wins += result
            result = 1.0 - result
            node = node.parent

    cdef void _extract_root_children_stats(self, Node* root):
        cdef int i
        for i in range(root.num_children):
            self.move_stats.append({
                'move': (root.children[i].a, root.children[i].b, root.children[i].c),
                'visits': root.children[i].visits,
                'wins': root.children[i].wins
            })

    def print_move_statistics(self):
        if not self.move_stats: return
        sorted_moves = sorted(self.move_stats, key=lambda x: x['visits'], reverse=True)
        total_v = sum(m['visits'] for m in sorted_moves)
        print("\n" + "="*75 + "\nMOVE STATISTICS\n" + "="*75)
        for rank, m in enumerate(sorted_moves, 1):
            move, v, w = m['move'], m['visits'], m['wins']
            print(f"{rank:<3} ({move[0]:2d},{move[1]:2d},{move[2]:2d}) | Visits: {v:<8} | Win%: {w/v*100:5.1f}%")

    def print_profile(self):
        total = self.timing_stats['total']
        print("\n" + "="*60 + "\nPERFORMANCE PROFILE\n" + "="*60)
        print(f"Total Iterations: {self.timing_stats['iterations']}")
        print(f"Total Time:      {total:.4f}s")
        for phase in ['selection', 'expansion', 'simulation', 'backprop', 'rollback']:
            t = self.timing_stats[phase]
            print(f"{phase.ljust(12)} {t:8.4f}s ({(t/total*100):4.1f}%)")