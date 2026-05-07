# Claude AI assisted with: Root parallelization architecture, worker orchestration, statistics merging
import concurrent.futures
import multiprocessing
import time
from collections import defaultdict
import numpy as np

from src.hex_engine.board.board_cython import HexBoard
from src.hex_engine.evaluation.evaluator import Evaluator
from src.hex_engine.search.mcts import MCTS

class SubsetEvaluator(Evaluator):
    """
    A custom wrapper that forces the MCTS engine to only explore a specific subset 
    of moves at the root level, guaranteeing zero overlap between parallel workers.
    """
    def __init__(self, search_radius, root_piece_count, allowed_root_moves):
        super().__init__(search_radius)
        self.root_piece_count = root_piece_count
        self.allowed_root_moves = allowed_root_moves
        
    def get_candidate_moves(self, board, top_n=10):
        # Hijack the candidate generation ONLY at the root node.
        # Using total piece count is 100% robust against complex turn mechanics.
        p1_count = board.get_pieces(1).shape[0]
        p2_count = board.get_pieces(2).shape[0]
        
        if (p1_count + p2_count) == self.root_piece_count:
            return self.allowed_root_moves
        
        # For all deeper simulated nodes, use the ultra-fast Numba evaluation
        return super().get_candidate_moves(board, top_n)

def _mcts_worker(worker_id, pieces_dict, turn, last_move, search_radius, iterations, exploration_constant, allowed_root_moves, root_piece_count):
    """
    Isolated worker process for Parallel MCTS.
    Reconstructs the Cython board to bypass multiprocessing pickling restrictions.
    """
    # 1. Reconstruct the C++ engine components safely in this new process
    board = HexBoard(pieces_dict=pieces_dict, turn=turn, last_move=last_move)
    
    # 2. Inject the restricted evaluator to force this worker down a unique subset of root branches
    evaluator = SubsetEvaluator(search_radius=search_radius, root_piece_count=root_piece_count, allowed_root_moves=allowed_root_moves)
    
    # 3. Run the isolated MCTS Search
    if iterations > 0:
        mcts = MCTS(evaluator=evaluator, exploration_constant=exploration_constant)
        mcts.search(board, iterations=iterations, profile=False)
        return mcts.move_stats
    return []

def _warmup_worker(search_radius):
    """A simple task to force the worker process to compile Numba JIT functions."""
    evaluator = Evaluator(search_radius=search_radius)
    evaluator.warmup()
    return True

class Analyzer:
    def __init__(self, num_workers=None, search_radius=15):
        # Default to leaving 1 CPU core free to keep your OS responsive
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self.search_radius = search_radius
        # Keep the executor alive so we don't pay process spin-up costs every move!
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers)
        
    def warmup(self, verbose=True):
        """Pre-warms all background workers by forcing them to compile Numba functions."""
        if verbose:
            print(f"Warming up {self.num_workers} parallel workers...")
        futures = [self.executor.submit(_warmup_worker, self.search_radius) for _ in range(self.num_workers)]
        concurrent.futures.wait(futures)
        if verbose:
            print("Workers ready.")

    def shutdown(self):
        """Cleans up background processes."""
        self.executor.shutdown()
        
    def analyze_move(self, board, total_iterations=100000, exploration_constant=1.414, verbose=True):
        """
        Spawns multiple MCTS workers, merges their trees, and returns the optimal move.
        Utilizes strict Root Parallelization to prevent duplicate processing.
        """
        if verbose:
            print(f"\n--- Starting Parallel Analysis ---")
            print(f"Workers: {self.num_workers} CPU Cores")
            print(f"Target Iterations: {total_iterations}")
        
        start_time = time.perf_counter()
        
        # 1. Generate the Top 25 initial candidate moves to distribute
        main_evaluator = Evaluator(search_radius=self.search_radius)
        root_candidates = main_evaluator.get_candidate_moves(board, top_n=25)
        
        # Split candidates as evenly as possible among workers
        worker_allocations = np.array_split(root_candidates, self.num_workers)
        
        # 2. Deconstruct the board for safe cross-process serialization
        p1_pieces = board.get_pieces(1)
        p2_pieces = board.get_pieces(2)
        
        pieces_dict = {}
        for p in p1_pieces:
            pieces_dict[(int(p[0]), int(p[1]), int(p[2]))] = 1
        for p in p2_pieces:
            pieces_dict[(int(p[0]), int(p[1]), int(p[2]))] = 2
            
        turn = int(board.turn)
        last_move = board.last_move
        if last_move is not None:
            last_move = (int(last_move[0]), int(last_move[1]), int(last_move[2]))
            
        root_piece_count = p1_pieces.shape[0] + p2_pieces.shape[0]
        
        # Calculate iteration load per worker
        iters_per_worker = total_iterations // self.num_workers
        
        # 3. Dispatch the workers to different CPU cores
        futures = []
        for i in range(self.num_workers):
            # Skip spawning a task if they were allocated 0 moves
            if len(worker_allocations[i]) == 0:
                continue
                
            future = self.executor.submit(
                _mcts_worker,
                i, 
                pieces_dict,
                turn,
                last_move,
                self.search_radius,
                iters_per_worker,
                exploration_constant,
                worker_allocations[i], # Provide the worker its specific slice of candidates
                root_piece_count       # Pass robust root tracker
            )
            futures.append(future)
        
        # 4. Merge Statistics from all parallel trees
        # Format: { move_tuple: {'visits': total_visits, 'wins': total_wins} }
        merged_stats = defaultdict(lambda: {'visits': 0, 'wins': 0.0})
        
        for future in concurrent.futures.as_completed(futures):
            worker_stats = future.result()
            for stat in worker_stats:
                move = stat['move']
                merged_stats[move]['visits'] += stat['visits']
                merged_stats[move]['wins'] += stat['wins']
                
        # 5. Find the mathematically best move across all combined trees
        best_move = None
        max_visits = -1
        
        if verbose:
            print("\nMerged Parallel Tree Statistics:")
            print(f"{'Move':<15} | {'Total Visits':<15} | {'Win Rate'}")
            print("-" * 45)
        
        # Sort display by highest visits
        for move, data in sorted(merged_stats.items(), key=lambda x: x[1]['visits'], reverse=True):
            visits = data['visits']
            wins = data['wins']
            win_rate = (wins / visits * 100) if visits > 0 else 0
            
            if verbose:
                print(f"({move[0]:2d},{move[1]:2d},{move[2]:2d})    | {visits:<15d} | {win_rate:5.1f}%")
            
            if visits > max_visits:
                max_visits = visits
                best_move = move
                
        end_time = time.perf_counter()
        speed = total_iterations / (end_time - start_time)
        
        if verbose:
            print("-" * 45)
            print(f"Completed {total_iterations} parallel simulations in {end_time - start_time:.3f} seconds.")
            print(f"Effective Search Speed: {speed:.0f} nodes/second")
            print(f"Optimal Move Suggested: {best_move}")
        
        return best_move

if __name__ == "__main__":
    # A quick standalone test to compare different thread counts!
    board = HexBoard()
    
    # Setup a skirmish to analyze
    board.do_move((0, 0, 0))   
    board.do_move((1, -1, 0))  
    board.do_move((2, -2, 0))  
    board.do_move((-1, 1, 0))
    
    thread_counts = [1, 2, 4, 8, 16]
    total_iterations = 100000
    
    print("==================================================")
    print("   THREAD SCALING COMPARISON")
    print("==================================================")
    print(f"Target Iterations: {total_iterations:,}")
    print("--------------------------------------------------")
    
    for cores in thread_counts:
        # Spin up an analyzer restricted to the specific core count
        analyzer = Analyzer(num_workers=cores)
        
        # Warmup the nodes (silently)
        print(f"Warming up {cores:2d} cores...", end="\r")
        analyzer.warmup(verbose=False)
        
        start_time = time.perf_counter()
        # Suppress verbose output to keep our loop summary table clean
        best_move = analyzer.analyze_move(board, total_iterations=total_iterations, verbose=False)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        nps = total_iterations / duration
        
        # Clear the "Warming up" carriage return and print the final stats
        print(f"Workers: {cores:2d} | Time: {duration:6.3f}s | Speed: {nps:7.0f} n/s | Best Move: {best_move}")
        
        analyzer.shutdown()
    print("==================================================")