import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
import time

# Import your actual engine
from src.hex_engine.board.board_cython import HexBoard
from src.hex_engine.evaluation.evaluator import Evaluator
from src.hex_engine.analysis.analysis import Analyzer

class InteractiveHexGame:
    def __init__(self, radius=4):
        self.radius = radius
        self.ai_thinking = False
        
        print("Initializing Engine...")
        self.board = HexBoard()
        self.evaluator = Evaluator(search_radius=15)
        self.evaluator.warmup()
        self.analyzer = Analyzer(num_workers=8, search_radius=15)
        self.analyzer.warmup(verbose=True)
        
        # Setup Plot
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title('Hex Engine Match')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.draw_board()
        
    def pixel_to_hex(self, x, y):
        """Reverses the Cartesian projection back into 3D Cube Coordinates."""
        c = y / 1.5
        a = (x / np.sqrt(3)) - (c / 2.0)
        b = -a - c
        
        # Rounding to nearest hex (Handling the fractional grid intersection)
        rx, ry, rz = round(a), round(b), round(c)
        x_diff, y_diff, z_diff = abs(rx - a), abs(ry - b), abs(rz - c)
        
        # Maintain the a + b + c = 0 constraint
        if x_diff > y_diff and x_diff > z_diff:
            rx = -ry - rz
        elif y_diff > z_diff:
            ry = -rx - rz
        else:
            rz = -rx - ry
            
        return (int(rx), int(ry), int(rz))

    def on_click(self, event):
        """Handles human interaction and triggers AI response."""
        if event.inaxes != self.ax: return
        
        # Ignore clicks if game is over, AI is calculating, or it's not the human's (Player 1) turn
        if self.board.check_win() or self.ai_thinking or self.board.get_current_player() != 1: 
            return 
        
        # 1. Map click to board coordinate
        coord = self.pixel_to_hex(event.xdata, event.ydata)
        
        # Out of bounds check
        if max(abs(coord[0]), abs(coord[1]), abs(coord[2])) > self.radius: return
        
        # Occupied check
        p1_pieces = [tuple(p) for p in self.board.get_pieces(1)]
        p2_pieces = [tuple(p) for p in self.board.get_pieces(2)]
        if coord in p1_pieces or coord in p2_pieces:
            print(f"Space {coord} is already occupied!")
            return
            
        # Execute Human Move
        print(f"\nPlayer played: {coord}")
        self.board.do_move(coord)
        self.draw_board()
        
        if self.board.check_win():
            plt.title("Game Over: You Win!", fontsize=18, fontweight='bold', color='green')
            self.fig.canvas.draw()
            return

        # 2. Trigger AI Turn(s)
        # Using a while loop so it perfectly respects Connect6 rules (AI plays twice after turn 1)
        self.ai_thinking = True
        while self.board.get_current_player() == 2 and not self.board.check_win():
            # Force UI to update text before AI freezes the thread
            plt.title("Parallel MCTS is analyzing...", fontsize=16, fontweight='bold', color='orange')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            start_time = time.perf_counter()
            best_move = self.analyzer.analyze_move(self.board, total_iterations=1000000, exploration_constant=1.414, verbose=True)
            end_time = time.perf_counter()
            
            print(f"Parallel analysis completed in {end_time - start_time:.3f}s. Plays: {best_move}")
            
            if best_move:
                self.board.do_move(best_move)
            else:
                break # Failsafe if no moves are generated
                
            self.draw_board()
            
            if self.board.check_win():
                plt.title("Game Over: Engine Wins!", fontsize=18, fontweight='bold', color='red')
                self.fig.canvas.draw()
                self.ai_thinking = False
                return
                
        self.ai_thinking = False

    def draw_board(self):
        """Renders the board state."""
        self.ax.clear()
        self.ax.set_aspect('equal')
        
        p1_pieces = [tuple(p) for p in self.board.get_pieces(1)]
        p2_pieces = [tuple(p) for p in self.board.get_pieces(2)]

        # Fetch the influence map from the evaluator
        w_inf, b_inf, chunk = self.evaluator.get_influence(self.board)
        # White is positive, Black is negative (differential field)
        inf_map = {tuple(chunk[i]): w_inf[i] - b_inf[i] for i in range(len(chunk))}
        # Absolute fluence field: total contention/interest
        abs_fluence_map = {tuple(chunk[i]): np.abs(w_inf[i]) + np.abs(b_inf[i]) for i in range(len(chunk))}
        
        # Fetch top candidate moves to highlight
        # Explicitly requesting only the top 5 moves for visualization
        candidates = self.evaluator.get_candidate_moves(self.board, top_n=25)
        # Convert candidates to tuples for proper comparison
        candidate_tuples = set(tuple(move) for move in candidates)

        for a in range(-self.radius, self.radius + 1):
            for b in range(-self.radius, self.radius + 1):
                c = -a - b
                if abs(c) <= self.radius:
                    # Pointy-topped hex cartesian math
                    x = np.sqrt(3) * (a + c / 2.0)
                    y = 1.5 * c
                    
                    coord = (a, b, c)
                    inf_val = inf_map.get(coord, 0.0)
                    abs_fluence = abs_fluence_map.get(coord, 0.0)
                    
                    if coord in p1_pieces:
                        hex_color = '#222222'
                        edge_color = '#000000'
                        text_color = '#ffffff'
                        label_text = f"{a},{b},{c}"
                    elif coord in p2_pieces:
                        hex_color = '#ffffff'
                        edge_color = '#000000'
                        text_color = '#000000'
                        label_text = f"{a},{b},{c}"
                    else:
                        hex_color = '#f0f0f0'
                        edge_color = '#cccccc'
                        text_color = '#777777'
                        
                        # Heatmap color coding based on influence values
                        if inf_val > 0.05:
                            hex_color = '#e0efff'  # Light Blue for White advantage
                            text_color = '#0044cc'
                        elif inf_val < -0.05:
                            hex_color = '#ffe0e0'  # Light Red for Black advantage
                            text_color = '#cc0000'
                            
                        # Format the influence to 2 decimal places with a +/- sign
                        # Also show absolute fluence field (total contention)
                        label_text = f"{a},{b},{c}\n{inf_val:+.2f}\n|Φ|:{abs_fluence:.2f}"
                        
                    # Highlight candidate moves with a thick gold outline
                    is_candidate = coord in candidate_tuples
                    if is_candidate and coord not in p1_pieces and coord not in p2_pieces:
                        edge_color = '#ffaa00' # Golden orange
                        current_linewidth = 3.5
                        zorder = 5
                    else:
                        current_linewidth = 1.5
                        zorder = 1
                        
                    # Fix: orientation=np.pi/3 rotates the hexes to have flat tops
                    # Fix: radius=0.98 closes the gaps while leaving a tiny border
                    hex_patch = RegularPolygon(
                        (x, y), 
                        numVertices=6, 
                        radius=0.98,
                        orientation=np.pi/3, 
                        facecolor=hex_color, 
                        edgecolor=edge_color, 
                        linewidth=current_linewidth,
                        zorder=zorder
                    )
                    self.ax.add_patch(hex_patch)
                    
                    # Add tiny coordinate labels
                    self.ax.text(x, y, label_text, ha='center', va='center', 
                                 size=7, color=text_color, weight='bold', zorder=zorder+1)

        if not self.board.check_win():
            if self.board.get_current_player() == 1:
                plt.title("Your Turn (Click a Hexagon)", fontsize=16, fontweight='bold')
            else:
                plt.title("Parallel MCTS is analyzing...", fontsize=16, fontweight='bold', color='orange')
            
        self.ax.autoscale_view()
        self.ax.axis('off')
        plt.tight_layout()
        self.fig.canvas.draw()

    def cleanup(self):
        """Shuts down background worker processes."""
        self.analyzer.shutdown()

if __name__ == "__main__":
    print("Starting Interactive Hex Match vs. Parallel MCTS Engine...")
    game = InteractiveHexGame(radius=5)
    try:
        plt.show()
    finally:
        game.cleanup()