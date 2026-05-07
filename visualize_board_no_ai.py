# Claude AI assisted with: creating a no AI version based on visualize_board.py
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np

# Import your engine components
from src.hex_engine.board.board_cython import HexBoard
from src.hex_engine.evaluation.evaluator import Evaluator

class InteractiveHexGame2P:
    def __init__(self, radius=4):
        self.radius = radius
        
        print("Initializing 2-Player Board...")
        self.board = HexBoard()
        # Keep the evaluator just to draw the cool influence heatmap
        self.evaluator = Evaluator(search_radius=15)
        self.evaluator.warmup()
        
        # Setup Plot
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title('Hex Engine Match (Local 2-Player)')
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
        """Handles human interaction for both players."""
        if event.inaxes != self.ax: return
        
        # Ignore clicks if game is over
        if self.board.check_win(): 
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
            
        # Execute Move
        current_player = self.board.get_current_player()
        player_name = "Black" if current_player == 1 else "White"
        
        print(f"\nPlayer {current_player} ({player_name}) played: {coord}")
        self.board.do_move(coord)
        self.draw_board()
        
        if self.board.check_win():
            win_color = 'black' if current_player == 1 else 'gray'
            plt.title(f"Game Over: Player {current_player} ({player_name}) Wins!", fontsize=18, fontweight='bold', color=win_color)
            self.fig.canvas.draw()
            return

    def draw_board(self):
        """Renders the board state."""
        self.ax.clear()
        self.ax.set_aspect('equal')
        
        p1_pieces = [tuple(p) for p in self.board.get_pieces(1)]
        p2_pieces = [tuple(p) for p in self.board.get_pieces(2)]

        # Fetch the influence map from the evaluator
        w_inf, b_inf, chunk = self.evaluator.get_influence(self.board)
        # White is positive, Black is negative
        inf_map = {tuple(chunk[i]): w_inf[i] - b_inf[i] for i in range(len(chunk))}
        
        # Fetch top candidate moves to highlight
        candidates = self.evaluator.get_candidate_moves(self.board, top_n=30)
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
                        label_text = f"{a},{b},{c}\n{inf_val:+.2f}"
                        
                    # Highlight candidate moves with a thick gold outline
                    is_candidate = coord in candidate_tuples
                    if is_candidate and coord not in p1_pieces and coord not in p2_pieces:
                        edge_color = '#ffaa00' # Golden orange
                        current_linewidth = 3.5
                        zorder = 5
                    else:
                        current_linewidth = 1.5
                        zorder = 1
                        
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
            current_player = self.board.get_current_player()
            player_name = "Black" if current_player == 1 else "White"
            color = 'black' if current_player == 1 else 'gray'
            plt.title(f"Player {current_player}'s Turn ({player_name}) - Click a Hexagon", fontsize=16, fontweight='bold', color=color)
            
        self.ax.autoscale_view()
        self.ax.axis('off')
        plt.tight_layout()
        self.fig.canvas.draw()

if __name__ == "__main__":
    print("Starting Interactive Hex Match (2-Player Local)...")
    game = InteractiveHexGame2P(radius=5)
    plt.show()