# Claude AI assisted with: Integration test design and board logic validation
from src.hex_engine.board.board_cython import HexBoard
from src.hex_engine.evaluation.evaluator import Evaluator

def run_integration_tests():
    print("--- Hex Engine Functional Test Suite ---")
    
    # 1. Initialization & Warmup
    print("\n1. Initializing Engine...")
    board = HexBoard()
    evaluator = Evaluator(search_radius=15)
    evaluator.warmup()
    print("   [OK] Evaluator JIT compiled successfully.")

    # 2. Empty Board Candidates
    print("\n2. Testing Candidate Generation (Empty Board)...")
    candidates = evaluator.get_candidate_moves(board, top_n=5)
    print(f"   -> Top candidates returned: {candidates.tolist()}")
    assert len(candidates) > 0, "Failed to generate candidates for empty board."
    print("   [OK] Candidate generation successful.")

    # 3. Advantage Shifting
    print("\n3. Testing Influence Evaluation...")
    # Player 1 (Black) places a piece
    board.do_move((0, 0, 0))
    score_p1 = evaluator.evaluate(board)
    print(f"   -> Score after P1 plays (0,0,0): {score_p1:.2f}")
    assert score_p1 > 0, "Player 1 should have a positive advantage."

    # Player 2 (White) gets two moves (Connect6 rules)
    board.do_move((1, -1, 0))
    board.do_move((2, -2, 0))
    score_p2 = evaluator.evaluate(board)
    print(f"   -> Score after P2 plays two pieces: {score_p2:.2f}")
    assert score_p2 < score_p1, "Advantage should shift toward Player 2."
    print("   [OK] Evaluation field shifts correctly.")

    # 4. Win Detection
    print("\n4. Testing O(1) 6-in-a-row Win Detection...")
    # Give P1 dummy moves so P2 can finish their line along the (1, -1, 0) axis
    board.do_move((10, -10, 0))
    board.do_move((11, -11, 0))
    
    # P2 gets two more
    board.do_move((3, -3, 0))
    board.do_move((4, -4, 0))
    
    # P1 gets two more
    board.do_move((12, -12, 0))
    board.do_move((13, -13, 0))
    
    # P2 gets the final two for the win
    board.do_move((5, -5, 0))
    is_win = board.do_move((6, -6, 0))
    
    print(f"   -> Board reports win: {is_win}")
    assert is_win == True, "do_move() failed to detect a 6-in-a-row win!"
    assert board.check_win() == True, "check_win() failed to verify the win state."
    print("   [OK] Win detection is flawless.")

    # 5. History Stack Rollback
    print("\n5. Testing History Stack Rollback...")
    board.undo_move()
    print(f"   -> Board reports win after undo: {board.check_win()}")
    assert board.check_win() == False, "Board should not be in a win state after undo!"
    
    # Verify the piece was actually removed from the C++ map
    pieces = board.get_pieces(player_id=2)
    assert not any((p == [6, -6, 0]).all() for p in pieces), "Piece was not removed from memory!"
    print("   [OK] Undo correctly reverses map and streak states.")

    print("\n========================================")
    print(" ALL FUNCTIONAL TESTS PASSED!")
    print("========================================")

if __name__ == "__main__":
    run_integration_tests()