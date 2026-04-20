import numpy as np


class TicTacToe4:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O

    def get_state(self):
        return tuple(self.board.reshape(16))

    def available_moves(self):
        return [i for i, val in enumerate(self.board.reshape(16)) if val == 0]

    def make_move(self, action):
        row, col = divmod(action, 4)
        self.board[row, col] = self.current_player
        if self.check_winner(self.current_player):
            return 1, True
        if len(self.available_moves()) == 0:
            return 0.5, True  # Draw
        self.current_player *= -1
        return 0, False

    def check_winner(self, p):
        b = self.board
        # Rows: any 3 consecutive
        for r in range(4):
            for c in range(2):
                if b[r, c] == b[r, c+1] == b[r, c+2] == p:
                    return True
        # Columns: any 3 consecutive
        for c in range(4):
            for r in range(2):
                if b[r, c] == b[r+1, c] == b[r+2, c] == p:
                    return True
        # Diagonal top-left → bottom-right
        for r in range(2):
            for c in range(2):
                if b[r, c] == b[r+1, c+1] == b[r+2, c+2] == p:
                    return True
        # Diagonal top-right → bottom-left
        for r in range(2):
            for c in range(2, 4):
                if b[r, c] == b[r+1, c-1] == b[r+2, c-2] == p:
                    return True
        return False

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.current_player = 1

    def print_board(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        print()
        for i in range(4):
            row = self.board[i]
            cells = " | ".join(symbols[val] for val in row)
            positions = "  ".join(f"{i*4+j+1:2d}" for j in range(4))
            print(f"  {cells}    {positions}")
        print()
