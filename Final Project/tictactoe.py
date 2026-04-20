import numpy as np


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O

    def get_state(self):
        return tuple(self.board.reshape(9))

    def available_moves(self):
        return [i for i, val in enumerate(self.board.reshape(9)) if val == 0]

    def make_move(self, action):
        row, col = divmod(action, 3)
        self.board[row, col] = self.current_player
        if self.check_winner(self.current_player):
            return 1, True
        if len(self.available_moves()) == 0:
            return 0.5, True  # Draw
        self.current_player *= -1
        return 0, False

    def check_winner(self, p):
        for i in range(3):
            if all(self.board[i, :] == p) or all(self.board[:, i] == p):
                return True
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == p:
            return True
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] == p:
            return True
        return False

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def print_board(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        print()
        for i, row in enumerate(self.board):
            cells = []
            for j, val in enumerate(row):
                idx = i * 3 + j
                cells.append(f"{symbols[val]}({idx})")
            print("  ".join(cells))
        print()
