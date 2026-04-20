from tictactoe import TicTacToe
from agent import load_agent


def play_vs_computer(agent):
    env = TicTacToe()
    print("\n--- Game Start ---")
    print("You are O  |  Computer is X")
    print("Board positions:")
    print("  0  1  2")
    print("  3  4  5")
    print("  6  7  8")

    while True:
        env.reset()
        done = False

        while not done:
            # ── Computer's turn (X = player 1) ──────────────────────────
            state = env.get_state()
            action = agent.choose_action(state, env.available_moves())
            reward, done = env.make_move(action)

            print(f"Computer (X) plays position {action}:")
            env.print_board()

            if done:
                if env.check_winner(1):
                    print("Computer wins!")
                else:
                    print("It's a draw!")
                break

            # ── Player's turn (O = player -1) ────────────────────────────
            valid_move = False
            while not valid_move:
                try:
                    move = int(input("Your move (0-8): "))
                    if move in env.available_moves():
                        valid_move = True
                    else:
                        print("That square is taken or out of range. Try again.")
                except ValueError:
                    print("Please enter a number between 0 and 8.")

            reward, done = env.make_move(move)
            env.print_board()

            if done:
                if env.check_winner(-1):
                    print("You win!")
                else:
                    print("It's a draw!")
                break

        if input("Play again? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    agent = load_agent("qtable.pkl")
    play_vs_computer(agent)
