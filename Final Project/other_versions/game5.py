from tictactoe5 import TicTacToe5
from agent import load_agent


def play_vs_computer(agent):
    env = TicTacToe5()
    print("\n--- 5x5 Tic-Tac-Toe (3-in-a-row wins) ---")
    print("Board positions:")
    for i in range(5):
        positions = "  ".join(f"{i*5+j+1:2d}" for j in range(5))
        print(f"   {positions}")

    while True:
        # Choose who goes first
        while True:
            choice = input("\nWho goes first? (p = player, a = agent): ").strip().lower()
            if choice in ("p", "a"):
                break
            print("Please enter 'p' or 'a'.")

        player_first = (choice == "p")
        if player_first:
            print("You go first! You are X  |  Computer is O")
        else:
            print("Agent goes first! Computer is X  |  You are O")

        env.reset()
        done = False
        player_turn = player_first

        while not done:
            if player_turn:
                # ── Player's turn ────────────────────────────────────────
                valid_move = False
                while not valid_move:
                    try:
                        move = int(input("Your move (1-25): ")) - 1
                        if move in env.available_moves():
                            valid_move = True
                        else:
                            print("That square is taken or out of range. Try again.")
                    except ValueError:
                        print("Please enter a number between 1 and 25.")

                _, done = env.make_move(move)
                env.print_board()

                if done:
                    player_id = 1 if player_first else -1
                    if env.check_winner(player_id):
                        print("You win!")
                    else:
                        print("It's a draw!")
                    break
            else:
                # ── Agent's turn ─────────────────────────────────────────
                state = env.get_state()
                action = agent.choose_action(state, env.available_moves())
                _, done = env.make_move(action)

                print(f"Computer plays position {action + 1}:")
                env.print_board()

                if done:
                    agent_id = 1 if not player_first else -1
                    if env.check_winner(agent_id):
                        print("Computer wins!")
                    else:
                        print("It's a draw!")
                    break

            player_turn = not player_turn

        if input("Play again? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    agent = load_agent("qtable5.pkl")
    play_vs_computer(agent)
