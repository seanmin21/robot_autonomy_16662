from tictactoe import TicTacToe
from agent import load_agent
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm

data = {
    "O1": [0.02717936, 0.32582323, 0.65109116, -2.93055366, 0.66739343, 3.32395032, 0.85388163],
    "O2": [0.08395651, 0.3653481, 0.338247, -2.65635308, 0.26792806, 3.01023283, 1.00907123],
    "O3": [0.09566458, 0.46567056, 0.22418855, -2.35520924, -0.20112449, 2.74888108, 1.33172064],
    "O4": [0.08222077, 0.65433193, 0.20484338, -1.9788316, -0.19975675, 2.57673945, 1.29079593],
    "O5": [0.07025379, 0.86675108, 0.21159245, -1.58441414, -0.18814389, 2.41977992, 1.23266756],

    "X1": [-0.56692782, 0.26172966, -0.06551953, -2.91744644, -0.19064872, 3.20180097, 0.33022845],
    "X2": [-0.54618934, 0.36615621, 0.06390144, -2.64768171, -0.1907015, 3.03291371, 0.51971415],
    "X3": [-0.54493342, 0.47661166, 0.14145441, -2.37585637, -0.19023456, 2.85898783, 0.58248814],
    "X4": [-0.51914914, 0.64345506, 0.18006523, -2.0467975, -0.19027603, 2.70032397, 0.6094833],
    "X5": [-0.43296065, 0.86525112, 0.18517362, -1.5995602, -0.2014892, 2.42366233, 0.63776388],

    "1": [-0.45067738, 0.28265142, 0.16382064, -2.80662844, -0.20560017, 3.09522544, 0.70419669],
    "2": [3.25696704e-02, 2.30555232e-01, -5.79317385e-03, -2.81683092e+00, -1.12204384e-03, 3.02052418e+00, 8.26276025e-01],
    "3": [9.61997021e-02, 2.71488161e-01, 1.39782616e-01, -2.77566572e+00, -1.15918533e-03, 3.01099066e+00, 1.02160341e+00],

    "4": [-0.08850364,  0.38954701, -0.09623303, -2.50753685,  0.00928853,  2.77477469, 0.63382277],

    "5": [-8.95488058e-03, 3.72719800e-01, 1.98762461e-02, -2.54039833e+00, -1.14541456e-03, 2.81989123e+00, 7.92279811e-01],

    "6": [0.0698005, 0.38442215, 0.11928718, -2.50846871, -0.09320199, 2.76728073, 1.06165961],

    "7": [-0.08701677,  0.49184465, -0.07721444, -2.33998312,  0.0094605,   2.88346292, 0.62935258],
    "8": [-0.02909579, 0.48906672, 0.03062725, -2.30645564, -0.09050243, 2.75880875, 0.88375314],

    "9": [0.03216658, 0.53120889, 0.13037212, -2.2488452, -0.09200739, 2.70056456, 1.04918757],
}

X_num = 1
O_num = 1

def execute_cmd(robot, cmd):
    global X_num, O_num
    # cmd id in form "X1", "O2"
    if cmd[0] == "X":
        robot.reset_joints()
        idx = cmd[1]
        robot.goto_joints(data["X" + str(X_num)])
        robot.close_gripper()
        robot.reset_joints()
        robot.goto_joints(data[idx])
        robot.open_gripper()
        robot.reset_joints()
        X_num += 1
    elif cmd[0] == "O":
        robot.reset_joints()
        idx = cmd[1]
        robot.goto_joints(data["O" + str(O_num)])
        robot.close_gripper()
        robot.reset_joints()
        robot.goto_joints(data[idx])
        robot.open_gripper()
        robot.reset_joints()
        O_num += 1
    
    


def play_vs_computer(agent):
    env = TicTacToe()
    print("\n--- Game Start ---")
    print("Board positions:")
    print("  1 | 2 | 3")
    print("  4 | 5 | 6")
    print("  7 | 8 | 9")

    robot = FrankaArm()

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
                        move = int(input("Your move (1-9): ")) - 1
                        if move in env.available_moves():
                            valid_move = True
                        else:
                            print("That square is taken or out of range. Try again.")
                    except ValueError:
                        print("Please enter a number between 1 and 9.")
                print(move)
                cmd = f"X{move + 1}" if player_first else f"O{move + 1}"
                execute_cmd(robot, cmd)
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
                print(action)
                cmd = f"X{action + 1}" if not player_first else f"O{action + 1}"
                execute_cmd(robot, cmd)
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

        # if input("Play again? (y/n): ").strip().lower() != "y":
        break


if __name__ == "__main__":
    agent = load_agent("qtable.pkl")
    play_vs_computer(agent)
