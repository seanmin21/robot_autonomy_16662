import random
from tictactoe5 import TicTacToe5
from agent import QAgent, save_agent


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_winner_flat(board, p):
    """Check 3-in-a-row win on a flat 25-element list for a 5x5 board."""
    # Rows
    for r in range(5):
        for c in range(3):
            if board[r*5+c] == board[r*5+c+1] == board[r*5+c+2] == p:
                return True
    # Columns
    for c in range(5):
        for r in range(3):
            if board[r*5+c] == board[(r+1)*5+c] == board[(r+2)*5+c] == p:
                return True
    # Diagonal top-left → bottom-right
    for r in range(3):
        for c in range(3):
            if board[r*5+c] == board[(r+1)*5+c+1] == board[(r+2)*5+c+2] == p:
                return True
    # Diagonal top-right → bottom-left
    for r in range(3):
        for c in range(2, 5):
            if board[r*5+c] == board[(r+1)*5+c-1] == board[(r+2)*5+c-2] == p:
                return True
    return False


def winning_moves(board_tuple, player):
    """Return positions where player can win immediately."""
    board = list(board_tuple)
    hits = []
    for i, v in enumerate(board):
        if v != 0:
            continue
        board[i] = player
        if _is_winner_flat(board, player):
            hits.append(i)
        board[i] = 0
    return hits


def shaped_reward(state, action, agent_player):
    """
    Mid-game reward shaping:
      +0.5 if this move blocks the opponent's immediate win
      +0.3 if this move creates our own immediate winning threat
    """
    board = list(state)
    opponent = -agent_player

    if action in winning_moves(state, opponent):
        return 0.5

    board[action] = agent_player
    if winning_moves(tuple(board), agent_player):
        return 0.3

    return 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def run_episode(agent, env, agent_player, opponent="random"):
    """
    Run one episode.
    opponent: "random"     — random legal moves
              "self"       — same agent (both sides learn)
    """
    env.reset()
    done = False
    prev_state = None
    prev_action = None

    while not done:
        state = env.get_state()

        if env.current_player == agent_player:
            action = agent.choose_action(state, env.available_moves())
            reward, done = env.make_move(action)
            next_state = env.get_state()

            if done:
                r = 1.0 if reward == 1 else 0.5
                agent.learn(state, action, r, next_state, done)
            else:
                r = shaped_reward(state, action, agent_player)
                agent.learn(state, action, r, next_state, done)

            prev_state = state
            prev_action = action

        else:
            # Opponent's turn
            if opponent == "self":
                action = agent.choose_action(state, env.available_moves())
            else:
                action = random.choice(env.available_moves())

            reward, done = env.make_move(action)
            next_state = env.get_state()

            if done and prev_state is not None:
                if reward == 1:
                    agent.learn(prev_state, prev_action, -1.0, next_state, done)
                elif reward == 0.5:
                    agent.learn(prev_state, prev_action, 0.5, next_state, done)


def train(episodes=500000):
    """
    Train the agent on 5x5 Tic-Tac-Toe (3-in-a-row wins).
    No minimax (state space is too large for 5x5).
    Uses self-play and random opponents, alternating sides each episode.
    """
    print(f"Training for {episodes} episodes on 5x5 board (3-in-a-row wins)...")
    print("50% self-play, 50% vs random. Alternating X/O each episode.\n")
    agent = QAgent(epsilon=0.5, alpha=0.4, gamma=0.95)
    env = TicTacToe5()

    for i in range(episodes):
        agent.epsilon = max(0.05, 0.5 * (1 - i / episodes))
        agent_player = 1 if i % 2 == 0 else -1
        opponent = "self" if i % 4 < 2 else "random"
        run_episode(agent, env, agent_player, opponent)

        if i % 100000 == 0:
            print(f"  Episode {i:>7} | epsilon: {agent.epsilon:.3f} "
                  f"| Q-table states: {len(agent.q_table)}")

    print(f"\nTraining complete. Q-table has {len(agent.q_table)} states.")
    return agent


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(agent, n_games=500):
    env = TicTacToe5()

    for agent_player, label in [(1, "agent=X (first)"), (-1, "agent=O (second)")]:
        print(f"\nEvaluating {n_games} games vs random — {label}...")
        results = {"win": 0, "draw": 0, "loss": 0}

        for _ in range(n_games):
            env.reset()
            done = False
            while not done:
                state = env.get_state()
                if env.current_player == agent_player:
                    action = agent.choose_action(state, env.available_moves())
                else:
                    action = random.choice(env.available_moves())
                reward, done = env.make_move(action)

            if reward == 0.5:
                results["draw"] += 1
            elif reward == 1 and env.current_player == agent_player:
                results["win"] += 1
            else:
                results["loss"] += 1

        total = n_games
        print(f"  Win:  {results['win']:>4}  ({100*results['win']/total:.1f}%)")
        print(f"  Draw: {results['draw']:>4}  ({100*results['draw']/total:.1f}%)")
        print(f"  Loss: {results['loss']:>4}  ({100*results['loss']/total:.1f}%)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trained_agent = train(episodes=500000)
    evaluate(trained_agent, n_games=500)
    save_agent(trained_agent, "qtable5.pkl")
