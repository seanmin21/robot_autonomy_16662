import random
from functools import lru_cache
from tictactoe import TicTacToe
from agent import QAgent, save_agent


# ── Minimax ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def minimax(state, is_maximizing):
    """
    Cached minimax over board states (tuples).
    Returns +1 if X wins, -1 if O wins, 0 for draw.
    """
    board = list(state)

    for p, score in [(1, 1), (-1, -1)]:
        for i in range(3):
            if all(board[i*3 + j] == p for j in range(3)):
                return score
            if all(board[j*3 + i] == p for j in range(3)):
                return score
        if board[0] == board[4] == board[8] == p:
            return score
        if board[2] == board[4] == board[6] == p:
            return score

    available = [i for i, v in enumerate(board) if v == 0]
    if not available:
        return 0

    if is_maximizing:
        best = -float("inf")
        for move in available:
            board[move] = 1
            best = max(best, minimax(tuple(board), False))
            board[move] = 0
        return best
    else:
        best = float("inf")
        for move in available:
            board[move] = -1
            best = min(best, minimax(tuple(board), True))
            board[move] = 0
        return best


def minimax_move(state, player):
    """
    Return the best move for the given player.
    player=1  → X, maximizing
    player=-1 → O, minimizing
    """
    board = list(state)
    available = [i for i, v in enumerate(board) if v == 0]

    if player == 1:
        best_score = -float("inf")
        best_action = None
        for move in available:
            board[move] = 1
            score = minimax(tuple(board), False)
            board[move] = 0
            if score > best_score:
                best_score = score
                best_action = move
    else:
        best_score = float("inf")
        best_action = None
        for move in available:
            board[move] = -1
            score = minimax(tuple(board), True)
            board[move] = 0
            if score < best_score:
                best_score = score
                best_action = move

    return best_action


# ── Training ──────────────────────────────────────────────────────────────────

def run_episode(agent, env, agent_player):
    """
    Run one training episode.
    agent_player=1  → agent is X (goes first)
    agent_player=-1 → agent is O (goes second)
    minimax plays as the other side.
    """
    env.reset()
    done = False
    prev_state = None
    prev_action = None

    while not done:
        state = env.get_state()

        if env.current_player == agent_player:
            # ── Agent's turn ─────────────────────────────────────────────
            action = agent.choose_action(state, env.available_moves())
            reward, done = env.make_move(action)
            next_state = env.get_state()

            if done:
                if reward == 1:
                    # Agent won
                    agent.learn(state, action, 1.0, next_state, done)
                else:
                    # Draw — agent made the last move
                    agent.learn(state, action, 0.5, next_state, done)
            else:
                # Game continues — neutral update
                agent.learn(state, action, 0.0, next_state, done)

            prev_state = state
            prev_action = action

        else:
            # ── Minimax's turn ───────────────────────────────────────────
            action = minimax_move(state, env.current_player)
            reward, done = env.make_move(action)
            next_state = env.get_state()

            if done and prev_state is not None:
                if reward == 1:
                    # Minimax won — penalise agent's last move
                    agent.learn(prev_state, prev_action, -1.0, next_state, done)
                elif reward == 0.5:
                    # Draw — minimax made the last move, still reward agent's last move
                    agent.learn(prev_state, prev_action, 0.5, next_state, done)


def train(episodes=50000):
    """
    Train the agent against minimax, alternating sides each episode so it
    learns to play well as both X (first) and O (second).
    """
    print(f"Training for {episodes} episodes against minimax...")
    print("Alternating sides each episode so agent learns as both X and O.\n")
    agent = QAgent(epsilon=0.3, alpha=0.5, gamma=0.9)
    env = TicTacToe()

    for i in range(episodes):
        # Decay epsilon from 0.3 down to 0.01 over training
        agent.epsilon = max(0.01, 0.3 * (1 - i / episodes))

        # Alternate: even episodes agent=X, odd episodes agent=O
        agent_player = 1 if i % 2 == 0 else -1
        run_episode(agent, env, agent_player)

        if i % 10000 == 0:
            print(f"  Episode {i:>6} | epsilon: {agent.epsilon:.3f} "
                  f"| Q-table states: {len(agent.q_table)}")

    print(f"\nTraining complete. Q-table has {len(agent.q_table)} states.")
    return agent


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(agent, n_games=500):
    """
    Evaluate the agent as both X and O against minimax.
    Against a perfect minimax, draws are the best achievable outcome.
    """
    env = TicTacToe()

    for agent_player, label in [(1, "agent=X (first)"), (-1, "agent=O (second)")]:
        print(f"\nEvaluating {n_games} games vs minimax — {label}...")
        results = {"win": 0, "draw": 0, "loss": 0}

        for _ in range(n_games):
            env.reset()
            done = False
            while not done:
                state = env.get_state()
                if env.current_player == agent_player:
                    action = agent.choose_action(state, env.available_moves())
                else:
                    action = minimax_move(state, env.current_player)
                reward, done = env.make_move(action)

            if reward == 0.5:
                results["draw"] += 1
            elif reward == 1 and env.current_player == agent_player:
                results["win"] += 1
            else:
                results["loss"] += 1

        total = n_games
        print(f"  Win:  {results['win']:>4}  ({100 * results['win'] / total:.1f}%)")
        print(f"  Draw: {results['draw']:>4}  ({100 * results['draw'] / total:.1f}%)")
        print(f"  Loss: {results['loss']:>4}  ({100 * results['loss'] / total:.1f}%)")

    print("\nAgainst a perfect minimax, 100% draws is the best achievable result.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trained_agent = train(episodes=50000)
    evaluate(trained_agent, n_games=500)
    save_agent(trained_agent, "qtable.pkl")