import numpy as np
import random
import pickle


class QAgent:
    def __init__(self, epsilon=0.3, alpha=0.5, gamma=0.9):
        self.q_table = {}       # state -> np.array of Q-values (size = board cells)
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(state))
        return self.q_table[state]

    def choose_action(self, state, available_moves):
        if random.random() < self.epsilon:
            return random.choice(available_moves)

        q_values = self.get_q_values(state)
        # Mask illegal moves so they are never chosen greedily
        masked_q = np.full(len(state), -np.inf)
        for move in available_moves:
            masked_q[move] = q_values[move]
        return int(np.argmax(masked_q))

    def learn(self, state, action, reward, next_state, done):
        old_q = self.get_q_values(state)[action]
        next_max = 0.0 if done else float(np.max(self.get_q_values(next_state)))
        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.q_table[state][action] = new_q


def save_agent(agent, filename="qtable.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(agent.q_table, f)
    print(f"Q-table saved to {filename}")


def load_agent(filename="qtable.pkl"):
    agent = QAgent(epsilon=0.0)  # epsilon=0 since we're loading a trained agent
    with open(filename, "rb") as f:
        agent.q_table = pickle.load(f)
    print(f"Q-table loaded from {filename} ({len(agent.q_table)} states)")
    return agent
