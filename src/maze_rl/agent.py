"""Simple agent implementations for bootstrapping experiments."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from src.maze_rl.env import AugmentedState, MazeEnv


class RandomWalkAgent:
    """Random agent that chooses a cardinal action uniformly."""

    def __init__(self, action_space: Sequence[int] = (0, 1, 2, 3)) -> None:
        self.action_space = action_space

    def choose_action(self, state: tuple[int, int], env: MazeEnv | None = None) -> int:
        """Return the next action; env is exposed for future extensions."""

        return random.choice(self.action_space)


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy action selection."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.num_actions = 5
        self.action_space = tuple(range(self.num_actions))
        self.q_table: dict[AugmentedState, np.ndarray] = {}

    def _get_or_init_state_q(self, state: AugmentedState) -> np.ndarray:
        if state not in self.q_table: # initialize the q-value for the state if it doesn't exist
            self.q_table[state] = np.zeros(self.num_actions, dtype=np.float64)
        return self.q_table[state]

    def get_q_value(self, state: AugmentedState, action: int) -> float:
        q_values = self._get_or_init_state_q(state) # get the q-value if it exists, otherwise initialize it to all zeros
        return float(q_values[action])

    def choose_action(self, state: AugmentedState, training: bool = True) -> int:
        q_values = self._get_or_init_state_q(state) # get the q-value if it exists, otherwise initialize it to all zeros

        if training and np.random.random() < self.epsilon:
            return random.choice(self.action_space)

        choices = np.flatnonzero(q_values == q_values.max())
        return random.choice(choices)

    def update_q_value(
        self,
        state: AugmentedState,
        action: int,
        reward: float,
        next_state: AugmentedState,
        done: bool,
    ) -> None:
        current_q = self.get_q_value(state, action)
        next_max = 0.0 if done else float(np.max(self._get_or_init_state_q(next_state)))
        target = reward + self.discount_factor * next_max
        updated_value = current_q + self.learning_rate * (target - current_q)
        self.q_table[state][action] = updated_value

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

