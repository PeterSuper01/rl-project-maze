"""Simple agent implementations for bootstrapping experiments."""

from __future__ import annotations

import random
from typing import Sequence

from src.maze_rl.env import MazeEnv


class RandomWalkAgent:
    """Random agent that chooses a cardinal action uniformly."""

    def __init__(self, action_space: Sequence[int] = (0, 1, 2, 3)) -> None:
        self.action_space = action_space

    def choose_action(self, state: tuple[int, int], env: MazeEnv | None = None) -> int:
        """Return the next action; env is exposed for future extensions."""

        return random.choice(self.action_space)

