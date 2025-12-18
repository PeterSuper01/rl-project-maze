"""Environment that encodes the maze scoring rules from the final exam prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

Coordinate = Tuple[int, int]

_ACTION_DELTAS: dict[int, Coordinate] = {
    0: (-1, 0),  # up
    1: (1, 0),  # down
    2: (0, -1),  # left
    3: (0, 1),  # right
}


def load_grid_from_csv(csv_path: Path | str) -> np.ndarray:
    """Load the score grid from `maze.csv` and return it as a numpy array."""

    df = pd.read_csv(csv_path, index_col=0)
    grid = df.to_numpy(dtype=np.float64)
    return grid


class MazeEnv:
    """Grid environment where visits accumulate unique scores and boundary hits end the game."""

    def __init__(
        self,
        grid_path: Path | str,
        start: Coordinate = (25, 25),
        max_steps: int | None = None,
    ) -> None:
        self.grid = load_grid_from_csv(grid_path)
        self.start = start
        self.max_steps = max_steps
        self.grid_min = 0
        self.grid_max = self.grid.shape[0] - 1
        self.grid_size = self.grid_max - self.grid_min + 1
        self.reset()

    def reset(self) -> Coordinate:
        """Reset the environment to the starting cell."""

        if not self._in_bounds(self.start):
            raise ValueError("starting coordinate is outside the allowed grid")

        self._state = self.start
        self.visited: set[Coordinate] = {self.start}
        self.unique_score = float(self.grid[self.start])
        self.steps = 0
        self.done = False
        return self._state

    @property
    def state(self) -> Coordinate:
        return self._state

    def step(self, action: int) -> Tuple[Coordinate, float, bool, dict]:
        """Take a step in the grid and return (state, reward, done, info)."""

        if self.done:
            raise RuntimeError("cannot step after episode is done")

        if action not in _ACTION_DELTAS:
            raise ValueError(f"invalid action {action}")

        delta = _ACTION_DELTAS[action]
        next_state = (self._state[0] + delta[0], self._state[1] + delta[1])
        if not self._in_bounds(next_state):
            raise ValueError("move would leave the grid")

        self._state = next_state
        self.steps += 1

        reward = 0.0
        if next_state not in self.visited:
            reward = float(self.grid[next_state])
            self.visited.add(next_state)
            self.unique_score += reward

        self.done = self._hits_boundary(next_state) or (
            self.max_steps is not None and self.steps >= self.max_steps
        )

        info = {
            "unique_score": self.unique_score,
            "steps": self.steps,
        }
        if self.done:
            info["final_score"] = self.final_score()

        return next_state, reward, self.done, info

    def final_score(self) -> float:
        """Compute the exam score formula once the episode finishes."""

        return (self.unique_score**1.2) - (self.steps**1.5)

    def _in_bounds(self, state: Coordinate) -> bool:
        return self.grid_min <= state[0] <= self.grid_max and self.grid_min <= state[1] <= self.grid_max

    def _hits_boundary(self, state: Coordinate) -> bool:
        return state[0] in {self.grid_min, self.grid_max} or state[1] in {self.grid_min, self.grid_max}

