"""Basic smoke tests for the maze scoring environment."""

from pathlib import Path

import pytest

from src.maze_rl.env import MazeEnv
from src.maze_rl.config import settings



def test_reset_initial_state() -> None:
    env = MazeEnv(grid_path=Path(settings.maze_path) / "maze.csv")
    state = env.reset()

    assert state == (25, 25)
    assert env.steps == 0
    assert env.unique_score == pytest.approx(env.grid[25, 25])


def test_boundary_completion() -> None:
    env = MazeEnv(grid_path=Path(settings.maze_path) / "maze.csv", start=(1, 1))
    env.reset()

    state, reward, done, info = env.step(0)  # move from row 1 to row 0

    assert state == (0, 1)
    assert reward == pytest.approx(env.grid[0, 1])
    assert done
    assert info["final_score"] == pytest.approx(env.final_score())
    assert env.steps == 1

