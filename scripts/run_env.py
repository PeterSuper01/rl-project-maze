"""Utility script that runs a random walk until the episode terminates."""

from __future__ import annotations

from pathlib import Path

from src.maze_rl.env import MazeEnv
from src.maze_rl.agent import RandomWalkAgent
from src.maze_rl.config import settings

def main() -> None:
    env = MazeEnv(grid_path=Path(settings.maze_path) / "maze.csv")
    agent = RandomWalkAgent()
    
    state = env.reset()
    while not env.done:
        action = agent.choose_action(state, env)
        state, _, _, _ = env.step(action)

    print("Episode finished.")
    print(f"Steps taken: {env.steps}")
    print(f"Unique score: {env.unique_score:.4f}")
    print(f"Final exam score: {env.final_score():.4f}")


if __name__ == "__main__":
    main()

