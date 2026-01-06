"""Train a tabular Q-learning agent against the maze environment."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from src.maze_rl.agent import QLearningAgent
from src.maze_rl.config import settings
from src.maze_rl.env import AugmentedState, MazeEnv



def run_training_episode(env: MazeEnv, agent: QLearningAgent) -> dict:
    env.reset()
    state = env.get_state()
    while not env.done:
        action = agent.choose_action(state, training=True)
        next_state, reward, done, info = env.step(action)
        agent.update_q_value(state, action, reward, next_state, done)
        state = next_state
    print(f"Training episode finished at state {state}")
    return info


def run_evaluation_episodes(
    env: MazeEnv, agent: QLearningAgent, num_episodes: int
) -> list[dict]:
    results: list[dict] = []
    for _ in range(num_episodes):
        print(f"Evaluation episode started")
        env.reset()
        state = env.get_state()
        while not env.done:
            action = agent.choose_action(state, training=False)
            print(f"Action: {action}")

            time.sleep(0)
            state, _, _, info = env.step(action)
            print(f"State: {env.get_state()}")
            print(f"Info: {info}")
        results.append(info)
        print(f"Evaluation episode finished")
    return results


def save_q_table_csv(q_table: dict[AugmentedState, np.ndarray], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row",
                "col",
                "neighbor_up_visited",
                "neighbor_down_visited",
                "neighbor_left_visited",
                "neighbor_right_visited",
                "value_up",
                "value_down",
                "value_left",
                "value_right",
                "value_stay",
            ]
        )
        for state, q_values in sorted(q_table.items()):
            writer.writerow([*state, *q_values.tolist()])


def main() -> None:
    env = MazeEnv(grid_path=settings.maze_path / "maze.csv")
    agent = QLearningAgent(
        learning_rate=settings.learning_rate,
        discount_factor=settings.discount_factor,
        epsilon_start=settings.epsilon_start,
        epsilon_end=settings.epsilon_end,
        epsilon_decay=settings.epsilon_decay,
    )
    best_score = float("-inf")
    best_model_path = settings.output_dir / "q_table_best.pkl"
    episodes_completed = 0

    for episode in range(1, settings.num_episodes + 1):
        info = run_training_episode(env, agent)
        episodes_completed = episode
        print(f"Training episode {episode} finished with score {info['final_score']}")
        if info["final_score"] > best_score:
            best_score = info["final_score"]
            save_q_table_csv(agent.q_table, settings.output_dir / "q_table_best.csv")

        agent.decay_epsilon()

        if episode % settings.log_frequency == 0:
            print(
                f"Episode {episode} | epsilon={agent.epsilon:.4f}"
                f"best score={best_score:.4f}"
                f"q-states={len(agent.q_table)}"
            )

        if episode % settings.eval_frequency == 0:
            print(f"Evaluation started at episode {episode}")
            eval_metrics = run_evaluation_episodes(env, agent, settings.eval_episodes)
            eval_scores = [m["final_score"] for m in eval_metrics]
            print(f"Evaluation complete. Mean score={np.mean(eval_scores):.4f}")

    print(
        f"Training complete ({episodes_completed} episodes). Best score={best_score:.4f} "
        f"(saved to {best_model_path})"
    )


if __name__ == "__main__":
    main()

