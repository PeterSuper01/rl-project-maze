"""Train a tabular Q-learning agent against the maze environment."""

from __future__ import annotations

import math
import pickle
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from src.maze_rl.agent import QLearningAgent
from src.maze_rl.config import settings
from src.maze_rl.env import MazeEnv


@dataclass
class EpisodeMetrics:
    final_score: float
    steps: int
    unique_score: float


def compute_stats(values: Sequence[float]) -> tuple[float, float]:
    return np.mean(values), np.std(values)


def run_training_episode(env: MazeEnv, agent: QLearningAgent) -> EpisodeMetrics:
    state = env.reset()
    print(f"Training episode started at state {state}")
    while not env.done:
        action = agent.choose_action(state, training=True)
        next_state, reward, done, info = env.step(action)
        agent.update_q_value(state, action, reward, next_state, done)
        state = next_state
    print(f"Training episode finished at state {state}")
    return EpisodeMetrics(
        final_score=info["final_score"],
        steps=env.steps,
        unique_score=env.unique_score,
    )


def run_evaluation_episodes(
    env: MazeEnv, agent: QLearningAgent, num_episodes: int
) -> list[EpisodeMetrics]:
    results: list[EpisodeMetrics] = []
    for _ in range(num_episodes):
        print(f"Evaluation episode started")
        state = env.reset()
        while not env.done:
            action = agent.choose_action(state, training=False)
            state, _, _, info = env.step(action)
        results.append(EpisodeMetrics(final_score=info["final_score"], steps=env.steps, unique_score=env.unique_score))
        print(f"Evaluation episode finished")
    return results


def save_q_table(
    agent: QLearningAgent, output_path, episode: int, final_score: float
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "q_table": agent.q_table,
        "metadata": {
            "episode": episode,
            "final_score": final_score,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "hyperparameters": {
                "learning_rate": settings.learning_rate,
                "discount_factor": settings.discount_factor,
                "epsilon_start": settings.epsilon_start,
                "epsilon_end": settings.epsilon_end,
                "epsilon_decay": settings.epsilon_decay,
            },
        },
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)


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
    recent_scores: deque[float] = deque(maxlen=settings.log_frequency)
    recent_steps: deque[float] = deque(maxlen=settings.log_frequency)
    unique_final_scores: set[float] = set()
    episodes_completed = 0

    try:
        for episode in range(1, settings.num_episodes + 1):
            metrics = run_training_episode(env, agent)
            episodes_completed = episode
            recent_scores.append(metrics.final_score)
            recent_steps.append(metrics.steps)
            unique_final_scores.add(metrics.final_score)
            if metrics.final_score > best_score:
                best_score = metrics.final_score
                save_q_table(agent, best_model_path, episode, best_score)

            agent.decay_epsilon()

            if settings.log_frequency > 0 and episode % settings.log_frequency == 0:
                score_mean, score_std = compute_stats(recent_scores)
                step_mean, step_std = compute_stats(recent_steps)
                print(
                    f"Episode {episode} | ε={agent.epsilon:.4f} | recent final score mean={score_mean:.4f} ± {score_std:.4f} | "
                    f"steps mean={step_mean:.1f} ± {step_std:.1f} | best score={best_score:.4f} | "
                    f"q-states={len(agent.q_table)} | unique finals={len(unique_final_scores)}"
                )

            if settings.eval_frequency > 0 and episode % settings.eval_frequency == 0:
                print(f"Evaluation started at episode {episode}")
                eval_metrics = run_evaluation_episodes(env, agent, settings.eval_episodes)
                eval_scores = [m.final_score for m in eval_metrics]
                eval_mean, eval_std = compute_stats(eval_scores)
                print(
                    f"Evaluation after episode {episode}: greedy mean score={eval_mean:.4f} ± {eval_std:.4f} over {len(eval_scores)} episodes"
                )
    except KeyboardInterrupt:
        print(f"Training interrupted at episode {episodes_completed}.")

    print(
        f"Training complete ({episodes_completed} episodes). Best score={best_score:.4f} "
        f"(saved to {best_model_path})"
    )


if __name__ == "__main__":
    main()

