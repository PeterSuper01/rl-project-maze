I’m planning a reinforcement-learning prototype for the statistical-computing
“maze score” problem described in the exam prompt.

## Problem
- The agent starts at `(25, 25)` on a `51×51` grid whose cells contain floating-
  point scores (see `maze.csv`).
- At each step the agent moves in one of the four cardinal directions. Once the
  agent reaches the boundary (a coordinate becomes `0` or `50`), the episode
  ends immediately and no moves beyond the grid are permitted.
- The cumulative score adds `s₍ᵢⱼ₎` only the first time cell `(i, j)` is visited.
- Final exam score is `(sum of unique values)¹·² − (number of steps)¹·⁵`.

## Repository layout

```
maze/
├── maze.csv             # provided grid scores (51 rows + header)
├── README.md            # this project overview
├── pyproject.toml       # python project metadata + deps
├── src/maze_rl/         # package implementing env + helpers
│   ├── __init__.py
│   ├── env.py
│   └── agent.py
├── scripts/             # runnable helpers for prototyping
│   └── run_env.py
└── tests/               # smoke tests for the environment
    └── test_env.py
```

## Getting started

1. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```
2. Run `python -m maze_rl.env` or `python scripts/run_env.py` to sample a trajectory
   from the random agent.
3. Extend an RL trainer (e.g., PPO, DQN) against `maze_rl.env.MazeEnv`.

## Next steps

- Wire up `gymnasium`/`ray[rllib]` wrappers for policy training.
- Experiment with sparse-vs-dense reward shaping based on `(sum of unique)¹·²`.
- Log trajectories to analyze which regions maximize the exam score.