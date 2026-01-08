from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    maze_path: Path
    score_exponent: float = 1.0
    step_exponent: float = 1.0
    num_episodes: int = 2500
    learning_rate: float = 0.3
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    eval_frequency: int = 100
    eval_episodes: int = 10
    log_frequency: int = 200
    output_dir: Path = Path("models")

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
