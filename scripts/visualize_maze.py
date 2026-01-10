from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.maze_rl.config import settings
from src.maze_rl.env import Coordinate, load_grid_from_csv


def visualize_maze(save_path: Path) -> None:
    """Render a heatmap of the maze grid with boundary and start annotations."""

    grid_path = Path(settings.maze_path) / "maze_plus_1.csv"
    grid = load_grid_from_csv(grid_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    heatmap = ax.imshow(grid, cmap="coolwarm", origin="upper")
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label("Cell value", rotation=270, labelpad=15)

    # Mark cells with negative values
    negative_mask = grid < 0
    negative_rows, negative_cols = np.where(negative_mask)
    if len(negative_rows) > 0:
        ax.scatter(
            negative_cols,
            negative_rows,
            marker="x",
            color="yellow",
            s=100,
            linewidths=2,
            zorder=2,
            label="negative values",
        )

    start_row, start_col = 25, 25
    ax.scatter(
        start_col,
        start_row,
        marker="*",
        color="red",
        edgecolor="black",
        s=200,
        zorder=3,
        label="start position",
    )

    ax.set_title("Maze Grid Heatmap")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks([0, grid.shape[1] - 1])
    ax.set_yticks([0, grid.shape[0] - 1])
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved maze visualization to {save_path}")


def plot_path_on_maze(path: list[Coordinate], save_path: Path) -> None:
    """Render a heatmap of the maze grid with the agent's path overlaid."""
    
    grid_path = Path(settings.maze_path) / "maze_plus_1.csv"
    grid = load_grid_from_csv(grid_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    heatmap = ax.imshow(grid, cmap="coolwarm", origin="upper")
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label("Cell value", rotation=270, labelpad=15)
    
    # Mark cells with negative values
    negative_mask = grid < 0
    negative_rows, negative_cols = np.where(negative_mask)
    if len(negative_rows) > 0:
        ax.scatter(
            negative_cols,
            negative_rows,
            marker="x",
            color="yellow",
            s=100,
            linewidths=2,
            zorder=2,
            label="negative values",
        )
    
    # Plot the path as a polyline
    if len(path) > 0:
        path_cols = [col for row, col in path]
        path_rows = [row for row, col in path]
        ax.plot(
            path_cols,
            path_rows,
            color="lime",
            linewidth=2,
            marker="o",
            markersize=4,
            zorder=3,
            label="agent path",
        )
        
        # Mark start position
        start_row, start_col = path[0]
        ax.scatter(
            start_col,
            start_row,
            marker="*",
            color="red",
            edgecolor="black",
            s=200,
            zorder=4,
            label="start position",
        )
        
        # Mark end position
        if len(path) > 1:
            end_row, end_col = path[-1]
            ax.scatter(
                end_col,
                end_row,
                marker="s",
                color="blue",
                edgecolor="black",
                s=200,
                zorder=4,
                label="end position",
            )
    
    ax.set_title("Maze Grid Heatmap with Agent Path")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks([0, grid.shape[1] - 1])
    ax.set_yticks([0, grid.shape[0] - 1])
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved path visualization to {save_path}")


if __name__ == "__main__":
    visualize_maze(save_path=Path(settings.maze_path) / "maze_visualization.png")
