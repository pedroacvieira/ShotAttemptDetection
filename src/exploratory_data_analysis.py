#!/usr/bin/env python3
"""
Exploratory Data Analysis for Shot Attempt Detection

This script loads and visualizes the basketball shot detection data to understand
the patterns and characteristics of the dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes
import seaborn as sns
import typer
from pathlib import Path
from typing import Optional, List

from src.preprocess import load_data
from src.features import extract_features_for_player


app = typer.Typer(help="Exploratory Data Analysis for Shot Attempt Detection")


def get_player_shot_times(shot_events_df: pd.DataFrame, player_id: str) -> List[float]:
    """Extract shot timestamps for a specific player.

    Args:
        shot_events_df: DataFrame with shot events
        player_id: Player ID to filter

    Returns:
        List of shot timestamps in seconds
    """
    if shot_events_df.empty or "player_id" not in shot_events_df.columns:
        return []
    return shot_events_df[shot_events_df["player_id"] == player_id]["timestamp_s"].values.tolist()


def mark_shot_attempts(ax: matplotlib.axes.Axes, shot_times: List[float]) -> None:
    """Mark shot attempt times on a plot with vertical lines.

    Args:
        ax: Matplotlib axes object
        shot_times: List of shot timestamps to mark
    """
    for shot_time in shot_times:
        ax.axvline(x=shot_time, color="red", linestyle="--", alpha=0.7, linewidth=2)


def plot_limb_distances(
    features_df: pd.DataFrame, shot_events_df: pd.DataFrame, player_id: str = "0", rolling_window_ms: float = 500.0
) -> None:
    """Plot hand distance and heel distance over time for a specific player.

    Args:
        features_df: DataFrame with extracted features
        shot_events_df: DataFrame with shot events
        player_id: Player ID to visualize
        rolling_window_ms: Rolling window size used for feature extraction
    """
    if features_df.empty:
        print(f"No data found for player {player_id}")
        return

    player_shots = get_player_shot_times(shot_events_df, player_id)
    fig, axes = plt.subplots(2, 1, figsize=(30, 20))

    # Plot 1: Hand distance
    axes[0].plot(
        features_df["timestamp_s"],
        features_df["hand_distance"],
        label="Hand Distance",
        alpha=0.7,
        color="green",
        linewidth=2,
    )
    axes[0].plot(
        features_df["timestamp_s"],
        features_df[f"hand_distance_rolling_max_{int(rolling_window_ms)}ms"],
        label=f"Hand Distance (rolling max {int(rolling_window_ms)}ms)",
        alpha=0.5,
        linestyle="--",
        color="darkgreen",
        linewidth=2,
    )
    axes[0].set_title(f"Hand Separation (Distance) - Player {player_id}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Hand Distance (normalized)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    mark_shot_attempts(axes[0], player_shots)

    # Plot 2: Heel distance
    axes[1].plot(
        features_df["timestamp_s"],
        features_df["heel_distance"],
        label="Heel Distance (normalized)",
        alpha=0.7,
        color="purple",
        linewidth=2,
    )
    axes[1].plot(
        features_df["timestamp_s"],
        features_df[f"heel_distance_rolling_max_{int(rolling_window_ms)}ms"],
        label=f"Heel Distance (rolling max {int(rolling_window_ms)}ms)",
        alpha=0.5,
        linestyle="--",
        color="indigo",
        linewidth=2,
    )
    axes[1].set_title(f"Heel Distance - Player {player_id}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Distance (normalized by bbox height)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    mark_shot_attempts(axes[1], player_shots)

    plt.tight_layout()
    plt.savefig(f"plots/player_{player_id}_limb_distances.png")


def plot_hand_rel_body(
    features_df: pd.DataFrame, shot_events_df: pd.DataFrame, player_id: str = "0", rolling_window_ms: float = 1200.0
) -> None:
    """Plot hand positions and velocities relative to body (head) and hand-hip ratios.

    Args:
        features_df: DataFrame with extracted features
        shot_events_df: DataFrame with shot events
        player_id: Player ID to visualize
        rolling_window_ms: Rolling window size used for feature extraction
    """
    if features_df.empty:
        print(f"No data found for player {player_id}")
        return

    player_shots = get_player_shot_times(shot_events_df, player_id)

    fig, axes = plt.subplots(3, 1, figsize=(30, 30))

    # Plot 1: Hand Y position relative to head
    axes[0].plot(
        features_df["timestamp_s"],
        features_df["left_hand_y_rel_head"],
        label="Left Hand Y rel. Head",
        alpha=0.7,
        color="blue",
        linewidth=2,
    )
    axes[0].plot(
        features_df["timestamp_s"],
        features_df["right_hand_y_rel_head"],
        label="Right Hand Y rel. Head",
        alpha=0.7,
        color="orange",
        linewidth=2,
    )
    axes[0].plot(
        features_df["timestamp_s"],
        features_df[f"left_hand_y_rel_head_rolling_max_{int(rolling_window_ms)}ms"],
        label=f"Left Hand Y rel. Head (rolling max {int(rolling_window_ms)}ms)",
        alpha=0.5,
        linestyle="--",
        color="darkblue",
        linewidth=2,
    )
    axes[0].plot(
        features_df["timestamp_s"],
        features_df[f"right_hand_y_rel_head_rolling_max_{int(rolling_window_ms)}ms"],
        label=f"Right Hand Y rel. Head (rolling max {int(rolling_window_ms)}ms)",
        alpha=0.5,
        linestyle="--",
        color="darkorange",
        linewidth=2,
    )
    axes[0].set_title(f"Hand Y Position relative to Head - Player {player_id}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Relative Y Position (normalized)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    mark_shot_attempts(axes[0], player_shots)

    # Plot 2: Hand Y velocity relative to head
    axes[1].plot(
        features_df["timestamp_s"],
        features_df["left_hand_y_velocity_rel_head"],
        label="Left Hand Y Velocity (rel. Head)",
        alpha=0.7,
        color="blue",
        linewidth=2,
    )
    axes[1].plot(
        features_df["timestamp_s"],
        features_df["right_hand_y_velocity_rel_head"],
        label="Right Hand Y Velocity (rel. Head)",
        alpha=0.7,
        color="purple",
        linewidth=2,
    )
    axes[1].plot(
        features_df["timestamp_s"],
        features_df[f"left_hand_y_velocity_rel_head_rolling_max_{int(rolling_window_ms)}ms"],
        label=f"Left Hand Y Velocity (rolling max {int(rolling_window_ms)}ms)",
        alpha=0.5,
        linestyle="--",
        color="darkblue",
        linewidth=2,
    )
    axes[1].plot(
        features_df["timestamp_s"],
        features_df[f"right_hand_y_velocity_rel_head_rolling_max_{int(rolling_window_ms)}ms"],
        label=f"Right Hand Y Velocity (rolling max {int(rolling_window_ms)}ms)",
        alpha=0.5,
        linestyle="--",
        color="indigo",
        linewidth=2,
    )
    axes[1].set_title(f"Hand Y Velocity relative to Head - Player {player_id}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Y Velocity (normalized)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    mark_shot_attempts(axes[1], player_shots)

    # Plot 3: Hand-hip ratios
    axes[2].plot(
        features_df["timestamp_s"],
        features_df["left_hand_hip_ratio"],
        label="Left Hand / Hip Ratio",
        alpha=0.7,
        color="blue",
        linewidth=2,
    )
    axes[2].plot(
        features_df["timestamp_s"],
        features_df["right_hand_hip_ratio"],
        label="Right Hand / Hip Ratio",
        alpha=0.7,
        color="purple",
        linewidth=2,
    )
    axes[2].set_title(f"Hand-Hip Position Ratios - Player {player_id}")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Ratio")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    mark_shot_attempts(axes[2], player_shots)

    plt.tight_layout()
    plt.savefig(f"plots/player_{player_id}_hand_rel_body.png")


def plot_hand_rel_bbox(features_df: pd.DataFrame, shot_events_df: pd.DataFrame, player_id: str = "0") -> None:
    """Plot hand Y positions relative to bounding box for a specific player.

    Args:
        features_df: DataFrame with extracted features
        shot_events_df: DataFrame with shot events
        player_id: Player ID to visualize
    """
    if features_df.empty:
        print(f"No data found for player {player_id}")
        return

    player_shots = get_player_shot_times(shot_events_df, player_id)

    fig, axes = plt.subplots(2, 1, figsize=(30, 20))

    # Plot 1: Left Hand Y position relative to bbox
    axes[0].plot(
        features_df["timestamp_s"],
        features_df["left_hand_y_to_bbox"],
        label="Left Hand Y to Bbox",
        alpha=0.7,
        color="blue",
        linewidth=2,
    )
    axes[0].set_title(f"Left Hand Y Position (relative to Bounding Box) - Player {player_id}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Normalized Position")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    mark_shot_attempts(axes[0], player_shots)

    # Plot 2: Right Hand Y position relative to bbox
    axes[1].plot(
        features_df["timestamp_s"],
        features_df["right_hand_y_to_bbox"],
        label="Right Hand Y to Bbox",
        alpha=0.7,
        color="purple",
        linewidth=2,
    )
    axes[1].set_title(f"Right Hand Y Position (relative to Bounding Box) - Player {player_id}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Normalized Position")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    mark_shot_attempts(axes[1], player_shots)

    plt.tight_layout()
    plt.savefig(f"plots/player_{player_id}_hand_rel_bbox.png")


def plot_hip_rel_bbox(
    features_df: pd.DataFrame, shot_events_df: pd.DataFrame, player_id: str = "0", rolling_window_ms: float = 500.0
) -> None:
    """Plot hip Y position and speed relative to bounding box for a specific player.

    Args:
        features_df: DataFrame with extracted features
        shot_events_df: DataFrame with shot events
        player_id: Player ID to visualize
        rolling_window_ms: Rolling window size used for feature extraction
    """
    if features_df.empty:
        print(f"No data found for player {player_id}")
        return

    player_shots = get_player_shot_times(shot_events_df, player_id)

    fig, axes = plt.subplots(2, 1, figsize=(30, 20))

    # Plot 1: Hip Y position to bbox
    axes[0].plot(
        features_df["timestamp_s"],
        features_df["hip_y_to_bbox"],
        label="Hip Y to Bbox",
        alpha=0.7,
        color="orange",
        linewidth=2,
    )
    axes[0].set_title(f"Hip Y Position (relative to Bounding Box) - Player {player_id}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Normalized Position")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    mark_shot_attempts(axes[0], player_shots)

    # Plot 2: Hip speed to bbox
    axes[1].plot(
        features_df["timestamp_s"],
        features_df["hip_speed_to_bbox"],
        label="Hip Speed (norm. by bbox)",
        alpha=0.7,
        color="blue",
        linewidth=2,
    )
    axes[1].plot(
        features_df["timestamp_s"],
        features_df[f"hip_speed_to_bbox_rolling_max_{int(rolling_window_ms)}ms"],
        label=f"Hip Speed (rolling max {int(rolling_window_ms)}ms)",
        alpha=0.5,
        linestyle="--",
        color="darkblue",
        linewidth=2,
    )
    axes[1].set_title(f"Hip Speed (normalized by BBox Height) - Player {player_id}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Speed (normalized)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    mark_shot_attempts(axes[1], player_shots)

    plt.tight_layout()
    plt.savefig(f"plots/player_{player_id}_hip_rel_bbox.png")


def plot_player_speed(
    features_df: pd.DataFrame, shot_events_df: pd.DataFrame, player_id: str = "0", rolling_window_ms: float = 500.0
) -> None:
    """Plot player absolute speed (XY) over time for a specific player.

    Args:
        features_df: DataFrame with extracted features
        shot_events_df: DataFrame with shot events
        player_id: Player ID to visualize
        rolling_window_ms: Rolling window size used for feature extraction
    """
    if features_df.empty:
        print(f"No data found for player {player_id}")
        return

    player_shots = get_player_shot_times(shot_events_df, player_id)

    fig, ax = plt.subplots(1, 1, figsize=(30, 10))

    # Plot: Player absolute speed
    ax.plot(
        features_df["timestamp_s"],
        features_df["player_abs_speed"],
        label="Player Absolute Speed",
        alpha=0.7,
        color="green",
        linewidth=2,
    )
    ax.plot(
        features_df["timestamp_s"],
        features_df[f"player_abs_speed_rolling_max_{int(rolling_window_ms)}ms"],
        label=f"Player Speed (rolling max {int(rolling_window_ms)}ms)",
        alpha=0.5,
        linestyle="--",
        color="darkgreen",
        linewidth=2,
    )
    ax.set_title(f"Player Absolute Speed (XY) - Player {player_id}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    mark_shot_attempts(ax, player_shots)

    plt.tight_layout()
    plt.savefig(f"plots/player_{player_id}_player_speed.png")


def plot_shot_timing_analysis(shot_events_df: pd.DataFrame) -> None:
    """Analyze and visualize shot timing patterns.

    Args:
        shot_events_df: DataFrame with shot events including timestamps and player IDs
    """
    if shot_events_df.empty:
        print("No shot events found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Shot timing distribution
    axes[0, 0].hist(shot_events_df["timestamp_s"], bins=30, alpha=0.7, edgecolor="black")
    axes[0, 0].set_title("Shot Event Time Distribution")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Number of Shots")
    axes[0, 0].grid(True, alpha=0.3)

    # Shot events per player
    if "player_id" in shot_events_df.columns:
        shot_counts = shot_events_df["player_id"].value_counts().sort_index()
        axes[0, 1].bar(range(len(shot_counts)), shot_counts.values, alpha=0.7, edgecolor="black")
        axes[0, 1].set_title("Shot Events per Player")
        axes[0, 1].set_xlabel("Player ID")
        axes[0, 1].set_ylabel("Shot Count")
        axes[0, 1].set_xticks(range(len(shot_counts)))
        axes[0, 1].set_xticklabels(shot_counts.index)
        axes[0, 1].grid(True, alpha=0.3)

    # Shot duration distribution
    if "end_timestamp_s" in shot_events_df.columns:
        shot_duration = shot_events_df["end_timestamp_s"] - shot_events_df["timestamp_s"]
        axes[1, 0].hist(shot_duration, bins=20, alpha=0.7, edgecolor="black")
        axes[1, 0].set_title("Shot Duration Distribution")
        axes[1, 0].set_xlabel("Duration (s)")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].grid(True, alpha=0.3)

    # Basket distribution
    if "basket_id" in shot_events_df.columns:
        basket_counts = shot_events_df["basket_id"].value_counts()
        axes[1, 1].bar(basket_counts.index, basket_counts.values, alpha=0.7, edgecolor="black")
        axes[1, 1].set_title("Shots by Basket")
        axes[1, 1].set_xlabel("Basket")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/shot_timing_analysis.png")


def plot_data_quality_overview(detections_df: pd.DataFrame, positions_df: pd.DataFrame) -> None:
    """Create overview visualizations of data quality metrics.

    Args:
        detections_df: DataFrame with skeletal detection data
        positions_df: DataFrame with player position data
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Detection confidence distribution
    axes[0, 0].hist(detections_df["confidence"], bins=30, alpha=0.7, edgecolor="black")
    axes[0, 0].set_title("Detection Confidence Distribution")
    axes[0, 0].set_xlabel("Confidence")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, alpha=0.3)

    # Player detection frequency
    player_counts = detections_df["player_id"].value_counts()
    axes[0, 1].bar(range(len(player_counts)), player_counts.values, alpha=0.7)
    axes[0, 1].set_title("Detections per Player")
    axes[0, 1].set_xlabel("Player ID")
    axes[0, 1].set_ylabel("Detection Count")
    axes[0, 1].set_xticks(range(len(player_counts)))
    axes[0, 1].set_xticklabels(player_counts.index)
    axes[0, 1].grid(True, alpha=0.3)

    # Timeline coverage
    det_time_range = detections_df["timestamp_s"].max() - detections_df["timestamp_s"].min()
    pos_time_range = positions_df["timestamp_s"].max() - positions_df["timestamp_s"].min()

    axes[1, 0].bar(["Detections", "Positions"], [det_time_range, pos_time_range], alpha=0.7)
    axes[1, 0].set_title("Data Timeline Coverage")
    axes[1, 0].set_ylabel("Time Range (s)")
    axes[1, 0].grid(True, alpha=0.3)

    # Missing data analysis
    missing_detections = detections_df.isnull().sum().sum()
    missing_positions = positions_df.isnull().sum().sum()

    axes[1, 1].bar(["Detections", "Positions"], [missing_detections, missing_positions], alpha=0.7)
    axes[1, 1].set_title("Missing Data Points")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/data_quality_overview.png")


def print_data_summary(
    detections_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    shot_events_df: pd.DataFrame,
) -> None:
    """Print comprehensive summary statistics of all datasets.

    Args:
        detections_df: DataFrame with skeletal detection data
        positions_df: DataFrame with player position data
        shot_events_df: DataFrame with shot events
    """
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print(f"\nDETECTIONS DATA:")
    print(f"  - Shape: {detections_df.shape}")
    print(f"  - Players: {detections_df['player_id'].nunique()}")
    print(f"  - Time range: {detections_df['timestamp_s'].min():.1f} - {detections_df['timestamp_s'].max():.1f} s")
    print(f"  - Duration: {detections_df['timestamp_s'].max() - detections_df['timestamp_s'].min():.1f} s")
    print(f"  - Avg confidence: {detections_df['confidence'].mean():.3f}")

    # Calculate timesteps per player for detections
    detections_sorted = detections_df.sort_values(["player_id", "timestamp_s"])
    det_timesteps = detections_sorted.groupby("player_id")["timestamp_s"].diff().dropna()
    if len(det_timesteps) > 0:
        print(
            f"  - Timesteps (per player): min={det_timesteps.min():.4f}s, median={det_timesteps.median():.4f}s, max={det_timesteps.max():.4f}s"
        )

    print(f"\nPOSITIONS DATA:")
    print(f"  - Shape: {positions_df.shape}")
    print(f"  - Players: {positions_df['player_id'].nunique()}")
    print(f"  - Time range: {positions_df['timestamp_s'].min():.1f} - {positions_df['timestamp_s'].max():.1f} s")
    print(f"  - Duration: {positions_df['timestamp_s'].max() - positions_df['timestamp_s'].min():.1f} s")
    print(f"  - Court bounds: X=[{positions_df['x in m'].min():.1f}, {positions_df['x in m'].max():.1f}]")
    print(f"                  Y=[{positions_df['y in m'].min():.1f}, {positions_df['y in m'].max():.1f}]")

    # Calculate timesteps per player for positions
    positions_sorted = positions_df.sort_values(["player_id", "timestamp_s"])
    pos_timesteps = positions_sorted.groupby("player_id")["timestamp_s"].diff().dropna()
    if len(pos_timesteps) > 0:
        print(
            f"  - Timesteps (per player): min={pos_timesteps.min():.4f}s, median={pos_timesteps.median():.4f}s, max={pos_timesteps.max():.4f}s"
        )

    print(f"\nSHOT EVENTS:")
    print(f"  - Total shots: {len(shot_events_df)}")
    if not shot_events_df.empty:
        if "successful" in shot_events_df.columns:
            success_rate = shot_events_df["successful"].mean() * 100
            print(f"  - Success rate: {success_rate:.1f}%")
        print(
            f"  - Time range: {shot_events_df['timestamp_s'].min():.1f} - {shot_events_df['timestamp_s'].max():.1f} s"
        )

    print("=" * 60)


@app.command()
def analyze(
    detections: Path = typer.Option("data/detections.csv", help="Path to detections CSV file"),
    positions: Path = typer.Option("data/player_positions.csv", help="Path to positions CSV file"),
    shots: Path = typer.Option("data/shot_events.json", help="Path to shot events JSON file"),
    player_id: Optional[str] = typer.Option(None, help="Focus analysis on specific player ID"),
) -> None:
    """Run comprehensive exploratory data analysis on basketball shot detection data.

    This command loads the three data sources and generates visualizations including:
    - Data quality overview
    - Shot timing patterns
    - Player-specific feature analysis (hand movements, body positions, speed)

    Args:
        detections: Path to skeletal detection CSV file
        positions: Path to player positions CSV file
        shots: Path to shot events JSON file
        player_id: Optional player ID to focus analysis on (defaults to most active player)
    """
    detections_df, positions_df, shot_events_df = load_data(detections, positions, shots)

    # Set up plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Print summary
    print_data_summary(detections_df, positions_df, shot_events_df)

    # Generate plots
    print("\nGenerating visualizations...")
    print("  - Data quality overview...")
    plot_data_quality_overview(detections_df, positions_df)
    print("  - Shot timing analysis...")
    plot_shot_timing_analysis(shot_events_df)

    # Player-specific analysis
    if player_id is None:
        player_id = str(detections_df["player_id"].value_counts().index[0])
        print(f"  - [Selecting most active player ID for visualization: {player_id}]")

    # Extract features once for the selected player
    rolling_window_ms = 1000.0
    print(f"  - Extracting features for player {player_id} (rolling window: {int(rolling_window_ms)}ms)...")
    features_df = extract_features_for_player(
        detections_df, player_id, positions_df, rolling_window_ms=rolling_window_ms
    )

    if features_df.empty:
        print(f"ERROR: No features could be extracted for player {player_id}")
        return

    print(f"    Features extracted: {features_df.shape[0]} samples, {features_df.shape[1]} features")

    # Generate all plots using the extracted features
    print(f"  - Player speed analysis for player {player_id}...")
    plot_player_speed(features_df, shot_events_df, player_id, rolling_window_ms)
    print(f"  - Limb distances analysis for player {player_id}...")
    plot_limb_distances(features_df, shot_events_df, player_id, rolling_window_ms)
    print(f"  - Hand relative to body analysis for player {player_id}...")
    plot_hand_rel_body(features_df, shot_events_df, player_id, rolling_window_ms)
    print(f"  - Hand relative to bbox analysis for player {player_id}...")
    plot_hand_rel_bbox(features_df, shot_events_df, player_id)
    print(f"  - Hip relative to bbox analysis for player {player_id}...")
    plot_hip_rel_bbox(features_df, shot_events_df, player_id, rolling_window_ms)

    print("Analysis complete!")


if __name__ == "__main__":
    app()
