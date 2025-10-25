#!/usr/bin/env python3
"""
Exploratory Data Analysis for Shot Attempt Detection

This script loads and visualizes the basketball shot detection data to understand
the patterns and characteristics of the dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import typer
from pathlib import Path
from typing import Optional

from src.preprocess import load_data
from src.features import extract_features_for_player


app = typer.Typer(help="Exploratory Data Analysis for Shot Attempt Detection")


def plot_player_trajectories(positions_df: pd.DataFrame, shot_events_df: pd.DataFrame, player_id: str, save_plots: bool = True):
    """Plot specific player's speed over time with shot event markers."""
    # Filter data for the specific player
    player_data = positions_df[positions_df["player_id"] == player_id].copy()

    if player_data.empty:
        print(f"No position data found for player {player_id}")
        return

    # Sort by timestamp
    player_data = player_data.sort_values("timestamp_s")

    # Calculate speed using X and Y positions
    player_data["speed"] = np.sqrt(
        player_data["x in m"].diff()**2 + player_data["y in m"].diff()**2
    ) / player_data["timestamp_s"].diff()

    # Calculate Y speed (variation in Y dimension)
    player_data["y_speed"] = np.abs(player_data["y in m"].diff()) / player_data["timestamp_s"].diff()

    # Calculate X speed (variation in X dimension)
    player_data["x_speed"] = np.abs(player_data["x in m"].diff()) / player_data["timestamp_s"].diff()

    # Filter shot events for this specific player
    if not shot_events_df.empty and "player_id" in shot_events_df.columns:
        player_shots = shot_events_df[shot_events_df["player_id"] == player_id]["timestamp_s"].values
    else:
        player_shots = []

    fig, axes = plt.subplots(3, 1, figsize=(30, 15))

    # Plot 1: Speed over time
    axes[0].plot(player_data["timestamp_s"], player_data["speed"], "g-", alpha=0.7, linewidth=2)
    axes[0].set_title(f"Player {player_id} - Speed Over Time")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].grid(True, alpha=0.3)

    # Mark shot events
    for shot_time in player_shots:
        axes[0].axvline(x=shot_time, color="red", linestyle=":", alpha=0.8, linewidth=2, label="Shot Event" if shot_time == player_shots[0] else "")

    if len(player_shots) > 0:
        axes[0].legend()

    # Plot 2: Y speed over time
    axes[1].plot(player_data["timestamp_s"], player_data["y_speed"], "m-", alpha=0.7, linewidth=2)
    axes[1].set_title(f"Player {player_id} - Y Speed Over Time")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Y Speed (m/s)")
    axes[1].grid(True, alpha=0.3)

    # Mark shot events
    for shot_time in player_shots:
        axes[1].axvline(x=shot_time, color="red", linestyle=":", alpha=0.8, linewidth=2, label="Shot Event" if shot_time == player_shots[0] else "")

    if len(player_shots) > 0:
        axes[1].legend()

    # Plot 3: X speed over time
    axes[2].plot(player_data["timestamp_s"], player_data["x_speed"], "b-", alpha=0.7, linewidth=2)
    axes[2].set_title(f"Player {player_id} - X Speed Over Time")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("X Speed (m/s)")
    axes[2].grid(True, alpha=0.3)

    # Mark shot events
    for shot_time in player_shots:
        axes[2].axvline(x=shot_time, color="red", linestyle=":", alpha=0.8, linewidth=2, label="Shot Event" if shot_time == player_shots[0] else "")

    if len(player_shots) > 0:
        axes[2].legend()

    plt.tight_layout()

    if save_plots:
        plt.savefig(f"plots/player_{player_id}_trajectories.png")
    else:
        plt.show()


def plot_hand_movement(
    detections_df: pd.DataFrame, shot_events_df: pd.DataFrame, player_id: str = "0", save_plots: bool = True
):
    """Plot hand movement patterns over time for a specific player using extracted features."""
    # Extract features for the player
    features_df = extract_features_for_player(detections_df, player_id, rolling_window_ms=500.0)

    if features_df.empty:
        print(f"No data found for player {player_id}")
        return

    # Filter shot events for this specific player
    if not shot_events_df.empty and "player_id" in shot_events_df.columns:
        player_shots = shot_events_df[shot_events_df["player_id"] == player_id]["timestamp_s"].values
    else:
        player_shots = []

    fig, axes = plt.subplots(3, 1, figsize=(30, 30))

    # Plot 1: Hand Y position relative to head
    axes[0].plot(
        features_df["timestamp_s"],
        features_df["left_hand_y_rel_head"],
        label="Left Hand Y rel. Head",
        alpha=0.7,
        color="blue"
    )
    axes[0].plot(
        features_df["timestamp_s"],
        features_df["right_hand_y_rel_head"],
        label="Right Hand Y rel. Head",
        alpha=0.7,
        color="orange"
    )
    axes[0].plot(
        features_df["timestamp_s"],
        features_df["left_hand_y_rel_head_rolling_max_500ms"],
        label="Left Hand Y rel. Head (rolling max)",
        alpha=0.5,
        linestyle="--",
        color="darkblue"
    )
    axes[0].plot(
        features_df["timestamp_s"],
        features_df["right_hand_y_rel_head_rolling_max_500ms"],
        label="Right Hand Y rel. Head (rolling max)",
        alpha=0.5,
        linestyle="--",
        color="darkorange"
    )
    axes[0].set_title(f"Hand Y Position relative to Head - Player {player_id}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Relative Y Position (normalized)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mark shot attempts
    for shot_time in player_shots:
        axes[0].axvline(x=shot_time, color="red", linestyle="--", alpha=0.7, linewidth=2)

    # Plot 2: Hand separation (distance)
    axes[1].plot(
        features_df["timestamp_s"],
        features_df["hand_distance"],
        label="Hand Distance",
        alpha=0.7,
        color="green"
    )
    axes[1].plot(
        features_df["timestamp_s"],
        features_df["hand_distance_rolling_max_500ms"],
        label="Hand Distance (rolling max)",
        alpha=0.5,
        linestyle="--",
        color="darkgreen"
    )
    axes[1].set_title(f"Hand Separation (Distance) - Player {player_id}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Hand Distance (normalized)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Mark shot attempts
    for shot_time in player_shots:
        axes[1].axvline(x=shot_time, color="red", linestyle="--", alpha=0.7, linewidth=2)

    # Plot 3: Hand Y velocity relative to hip
    axes[2].plot(
        features_df["timestamp_s"],
        features_df["left_hand_y_velocity_rel_hip"],
        label="Left Hand Y Velocity (rel. Hip)",
        alpha=0.7,
        color="blue"
    )
    axes[2].plot(
        features_df["timestamp_s"],
        features_df["right_hand_y_velocity_rel_hip"],
        label="Right Hand Y Velocity (rel. Hip)",
        alpha=0.7,
        color="purple"
    )
    axes[2].plot(
        features_df["timestamp_s"],
        features_df["left_hand_y_velocity_rel_hip_rolling_max_500ms"],
        label="Left Hand Y Velocity (rolling max)",
        alpha=0.5,
        linestyle="--",
        color="darkblue"
    )
    axes[2].plot(
        features_df["timestamp_s"],
        features_df["right_hand_y_velocity_rel_hip_rolling_max_500ms"],
        label="Right Hand Y Velocity (rolling max)",
        alpha=0.5,
        linestyle="--",
        color="indigo"
    )
    axes[2].set_title(f"Hand Y Velocity relative to Hip - Player {player_id}")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Y Velocity (normalized)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Mark shot attempts
    for shot_time in player_shots:
        axes[2].axvline(x=shot_time, color="red", linestyle="--", alpha=0.7, linewidth=2)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"plots/player_{player_id}_hand_movement.png")
    else:
        plt.show()


def plot_body_movement(
    detections_df: pd.DataFrame, shot_events_df: pd.DataFrame, player_id: str = "0", save_plots: bool = True
):
    """Plot body movement patterns (hips and heels) over time for a specific player using extracted features."""
    # Extract features for the player
    features_df = extract_features_for_player(detections_df, player_id, rolling_window_ms=500.0)

    if features_df.empty:
        print(f"No data found for player {player_id}")
        return

    # Filter shot events for this specific player
    if not shot_events_df.empty and "player_id" in shot_events_df.columns:
        player_shots = shot_events_df[shot_events_df["player_id"] == player_id]["timestamp_s"].values
    else:
        player_shots = []

    fig, axes = plt.subplots(2, 1, figsize=(30, 20))

    # Plot 1: Hip Y speed
    axes[0].plot(
        features_df["timestamp_s"],
        features_df["hip_y_speed"],
        label="Hip Y Speed",
        alpha=0.7,
        color="blue"
    )
    axes[0].plot(
        features_df["timestamp_s"],
        features_df["hip_y_speed_rolling_max_500ms"],
        label="Hip Y Speed (rolling max)",
        alpha=0.5,
        linestyle="--",
        color="darkblue"
    )
    axes[0].set_title(f"Hip Y Speed - Player {player_id}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (normalized)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mark shot attempts
    for shot_time in player_shots:
        axes[0].axvline(x=shot_time, color="red", linestyle="--", alpha=0.7, linewidth=2)

    # Plot 2: Heel distance (normalized)
    axes[1].plot(
        features_df["timestamp_s"],
        features_df["heel_distance"],
        label="Heel Distance (normalized)",
        alpha=0.7,
        color="green"
    )
    axes[1].plot(
        features_df["timestamp_s"],
        features_df["heel_distance_rolling_max_500ms"],
        label="Heel Distance (rolling max)",
        alpha=0.5,
        linestyle="--",
        color="darkgreen"
    )
    axes[1].set_title(f"Heel Distance - Player {player_id}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Distance (normalized by bbox height)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Mark shot attempts
    for shot_time in player_shots:
        axes[1].axvline(x=shot_time, color="red", linestyle="--", alpha=0.7, linewidth=2)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"plots/player_{player_id}_body_movement.png")
    else:
        plt.show()


def plot_player_positions(
    detections_df: pd.DataFrame, positions_df: pd.DataFrame, shot_events_df: pd.DataFrame, player_id: str = "0", save_plots: bool = True
):
    """Plot extracted feature ratios and normalized positions for a specific player."""
    # Extract features for the player
    features_df = extract_features_for_player(detections_df, player_id, rolling_window_ms=500.0)

    if features_df.empty:
        print(f"No data found for player {player_id}")
        return

    # Filter shot events for this specific player
    if not shot_events_df.empty and "player_id" in shot_events_df.columns:
        player_shots = shot_events_df[shot_events_df["player_id"] == player_id]["timestamp_s"].values
    else:
        player_shots = []

    fig, axes = plt.subplots(2, 1, figsize=(30, 20))

    # Plot 1: Hand-hip ratios
    axes[0].plot(
        features_df["timestamp_s"],
        features_df["left_hand_hip_ratio"],
        label="Left Hand / Hip Ratio",
        alpha=0.7,
        color="blue"
    )
    axes[0].plot(
        features_df["timestamp_s"],
        features_df["right_hand_hip_ratio"],
        label="Right Hand / Hip Ratio",
        alpha=0.7,
        color="purple"
    )
    axes[0].set_title(f"Hand-Hip Position Ratios - Player {player_id}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Ratio")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mark shot attempts
    for shot_time in player_shots:
        axes[0].axvline(x=shot_time, color="red", linestyle="--", alpha=0.7, linewidth=2)

    # Plot 2: All normalized position features together
    axes[1].plot(
        features_df["timestamp_s"],
        features_df["left_hand_y_to_bbox"],
        label="Left Hand Y to Bbox",
        alpha=0.7,
        color="blue"
    )
    axes[1].plot(
        features_df["timestamp_s"],
        features_df["right_hand_y_to_bbox"],
        label="Right Hand Y to Bbox",
        alpha=0.7,
        color="purple"
    )
    axes[1].plot(
        features_df["timestamp_s"],
        features_df["hip_y_to_bbox"],
        label="Hip Y to Bbox",
        alpha=0.7,
        color="orange"
    )
    axes[1].set_title(f"Normalized Y Positions (relative to Bounding Box) - Player {player_id}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Normalized Position")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Mark shot attempts
    for shot_time in player_shots:
        axes[1].axvline(x=shot_time, color="red", linestyle="--", alpha=0.7, linewidth=2)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"plots/player_{player_id}_positions.png")
    else:
        plt.show()


def plot_shot_timing_analysis(shot_events_df: pd.DataFrame, save_plots: bool = True):
    """Analyze shot timing patterns."""
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
    if save_plots:
        plt.savefig("plots/shot_timing_analysis.png")
    else:
        plt.show()


def plot_data_quality_overview(detections_df: pd.DataFrame, positions_df: pd.DataFrame, save_plots: bool = True):
    """Overview of data quality metrics."""
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
    if save_plots:
        plt.savefig("plots/data_quality_overview.png")
    else:
        plt.show()


def print_data_summary(
    detections_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    shot_events_df: pd.DataFrame,
):
    """Print summary statistics of the datasets."""
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
        print(f"  - Timesteps (per player): min={det_timesteps.min():.4f}s, median={det_timesteps.median():.4f}s, max={det_timesteps.max():.4f}s")

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
        print(f"  - Timesteps (per player): min={pos_timesteps.min():.4f}s, median={pos_timesteps.median():.4f}s, max={pos_timesteps.max():.4f}s")

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
    save_plots: bool = typer.Option(True, help="Save plots instead of displaying them"),
):
    """
    Run exploratory data analysis on basketball shot detection data.

    This command loads the three data sources and generates comprehensive
    visualizations to understand the dataset characteristics.
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
    plot_data_quality_overview(detections_df, positions_df, save_plots)
    print("  - Shot timing analysis...")
    plot_shot_timing_analysis(shot_events_df, save_plots)

    # Player-specific analysis
    if player_id is None:
        player_id = str(detections_df["player_id"].value_counts().index[0])
        print(f"  - [Selecting most active player ID for visualization: {player_id}]")

    print(f"  - Trajectory analysis for player {player_id}...")
    plot_player_trajectories(positions_df, shot_events_df, player_id, save_plots)
    print(f"  - Position comparison analysis for player {player_id}...")
    plot_player_positions(detections_df, positions_df, shot_events_df, player_id, save_plots)
    print(f"  - Hand movement analysis for player {player_id}...")
    plot_hand_movement(detections_df, shot_events_df, player_id, save_plots)
    print(f"  - Body movement analysis for player {player_id}...")
    plot_body_movement(detections_df, shot_events_df, player_id, save_plots)

    print("Analysis complete!")


if __name__ == "__main__":
    app()
