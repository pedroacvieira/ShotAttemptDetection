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


app = typer.Typer(help="Exploratory Data Analysis for Shot Attempt Detection")


def plot_player_trajectories(positions_df: pd.DataFrame, shot_events_df: pd.DataFrame, player_id: int, save_plots: bool = True):
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
        player_shots = shot_events_df[shot_events_df["player_id"] == str(player_id)]["timestamp_s"].values
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
    detections_df: pd.DataFrame, shot_events_df: pd.DataFrame, player_id: int = 0, save_plots: bool = True
):
    """Plot hand movement patterns over time for a specific player."""
    player_data = detections_df[detections_df["player_id"] == player_id].copy()

    if player_data.empty:
        print(f"No data found for player {player_id}")
        return

    # Sort by timestamp
    player_data = player_data.sort_values("timestamp_s")

    # Filter shot events for this specific player
    if not shot_events_df.empty and "player_id" in shot_events_df.columns:
        player_shots = shot_events_df[shot_events_df["player_id"] == str(player_id)]["timestamp_s"].values
    else:
        player_shots = []

    fig, axes = plt.subplots(3, 1, figsize=(30, 30))

    # Hand positions over time with smoothing
    left_hand_rel = player_data["left-hand-y"] - player_data["face-y"]
    right_hand_rel = player_data["right-hand-y"] - player_data["face-y"]
    left_hand_smooth = pd.Series(left_hand_rel).rolling(window=20, min_periods=1).mean()
    right_hand_smooth = pd.Series(right_hand_rel).rolling(window=20, min_periods=1).mean()

    axes[0].plot(
        player_data["timestamp_s"],
        left_hand_smooth,
        label="Left Hand rel. Head",
        alpha=0.7,
    )
    axes[0].plot(
        player_data["timestamp_s"],
        right_hand_smooth,
        label="Right Hand rel. Head",
        alpha=0.7,
    )
    axes[0].set_title(f"Vertical Hand Movement rel. Head - Player {player_id}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Relative Y Position (normalized)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mark shot attempts
    for shot_time in player_shots:
        axes[0].axvline(x=shot_time, color="red", linestyle="--", alpha=0.7, linewidth=2)

    # Hand separation (arm spread) with smoothing
    hand_separation = np.sqrt(
        (player_data["left-hand-x"] - player_data["right-hand-x"]) ** 2
        + (player_data["left-hand-y"] - player_data["right-hand-y"]) ** 2
    )
    # Apply smoothing with rolling window
    hand_separation_smooth = pd.Series(hand_separation).rolling(window=20, min_periods=1).mean()
    axes[1].plot(player_data["timestamp_s"], hand_separation_smooth, "green", alpha=0.7)
    axes[1].set_title(f"Hand Separation - Player {player_id}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Hand Separation (normalized)")
    axes[1].grid(True, alpha=0.3)

    # Mark shot attempts
    for shot_time in player_shots:
        axes[1].axvline(x=shot_time, color="red", linestyle="--", alpha=0.7, linewidth=2)

    # Plot 3: Hand Y speed over time (relative to hip)
    left_hand_y_speed = np.abs((player_data["left-hand-y"] - player_data["hip-center-y"]).diff()) / player_data["timestamp_s"].diff()
    right_hand_y_speed = np.abs((player_data["right-hand-y"] - player_data["hip-center-y"]).diff()) / player_data["timestamp_s"].diff()

    # Apply smoothing
    left_hand_y_speed_smooth = pd.Series(left_hand_y_speed).rolling(window=15, min_periods=1).mean()
    right_hand_y_speed_smooth = pd.Series(right_hand_y_speed).rolling(window=15, min_periods=1).mean()

    axes[2].plot(
        player_data["timestamp_s"],
        left_hand_y_speed_smooth,
        label="Left Hand Y Speed (rel. Hip)",
        alpha=0.7,
        color="blue"
    )
    axes[2].plot(
        player_data["timestamp_s"],
        right_hand_y_speed_smooth,
        label="Right Hand Y Speed (rel. Hip)",
        alpha=0.7,
        color="purple"
    )
    axes[2].set_title(f"Hand Y Speed rel. Hip - Player {player_id}")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Y Speed (normalized)")
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
    detections_df: pd.DataFrame, shot_events_df: pd.DataFrame, player_id: int = 0, save_plots: bool = True
):
    """Plot body movement patterns (hips and heels) over time for a specific player."""
    player_data = detections_df[detections_df["player_id"] == player_id].copy()

    if player_data.empty:
        print(f"No data found for player {player_id}")
        return

    # Sort by timestamp
    player_data = player_data.sort_values("timestamp_s")

    # Filter shot events for this specific player
    if not shot_events_df.empty and "player_id" in shot_events_df.columns:
        player_shots = shot_events_df[shot_events_df["player_id"] == str(player_id)]["timestamp_s"].values
    else:
        player_shots = []

    fig, axes = plt.subplots(3, 1, figsize=(30, 30))

    # Hip stability - Y speed and Y position
    hip_y_speed = np.abs(player_data["hip-center-y"].diff()) / player_data["timestamp_s"].diff()
    axes[0].plot(
        player_data["timestamp_s"],
        hip_y_speed,
        label="Hip Y Speed",
        alpha=0.7,
        color="blue"
    )
    axes[0].plot(
        player_data["timestamp_s"],
        player_data["hip-center-y"],
        label="Hip Y Position",
        alpha=0.7,
        color="orange"
    )
    axes[0].set_title(f"Hip Y Movement - Player {player_id}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Position/Speed (normalized)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mark shot attempts
    for shot_time in player_shots:
        axes[0].axvline(x=shot_time, color="red", linestyle="--", alpha=0.7, linewidth=2)

    # Heel distance over time with smoothing
    heel_distance = np.sqrt(
        (player_data["left-heel-x"] - player_data["right-heel-x"]) ** 2
        + (player_data["left-heel-y"] - player_data["right-heel-y"]) ** 2
    ) / (player_data["bbox-ul-y"] - player_data["bbox-lr-y"])
    heel_distance = np.abs(heel_distance)

    # Apply smoothing with rolling window
    heel_distance_smooth = pd.Series(heel_distance).rolling(window=30, min_periods=1).mean()

    # Calculate heel distance speed (derivative)
    heel_distance_speed = np.abs(heel_distance_smooth.diff()) / player_data["timestamp_s"].diff()
    heel_distance_speed_smooth = pd.Series(heel_distance_speed).rolling(window=15, min_periods=1).mean()

    # Calculate heel distance acceleration (second derivative)
    heel_distance_acceleration = np.abs(heel_distance_speed_smooth.diff()) / player_data["timestamp_s"].diff()
    heel_distance_acceleration_smooth = pd.Series(heel_distance_acceleration).rolling(window=10, min_periods=1).mean()

    axes[1].plot(player_data["timestamp_s"], heel_distance_smooth, "green", alpha=0.7, label="Heel Distance")
    axes[1].plot(player_data["timestamp_s"], heel_distance_speed_smooth, "orange", alpha=0.7, label="Heel Distance Speed")
    axes[1].plot(player_data["timestamp_s"], heel_distance_acceleration_smooth, "red", alpha=0.7, label="Heel Distance Acceleration")
    axes[1].set_title(f"Heel Distance, Speed & Acceleration - Player {player_id}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Distance/Speed/Acceleration (normalized)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Mark shot attempts
    for shot_time in player_shots:
        axes[1].axvline(x=shot_time, color="red", linestyle="--", alpha=0.7, linewidth=2)

    # Heel X speed (left and right) with smoothing
    left_heel_x_speed = np.abs(player_data["left-heel-x"].diff()) / player_data["timestamp_s"].diff()
    right_heel_x_speed = np.abs(player_data["right-heel-x"].diff()) / player_data["timestamp_s"].diff()
    left_heel_x_speed_smooth = pd.Series(left_heel_x_speed).rolling(window=20, min_periods=1).mean()
    right_heel_x_speed_smooth = pd.Series(right_heel_x_speed).rolling(window=20, min_periods=1).mean()

    axes[2].plot(player_data["timestamp_s"], left_heel_x_speed_smooth, "blue", alpha=0.7, label="Left Heel X Speed")
    axes[2].plot(player_data["timestamp_s"], right_heel_x_speed_smooth, "purple", alpha=0.7, label="Right Heel X Speed")
    axes[2].set_title(f"Heel X Speed - Player {player_id}")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("X Speed (normalized)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Mark shot attempts
    for shot_time in player_shots:
        axes[2].axvline(x=shot_time, color="red", linestyle="--", alpha=0.7, linewidth=2)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"plots/player_{player_id}_body_movement.png")
    else:
        plt.show()


def plot_player_positions(
    detections_df: pd.DataFrame, positions_df: pd.DataFrame, shot_events_df: pd.DataFrame, player_id: int = 0, save_plots: bool = True
):
    """Plot player hip coordinates vs court positions over time for a specific player."""
    # Get detection data for the player
    player_detections = detections_df[detections_df["player_id"] == player_id].copy()
    # Get position data for the player
    player_positions = positions_df[positions_df["player_id"] == player_id].copy()

    if player_detections.empty and player_positions.empty:
        print(f"No data found for player {player_id}")
        return

    # Sort by timestamp
    if not player_detections.empty:
        player_detections = player_detections.sort_values("timestamp_s")
    if not player_positions.empty:
        player_positions = player_positions.sort_values("timestamp_s")

    # Filter shot events for this specific player
    if not shot_events_df.empty and "player_id" in shot_events_df.columns:
        player_shots = shot_events_df[shot_events_df["player_id"] == str(player_id)]["timestamp_s"].values
    else:
        player_shots = []

    fig, axes = plt.subplots(2, 1, figsize=(30, 20))

    # Calculate min-max range from position data
    if not player_positions.empty:
        x_min, x_max = player_positions["x in m"].min(), player_positions["x in m"].max()
        y_min, y_max = player_positions["y in m"].min(), player_positions["y in m"].max()
    else:
        x_min, x_max = 0.0, 1.0
        y_min, y_max = 0.0, 1.0

    # Plot 1: Hip X vs X position from positions data
    if not player_detections.empty:
        # Map hip-center-x from its range to position X range
        hip_x_min, hip_x_max = player_detections["hip-center-x"].min(), player_detections["hip-center-x"].max()
        if hip_x_max != hip_x_min:
            normalized_hip_x = (player_detections["hip-center-x"] - hip_x_min) / (hip_x_max - hip_x_min)
            mapped_hip_x = normalized_hip_x * (x_max - x_min) + x_min
        else:
            mapped_hip_x = player_detections["hip-center-x"]

        axes[0].plot(
            player_detections["timestamp_s"],
            mapped_hip_x,
            label="Hip X (mapped to position range)",
            alpha=0.7,
            color="blue"
        )

    if not player_positions.empty:
        axes[0].plot(
            player_positions["timestamp_s"],
            player_positions["x in m"],
            label="X Position (positions)",
            alpha=0.7,
            color="orange"
        )

    axes[0].set_title(f"Hip X vs X Position - Player {player_id}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("X Coordinate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mark shot attempts
    for shot_time in player_shots:
        axes[0].axvline(x=shot_time, color="red", linestyle="--", alpha=0.7, linewidth=2)

    # Plot 2: Hip Y vs Y position from positions data
    if not player_detections.empty:
        # Map hip-center-y from its range to position Y range
        hip_y_min, hip_y_max = player_detections["hip-center-y"].min(), player_detections["hip-center-y"].max()
        if hip_y_max != hip_y_min:
            normalized_hip_y = (player_detections["hip-center-y"] - hip_y_min) / (hip_y_max - hip_y_min)
            mapped_hip_y = normalized_hip_y * (y_max - y_min) + y_min
        else:
            mapped_hip_y = player_detections["hip-center-y"]

        axes[1].plot(
            player_detections["timestamp_s"],
            mapped_hip_y,
            label="Hip Y (mapped to position range)",
            alpha=0.7,
            color="blue"
        )

    if not player_positions.empty:
        axes[1].plot(
            player_positions["timestamp_s"],
            player_positions["y in m"],
            label="Y Position (positions)",
            alpha=0.7,
            color="orange"
        )

    axes[1].set_title(f"Hip Y vs Y Position - Player {player_id}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Y Coordinate")
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

    # Success rate
    if "successful" in shot_events_df.columns:
        success_rate = shot_events_df["successful"].value_counts()
        axes[0, 1].pie(success_rate.values, labels=success_rate.index, autopct="%1.1f%%")
        axes[0, 1].set_title("Shot Success Rate")

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
        axes[1, 1].bar(basket_counts.index, basket_counts.values, alpha=0.7)
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

    print(f"\nPOSITIONS DATA:")
    print(f"  - Shape: {positions_df.shape}")
    print(f"  - Players: {positions_df['player_id'].nunique()}")
    print(f"  - Time range: {positions_df['timestamp_s'].min():.1f} - {positions_df['timestamp_s'].max():.1f} s")
    print(f"  - Duration: {positions_df['timestamp_s'].max() - positions_df['timestamp_s'].min():.1f} s")
    print(f"  - Court bounds: X=[{positions_df['x in m'].min():.1f}, {positions_df['x in m'].max():.1f}]")
    print(f"                  Y=[{positions_df['y in m'].min():.1f}, {positions_df['y in m'].max():.1f}]")

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
    player_focus: Optional[int] = typer.Option(None, help="Focus analysis on specific player ID"),
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

    print("\nGenerating visualizations...")

    # Generate plots
    print("  - Data quality overview...")
    plot_data_quality_overview(detections_df, positions_df, save_plots)

    print("  - Shot timing analysis...")
    plot_shot_timing_analysis(shot_events_df, save_plots)

    # Player trajectories analysis
    if player_focus is not None:
        print(f"  - Trajectory analysis for player {player_focus}...")
        plot_player_trajectories(positions_df, shot_events_df, player_focus, save_plots)
    else:
        # Analyze the player with most position data
        top_player = positions_df["player_id"].value_counts().index[0]
        print(f"  - Trajectory analysis for most active player ({top_player})...")
        plot_player_trajectories(positions_df, shot_events_df, top_player, save_plots)

    # Player-specific analysis
    if player_focus is not None:
        print(f"  - Position comparison analysis for player {player_focus}...")
        plot_player_positions(detections_df, positions_df, shot_events_df, player_focus, save_plots)
        print(f"  - Hand movement analysis for player {player_focus}...")
        plot_hand_movement(detections_df, shot_events_df, player_focus, save_plots)
        print(f"  - Body movement analysis for player {player_focus}...")
        plot_body_movement(detections_df, shot_events_df, player_focus, save_plots)
    else:
        # Analyze the player with most detections
        top_player = detections_df["player_id"].value_counts().index[0]
        print(f"  - Position comparison analysis for most active player ({top_player})...")
        plot_player_positions(detections_df, positions_df, shot_events_df, top_player, save_plots)
        print(f"  - Hand movement analysis for most active player ({top_player})...")
        plot_hand_movement(detections_df, shot_events_df, top_player, save_plots)
        print(f"  - Body movement analysis for most active player ({top_player})...")
        plot_body_movement(detections_df, shot_events_df, top_player, save_plots)

    print("Analysis complete!")


if __name__ == "__main__":
    app()
