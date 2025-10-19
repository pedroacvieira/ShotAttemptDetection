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


def plot_player_trajectories(positions_df: pd.DataFrame, shot_events_df: pd.DataFrame):
    """Plot player trajectories on court."""
    plt.figure(figsize=(15, 10))

    # Plot trajectories for each player
    unique_players = positions_df["player_id"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_players)))

    for i, player in enumerate(unique_players):
        player_data = positions_df[positions_df["player_id"] == player]
        plt.scatter(
            player_data["x in m"],
            player_data["y in m"],
            c=[colors[i]],
            alpha=0.6,
            s=1,
            label=f"Player {player}",
        )

    # Add shot events as larger markers
    if not shot_events_df.empty:
        # Note: Shot events don't have position data directly, so we'll mark them differently
        plt.scatter([], [], c="red", s=100, marker="*", label="Shot Events")

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Player Trajectories on Court")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("output/player_trajectories.png")


def plot_skeleton_movement(detections_df: pd.DataFrame, player_id: int = 0):
    """Plot skeleton keypoint movement over time for a specific player."""
    player_data = detections_df[detections_df["player_id"] == player_id].copy()

    if player_data.empty:
        print(f"No data found for player {player_id}")
        return

    # Sort by timestamp
    player_data = player_data.sort_values("timestamp_s")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Hand positions over time
    axes[0, 0].plot(
        player_data["timestamp_s"],
        player_data["left-hand-y"] - player_data["face-y"],
        label="Left Hand rel. Head",
        alpha=0.7,
    )
    axes[0, 0].plot(
        player_data["timestamp_s"],
        player_data["right-hand-y"] - player_data["face-y"],
        label="Right Hand rel. Head",
        alpha=0.7,
    )
    axes[0, 0].set_title(f"Vertical Hand Movement rel. Head - Player {player_id}")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Relative Y Position (normalized)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Hand separation (arm spread)
    hand_separation = np.sqrt(
        (player_data["left-hand-x"] - player_data["right-hand-x"]) ** 2
        + (player_data["left-hand-y"] - player_data["right-hand-y"]) ** 2
    )
    axes[0, 1].plot(player_data["timestamp_s"], hand_separation, "green", alpha=0.7)
    axes[0, 1].set_title(f"Hand Separation - Player {player_id}")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Hand Separation (normalized)")
    axes[0, 1].grid(True, alpha=0.3)

    # Hip stability
    axes[1, 0].plot(
        player_data["timestamp_s"],
        player_data["hip-center-x"],
        label="Hip X",
        alpha=0.7,
    )
    axes[1, 0].plot(
        player_data["timestamp_s"],
        player_data["hip-center-y"],
        label="Hip Y",
        alpha=0.7,
    )
    axes[1, 0].set_title(f"Hip Position - Player {player_id}")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Position (normalized)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Confidence over time
    axes[1, 1].plot(
        player_data["timestamp_s"], player_data["confidence"], "red", alpha=0.7
    )
    axes[1, 1].set_title(f"Detection Confidence - Player {player_id}")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Confidence")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"output/skeleton_movement_player_{player_id}.png")


def plot_shot_timing_analysis(shot_events_df: pd.DataFrame):
    """Analyze shot timing patterns."""
    if shot_events_df.empty:
        print("No shot events found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Shot timing distribution
    axes[0, 0].hist(
        shot_events_df["timestamp_s"], bins=30, alpha=0.7, edgecolor="black"
    )
    axes[0, 0].set_title("Shot Event Time Distribution")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Number of Shots")
    axes[0, 0].grid(True, alpha=0.3)

    # Success rate
    if "successful" in shot_events_df.columns:
        success_rate = shot_events_df["successful"].value_counts()
        axes[0, 1].pie(
            success_rate.values, labels=success_rate.index, autopct="%1.1f%%"
        )
        axes[0, 1].set_title("Shot Success Rate")

    # Shot duration distribution
    if "end_timestamp_s" in shot_events_df.columns:
        shot_duration = (
            shot_events_df["end_timestamp_s"] - shot_events_df["timestamp_s"]
        )
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
    plt.savefig("output/shot_timing_analysis.png")


def plot_data_quality_overview(detections_df: pd.DataFrame, positions_df: pd.DataFrame):
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
    det_time_range = (
        detections_df["timestamp_s"].max() - detections_df["timestamp_s"].min()
    )
    pos_time_range = (
        positions_df["timestamp_s"].max() - positions_df["timestamp_s"].min()
    )

    axes[1, 0].bar(
        ["Detections", "Positions"], [det_time_range, pos_time_range], alpha=0.7
    )
    axes[1, 0].set_title("Data Timeline Coverage")
    axes[1, 0].set_ylabel("Time Range (s)")
    axes[1, 0].grid(True, alpha=0.3)

    # Missing data analysis
    missing_detections = detections_df.isnull().sum().sum()
    missing_positions = positions_df.isnull().sum().sum()

    axes[1, 1].bar(
        ["Detections", "Positions"], [missing_detections, missing_positions], alpha=0.7
    )
    axes[1, 1].set_title("Missing Data Points")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/data_quality_overview.png")


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
    print(
        f"  - Time range: {detections_df['timestamp_s'].min():.1f} - {detections_df['timestamp_s'].max():.1f} s"
    )
    print(
        f"  - Duration: {detections_df['timestamp_s'].max() - detections_df['timestamp_s'].min():.1f} s"
    )
    print(f"  - Avg confidence: {detections_df['confidence'].mean():.3f}")

    print(f"\nPOSITIONS DATA:")
    print(f"  - Shape: {positions_df.shape}")
    print(f"  - Players: {positions_df['player_id'].nunique()}")
    print(
        f"  - Time range: {positions_df['timestamp_s'].min():.1f} - {positions_df['timestamp_s'].max():.1f} s"
    )
    print(
        f"  - Duration: {positions_df['timestamp_s'].max() - positions_df['timestamp_s'].min():.1f} s"
    )
    print(
        f"  - Court bounds: X=[{positions_df['x in m'].min():.1f}, {positions_df['x in m'].max():.1f}]"
    )
    print(
        f"                  Y=[{positions_df['y in m'].min():.1f}, {positions_df['y in m'].max():.1f}]"
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
    detections: Path = typer.Option(
        "data/detections.csv", help="Path to detections CSV file"
    ),
    positions: Path = typer.Option(
        "data/player_positions.csv", help="Path to positions CSV file"
    ),
    shots: Path = typer.Option(
        "data/shot_events.json", help="Path to shot events JSON file"
    ),
    player_focus: Optional[int] = typer.Option(
        None, help="Focus analysis on specific player ID"
    ),
    save_plots: bool = typer.Option(
        False, help="Save plots instead of displaying them"
    ),
):
    """
    Run exploratory data analysis on basketball shot detection data.

    This command loads the three data sources and generates comprehensive
    visualizations to understand the dataset characteristics.
    """
    detections_df, positions_df, shot_events_df = load_data(
        detections, positions, shots, player_focus
    )

    # Set up plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    if save_plots:
        plt.ioff()  # Turn off interactive mode

    # Print summary
    print_data_summary(detections_df, positions_df, shot_events_df)

    print("\nGenerating visualizations...")

    # Generate plots
    print("  - Data quality overview...")
    plot_data_quality_overview(detections_df, positions_df)

    print("  - Player trajectories...")
    plot_player_trajectories(positions_df, shot_events_df)

    print("  - Shot timing analysis...")
    plot_shot_timing_analysis(shot_events_df)

    # Player-specific analysis
    if player_focus is not None:
        print(f"  - Skeleton analysis for player {player_focus}...")
        plot_skeleton_movement(detections_df, player_focus)
    else:
        # Analyze the player with most detections
        top_player = detections_df["player_id"].value_counts().index[0]
        print(f"  - Skeleton analysis for most active player ({top_player})...")
        plot_skeleton_movement(detections_df, top_player)

    if save_plots:
        print("Plots saved to current directory")
        plt.ion()  # Turn interactive mode back on

    print("Analysis complete!")


if __name__ == "__main__":
    app()
