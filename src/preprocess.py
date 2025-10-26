"""Data preprocessing and loading utilities for basketball shot detection.

This module provides functions to load and normalize data from various sources:
- Skeletal detection data (keypoints)
- Player position data (X, Y coordinates)
- Ground truth shot events
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def load_detections(file_path: Path) -> pd.DataFrame:
    """Load skeleton detection data from CSV file.

    Converts timestamp_ms to timestamp_s for consistency.

    Args:
        file_path: Path to detections CSV file with skeletal keypoint data

    Returns:
        DataFrame with detection data including timestamp_s column
    """
    df = pd.read_csv(file_path)
    df["timestamp_s"] = df["timestamp_ms"] / 1000
    return df


def load_positions(file_path: Path) -> pd.DataFrame:
    """Load player position data from CSV file.

    Converts timestamp_ms to timestamp_s for consistency.

    Args:
        file_path: Path to positions CSV file with player X,Y coordinates

    Returns:
        DataFrame with position data including timestamp_s column
    """
    df = pd.read_csv(file_path)
    df["timestamp_s"] = df["timestamp_ms"] / 1000
    return df


def load_shot_events(file_path: Path) -> pd.DataFrame:
    """Load ground truth shot events from JSON file.

    Args:
        file_path: Path to JSON file containing ground truth shot events

    Returns:
        DataFrame with shot event data
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    return df


def load_data(
    detections: Path, positions: Path, shots: Optional[Path]
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load and normalize all basketball shot detection data.

    Loads skeletal detections, player positions, and optionally shot events.
    Normalizes all timestamps to start at 0.0 seconds for consistency across datasets.

    Args:
        detections: Path to detections CSV file with skeletal keypoints
        positions: Path to positions CSV file with player coordinates
        shots: Path to shot events JSON file (optional, can be None)

    Returns:
        Tuple of (detections_df, positions_df, shot_events_df) where shot_events_df
        may be None if shots parameter is None

    Raises:
        Exception: If data loading fails (e.g., file not found, invalid format)
    """
    print("Loading basketball shot detection data...")
    try:
        detections_df = load_detections(detections)
        positions_df = load_positions(positions)
        shot_events_df = load_shot_events(shots) if shots is not None else None
        print("Data loaded successfully!")

        # Make timestamps relative to the earliest timestamp across all datasets
        min_timestamp = min(detections_df["timestamp_s"].min(), positions_df["timestamp_s"].min())

        detections_df["timestamp_s"] = detections_df["timestamp_s"] - min_timestamp
        positions_df["timestamp_s"] = positions_df["timestamp_s"] - min_timestamp
        if shot_events_df is not None and not shot_events_df.empty:
            shot_events_df["timestamp_s"] = shot_events_df["timestamp_s"] - min_timestamp

            if "end_timestamp_s" in shot_events_df.columns:
                shot_events_df["end_timestamp_s"] = shot_events_df["end_timestamp_s"] - min_timestamp

        print(f"Normalized timestamps to start at 0.0s (offset: {min_timestamp:.3f}s)")
        return detections_df, positions_df, shot_events_df

    except Exception as e:
        print(f"ERROR: Failed to load data - {e}")
        raise
