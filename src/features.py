#!/usr/bin/env python3
"""
Feature Extraction for Basketball Shot Attempt Detection

This module implements feature extraction functions for analyzing basketball
player movements and detecting shot attempts from skeletal and position data.
"""

import pandas as pd
import numpy as np
import typer
from pathlib import Path
from typing import Optional
from scipy.signal import savgol_filter

from src.preprocess import load_data


def apply_savgol_smoothing(df: pd.DataFrame, columns: list, window_length: int = 10, polyorder: int = 2) -> pd.DataFrame:
    """Apply Savitzky-Golay filter to smooth data per player.

    Args:
        df: DataFrame with player_id and columns to smooth
        columns: List of column names to apply smoothing to
        window_length: Length of filter window (must be odd, default: 10)
        polyorder: Order of polynomial (default: 2)

    Returns:
        DataFrame with smoothed columns
    """
    result_df = df.copy()

    # Apply smoothing per player to avoid cross-player contamination
    for player_id in df["player_id"].unique():
        player_mask = df["player_id"] == player_id
        player_data = df[player_mask].copy()
        player_data = player_data.sort_values("timestamp_s")

        for col in columns:
            if col in player_data.columns and len(player_data) >= window_length:
                # Apply Savitzky-Golay filter
                smoothed = savgol_filter(player_data[col].values, window_length, polyorder)
                result_df.loc[player_mask, col] = smoothed

    return result_df


def extract_hand_y_rel_head(detections_df: pd.DataFrame) -> pd.DataFrame:
    """Extract hand Y position relative to head."""
    features_df = detections_df.copy()
    features_df["left_hand_y_rel_head"] = detections_df["left-hand-y"] - detections_df["face-y"]
    features_df["right_hand_y_rel_head"] = detections_df["right-hand-y"] - detections_df["face-y"]

    # Apply smoothing to positional features
    features_df = apply_savgol_smoothing(features_df, ["left_hand_y_rel_head", "right_hand_y_rel_head"])

    return features_df[["player_id", "timestamp_s", "left_hand_y_rel_head", "right_hand_y_rel_head"]]


def extract_hand_y_velocity_rel_head(detections_df: pd.DataFrame) -> pd.DataFrame:
    """Extract hand Y velocity relative to head."""
    # First get hand positions relative to head using existing function
    hand_rel_head_df = extract_hand_y_rel_head(detections_df)

    # Sort by player and timestamp
    features_df = hand_rel_head_df.sort_values(["player_id", "timestamp_s"])

    # Calculate time differences per player
    features_df["time_diff"] = features_df.groupby("player_id")["timestamp_s"].diff()

    # Calculate position differences per player
    features_df["left_hand_rel_head_diff"] = features_df.groupby("player_id")["left_hand_y_rel_head"].diff()
    features_df["right_hand_rel_head_diff"] = features_df.groupby("player_id")["right_hand_y_rel_head"].diff()

    # Calculate velocities
    features_df["left_hand_y_velocity_rel_head"] = np.abs(features_df["left_hand_rel_head_diff"]) / features_df["time_diff"]
    features_df["right_hand_y_velocity_rel_head"] = np.abs(features_df["right_hand_rel_head_diff"]) / features_df["time_diff"]

    return features_df[["player_id", "timestamp_s", "left_hand_y_velocity_rel_head", "right_hand_y_velocity_rel_head"]]


def extract_hand_distance(detections_df: pd.DataFrame) -> pd.DataFrame:
    """Extract distance between hands (hand separation)."""
    features_df = detections_df.copy()

    features_df["hand_distance"] = np.sqrt(
        (detections_df["left-hand-x"] - detections_df["right-hand-x"]) ** 2 +
        (detections_df["left-hand-y"] - detections_df["right-hand-y"]) ** 2
    )

    # Apply smoothing to positional feature
    features_df = apply_savgol_smoothing(features_df, ["hand_distance"])

    return features_df[["player_id", "timestamp_s", "hand_distance"]]


def extract_heel_distance(detections_df: pd.DataFrame) -> pd.DataFrame:
    """Extract distance between heels (normalized by bounding box height)."""
    features_df = detections_df.copy()

    # Calculate raw heel distance
    raw_heel_distance = np.sqrt(
        (detections_df["left-heel-x"] - detections_df["right-heel-x"]) ** 2 +
        (detections_df["left-heel-y"] - detections_df["right-heel-y"]) ** 2
    )

    # Normalize by bounding box height
    bbox_height = detections_df["bbox-ul-y"] - detections_df["bbox-lr-y"]
    features_df["heel_distance"] = np.abs(raw_heel_distance / bbox_height)

    # Apply smoothing to positional feature
    features_df = apply_savgol_smoothing(features_df, ["heel_distance"])

    return features_df[["player_id", "timestamp_s", "heel_distance"]]


def extract_hand_y_to_bbox(detections_df: pd.DataFrame) -> pd.DataFrame:
    """Extract hand Y position relative to bounding box."""
    features_df = detections_df.copy()

    bbox_height = detections_df["bbox-ul-y"] - detections_df["bbox-lr-y"]
    bbox_top = detections_df["bbox-ul-y"]

    # Calculate hand positions relative to bounding box (normalized)
    features_df["left_hand_y_to_bbox"] = (detections_df["left-hand-y"] - bbox_top) / bbox_height
    features_df["right_hand_y_to_bbox"] = (detections_df["right-hand-y"] - bbox_top) / bbox_height

    # Apply smoothing to positional features
    features_df = apply_savgol_smoothing(features_df, ["left_hand_y_to_bbox", "right_hand_y_to_bbox"])

    return features_df[["player_id", "timestamp_s", "left_hand_y_to_bbox", "right_hand_y_to_bbox"]]


def extract_hip_y_to_bbox(detections_df: pd.DataFrame) -> pd.DataFrame:
    """Extract hip Y position relative to bounding box."""
    features_df = detections_df.copy()

    bbox_height = detections_df["bbox-ul-y"] - detections_df["bbox-lr-y"]
    bbox_top = detections_df["bbox-ul-y"]

    # Calculate hip position relative to bounding box (normalized)
    features_df["hip_y_to_bbox"] = (detections_df["hip-center-y"] - bbox_top) / bbox_height

    # Apply smoothing to positional feature
    features_df = apply_savgol_smoothing(features_df, ["hip_y_to_bbox"])

    return features_df[["player_id", "timestamp_s", "hip_y_to_bbox"]]


def extract_hip_speed_to_bbox(detections_df: pd.DataFrame) -> pd.DataFrame:
    """Extract hip speed normalized by bounding box height."""
    features_df = detections_df.copy()

    # Sort by player and timestamp
    features_df = features_df.sort_values(["player_id", "timestamp_s"])

    # Calculate time and position differences per player
    features_df["time_diff"] = features_df.groupby("player_id")["timestamp_s"].diff()
    features_df["hip_y_diff"] = features_df.groupby("player_id")["hip-center-y"].diff()

    # Normalize by bounding box height (take absolute value)
    bbox_height = np.abs(features_df["bbox-ul-y"] - features_df["bbox-lr-y"])

    # Calculate hip speed normalized by bbox height (absolute value)
    features_df["hip_speed_to_bbox"] = np.abs(features_df["hip_y_diff"] / (features_df["time_diff"] * bbox_height))

    return features_df[["player_id", "timestamp_s", "hip_speed_to_bbox"]]


def extract_hand_hip_ratio(detections_df: pd.DataFrame) -> pd.DataFrame:
    """Extract ratio between hand positions and hip position."""
    features_df = detections_df.copy()

    # Calculate ratios (with small epsilon to avoid division by zero)
    hip_y = detections_df["hip-center-y"]
    epsilon = 1e-8

    features_df["left_hand_hip_ratio"] = detections_df["left-hand-y"] / (hip_y + epsilon)
    features_df["right_hand_hip_ratio"] = detections_df["right-hand-y"] / (hip_y + epsilon)

    # Apply smoothing to positional features
    features_df = apply_savgol_smoothing(features_df, ["left_hand_hip_ratio", "right_hand_hip_ratio"])

    return features_df[["player_id", "timestamp_s", "left_hand_hip_ratio", "right_hand_hip_ratio"]]


def extract_player_abs_speed(detections_df: pd.DataFrame, positions_df: pd.DataFrame) -> pd.DataFrame:
    """Extract player absolute speed from X and Y position data.

    Args:
        detections_df: DataFrame with skeletal detection data (for alignment)
        positions_df: DataFrame with player position data (x in m, y in m)

    Returns:
        DataFrame with player absolute speed feature
    """
    if positions_df is None or positions_df.empty:
        # Return empty feature if positions data not available
        result_df = detections_df[["player_id", "timestamp_s"]].copy()
        result_df["player_abs_speed"] = np.nan
        return result_df[["player_id", "timestamp_s", "player_abs_speed"]]

    # Calculate speed from positions data
    positions_features = positions_df.copy()
    positions_features = positions_features.sort_values(["player_id", "timestamp_s"])

    # Apply smoothing to positional data before computing differences
    positions_features = apply_savgol_smoothing(positions_features, ["x in m", "y in m"])

    # Calculate time and position differences per player
    positions_features["time_diff"] = positions_features.groupby("player_id")["timestamp_s"].diff()
    positions_features["x_diff"] = positions_features.groupby("player_id")["x in m"].diff()
    positions_features["y_diff"] = positions_features.groupby("player_id")["y in m"].diff()

    # Calculate absolute speed (Euclidean distance / time)
    positions_features["player_abs_speed"] = np.sqrt(
        positions_features["x_diff"]**2 + positions_features["y_diff"]**2
    ) / positions_features["time_diff"]

    # Merge with detections data to align timestamps
    result_df = detections_df[["player_id", "timestamp_s"]].merge(
        positions_features[["player_id", "timestamp_s", "player_abs_speed"]],
        on=["player_id", "timestamp_s"],
        how="left"
    )

    return result_df[["player_id", "timestamp_s", "player_abs_speed"]]


def apply_rolling_max(features_df: pd.DataFrame, feature_columns: list, window_ms: float = 500.0) -> pd.DataFrame:
    """Apply rolling maximum over specified time window for temporal dynamics.

    Args:
        features_df: DataFrame with features and timestamps
        feature_columns: List of column names to apply rolling max to
        window_ms: Time window in milliseconds (default: 500ms)

    Returns:
        DataFrame with additional rolling max columns
    """
    result_df = features_df.copy()

    # Convert window from milliseconds to seconds
    window_s = window_ms / 1000.0

    # Group by player to avoid cross-player calculations
    for player_id in features_df["player_id"].unique():
        player_mask = features_df["player_id"] == player_id
        player_data = features_df[player_mask].copy()
        player_data = player_data.sort_values("timestamp_s")

        # For each feature column, calculate rolling max
        for col in feature_columns:
            if col in player_data.columns:
                # Convert timestamp to datetime for time-based rolling
                player_data_indexed = player_data.set_index(pd.to_datetime(player_data["timestamp_s"], unit="s"))
                rolling_max = player_data_indexed[col].rolling(f"{window_ms}ms", min_periods=1).max()

                # Add rolling max column
                result_df.loc[player_mask, f"{col}_rolling_max_{int(window_ms)}ms"] = rolling_max.values

    return result_df


def extract_all_features(detections_df: pd.DataFrame, positions_df: pd.DataFrame = None, rolling_window_ms: float = 500.0) -> pd.DataFrame:
    """Extract all basketball shot detection features.

    Args:
        detections_df: DataFrame with skeletal detection data
        positions_df: DataFrame with player position data (optional)
        rolling_window_ms: Time window for rolling maximum features (default: 500ms)

    Returns:
        DataFrame with all extracted features
    """
    # Start with base data
    result_df = detections_df[["player_id", "timestamp_s"]].copy()

    # Extract individual feature sets
    hand_y_rel_head = extract_hand_y_rel_head(detections_df)
    hand_y_velocity = extract_hand_y_velocity_rel_head(detections_df)
    hand_distance = extract_hand_distance(detections_df)
    heel_distance = extract_heel_distance(detections_df)
    hand_y_bbox = extract_hand_y_to_bbox(detections_df)
    hip_y_bbox = extract_hip_y_to_bbox(detections_df)
    hip_speed = extract_hip_speed_to_bbox(detections_df)
    hand_hip_ratio = extract_hand_hip_ratio(detections_df)
    player_speed = extract_player_abs_speed(detections_df, positions_df)

    # Merge all features
    feature_dfs = [
        hand_y_rel_head, hand_y_velocity, hand_distance, hip_speed,
        heel_distance, hand_y_bbox, hip_y_bbox, hand_hip_ratio, player_speed
    ]

    for feature_df in feature_dfs:
        result_df = result_df.merge(
            feature_df,
            on=["player_id", "timestamp_s"],
            how="left"
        )

    # Apply rolling maximum to key features
    key_features = [
        "left_hand_y_rel_head", "right_hand_y_rel_head",
        "left_hand_y_velocity_rel_head", "right_hand_y_velocity_rel_head",
        "hand_distance", "hip_speed_to_bbox", "heel_distance", "player_abs_speed"
    ]

    result_df = apply_rolling_max(result_df, key_features, rolling_window_ms)

    # Remove rows with NaN values
    result_df = result_df.dropna()

    return result_df


def extract_features_for_player(detections_df: pd.DataFrame, player_id: str,
                              positions_df: pd.DataFrame = None, rolling_window_ms: float = 500.0) -> pd.DataFrame:
    """Extract features for a specific player.

    Args:
        detections_df: DataFrame with skeletal detection data
        player_id: ID of the player to extract features for
        positions_df: DataFrame with player position data (optional)
        rolling_window_ms: Time window for rolling maximum features (default: 500ms)

    Returns:
        DataFrame with features for the specified player
    """
    player_detections = detections_df[detections_df["player_id"] == player_id].copy()

    if player_detections.empty:
        print(f"No data found for player {player_id}")
        return pd.DataFrame()

    # Filter positions for the player if available
    player_positions = None
    if positions_df is not None and not positions_df.empty:
        player_positions = positions_df[positions_df["player_id"] == player_id].copy()

    return extract_all_features(player_detections, player_positions, rolling_window_ms)


app = typer.Typer(help="Feature Extraction for Basketball Shot Attempt Detection")


@app.command()
def extract(
    detections: Path = typer.Option("data/detections.csv", help="Path to detections CSV file"),
    positions: Path = typer.Option("data/player_positions.csv", help="Path to positions CSV file"),
    shots: Path = typer.Option("data/shot_events.json", help="Path to shot events JSON file"),
    player_id: Optional[str] = typer.Option(None, help="Extract features for specific player ID only"),
    rolling_window_ms: float = typer.Option(500.0, help="Rolling window size in milliseconds"),
    output: Optional[Path] = typer.Option(None, help="Output CSV file path for extracted features"),
):
    """
    Extract basketball shot detection features from skeletal detection data.

    This command loads detection data and extracts movement-based features
    that can be used for shot attempt detection models.
    """
    # Load data
    print("Loading data...")
    detections_df, positions_df, shot_events_df = load_data(detections, positions, shots)

    if detections_df.empty:
        print("No detection data found!")
        return

    # Extract features
    if player_id is not None:
        print(f"Extracting features for player {player_id}...")
        features_df = extract_features_for_player(detections_df, player_id, positions_df, rolling_window_ms)

        if features_df.empty:
            print(f"No data found for player {player_id}")
            return

        print(f"Player {player_id} features shape: {features_df.shape}")
    else:
        print("Extracting features for all players...")
        features_df = extract_all_features(detections_df, positions_df, rolling_window_ms)
        print(f"Extracted features shape: {features_df.shape}")

    # Display feature information
    print(f"Feature columns: {features_df.columns.tolist()}")
    print(f"Players in features: {sorted(features_df['player_id'].unique())}")
    print(f"Time range: {features_df['timestamp_s'].min():.2f} - {features_df['timestamp_s'].max():.2f} seconds")

    # Save features if output path specified
    if output:
        print(f"Saving features to {output}...")
        features_df.to_csv(output, index=False)
        print("Features saved successfully!")
    else:
        print("\nFirst few rows of features:")
        print(features_df.head())


@app.command()
def info(
    detections: Path = typer.Option("data/detections.csv", help="Path to detections CSV file"),
    positions: Path = typer.Option("data/player_positions.csv", help="Path to positions CSV file"),
    shots: Path = typer.Option("data/shot_events.json", help="Path to shot events JSON file"),
):
    """
    Display information about available data for feature extraction.
    """
    # Load data
    print("Loading data...")
    detections_df, positions_df, shot_events_df = load_data(detections, positions, shots)

    print("=" * 60)
    print("FEATURE EXTRACTION DATA INFO")
    print("=" * 60)

    if not detections_df.empty:
        print(f"\nDETECTIONS DATA:")
        print(f"  - Shape: {detections_df.shape}")
        print(f"  - Players: {sorted(detections_df['player_id'].unique())}")
        print(f"  - Available keypoints: {[col for col in detections_df.columns if '-' in col and col != 'timestamp_s']}")
        print(f"  - Time range: {detections_df['timestamp_s'].min():.2f} - {detections_df['timestamp_s'].max():.2f} s")
    else:
        print("\nNo detection data found!")

    print("\nAVAILABLE FEATURES:")
    feature_info = {
        "hand_y_rel_head": "Hand Y positions relative to head",
        "hand_y_velocity_rel_head": "Hand Y velocity relative to head",
        "hand_distance": "Distance between left and right hands",
        "heel_distance": "Distance between heels (normalized)",
        "hand_y_to_bbox": "Hand Y positions relative to bounding box",
        "hip_y_to_bbox": "Hip Y position relative to bounding box",
        "hip_speed_to_bbox": "Hip speed normalized by bounding box height",
        "hand_hip_ratio": "Ratio between hand and hip Y positions",
        "player_abs_speed": "Player absolute speed from X and Y positions",
        "rolling_max_*": "Rolling maximum values over time window"
    }

    for feature, description in feature_info.items():
        print(f"  - {feature}: {description}")

    print("=" * 60)


if __name__ == "__main__":
    app()