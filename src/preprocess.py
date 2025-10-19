import pandas as pd
import json
from pathlib import Path


def load_detections(file_path: Path) -> pd.DataFrame:
    """Load skeleton detection data."""
    df = pd.read_csv(file_path)
    df["timestamp_s"] = df["timestamp_ms"] / 1000
    return df


def load_positions(file_path: Path) -> pd.DataFrame:
    """Load player position data."""
    df = pd.read_csv(file_path)
    df["timestamp_s"] = df["timestamp_ms"] / 1000
    return df


def load_shot_events(file_path: Path) -> pd.DataFrame:
    """Load ground truth shot events."""
    with open(file_path, "r") as f:
        data = json.load(f)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    return df


def load_data(detections: Path, positions: Path, shots: Path):
    print("Loading basketball shot detection data...")
    try:
        detections_df = load_detections(detections)
        positions_df = load_positions(positions)
        shot_events_df = load_shot_events(shots)
        print("Data loaded successfully!")

        # Make timestamps relative to the earliest timestamp across all datasets
        min_timestamp = min(detections_df["timestamp_s"].min(), positions_df["timestamp_s"].min())

        detections_df["timestamp_s"] = detections_df["timestamp_s"] - min_timestamp
        positions_df["timestamp_s"] = positions_df["timestamp_s"] - min_timestamp
        if not shot_events_df.empty:
            shot_events_df["timestamp_s"] = shot_events_df["timestamp_s"] - min_timestamp

            if "end_timestamp_s" in shot_events_df.columns:
                shot_events_df["end_timestamp_s"] = shot_events_df["end_timestamp_s"] - min_timestamp

        print(f"Normalized timestamps to start at 0.0s (offset: {min_timestamp:.3f}s)")
        return detections_df, positions_df, shot_events_df

    except Exception as e:
        print(f"Error loading data: {e}")
        return
