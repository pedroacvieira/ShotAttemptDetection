#!/usr/bin/env python3
"""
Shot Attempt Detection using Random Forest

This module implements training and detection of basketball shot attempts
using Random Forest classifier on extracted skeletal and motion features.
"""

import pandas as pd
import numpy as np
import typer
import joblib
from pathlib import Path
from typing import Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit

from src.preprocess import load_data
from src.features import extract_all_features


def create_shot_labels(
    features_df: pd.DataFrame, shot_events_df: pd.DataFrame, offset_ms: int = 0
) -> pd.DataFrame:
    """Create binary labels for shot attempts.

    A frame is labeled as a shot (1) if it falls within the shot event time range
    (between timestamp_s and end_timestamp_s) for that player.

    Args:
        features_df: DataFrame with extracted features and timestamps
        shot_events_df: DataFrame with ground truth shot events (must have timestamp_s and end_timestamp_s)
        offset_ms: Additional time offset in milliseconds to adjust label alignment (default: 0)

    Returns:
        DataFrame with added 'is_shot' binary column (0=no shot, 1=shot)
    """
    # To align labels with the center of the feature window, subtract half of it
    offset_s = offset_ms / 1000.0
    labeled_df = features_df.copy()
    labeled_df["is_shot"] = 0

    # For each shot event, label frames within the shot time range
    for _, shot in shot_events_df.iterrows():
        shot_start = shot["timestamp_s"]
        shot_end = shot.get("end_timestamp_s", shot_start)  # Fallback to start time if end not available
        player_id = str(shot["player_id"])

        # Find frames within shot event time range for this player
        player_mask = labeled_df["player_id"] == player_id
        time_mask = (labeled_df["timestamp_s"] >= shot_start + offset_s) & (
            labeled_df["timestamp_s"] <= shot_end + offset_s
        )

        labeled_df.loc[player_mask & time_mask, "is_shot"] = 1

    return labeled_df


def group_predictions(predictions_df: pd.DataFrame, min_gap_ms: float = 1000.0) -> pd.DataFrame:
    """Group nearby predictions into single shot events.

    Predictions separated by less than min_gap_ms are grouped together,
    and the median of the original timestamp_ms is used as the shot time.
    This reduces duplicate detections from consecutive frames of the same shot.

    Args:
        predictions_df: DataFrame with predicted shots (must have player_id, timestamp_s, timestamp_ms)
        min_gap_ms: Minimum time gap between separate shot events in milliseconds (default: 1000ms)

    Returns:
        DataFrame with grouped shot events containing timestamp_ms and player_id columns
    """
    if predictions_df.empty:
        return pd.DataFrame(columns=["timestamp_ms", "player_id"])

    min_gap_s = min_gap_ms / 1000.0
    grouped_shots = []

    # Process each player separately
    for player_id in predictions_df["player_id"].unique():
        player_preds = predictions_df[predictions_df["player_id"] == player_id].copy()
        player_preds = player_preds.sort_values("timestamp_s")

        if player_preds.empty:
            continue

        # Group predictions within time window
        current_group_indices = [0]

        for i in range(1, len(player_preds)):
            current_time = player_preds.iloc[i]["timestamp_s"]
            prev_time = player_preds.iloc[i - 1]["timestamp_s"]

            if current_time - prev_time <= min_gap_s:
                current_group_indices.append(i)
            else:
                # Save current group and start new one
                # Use median of original timestamp_ms values
                group_timestamps_ms = player_preds.iloc[current_group_indices]["timestamp_ms"].values
                median_timestamp_ms = int(np.median(group_timestamps_ms))
                grouped_shots.append({"timestamp_ms": median_timestamp_ms, "player_id": player_id})
                current_group_indices = [i]

        # Save last group
        if current_group_indices:
            group_timestamps_ms = player_preds.iloc[current_group_indices]["timestamp_ms"].values
            median_timestamp_ms = int(np.median(group_timestamps_ms))
            grouped_shots.append({"timestamp_ms": median_timestamp_ms, "player_id": player_id})

    return pd.DataFrame(grouped_shots)


app = typer.Typer(help="Shot Attempt Detection using Random Forest")


@app.command()
def train(
    detections: Path = typer.Option("data/detections.csv", help="Path to detections CSV file"),
    positions: Path = typer.Option("data/player_positions.csv", help="Path to positions CSV file"),
    shots: Path = typer.Option("data/shot_events.json", help="Path to shot events JSON file"),
    output_model: Path = typer.Option("models/shot_detector.pkl", help="Path to save trained model"),
    rolling_window_ms: float = typer.Option(1200.0, help="Rolling window size in milliseconds"),
    label_offset_ms: int = typer.Option(0, help="Label offset in milliseconds"),
    n_estimators: int = typer.Option(100, help="Number of trees in random forest"),
    max_depth: Optional[int] = typer.Option(None, help="Maximum depth of trees"),
    test_size: float = typer.Option(0.2, help="Fraction of data for testing"),
    random_state: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Train a Random Forest classifier to detect shot attempts.

    This command loads skeletal data, extracts features, creates labels from
    ground truth shot events, and trains a Random Forest model. The model is
    evaluated on a temporal test split and saved with metadata.

    Args:
        detections: Path to detections CSV file with skeletal keypoints
        positions: Path to player positions CSV file
        shots: Path to ground truth shot events JSON file
        output_model: Path where trained model will be saved
        rolling_window_ms: Rolling window size for temporal features (milliseconds)
        label_offset_ms: Time offset for label alignment (milliseconds)
        n_estimators: Number of trees in the random forest
        max_depth: Maximum depth of each tree (None for unlimited)
        test_size: Fraction of data reserved for testing (0.0-1.0)
        random_state: Random seed for reproducibility
    """
    print("=" * 70)
    print("TRAINING SHOT ATTEMPT DETECTOR")
    print("=" * 70)

    # Load data
    print("\n[1/7] Loading data...")
    detections_df, positions_df, shot_events_df = load_data(detections, positions, shots)

    if detections_df.empty:
        print("ERROR: No detection data found!")
        return

    if shot_events_df.empty:
        print("ERROR: No shot events found!")
        return

    print(f"  - Detections: {len(detections_df)} frames")
    print(f"  - Shot events: {len(shot_events_df)} shots")
    print(f"  - Players: {sorted(detections_df['player_id'].unique())}")

    # Extract features
    print(f"\n[2/7] Extracting features (rolling window: {rolling_window_ms}ms)...")
    features_df = extract_all_features(detections_df, positions_df, rolling_window_ms)
    print(f"  - Features shape: {features_df.shape}")
    print(f"  - Feature columns: {len(features_df.columns) - 2} features")

    # Create labels
    print(f"\n[3/7] Creating labels...")
    labeled_df = create_shot_labels(features_df, shot_events_df, label_offset_ms)

    n_shots = labeled_df["is_shot"].sum()
    n_total = len(labeled_df)
    shot_ratio = n_shots / n_total * 100

    print(f"  - Total frames: {n_total}")
    print(f"  - Shot frames: {n_shots} ({shot_ratio:.2f}%)")
    print(f"  - Non-shot frames: {n_total - n_shots} ({100-shot_ratio:.2f}%)")

    # Prepare data for training
    print("\n[4/7] Preparing training data...")

    # Sort by timestamp to ensure temporal ordering
    labeled_df = labeled_df.sort_values("timestamp_s").reset_index(drop=True)

    feature_columns = [
        col for col in labeled_df.columns if col not in ["player_id", "timestamp_s", "timestamp_ms", "is_shot"]
    ]

    X = labeled_df[feature_columns].values
    y = labeled_df["is_shot"].values

    # Split into train and test sets using TimeSeriesSplit for temporal separation
    # This ensures no frames from the same shot event are in both train and test
    # and that test data comes after training data (more realistic evaluation)
    n_splits = max(2, int(1 / test_size))  # Calculate splits to approximate desired test_size
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Use the last split (largest training set, most recent data for testing)
    train_idx, test_idx = list(tscv.split(X))[-1]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"  - Training samples: {len(X_train)} ({len(X_train[y_train==1])} shots)")
    print(f"  - Test samples: {len(X_test)} ({len(X_test[y_test==1])} shots)")
    print(f"  - Using temporal split: train on earlier data, test on later data")

    # Train Random Forest
    print(f"\n[5/7] Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",  # Handle class imbalance
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    rf_model.fit(X_train, y_train)
    print("  - Training complete!")

    # Evaluate on test set
    print("\n[6/7] Evaluating model...")
    y_pred = rf_model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    print("\n--- Test Set Performance ---")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives:  {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives:  {cm[1,1]}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["No Shot", "Shot"], zero_division=0))

    # Feature importance
    print("\n--- Top 10 Most Important Features ---")
    feature_importance = pd.DataFrame(
        {"feature": feature_columns, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:40s} {row['importance']:.4f}")

    # Save model and metadata
    print(f"\n[7/7] Saving model to {output_model}...")
    output_model.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "model": rf_model,
        "feature_columns": feature_columns,
        "rolling_window_ms": rolling_window_ms,
        "metadata": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
        },
    }

    joblib.dump(model_data, output_model)
    print("  - Model saved successfully!")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


@app.command()
def detect(
    detections: Path = typer.Option("data/detections.csv", help="Path to detections CSV file"),
    positions: Path = typer.Option("data/player_positions.csv", help="Path to positions CSV file"),
    model_path: Path = typer.Option("models/shot_detector.pkl", help="Path to trained model"),
    output: Path = typer.Option("output/predicted_shots.csv", help="Output CSV file for predictions"),
    min_gap_ms: float = typer.Option(1000.0, help="Minimum gap between separate shots (ms)"),
    confidence_threshold: float = typer.Option(0.5, help="Confidence threshold for predictions"),
) -> None:
    """Detect shot attempts using a trained Random Forest model.

    This command loads a trained model and uses it to predict shot attempts
    from skeletal detection data. Nearby predictions are grouped to avoid
    duplicate detections, and results are saved to a CSV file.

    Args:
        detections: Path to detections CSV file with skeletal keypoints
        positions: Path to player positions CSV file
        model_path: Path to trained model file (.pkl)
        output: Path where predictions CSV will be saved
        min_gap_ms: Minimum time gap between separate shots (milliseconds)
        confidence_threshold: Probability threshold for positive predictions (0.0-1.0)
    """
    print("=" * 70)
    print("DETECTING SHOT ATTEMPTS")
    print("=" * 70)

    # Load model
    print(f"\n[1/5] Loading model from {model_path}...")
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        print("Please train a model first using: poetry run detect_shots train")
        return

    model_data = joblib.load(model_path)
    rf_model = model_data["model"]
    feature_columns = model_data["feature_columns"]
    rolling_window_ms = model_data["rolling_window_ms"]

    print(f"  - Model loaded successfully")
    print(f"  - Features: {len(feature_columns)}")
    print(f"  - Rolling window: {rolling_window_ms}ms")

    # Load data
    print("\n[2/5] Loading data...")
    detections_df, positions_df, _ = load_data(detections, positions, None)

    if detections_df.empty:
        print("ERROR: No detection data found!")
        return

    print(f"  - Detections: {len(detections_df)} frames")
    print(f"  - Players: {sorted(detections_df['player_id'].unique())}")

    # Extract features
    print(f"\n[3/5] Extracting features...")
    features_df = extract_all_features(detections_df, positions_df, rolling_window_ms)
    print(f"  - Features shape: {features_df.shape}")

    # Make predictions
    print(f"\n[4/5] Making predictions (threshold: {confidence_threshold})...")
    X = features_df[feature_columns].values

    # Get prediction probabilities
    y_proba = rf_model.predict_proba(X)[:, 1]  # Probability of shot class
    y_pred = (y_proba >= confidence_threshold).astype(int)

    # Filter to predicted shots
    shot_indices = np.where(y_pred == 1)[0]
    predicted_shots = features_df.iloc[shot_indices][["player_id", "timestamp_s", "timestamp_ms"]].copy()

    print(f"  - Raw predictions: {len(predicted_shots)} frames")

    # Group nearby predictions
    print(f"\n[5/5] Grouping predictions (min gap: {min_gap_ms}ms)...")
    grouped_shots = group_predictions(predicted_shots, min_gap_ms)

    print(f"  - Grouped shot events: {len(grouped_shots)}")

    # Display per-player summary
    if not grouped_shots.empty:
        print("\n--- Shot Events per Player ---")
        player_counts = grouped_shots["player_id"].value_counts().sort_index()
        for player_id, count in player_counts.items():
            print(f"  Player {player_id}: {count} shots")

    # Save predictions
    print(f"\nSaving predictions to {output}...")
    output.parent.mkdir(parents=True, exist_ok=True)

    if grouped_shots.empty:
        # Save empty file with headers
        pd.DataFrame(columns=["timestamp_ms", "player_id"]).to_csv(output, index=False)
        print("  - No shots detected. Empty file saved.")
    else:
        grouped_shots[["timestamp_ms", "player_id"]].to_csv(output, index=False)
        print(f"  - Saved {len(grouped_shots)} shot events")

    print("\n" + "=" * 70)
    print("DETECTION COMPLETE")
    print("=" * 70)


@app.command()
def info(
    model_path: Path = typer.Option("models/shot_detector.pkl", help="Path to trained model"),
) -> None:
    """Display information about a trained model.

    This command loads a trained model and displays its parameters,
    feature importance, and performance metrics.

    Args:
        model_path: Path to trained model file (.pkl)
    """
    print("=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)

    if not model_path.exists():
        print(f"\nERROR: Model file not found at {model_path}")
        return

    model_data = joblib.load(model_path)
    rf_model = model_data["model"]
    feature_columns = model_data["feature_columns"]
    metadata = model_data.get("metadata", {})

    print(f"\nModel: Random Forest Classifier")
    print(f"Path: {model_path}")
    print(f"\n--- Model Parameters ---")
    print(f"Number of trees: {rf_model.n_estimators}")
    print(f"Max depth: {rf_model.max_depth}")
    print(f"Number of features: {len(feature_columns)}")

    print(f"\n--- Feature Extraction Parameters ---")
    print(f"Rolling window: {model_data.get('rolling_window_ms', 'N/A')}ms")

    if metadata:
        print(f"\n--- Test Set Performance ---")
        print(f"Precision: {metadata.get('test_precision', 'N/A'):.3f}")
        print(f"Recall:    {metadata.get('test_recall', 'N/A'):.3f}")
        print(f"F1-Score:  {metadata.get('test_f1', 'N/A'):.3f}")

    print(f"\n--- Feature Importance (Top 10) ---")
    feature_importance = pd.DataFrame(
        {"feature": feature_columns, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:40s} {row['importance']:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    app()
