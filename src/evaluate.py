#!/usr/bin/env python3
"""
Shot Detection Evaluation

This script evaluates predicted shot attempts against ground truth data
using precision, recall, and F1-score metrics with temporal tolerance.
"""

import json
import pandas as pd
import numpy as np
import numpy.typing as npt
import typer
from pathlib import Path
from typing import Tuple, Dict
import matplotlib.pyplot as plt

app = typer.Typer(help="Evaluate shot detection predictions against ground truth")


def load_ground_truth(file_path: Path) -> pd.DataFrame:
    """Load ground truth shot events from JSON file.

    Args:
        file_path: Path to the JSON file containing ground truth shot events

    Returns:
        DataFrame with shot events including timestamp_ms and end_timestamp_ms columns
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["timestamp_ms"] = (df["timestamp_s"] * 1000).astype(int)

    # Load end timestamp if available, otherwise use start timestamp
    if "end_timestamp_s" in df.columns:
        df["end_timestamp_ms"] = (df["end_timestamp_s"] * 1000).astype(int)
    else:
        df["end_timestamp_ms"] = df["timestamp_ms"]

    return df


def load_predictions(file_path: Path) -> pd.DataFrame:
    """Load predicted shots from CSV file.

    Args:
        file_path: Path to the CSV file containing predictions

    Returns:
        DataFrame with predicted shot events
    """
    df = pd.read_csv(file_path)
    return df


def evaluate_temporal_matching(
    predictions: pd.DataFrame, ground_truth: pd.DataFrame, tolerance_ms: int = 300, match_player_id: bool = True
) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_], int, int]:
    """Evaluate predictions against ground truth with temporal tolerance.

    Predictions are matched to ground truth shot events considering the full
    time range of each shot (from timestamp_ms to end_timestamp_ms). A prediction
    is considered a match if it falls within the shot event time range, with an
    optional tolerance buffer applied before the start and after the end.

    Args:
        predictions: DataFrame with timestamp_ms and player_id columns
        ground_truth: DataFrame with timestamp_ms, end_timestamp_ms, and player_id columns
        tolerance_ms: Temporal tolerance buffer in milliseconds (extends range on both sides)
        match_player_id: Whether to match player IDs in addition to timestamps

    Returns:
        Tuple of (true_positives_mask, matched_gt_mask, num_predictions, num_ground_truth)
    """
    pred_times = predictions["timestamp_ms"].values
    gt_start_times = ground_truth["timestamp_ms"].values
    gt_end_times = ground_truth["end_timestamp_ms"].values

    if match_player_id and "player_id" in predictions.columns and "player_id" in ground_truth.columns:
        pred_players = predictions["player_id"].astype(str).values
        gt_players = ground_truth["player_id"].astype(str).values
    else:
        pred_players = None
        gt_players = None

    # Track which predictions are true positives
    tp_mask = np.zeros(len(predictions), dtype=bool)
    # Track which ground truth events have been matched
    matched_gt_mask = np.zeros(len(ground_truth), dtype=bool)

    # For each prediction, check if it falls within any shot event time range
    for i, pred_time in enumerate(pred_times):
        # Check if prediction falls within shot event time range (with tolerance buffer)
        # Prediction matches if: (gt_start - tolerance) <= pred_time <= (gt_end + tolerance)
        within_range = (pred_time >= gt_start_times - tolerance_ms) & (pred_time <= gt_end_times + tolerance_ms)

        if not np.any(within_range):
            continue

        # If player matching is enabled, also check player IDs
        if pred_players is not None and gt_players is not None:
            pred_player = pred_players[i]
            player_matches = (gt_players == pred_player) | (pred_player == "Undefined") | (gt_players == "Undefined")
            valid_matches = within_range & player_matches & (~matched_gt_mask)
        else:
            valid_matches = within_range & (~matched_gt_mask)

        if np.any(valid_matches):
            # Find the closest valid match (closest to midpoint of shot event)
            valid_indices = np.where(valid_matches)[0]
            gt_midpoints = (gt_start_times[valid_indices] + gt_end_times[valid_indices]) / 2
            time_diffs = np.abs(gt_midpoints - pred_time)
            closest_idx = valid_indices[np.argmin(time_diffs)]

            tp_mask[i] = True
            matched_gt_mask[closest_idx] = True

    return tp_mask, matched_gt_mask, len(predictions), len(ground_truth)


def calculate_metrics(
    tp_mask: npt.NDArray[np.bool_], matched_gt_mask: npt.NDArray[np.bool_], num_predictions: int, num_ground_truth: int
) -> Dict[str, float]:
    """Calculate precision, recall, and F1-score.

    Args:
        tp_mask: Boolean mask indicating true positive predictions
        matched_gt_mask: Boolean mask indicating matched ground truth events
        num_predictions: Total number of predictions
        num_ground_truth: Total number of ground truth events

    Returns:
        Dictionary containing evaluation metrics
    """
    true_positives = np.sum(tp_mask)
    false_positives = num_predictions - true_positives
    false_negatives = num_ground_truth - np.sum(matched_gt_mask)

    precision = true_positives / num_predictions if num_predictions > 0 else 0.0
    recall = true_positives / num_ground_truth if num_ground_truth > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "num_predictions": num_predictions,
        "num_ground_truth": num_ground_truth,
    }


def plot_temporal_alignment(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    tp_mask: npt.NDArray[np.bool_],
    matched_gt_mask: npt.NDArray[np.bool_],
) -> None:
    """Plot temporal alignment between predictions and ground truth.

    Creates two subplots:
    1. Timeline showing shot event ranges (horizontal bars) and prediction points
    2. Histogram of time differences relative to shot event midpoints

    Args:
        predictions: DataFrame with predicted shot events (timestamp_ms)
        ground_truth: DataFrame with ground truth shot events (timestamp_ms, end_timestamp_ms)
        tp_mask: Boolean mask indicating true positive predictions
        matched_gt_mask: Boolean mask indicating matched ground truth events
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Convert to relative timestamps (minutes)
    min_time = min(predictions["timestamp_ms"].min(), ground_truth["timestamp_ms"].min())
    pred_times_rel = (predictions["timestamp_ms"] - min_time) / 60000  # minutes
    gt_start_times_rel = (ground_truth["timestamp_ms"] - min_time) / 60000  # minutes
    gt_end_times_rel = (ground_truth["end_timestamp_ms"] - min_time) / 60000  # minutes

    # Plot 1: Timeline view with shot event ranges
    # Ground truth events as horizontal bars showing duration
    for i in range(len(ground_truth)):
        start = gt_start_times_rel.iloc[i]
        end = gt_end_times_rel.iloc[i]
        color = "green" if matched_gt_mask[i] else "red"
        axes[0].plot([start, end], [1, 1], color=color, linewidth=4, alpha=0.6, solid_capstyle="round")

    # Add legend markers for ground truth
    axes[0].plot([], [], color="green", linewidth=4, alpha=0.6, label=f"Matched GT ({np.sum(matched_gt_mask)})")
    axes[0].plot([], [], color="red", linewidth=4, alpha=0.6, label=f"Missed GT ({np.sum(~matched_gt_mask)})")

    # Predictions
    axes[0].scatter(
        pred_times_rel[tp_mask],
        [0] * np.sum(tp_mask),
        c="green",
        s=80,
        alpha=0.7,
        label=f"True Positives ({np.sum(tp_mask)})",
        marker="o",
    )
    axes[0].scatter(
        pred_times_rel[~tp_mask],
        [0] * np.sum(~tp_mask),
        c="orange",
        s=80,
        alpha=0.7,
        label=f"False Positives ({np.sum(~tp_mask)})",
        marker="o",
    )

    axes[0].set_ylim(-0.5, 1.5)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Predictions", "Ground Truth"])
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_title("Temporal Alignment of Predictions vs Ground Truth")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Histogram of time differences for matched events
    if np.sum(tp_mask) > 0:
        matched_pred_times = predictions.loc[tp_mask, "timestamp_ms"].values

        # Calculate time differences relative to shot event midpoint
        time_diffs = []
        gt_start_times = ground_truth["timestamp_ms"].values
        gt_end_times = ground_truth["end_timestamp_ms"].values

        for pred_time in matched_pred_times:
            # Find the matched ground truth event
            gt_midpoints = (gt_start_times + gt_end_times) / 2
            closest_gt_idx = np.argmin(np.abs(gt_midpoints - pred_time))

            # Calculate difference from midpoint of shot event
            time_diff = pred_time - gt_midpoints[closest_gt_idx]
            time_diffs.append(time_diff)

        axes[1].hist(time_diffs, bins=20, alpha=0.7, edgecolor="black", color="steelblue")
        axes[1].set_xlabel("Time Difference (ms): Prediction - Shot Event Midpoint")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of Time Differences for Matched Events")
        axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=2, label="Shot Event Midpoint")

        # Add shaded region showing typical shot event duration
        if len(gt_start_times) > 0:
            avg_duration = np.mean(gt_end_times - gt_start_times)
            axes[1].axvspan(
                -avg_duration / 2,
                avg_duration / 2,
                alpha=0.2,
                color="green",
                label=f"Avg Shot Duration: {avg_duration:.0f}ms",
            )

        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(
            0.5, 0.5, "No matched events to display", transform=axes[1].transAxes, ha="center", va="center", fontsize=14
        )
        axes[1].set_title("Distribution of Time Differences for Matched Events")

    plt.tight_layout()
    plt.savefig("plots/evaluation_temporal_alignment.png", dpi=150, bbox_inches="tight")


def print_detailed_results(metrics: Dict[str, float], tolerance_ms: int) -> None:
    """Print detailed evaluation results to console.

    Args:
        metrics: Dictionary containing evaluation metrics
        tolerance_ms: Temporal tolerance used for matching (in milliseconds)
    """
    print("=" * 70)
    print("SHOT DETECTION EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nDATASET SUMMARY:")
    print(f"  - Ground Truth Events: {metrics['num_ground_truth']}")
    print(f"  - Predicted Events: {metrics['num_predictions']}")
    print(f"  - Temporal Tolerance: {tolerance_ms}ms")

    print(f"\nMATCHING RESULTS:")
    print(f"  - True Positives:  {metrics['true_positives']:3d}")
    print(f"  - False Positives: {metrics['false_positives']:3d}")
    print(f"  - False Negatives: {metrics['false_negatives']:3d}")

    print(f"\nPERFORMANCE METRICS:")
    print(f"  - Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"  - Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"  - F1-Score:  {metrics['f1_score']:.3f} ({metrics['f1_score']*100:.1f}%)")

    print("=" * 70)


@app.command()
def evaluate(
    predictions: Path = typer.Option("output/predicted_shots.csv", help="Path to predicted shots CSV"),
    ground_truth: Path = typer.Option("data/shot_events.json", help="Path to ground truth JSON"),
    tolerance_ms: int = typer.Option(300, help="Temporal tolerance in milliseconds"),
    match_player_id: bool = typer.Option(True, help="Whether to match player IDs"),
    verbose: bool = typer.Option(False, help="Show detailed per-event analysis"),
) -> None:
    """Evaluate shot detection predictions against ground truth.

    This command loads both prediction and ground truth files, performs temporal
    matching within the specified tolerance, and calculates standard classification metrics.

    Args:
        predictions: Path to predicted shots CSV file
        ground_truth: Path to ground truth JSON file
        tolerance_ms: Temporal tolerance for matching in milliseconds
        match_player_id: Whether to require player ID matching
        verbose: Whether to show detailed per-event analysis
    """
    print("Loading evaluation data...")

    try:
        # Load data
        pred_df = load_predictions(predictions)
        gt_df = load_ground_truth(ground_truth)

        print(f"Loaded {len(pred_df)} predictions and {len(gt_df)} ground truth events")

        # Perform evaluation
        tp_mask, matched_gt_mask, num_pred, num_gt = evaluate_temporal_matching(
            pred_df, gt_df, tolerance_ms, match_player_id
        )

        # Calculate metrics
        metrics = calculate_metrics(tp_mask, matched_gt_mask, num_pred, num_gt)

        # Print results
        print_detailed_results(metrics, tolerance_ms)

        # Generate plots
        print("\nGenerating evaluation plots...")
        plot_temporal_alignment(pred_df, gt_df, tp_mask, matched_gt_mask)
        print("Plots saved to plots/evaluation_temporal_alignment.png")

        # Verbose analysis
        if verbose:
            print("\nDETAILED ANALYSIS:")
            print(f"\nTrue Positive Events:")
            tp_events = pred_df[tp_mask][["timestamp_ms", "player_id"]].head(10)
            print(tp_events.to_string(index=False))

            print(f"\nFalse Positive Events (first 10):")
            fp_events = pred_df[~tp_mask][["timestamp_ms", "player_id"]].head(10)
            print(fp_events.to_string(index=False))

            print(f"\nMissed Ground Truth Events (first 10):")
            missed_events = gt_df[~matched_gt_mask][["timestamp_s", "player_id"]].head(10)
            print(missed_events.to_string(index=False))

    except FileNotFoundError as e:
        print(f"ERROR: Could not find file - {e}")
    except Exception as e:
        print(f"ERROR: Evaluation failed - {e}")
        raise


if __name__ == "__main__":
    app()
