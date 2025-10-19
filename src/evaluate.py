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
from typing import Any
import matplotlib.pyplot as plt
from numpy import ndarray, dtype

app = typer.Typer(help="Evaluate shot detection predictions against ground truth")


def load_ground_truth(file_path: Path) -> pd.DataFrame:
    """Load ground truth shot events from JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["timestamp_ms"] = (df["timestamp_s"] * 1000).astype(int)
    return df


def load_predictions(file_path: Path) -> pd.DataFrame:
    """Load predicted shots from CSV file."""
    df = pd.read_csv(file_path)
    return df


def evaluate_temporal_matching(
    predictions: pd.DataFrame, ground_truth: pd.DataFrame, tolerance_ms: int = 500, match_player_id: bool = True
) -> tuple[ndarray[tuple[int], dtype[Any]], ndarray[tuple[int], dtype[Any]], int, int]:
    """
    Evaluate predictions against ground truth with temporal tolerance.

    Args:
        predictions: DataFrame with timestamp_ms and player_id columns
        ground_truth: DataFrame with timestamp_ms and player_id columns
        tolerance_ms: Temporal tolerance in milliseconds (�tolerance_ms)
        match_player_id: Whether to match player IDs in addition to timestamps

    Returns:
        Tuple of (true_positives_mask, matched_gt_mask, num_predictions, num_ground_truth)
    """
    pred_times = predictions["timestamp_ms"].values
    gt_times = ground_truth["timestamp_ms"].values

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

    # For each prediction, find the closest ground truth event within tolerance
    for i, pred_time in enumerate(pred_times):
        # Calculate time differences
        time_diffs = np.abs(gt_times - pred_time)

        # Find events within tolerance
        within_tolerance = time_diffs <= tolerance_ms

        if not np.any(within_tolerance):
            continue

        # If player matching is enabled, also check player IDs
        if pred_players is not None and gt_players is not None:
            pred_player = pred_players[i]
            player_matches = (gt_players == pred_player) | (pred_player == "Undefined") | (gt_players == "Undefined")
            valid_matches = within_tolerance & player_matches & (~matched_gt_mask)
        else:
            valid_matches = within_tolerance & (~matched_gt_mask)

        if np.any(valid_matches):
            # Find the closest valid match
            valid_indices = np.where(valid_matches)[0]
            closest_idx = valid_indices[np.argmin(time_diffs[valid_indices])]

            tp_mask[i] = True
            matched_gt_mask[closest_idx] = True

    return tp_mask, matched_gt_mask, len(predictions), len(ground_truth)


def calculate_metrics(
    tp_mask: npt.NDArray, matched_gt_mask: npt.NDArray, num_predictions: int, num_ground_truth: int
) -> dict:
    """Calculate precision, recall, and F1-score."""
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
    tp_mask: npt.NDArray,
    matched_gt_mask: npt.NDArray,
    save_plots: bool = True,
):
    """Plot temporal alignment between predictions and ground truth."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Convert to relative timestamps (minutes)
    min_time = min(predictions["timestamp_ms"].min(), ground_truth["timestamp_ms"].min())
    pred_times_rel = (predictions["timestamp_ms"] - min_time) / 60000  # minutes
    gt_times_rel = (ground_truth["timestamp_ms"] - min_time) / 60000  # minutes

    # Plot 1: Timeline view
    # Ground truth events
    axes[0].scatter(
        gt_times_rel[matched_gt_mask],
        [1] * np.sum(matched_gt_mask),
        c="green",
        s=100,
        alpha=0.7,
        label=f"Matched GT ({np.sum(matched_gt_mask)})",
        marker="s",
    )
    axes[0].scatter(
        gt_times_rel[~matched_gt_mask],
        [1] * np.sum(~matched_gt_mask),
        c="red",
        s=100,
        alpha=0.7,
        label=f"Missed GT ({np.sum(~matched_gt_mask)})",
        marker="s",
    )

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

        # Calculate time differences for matched pairs
        time_diffs = []
        for pred_time in matched_pred_times:
            gt_times = ground_truth["timestamp_ms"].values
            closest_gt_idx = np.argmin(np.abs(gt_times - pred_time))
            time_diff = pred_time - gt_times[closest_gt_idx]
            time_diffs.append(time_diff)

        axes[1].hist(time_diffs, bins=20, alpha=0.7, edgecolor="black")
        axes[1].set_xlabel("Time Difference (ms): Prediction - Ground Truth")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of Time Differences for Matched Events")
        axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Perfect Match")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(
            0.5, 0.5, "No matched events to display", transform=axes[1].transAxes, ha="center", va="center", fontsize=14
        )
        axes[1].set_title("Distribution of Time Differences for Matched Events")

    plt.tight_layout()
    if save_plots:
        plt.savefig("output/evaluation_temporal_alignment.png", dpi=150, bbox_inches="tight")
    else:
        plt.show()


def print_detailed_results(metrics: dict, tolerance_ms: int):
    """Print detailed evaluation results."""
    print("=" * 70)
    print("SHOT DETECTION EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nDATASET SUMMARY:")
    print(f"  - Ground Truth Events: {metrics['num_ground_truth']}")
    print(f"  - Predicted Events: {metrics['num_predictions']}")
    print(f"  - Temporal Tolerance: �{tolerance_ms}ms")

    print(f"\nMATCHING RESULTS:")
    print(f"  - True Positives:  {metrics['true_positives']:3d}")
    print(f"  - False Positives: {metrics['false_positives']:3d}")
    print(f"  - False Negatives: {metrics['false_negatives']:3d}")

    print(f"\nPERFORMANCE METRICS:")
    print(f"  - Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"  - Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"  - F1-Score:  {metrics['f1_score']:.3f} ({metrics['f1_score']*100:.1f}%)")

    # Performance interpretation
    print(f"\nINTERPRETATION:")
    if metrics["f1_score"] >= 0.8:
        print("  Excellent performance!")
    elif metrics["f1_score"] >= 0.6:
        print("  Good performance with room for improvement")
    elif metrics["f1_score"] >= 0.4:
        print("  Moderate performance - consider algorithm improvements")
    else:
        print("  Poor performance - significant improvements needed")
    print("=" * 70)


@app.command()
def evaluate(
    predictions: Path = typer.Option("output/predicted_shots.csv", help="Path to predicted shots CSV"),
    ground_truth: Path = typer.Option("data/shot_events.json", help="Path to ground truth JSON"),
    tolerance_ms: int = typer.Option(500, help="Temporal tolerance in milliseconds"),
    match_player_id: bool = typer.Option(True, help="Whether to match player IDs"),
    save_plots: bool = typer.Option(True, help="Save evaluation plots"),
    verbose: bool = typer.Option(False, help="Show detailed per-event analysis"),
):
    """
    Evaluate shot detection predictions against ground truth.

    This command loads both prediction and ground truth files, performs temporal
    matching within the specified tolerance, and calculates standard classification metrics.
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
        if save_plots or not save_plots:  # Always generate for now
            print("\nGenerating evaluation plots...")
            plot_temporal_alignment(pred_df, gt_df, tp_mask, matched_gt_mask, save_plots)
            if save_plots:
                print("Plots saved to output/evaluation_temporal_alignment.png")

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
        print(f"L Error: Could not find file - {e}")
    except Exception as e:
        print(f"L Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    app()
