#!/usr/bin/env python3
"""
Fake Ground Truth Generator

This script loads the ground truth shot events JSON file and converts it to the
predicted shots CSV format for testing the evaluation pipeline.
"""

import json
import pandas as pd
import typer
from pathlib import Path

app = typer.Typer(help="Generate fake predictions from ground truth data")


@app.command()
def generate(
    input_file: Path = typer.Option("data/shot_events.json", help="Path to shot events JSON file"),
    output_file: Path = typer.Option("output/predicted_shots.csv", help="Path to output predicted shots CSV"),
):
    """
    Convert ground truth shot events to predicted shots CSV format.

    This creates a fake predictions file that should achieve perfect scores
    when evaluated against the same ground truth data.
    """
    print(f"Loading shot events from {input_file}...")

    try:
        # Load JSON data
        with open(input_file, "r") as f:
            shot_events = json.load(f)

        print(f"Found {len(shot_events)} shot events")

        # Convert to DataFrame
        df = pd.DataFrame(shot_events)

        # Convert timestamp_s to timestamp_ms
        df["timestamp_ms"] = (df["timestamp_s"] * 1000).astype(int)

        # Create the output dataframe with only required columns
        output_df = pd.DataFrame(
            {
                "timestamp_ms": df["timestamp_ms"],
                "player_id": df.get("player_id", "Unknown"),  # Use 'Unknown' if player_id doesn't exist
            }
        )

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        output_df.to_csv(output_file, index=False)

        print(f"✅ Generated {len(output_df)} predictions saved to {output_file}")
        print(f"\nOutput format preview:")
        print(output_df.head())

    except FileNotFoundError:
        print(f"❌ Error: Could not find input file {input_file}")
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {input_file}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    app()
