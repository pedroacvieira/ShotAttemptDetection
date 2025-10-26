# 🏀 Shot Attempt Detection Challenge

**Author:** Pedro Andrade Vieira  
**Date:** _Something_ October 2025  
**Kinexon Coding Challenge Submission**

---

## 🧭 Overview

This project implements a **machine learning approach for detecting shot attempts in basketball games** using time-series data of player positions and skeleton keypoints.

The solution includes:
1. **Feature engineering pipeline** extracting basketball-specific features from skeletal keypoint data
2. **Random Forest classifier** trained on labeled shot events with temporal train/test split
3. **Comprehensive evaluation framework** with temporal matching and visualization
4. **Proposed cloud architecture** for scaling shot detection from raw video streams

The approach demonstrates a **reasonable, explainable, and extensible ML solution** using established computer vision and machine learning techniques.

---

## 📁 Repository Structure

```
.
├── data/
│   ├── detections.csv           # Skeleton keypoint detections
│   ├── player_positions.csv     # Player X,Y coordinates
│   └── shot_events.json         # Ground truth shot events
│
├── src/
│   ├── exploratory_data_analysis.py    # EDA plotting and visualization
│   ├── preprocess.py                   # Data loading & normalization
│   ├── features.py                     # Feature extraction from keypoints
│   ├── detect_shots.py                 # Random Forest training & detection
│   ├── evaluate.py                     # Evaluation metrics & visualization
│   └── fake_results.py                 # Testing utility for evaluation
│
├── models/
│   └── shot_detector.pkl        # Trained Random Forest model
│
├── output/
│   └── predicted_shots.csv      # Detection results
│
├── plots/
│   └── evaluation_temporal_alignment.png    # Evaluation visualization
│
├── pyproject.toml               # Poetry dependencies
└── README.md
```

---

## ⚙️ Setup & Usage

### 1. Environment
```bash
git clone https://github.com/pedroacvieira/ShotAttemptDetection.git
cd shot-attempt-detection
poetry install
```

### 2. Exploratory Data Analysis (Optional)
```bash
# View data distribution and visualizations
poetry run eda
```

### 3. Train Model
```bash
# Train Random Forest classifier with default parameters
poetry run detection train

# Or with custom parameters
poetry run detection train --rolling-window-ms 1200 --n-estimators 100 --max-depth 20
```

### 4. Run Detection
```bash
# Detect shots using trained model
poetry run detection detect

# Or with custom confidence threshold
poetry run detection detect --confidence-threshold 0.6 --min-gap-ms 1000
```

### 5. Evaluate Results
```bash
# Evaluate predictions against ground truth
poetry run evaluate

# Or with custom parameters
poetry run evaluate --tolerance-ms 500 --match-player-id --verbose
```

### 6. View Model Information
```bash
# Display model parameters and feature importance
poetry run detection info
```

---

## 🧩 Approach

### 1. Data Understanding

The provided data sources:
- **`detections.csv`** — bounding boxes and skeleton keypoints (normalized)
- **`player_positions.csv`** — player coordinates and motion data
- **`shot_events.json`** — ground-truth shot attempts with:
  - `timestamp_s` and `end_timestamp_s`: Start and end of each shot event
  - `player_id`: Player who attempted the shot

All data are synchronized on timestamp (ms).

---

### 2. Feature Engineering

The `features.py` module extracts basketball-specific features from skeletal keypoint data. All features are computed per-player to avoid cross-player contamination in time-series calculations.

**Hand Movement Features:**
- **Hand Y relative to head**: Vertical distance of each hand from face position (higher values = hands raised above head)
- **Hand Y velocity relative to head**: Rate of change in vertical hand position relative to head (captures upward/downward motion)
- **Hand distance**: Euclidean distance between left and right hands (wider separation common in shooting motions)

**Body Position Features:**
- **Heel distance**: Distance between heels normalized by bounding box height (wider stance during shooting preparation)
- **Hand Y to bbox**: Hand positions relative to bounding box, normalized (0 = top of bbox)
- **Hip Y to bbox**: Hip position relative to bounding box, normalized (tracks body center-of-mass)
- **Hand-hip ratio**: Ratio of hand position to hip position (values > 1 indicate hands above hips)

**Motion Features:**
- **Hip speed to bbox**: Vertical hip movement rate normalized by bounding box height (captures body motion dynamics)
- **Player absolute speed**: Euclidean speed from X,Y position data (meters per second)

**Temporal Features:**
- **Rolling maximum**: Maximum values over configurable time windows (default: 500ms-1200ms depending on context)
  - Applied to key features to capture temporal dynamics and peak movements
  - Helps detect the characteristic motion patterns of shooting attempts

**Data Smoothing:**
- **Savitzky-Golay filter** (window=9, polynomial=2) applied to positional features to reduce noise
- Applied per-player before calculating velocity and speed features

All features are merged with original timestamp_ms preserved for evaluation against ground truth.

---

### 3. Training and Detection

The `detect_shots.py` module implements a **Random Forest classifier** for shot attempt detection with a complete training and inference pipeline.

#### Training Pipeline

1. **Feature Extraction:**
   - Extract all features using configurable rolling window (default: 1200ms)
   - Features include hand positions, velocities, body posture, and temporal dynamics

2. **Label Creation:**
   - Binary labels (0=no shot, 1=shot) created from ground truth shot events
   - Frames within shot event time range (`timestamp_s` to `end_timestamp_s`) labeled as shots
   - Labels aligned with feature window center for temporal consistency
   - Optional offset parameter to compensate for early/late model firing

3. **Temporal Train/Test Split:**
   - Uses `TimeSeriesSplit` from scikit-learn to ensure temporal ordering
   - Test data comes chronologically after training data (realistic evaluation)
   - Prevents information leakage from future data

4. **Model Training:**
   - Random Forest classifier with configurable parameters (default: 100 estimators)
   - Class-weighted training to handle class imbalance (shots are rare events)
   - Feature importance analysis to identify most predictive features

5. **Evaluation:**
   - Precision, Recall, F1-Score on test set
   - Confusion matrix and classification report
   - Top-10 most important features displayed

#### Detection Pipeline

1. **Load Trained Model:**
   - Model includes feature definitions and rolling window parameters
   - Ensures consistency between training and inference

2. **Feature Extraction:**
   - Same feature extraction pipeline applied to new data
   - Uses stored rolling window size from training

3. **Prediction:**
   - Configurable confidence threshold (default: 0.5)
   - Returns probability scores for each frame

4. **Temporal Grouping:**
   - Groups nearby predictions within minimum gap (default: 1000ms)
   - Uses median timestamp_ms of grouped frames as final shot time
   - Reduces duplicate detections from consecutive frames

**Output format:**
```csv
timestamp_ms,player_id
152345,23
187900,5
```

---

### 4. Evaluation

The `evaluate.py` module provides comprehensive evaluation of predictions against ground truth shot events using temporal matching with configurable tolerance.

#### Evaluation Methodology

1. **Temporal Matching with Shot Event Ranges:**
   - Ground truth shot events span time ranges from `timestamp_s` to `end_timestamp_s`
   - Predictions matched if they fall within the shot event time range: `[start - tolerance, end + tolerance]`
   - Tolerance buffer (default: ±500ms) extends the acceptable range on both sides
   - Each prediction matched to closest unmatched ground truth event (by midpoint distance)
   - Prevents multiple predictions from matching the same ground truth event

2. **Player ID Matching (Optional):**
   - Can require player IDs to match in addition to timestamps
   - Handles "Undefined" player IDs gracefully
   - Configurable via `--match-player-id` flag

3. **Metrics Calculation:**
   - **True Positives**: Predictions within tolerance of ground truth
   - **False Positives**: Predictions with no matching ground truth
   - **False Negatives**: Ground truth events with no matching prediction
   - **Precision**: TP / (TP + FP) - accuracy of predictions
   - **Recall**: TP / (TP + FN) - coverage of ground truth events
   - **F1-Score**: Harmonic mean of precision and recall

#### Visualization

Two plots generated (saved to `plots/evaluation_temporal_alignment.png`):

1. **Temporal Alignment Timeline:**
   - Ground truth shot events shown as **horizontal bars** spanning their duration (start to end)
   - Matched events in green, missed events in red
   - Predictions shown as points (true positives in green, false positives in orange)
   - Time axis in minutes for easy interpretation
   - Clearly visualizes whether predictions fall within shot event windows

2. **Time Difference Distribution:**
   - Histogram of time differences: Prediction - Shot Event Midpoint
   - Shot event midpoint = (timestamp_s + end_timestamp_s) / 2
   - Red dashed line at x=0 indicates perfect alignment with midpoint
   - Green shaded region shows average shot event duration
   - Shows temporal bias (early/late predictions relative to shot center)
   - Only includes matched events

#### Output

Detailed console output includes:
- Dataset summary (number of events, tolerance used)
- Matching results (TP, FP, FN counts)
- Performance metrics (precision, recall, F1-score as percentages)
- Optional verbose mode for per-event analysis

---

## ☁️ System Design Proposal

### Goal
Design a **cloud-based architecture** to automatically detect shot attempts from **video streams** at scale.

---

### 🔧 High-Level Architecture

```
                   ┌────────────────────────┐
                   │  Video Upload / Stream │
                   └──────────┬─────────────┘
                              ▼
       ┌────────────────────────────────────────────┐
       │ Pose Estimation Service (GPU)              │
       │ - Extracts skeleton keypoints from frames  │
       │ - Outputs detections.csv                   │
       └────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────┐
        │ Shot Detection Service (CPU)           │
        │ - Runs heuristic / ML model            │
        │ - Produces shot events (timestamp, id) │
        └────────────────────────────────────────┘
                              │
                              ▼
         ┌──────────────────────────────────────┐
         │ Event Storage & API                  │
         │ - Stores metadata in Postgres        │
         │ - Exposes REST/GraphQL endpoints     │
         └──────────────────────────────────────┘
                              │
                              ▼
          ┌────────────────────────────────┐
          │ Dashboard / Analytics UI       │
          │ - View shots overlayed on video│
          └────────────────────────────────┘
```

---

### ☁️ Scalability & Deployment
| Component | Technology | Notes |
|------------|-------------|-------|
| **Video storage** | AWS S3 / GCP Cloud Storage | Raw input |
| **Messaging** | Kafka / PubSub | Decouple video and detection stages |
| **Compute** | Kubernetes pods | Autoscale GPU (pose) and CPU (detection) |
| **Database** | Postgres + TimescaleDB | Store time-series shot events |
| **Monitoring** | Prometheus + Grafana | Track latency & throughput |
| **API layer** | FastAPI / Flask | Serve event data to dashboard |

---

### ⚙️ Future Improvements
- Upgrade to **deep learning models** (LSTM / Transformer) to capture longer temporal sequences
- Implement **hyperparameter tuning** (GridSearchCV, Bayesian optimization) for Random Forest
- Add **feature selection** techniques to identify minimal feature set
- Use **multi-view camera data** for 3D motion context and occlusion handling
- Implement **online learning** for model adaptation to different game styles
- Add **per-shot confidence scores** with uncertainty estimation
- Enable **real-time streaming inference** for live game analytics
- Extend to detect **shot types** (3-pointer, layup, free throw) as multi-class problem

---

## 🧠 Assumptions
- Shot attempts exhibit characteristic motion patterns detectable from skeletal keypoints
- Hand elevation, body stability, and temporal dynamics are predictive features
- Each `player_id` in the data corresponds to a single tracked player throughout the game
- All timestamps are synchronized across detection and position data sources
- Ground truth shot events span time ranges (timestamp_s to end_timestamp_s)
- Shot attempts are relatively rare events (class imbalance addressed via weighting)

---

## 🧾 Example Output (predicted_shots.csv)
```
timestamp_ms,player_id
152345,23
187900,5
289532,23
...
```

---

## 📈 Next Steps
1. Collect more training data from additional games to improve generalization
2. Experiment with deep learning architectures (LSTM, TCN, Transformers) for temporal modeling
3. Implement cross-validation and hyperparameter tuning for optimal performance
4. Extend architecture for multi-camera input and 3D pose estimation
5. Evaluate robustness on unseen games, different venues, and partial occlusions
6. Add shot type classification (3-pointer, layup, dunk, free throw)
7. Deploy model in cloud environment for scalable inference

---

## 💬 Contact
If you have any questions or would like to discuss the approach, feel free to reach out during the interview.
