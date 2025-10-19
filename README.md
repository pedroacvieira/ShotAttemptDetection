# 🏀 Shot Attempt Detection Challenge

**Author:** Pedro Andrade Vieira  
**Date:** _Something_ October 2025  
**Kinexon Coding Challenge Submission**

---

## 🧭 Overview

This project implements a **first approach for detecting shot attempts in basketball games** using time-series data of player positions and skeleton keypoints.  
The solution includes:
1. A **baseline detection algorithm** based on pose and motion heuristics.
2. An **evaluation** against provided ground-truth shot events.
3. A **proposed system architecture** for scaling shot detection from raw video in a cloud environment.

The goal is not to achieve perfect accuracy but to demonstrate a **reasonable, explainable, and extensible approach**.

---

## 📁 Repository Structure

```
.
├── data/
│   ├── detections.csv
│   ├── player_positions.csv
│   └── shot_events.json
│
├── shot-attempt-detection/
│   ├── exploratory_data_analysis.py    # Print and plot data insights
│   ├── preprocess.py                   # Data loading & synchronization
│   ├── features.py                     # Pose/position feature computation
│   ├── detect_shots.py                 # Heuristic detection algorithm
│   └── evaluate.py                     # Evaluation against ground truth
│
├── output/
│   └── predicted_shots.csv    # Model output
│
├── architecture_diagram.png   # Cloud system design
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

### 2. Run EDA (Optional)
```bash
poetry run eda --detections data/detections.csv --positions data/player_positions.csv --output output/predicted_shots.csv
```

### 3. Run Detection
```bash
poetry run detect_shots --detections data/detections.csv --positions data/player_positions.csv --output output/predicted_shots.csv
```

### 4. Evaluate Results
```bash
poetry run evaluate --pred output/predicted_shots.csv --truth data/shot_events.json
```

---

## 🧩 Approach

### 1. Data Understanding

The provided data sources:
- **`detections.csv`** — bounding boxes and skeleton keypoints (normalized)
- **`player_positions.csv`** — player coordinates and motion data
- **`shot_events.json`** — ground-truth timestamps and player IDs for shot attempts

All data are synchronized on timestamp (ms).

---

### 2. Feature Engineering

For each player over time:
- **Vertical motion**: difference in y-coordinates of hands over time
  → Detects upward motion compared to head/hip.
Lower body stability (hip vs. feet)
- **Player velocity**: computed from position deltas  
  → Shots usually occur when horizontal motion is low.  
- **Temporal smoothing**: rolling median to reduce noise.

---

### 3. Detection Logic (Heuristic Baseline)

1. **Arm raise detection:**  
   Identify time windows where arm angle < threshold (arms above shoulders).

2. **Low movement filtering:**  
   Only keep frames where player speed < 25th percentile (indicating stationary shooting stance).

3. **Temporal grouping:**  
   Combine continuous detections into a single “shot window.”

4. **Timestamp assignment:**  
   Assign the midpoint of each window as the **shot timestamp**.

**Output format:**
```
timestamp_ms, player_id
```

---

### 4. Evaluation

Using `shot_events.json` as reference:
- A predicted event is **correct** if it falls within ±500 ms of a ground-truth shot.
- Metrics:
  - Precision
  - Recall
  - F1-score

Visualization plots are optionally generated for comparison.

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
- Integrate a **temporal ML model** (LSTM / Transformer) to learn shot dynamics.
- Use **multi-view camera data** for 3D motion context.
- Add **confidence scores** per prediction.
- Enable **real-time streaming inference** for live game analytics.

---

## 🧠 Assumptions
- Shot attempts involve visible arm elevation and minimal lateral motion.
- Each `player_id` in the data corresponds to a single tracked player.
- All timestamps are synchronized across sources.

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
1. Replace heuristic with an ML classifier trained on temporal pose sequences.  
2. Extend architecture for multi-camera input.  
3. Evaluate robustness on unseen games and partial occlusions.

---

## 💬 Contact
If you have any questions or would like to discuss the approach, feel free to reach out during the interview.
