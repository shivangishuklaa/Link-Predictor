# Link Prediction on Social Graphs

Given a directed social network snapshot at time **T**, predict which edges will form at **T+1** — framed as binary classification over node pairs using graph topology features.

**Live demo →** [link-predictor-xcyhmdkyxnqq9cs5javwxs.streamlit.app](https://link-predictor-xcyhmdkyxnqq9cs5javwxs.streamlit.app/)

---

## Problem

Social networks are dynamic. Edges appear and disappear as users follow, unfollow, and reconnect. Link prediction answers: *given what we know about the graph right now, which connections are likely to form next?*

This has direct applications in friend/follow recommendation, fraud detection, knowledge graph completion, and protein interaction networks.

---

## Dataset

A large-scale directed graph representing follower/following relationships.

| | |
|---|---|
| Nodes | ~1.8 million |
| Edges | ~30 million |
| Graph type | Directed (u → v means u follows v) |
| Task | Binary classification — edge or no edge |

Negative examples are randomly sampled non-edges, matched 1:1 with positive edges for balanced classes. `random.seed(42)` is set once and never reset, so train/test splits are fully reproducible across notebooks.

---

## Pipeline

```
EDA.ipynb
  Load raw edge list → build DiGraph → analyse degree distributions
  → weakly connected components → sample negative edges → 80/20 split

Feature_eng.ipynb
  Load train graph → compute Katz centrality + HITS scores
  → build 20-feature matrix for all train/test pairs → serialise to CSV

Model_Training.ipynb
  Load feature matrix → train 8 classifiers → evaluate on held-out test set
  → RandomizedSearchCV on top-2 → save models + LinkPredictor class
```

---

## Features

20 graph-derived features per node pair, grouped by type:

| Type | Features |
|------|----------|
| Similarity | Jaccard similarity of follower sets, Jaccard similarity of followee sets |
| Distance | Cosine distance of follower sets, cosine distance of followee sets |
| Structural | In-degree and out-degree of source and destination |
| Overlap | Shared followers, shared followees |
| Graph | Shortest directed path length, same weakly connected component, Adamic-Adar index |
| Social | Is destination already following source |
| Centrality | Katz centrality (src, dst), HITS hub score (src, dst), HITS authority score (src, dst) |

All centrality features (Katz, HITS) are computed on the **training graph only** — no leakage from test edges.

---

## Results

Eight classifiers trained on the same features, evaluated on a held-out 20% split:

| Model | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| **LightGBM** | **0.9412** | 0.9435 | 0.9390 | **0.9821** |
| XGBoost | 0.9389 | 0.9410 | 0.9369 | 0.9798 |
| Random Forest | 0.9301 | 0.9312 | 0.9290 | 0.9712 |
| Extra Trees | 0.9288 | 0.9305 | 0.9271 | 0.9695 |
| Gradient Boosting | 0.9154 | 0.9180 | 0.9128 | 0.9541 |
| KNN | 0.8923 | 0.8950 | 0.8897 | 0.9312 |
| Decision Tree | 0.8741 | 0.8810 | 0.8673 | 0.9041 |
| Logistic Regression | 0.8102 | 0.8245 | 0.7963 | 0.8734 |

LightGBM and XGBoost were further tuned with `RandomizedSearchCV` (20 iterations, 3-fold CV, scoring on ROC-AUC), giving ~0.3–0.5% additional lift.

---

## Project Structure

```
├── app.py                  # Streamlit demo app
├── EDA.ipynb               # Graph construction, EDA, negative sampling
├── Feature_eng.ipynb       # Feature engineering — Katz, HITS, 20-feature matrix
├── Model_Training.ipynb    # Training, evaluation, tuning, inference class
├── requirements.txt
├── .gitignore
└── data/                   # gitignored — not committed
    ├── train.csv
    ├── X_train_pos.csv / X_train_neg.csv
    ├── X_test_pos.csv  / X_test_neg.csv
    ├── katz.pkl
    ├── hits.pkl
    ├── best_model.pkl
    └── predictor.pkl
```

---

## Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/link-prediction.git
cd link-prediction
pip install -r requirements.txt
streamlit run app.py
```

The app runs on the demo graph (300-node synthetic scale-free network) — no data files needed.

---

## Stack

NetworkX · scikit-learn · XGBoost · LightGBM · pandas · NumPy · Streamlit · joblib
