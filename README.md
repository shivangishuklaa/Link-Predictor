# 🔗 Link Prediction on Social Graphs

Predicting whether a directed edge **(u → v)** will form between two nodes in a large-scale social network, using graph topology features and an ensemble of machine learning classifiers.



---

## 📊 Dataset

**Source:** [Facebook Recruiting Competition — Kaggle](https://www.kaggle.com/c/FacebookRecruiting)

> Download `train.csv` from the competition page and place it at `data/train.csv`. The file contains two columns — `source_node` and `destination_node` — representing a directed follower/following graph.

| Stat | Value |
|------|-------|
| Total edges | ~30 million |
| Unique nodes | ~1.8 million |
| Graph type | Directed (follower/following) |
| Task | Binary classification (edge / no edge) |

Negative edges are sampled randomly with `random.seed(42)` to guarantee exact 50/50 class balance and full reproducibility.

---

## 🚀 Live Demo

**Deployed app:** [https://link-predictor-xcyhmdkyxnqq9cs5javwxs.streamlit.app/](https://link-predictor-xcyhmdkyxnqq9cs5javwxs.streamlit.app/)

Or run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The Streamlit app includes:
- **Overview** — pipeline summary and demo graph statistics
- **Live Predictor** — enter any two node IDs and get a real-time prediction with feature breakdown and ego-graph visualisation
- **EDA & Features** — degree distributions, WCC analysis, feature catalogue, centrality charts
- **Models** — leaderboard, metric charts, and ROC curves for all 8 classifiers
- **Project** — problem context, notebook guide, and key results

---

## 🛠️ Pipeline

```
EDA.ipynb
  └─ Load raw CSV → build DiGraph → degree analysis → negative sampling → train/test split

Feature_eng.ipynb
  └─ Load train graph → compute Katz & HITS → build 20-feature matrix → save CSVs

Model_Training.ipynb
  └─ Load features → train 8 classifiers → evaluate → tune top-2 → save models + LinkPredictor
```

---

## 🧠 Features (20 total)

| Category | Features |
|----------|----------|
| **Similarity** | Jaccard (followers), Jaccard (followees) |
| **Distance** | Cosine distance (followers), Cosine distance (followees) |
| **Structural** | #Followers src/dst, #Followees src/dst |
| **Overlap** | Intersection of followers, Intersection of followees |
| **Graph** | Shortest path, Same WCC, Adamic-Adar index |
| **Social** | Is following back |
| **Centrality** | Katz (src/dst), HITS hub (src/dst), HITS authority (src/dst) |

---

## 🤖 Models & Results

Eight classifiers trained and compared on the same 80/20 split:

| Model | F1 | ROC-AUC |
|-------|----|---------|
| **LightGBM** ⭐ | 0.9412 | 0.9821 |
| XGBoost | 0.9389 | 0.9798 |
| Random Forest | 0.9301 | 0.9712 |
| Extra Trees | 0.9288 | 0.9695 |
| Gradient Boosting | 0.9154 | 0.9541 |
| KNN | 0.8923 | 0.9312 |
| Decision Tree | 0.8741 | 0.9041 |
| Logistic Regression | 0.8102 | 0.8734 |

Top-2 models (LightGBM, XGBoost) further tuned with `RandomizedSearchCV` (20 iterations, 3-fold CV on ROC-AUC), yielding ~0.3–0.5% additional improvement.

---

## 📁 Project Structure

```
├── app.py                  # Streamlit demo app
├── EDA.ipynb               # Graph construction, EDA, negative sampling
├── Feature_eng.ipynb       # Feature computation (Katz, HITS, graph features)
├── Model_Training.ipynb    # Training, evaluation, tuning, inference class
├── requirements.txt
├── .gitignore
└── data/                   # (gitignored) CSVs, pickles, saved models
    ├── train.csv
    ├── X_train_pos/neg.csv
    ├── X_test_pos/neg.csv
    ├── katz.pkl
    ├── hits.pkl
    ├── best_model.pkl
    └── predictor.pkl
```

---

## ⚙️ Tech Stack

`NetworkX` · `scikit-learn` · `XGBoost` · `LightGBM` · `pandas` · `NumPy` · `Streamlit` · `joblib`

---

## 🔑 Key Design Decisions

- **No data leakage** — Katz and HITS centralities are computed on the *training graph only*; test edges are never seen during feature computation.
- **Reproducibility** — `random.seed(42)` is set once and never reset between train/test sampling; same seed used in both `Feature_eng.ipynb` and `Model_Training.ipynb`.
- **Production-ready inference** — `LinkPredictor` class holds all state explicitly (no globals), handles unseen nodes gracefully, and is serialised with `joblib` for deployment.
