"""
ML Service — Training Script
Trains a Random Forest + Logistic Regression on synthetic study data.
Saves models to models/saved/ for the prediction API to load.
"""

import os
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
os.makedirs(SAVE_DIR, exist_ok=True)

DIFFICULTY_MAP = {"easy": 0, "medium": 1, "hard": 2}
DIFF_FACTOR = {0: 1.5, 1: 1.0, 2: 0.7}


def ebbinghaus_retention(t_hours, score, review_count, difficulty, study_duration):
    """Ground-truth label generator (Ebbinghaus + noise)."""
    diff_factor = DIFF_FACTOR.get(difficulty, 1.0)
    s = 24 * diff_factor * (1 + score / 100) * (1 + review_count * 0.3) * (1 + study_duration / 200)
    r = math.exp(-t_hours / s)
    return max(0.0, min(1.0, r))


def generate_dataset(n=5000, seed=42):
    """Generate synthetic study session data with realistic distributions."""
    rng = np.random.default_rng(seed)

    time_since = rng.exponential(scale=72, size=n).clip(0.5, 720)  # hours
    quiz_score = rng.beta(5, 2, size=n) * 100                       # skewed high
    difficulty = rng.integers(0, 3, size=n)
    review_count = rng.integers(0, 11, size=n)
    study_duration = rng.integers(10, 121, size=n)

    # Compute labels with small Gaussian noise
    retention = np.array([
        ebbinghaus_retention(t, s, r, d, dur)
        for t, s, r, d, dur in zip(time_since, quiz_score, difficulty, review_count, study_duration)
    ])
    noise = rng.normal(0, 0.02, size=n)
    retention_noisy = np.clip(retention + noise, 0.0, 1.0)

    df = pd.DataFrame({
        "time_since_last_review": time_since,
        "quiz_score": quiz_score,
        "difficulty": difficulty,
        "review_count": review_count,
        "study_duration": study_duration,
        "retention": retention_noisy,
    })
    return df


def train():
    print("Generating synthetic dataset…")
    df = generate_dataset(n=8000)

    # Save dataset sample
    df.to_csv(os.path.join(SAVE_DIR, "..", "sample_dataset.csv"), index=False)
    print(f"Dataset: {len(df)} rows, retention range [{df['retention'].min():.3f}, {df['retention'].max():.3f}]")

    FEATURES = ["time_since_last_review", "quiz_score", "difficulty", "review_count", "study_duration"]
    X = df[FEATURES].values
    y = df["retention"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── Scaler ────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Random Forest (primary model) ─────────────────────────────────────────
    print("\nTraining Random Forest…")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2  = r2_score(y_test, rf_preds)
    print(f"  Random Forest — MAE: {rf_mae:.4f}  R²: {rf_r2:.4f}")

    # ── Logistic Regression (secondary — binary: will_forget in 7 days) ───────
    print("\nTraining Logistic Regression (will forget in 7d?)…")
    y_binary = (y < 0.5).astype(int)
    _, _, y_train_b, y_test_b = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    lr = LogisticRegression(max_iter=500, C=1.0)
    lr.fit(X_train_scaled, y_train_b)
    lr_acc = lr.score(X_test_scaled, y_test_b)
    print(f"  Logistic Regression accuracy: {lr_acc:.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    joblib.dump(rf,     os.path.join(SAVE_DIR, "random_forest.joblib"))
    joblib.dump(lr,     os.path.join(SAVE_DIR, "logistic_regression.joblib"))
    joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.joblib"))

    # Save feature importance
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature importances:")
    print(importance.to_string(index=False))

    print(f"\nModels saved to {SAVE_DIR}")
    return {"rf_mae": rf_mae, "rf_r2": rf_r2, "lr_acc": lr_acc}


if __name__ == "__main__":
    metrics = train()
    print("\nTraining complete:", metrics)
