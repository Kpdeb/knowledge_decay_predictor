"""
Prediction helpers — loads saved models and runs inference.
"""

import os
import math
import numpy as np
import joblib
from typing import Optional

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")

_rf     = None
_lr     = None
_scaler = None


def _load_models():
    global _rf, _lr, _scaler
    rf_path = os.path.join(MODEL_DIR, "random_forest.joblib")
    lr_path = os.path.join(MODEL_DIR, "logistic_regression.joblib")
    sc_path = os.path.join(MODEL_DIR, "scaler.joblib")

    if os.path.exists(rf_path):
        _rf     = joblib.load(rf_path)
        _lr     = joblib.load(lr_path)
        _scaler = joblib.load(sc_path)
        return True
    return False


def _rule_based(t_hours: float, score: float, difficulty: int,
                review_count: int, study_duration: int) -> float:
    """Fallback: Ebbinghaus formula."""
    diff_factor = {0: 1.5, 1: 1.0, 2: 0.7}.get(difficulty, 1.0)
    s = 24 * diff_factor * (1 + score / 100) * (1 + review_count * 0.3) * (1 + study_duration / 200)
    return max(0.0, min(1.0, math.exp(-t_hours / s)))


def predict_retention(
    time_since_last_review: float,
    quiz_score: float,
    difficulty: int,
    review_count: int,
    study_duration: int,
) -> dict:
    """
    Returns retention probability from both rule-based and ML models.
    """
    rule_ret = _rule_based(time_since_last_review, quiz_score, difficulty, review_count, study_duration)

    ml_ret: Optional[float] = None
    will_forget: Optional[bool] = None

    if _rf is not None or _load_models():
        features = np.array([[time_since_last_review, quiz_score, difficulty, review_count, study_duration]])
        ml_ret = float(np.clip(_rf.predict(features)[0], 0.0, 1.0))

        features_scaled = _scaler.transform(features)
        will_forget = bool(_lr.predict(features_scaled)[0])

    return {
        "retention_rule_based": round(rule_ret, 4),
        "retention_ml": round(ml_ret, 4) if ml_ret is not None else None,
        "retention_probability": round(ml_ret if ml_ret is not None else rule_ret, 4),
        "will_forget_in_7_days": will_forget,
        "model_used": "random_forest" if ml_ret is not None else "rule_based",
    }
