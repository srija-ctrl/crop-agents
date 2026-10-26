"""Steps 2-3: train a lightweight classifier and expose it as a callable tool."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, LABEL, MODEL_PATH, OUTPUT_DIR, RANDOM_SEED, TEST_SIZE


def train_and_save(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    X, y = df[FEATURES], df[LABEL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_seed": RANDOM_SEED,
        "feature_importance": {
            f: round(float(v), 4) for f, v in sorted(zip(FEATURES, clf.feature_importances_), key=lambda t: -t[1])
        },
        "report": classification_report(y_test, y_pred, zero_division=0),
    }

    # nearest-neighbour index on standardised training rows (similarity evidence)
    scaler = StandardScaler().fit(X_train)
    nn = NearestNeighbors(n_neighbors=1).fit(scaler.transform(X_train))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"clf": clf, "scaler": scaler, "nn": nn, "train_labels": y_train.to_numpy(), "metrics": metrics},
        model_path,
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


class CropPredictorTool:
    """The dataset-derived tool called by the Crop Recommendation Agent.
    The crop label ALWAYS comes from here, never from the LLM."""

    name = "crop_predictor"

    def __init__(self, bundle: Dict[str, Any]):
        self.clf = bundle["clf"]
        self.scaler = bundle["scaler"]
        self.nn = bundle["nn"]
        self.train_labels = bundle["train_labels"]
        self.metrics = bundle["metrics"]

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "CropPredictorTool":
        return cls(joblib.load(path))

    def predict(self, measurements: Dict[str, float], top_k: int = 3) -> Dict[str, Any]:
        row = pd.DataFrame([{f: float(measurements[f]) for f in FEATURES}])
        proba = self.clf.predict_proba(row)[0]
        order = np.argsort(proba)[::-1][:top_k]
        top = [{"crop": str(self.clf.classes_[i]), "probability": round(float(proba[i]), 4)} for i in order]

        dist, idx = self.nn.kneighbors(self.scaler.transform(row))
        nn_label = str(self.train_labels[idx[0][0]])
        d = float(dist[0][0])
        return {
            "label": top[0]["crop"],
            "probability": top[0]["probability"],
            "top_k": top,
            "nearest_neighbor": {
                "crop": nn_label,
                "distance": round(d, 4),
                "similarity": round(1.0 / (1.0 + d), 4),
                "agrees_with_label": nn_label == top[0]["crop"],
            },
        }
