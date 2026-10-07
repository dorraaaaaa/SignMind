"""
Gesture classifier — scikit-learn MLPClassifier.

Même architecture que la version Keras (128 → 64 → 32),
mais sans dépendance TensorFlow :
  - sklearn est déjà installé (0 Mo à télécharger)
  - Entraînement 3× plus rapide sur CPU
  - Précision identique sur données tabulaires (landmarks normalisés)
  - Modèle sauvegardé en .pkl (~50 Ko vs ~5 Mo pour .keras)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier


class GestureModel:
    """
    Wrapper autour de sklearn MLPClassifier.

    Usage entraînement
    ------------------
    gm = GestureModel(config)
    gm.build(num_classes=29)
    gm.class_names = ["A", "B", ..., "SPACE", "DELETE", "NOTHING"]
    gm.clf.fit(X_train, y_train)
    gm.save("models/gesture_model.pkl", "models/classes.json")

    Usage inférence
    ---------------
    gm = GestureModel(config).load("models/gesture_model.pkl", "models/classes.json")
    gesture, confidence = gm.predict(landmarks)
    """

    def __init__(self, config: dict):
        self.config = config
        self.clf: MLPClassifier = None
        self.class_names: List[str] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, num_classes: int) -> MLPClassifier:
        """Instancie et retourne le MLPClassifier configuré."""
        cfg = self.config["model"]

        self.clf = MLPClassifier(
            hidden_layer_sizes=tuple(cfg["hidden_layers"]),   # (128, 64, 32)
            activation="relu",
            solver="adam",
            alpha=1e-4,                                        # L2 regularisation
            learning_rate_init=cfg["learning_rate"],
            max_iter=cfg["epochs"],
            batch_size=cfg["batch_size"],
            random_state=42,
            early_stopping=True,
            validation_fraction=cfg["validation_split"],
            n_iter_no_change=15,                               # patience
            verbose=True,
        )
        return self.clf

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_path: str, classes_path: str) -> None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self.clf, f)
        with open(classes_path, "w") as f:
            json.dump(self.class_names, f, indent=2)

    def load(self, model_path: str, classes_path: str) -> "GestureModel":
        with open(model_path, "rb") as f:
            self.clf = pickle.load(f)
        with open(classes_path) as f:
            self.class_names = json.load(f)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Classifie un vecteur de landmarks normalisé.

        Returns
        -------
        (gesture_name, confidence)   ex: ("A", 0.97)
        """
        probs = self.clf.predict_proba(landmarks.reshape(1, -1))[0]
        idx   = int(np.argmax(probs))
        return self.class_names[idx], float(probs[idx])