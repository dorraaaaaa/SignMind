"""
Script d'entraînement — scikit-learn MLPClassifier.
Aucune dépendance TensorFlow requise.

Usage:  python train.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from core.model import GestureModel


# ── Chargement du dataset ─────────────────────────────────────────────────────

def load_dataset(data_dir: str):
    root         = Path(data_dir)
    gesture_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

    if not gesture_dirs:
        raise FileNotFoundError(
            f"Aucun dossier de gestes dans '{data_dir}'.\n"
            "Lance  python main.py  → option 1  d'abord."
        )

    X_parts, y_parts = [], []

    for gdir in gesture_dirs:
        files           = sorted(gdir.glob("batch_*.npy"))
        gesture_samples = 0
        for f in files:
            batch = np.load(f)
            X_parts.append(batch)
            y_parts.extend([gdir.name] * len(batch))
            gesture_samples += len(batch)

        tag = "✓" if gesture_samples >= 50 else "⚠ peu"
        print(f"  {tag}  {gdir.name:<12}  {gesture_samples:>5} samples")

    return np.vstack(X_parts), np.array(y_parts), [d.name for d in gesture_dirs]


# ── Entraînement ──────────────────────────────────────────────────────────────

def train() -> None:
    print("\n" + "=" * 54)
    print("  ASL TRANSLATOR  —  Entraînement du modèle")
    print("=" * 54)

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # 1. Données
    print("\n[1/4]  Chargement du dataset …")
    X, y_raw, classes = load_dataset(config["data"]["gestures_dir"])
    print(f"\n       Total samples : {len(X)}")
    print(f"       Classes       : {len(classes)}  →  {classes}")

    if len(classes) < 2:
        print("\n[ERREUR] Il faut au moins 2 classes. Lance d'abord l'option 1.")
        return

    # 2. Encodage
    print("\n[2/4]  Encodage des labels …")
    le = LabelEncoder()
    le.fit(classes)
    y = le.transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )
    print(f"       Train : {len(X_train)}  |  Test : {len(X_test)}")

    # 3. Entraînement
    print("\n[3/4]  Entraînement MLP …  (patience=15 itérations sans amélioration)")
    gm = GestureModel(config)
    clf = gm.build(num_classes=len(classes))
    gm.class_names = list(le.classes_)

    clf.fit(X_train, y_train)

    # 4. Évaluation + sauvegarde
    print("\n[4/4]  Évaluation & sauvegarde …")
    acc_train = clf.score(X_train, y_train)
    acc_test  = clf.score(X_test,  y_test)

    print(f"\n       Précision train : {acc_train:.2%}")
    print(f"       Précision test  : {acc_test:.2%}")

    if acc_test < 0.90:
        print("\n  ⚠  Précision < 90%. Conseils :")
        print("     → Collecte plus de samples (300+ par geste)")
        print("     → Vérifie que chaque geste est bien distinct")

    Path("models").mkdir(exist_ok=True)
    gm.save("models/gesture_model.pkl", "models/classes.json")

    print(f"\n  ✓  Modèle sauvegardé  →  models/gesture_model.pkl")
    print(f"  ✓  Classes            →  models/classes.json")
    print(f"\n  Prochaine étape  →  python main.py  → option 3")

    # Rapport détaillé
    print("\n── Rapport par classe ────────────────────────────────")
    y_pred = clf.predict(X_test)
    present = sorted(set(y_test))
    names   = [le.classes_[i] for i in present]
    print(classification_report(y_test, y_pred, labels=present, target_names=names))

    _plot_loss(clf)


def _plot_loss(clf) -> None:
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(clf.loss_curve_, label="Train loss", color="#1D9E75")
        if hasattr(clf, "validation_scores_") and clf.validation_scores_:
            # sklearn stores val accuracy, not loss — invert for display
            val_loss = [1 - s for s in clf.validation_scores_]
            plt.plot(val_loss, label="Val loss (approx)", color="#F59E0B", linestyle="--")
        plt.title("Courbe de perte — MLP")
        plt.xlabel("Itérations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        Path("models").mkdir(exist_ok=True)
        plt.savefig("models/training_curves.png", dpi=150)
        plt.close()
        print(f"\n  ✓  Courbe sauvegardée  →  models/training_curves.png")
        print(f"     Ouvre ce fichier pour voir le graphe d'entraînement.")
    except Exception as exc:
        print(f"  [WARN] Impossible de tracer la courbe : {exc}")


if __name__ == "__main__":
    train()