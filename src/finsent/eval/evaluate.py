"""Évaluation commune à tous les modèles du projet.

Métrique principale : macro-F1 (les classes sont déséquilibrées, "neutral" domine).
On calcule aussi l'accuracy, la F1 par classe et la matrice de confusion.

Usage, dans un run MLflow ouvert :
    evaluate(y_true, y_pred, prefix="test")
"""

import matplotlib

matplotlib.use("Agg")  # dessine dans un fichier, sans ouvrir de fenêtre

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score

from finsent.data.schema import LABELS

# Ordre fixe des classes : 0=negative, 1=neutral, 2=positive (cf. schema.LABEL2ID)
LABEL_IDS = list(range(len(LABELS)))


def check_labels(y_true, y_pred) -> None:
    """Vérifie que vérité et prédictions sont des label_id connus et de même longueur."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"longueurs différentes : {len(y_true)} vrais vs {len(y_pred)} prédits")
    unknown = (set(y_true) | set(y_pred)) - set(LABEL_IDS)
    if unknown:
        raise ValueError(f"label_id inconnus : {unknown} (attendus : {LABEL_IDS})")


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """macro-F1, accuracy et F1 de chaque classe."""
    check_labels(y_true, y_pred)
    per_class = f1_score(y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro",
                                   zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        **{f"f1_{label}": float(score) for label, score in zip(LABELS, per_class)},
    }


def confusion_table(y_true, y_pred) -> pd.DataFrame:
    """Matrice de confusion : lignes = vraie classe, colonnes = classe prédite."""
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    return pd.DataFrame(cm, index=[f"true_{label}" for label in LABELS],
                        columns=[f"pred_{label}" for label in LABELS])


def plot_confusion(table: pd.DataFrame, title: str) -> plt.Figure:
    """Image de la matrice de confusion (comptes bruts)."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(table.to_numpy(), display_labels=LABELS).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def evaluate(y_true, y_pred, prefix: str) -> dict[str, float]:
    """Calcule les métriques sur un jeu (`prefix` = "val", "test", "test_fr"...).

    Si un run MLflow est ouvert, enregistre aussi les métriques (préfixées)
    et la matrice de confusion (CSV + PNG) dans le dossier d'artefacts `prefix/`.
    """
    metrics = compute_metrics(y_true, y_pred)
    table = confusion_table(y_true, y_pred)

    if mlflow.active_run() is not None:
        mlflow.log_metrics({f"{prefix}_{name}": value for name, value in metrics.items()})
        mlflow.log_text(table.to_csv(), f"{prefix}/confusion_matrix.csv")
        fig = plot_confusion(table, title=prefix)
        mlflow.log_figure(fig, f"{prefix}/confusion_matrix.png")
        plt.close(fig)  # libère la mémoire (important quand on évalue beaucoup de runs)

    return metrics
