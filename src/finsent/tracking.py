"""Conventions MLflow communes à tout le projet.

- backend local SQLite : mlflow.db à la racine
- une expérience par étape
- tags obligatoires pour tout run de modèle (pas pour la préparation des données)
"""

import warnings

import mlflow
import pandas as pd

from finsent import ROOT

TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"

EXPERIMENTS = {
    "data": "00-data-prep",
    "baseline": "01-baseline",
    "finetune": "02-finetune",
    "cross_lingual": "03-cross-lingual",
}

REQUIRED_TAGS = ["model_family", "model_name", "train_lang", "train_domain",
                 "eval_lang", "eval_domain", "seed"]


def start_run(stage: str, run_name: str, tags: dict | None = None) -> mlflow.ActiveRun:
    """Ouvre un run dans l'expérience de l'étape `stage` (à utiliser avec `with`)."""
    tags = {k: str(v) for k, v in (tags or {}).items()}
    if stage != "data":
        missing = [t for t in REQUIRED_TAGS if t not in tags]
        if missing:
            raise ValueError(f"tags MLflow obligatoires manquants : {missing}")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENTS[stage])
    return mlflow.start_run(run_name=run_name, tags=tags)


def log_dataset(df: pd.DataFrame, name: str, source: str, context: str) -> None:
    """Enregistre un jeu de données avec mlflow.log_input (traçabilité)."""
    with warnings.catch_warnings():  # avertissements MLflow sans conséquence ici
        warnings.simplefilter("ignore", UserWarning)
        dataset = mlflow.data.from_pandas(df, source=source, name=name)
        mlflow.log_input(dataset, context=context)
