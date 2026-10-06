"""Schéma commun à tous les jeux de données préparés (FPB, FiQA, tweets, FR, TN)."""

import pandas as pd

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}  # negative=0, neutral=1, positive=2
COLUMNS = ["text", "label", "label_id", "lang", "domain", "split"]
SPLITS = ["train", "val", "test"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie que `df` respecte le schéma commun ; lève une erreur sinon."""
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"colonnes manquantes : {sorted(missing)}")
    if df["text"].isna().any() or (df["text"].str.strip() == "").any():
        raise ValueError("textes vides")
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"labels inconnus : {unknown}")
    if not (df["label"].map(LABEL2ID) == df["label_id"]).all():
        raise ValueError("label_id incohérent avec label")
    if not set(df["split"]) <= set(SPLITS):
        raise ValueError(f"splits inconnus : {set(df['split']) - set(SPLITS)}")
    return df
