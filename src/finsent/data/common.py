"""Étapes de préparation communes à tous les jeux de données (FPB, tweets, FiQA...).

Ordre à respecter dans chaque prepare_*.py :
nettoyage (doublons, conflits) AVANT le split, pour qu'une même phrase
ne puisse pas se retrouver en train ET en test.
"""

import re

import ftfy
import pandas as pd
from sklearn.model_selection import train_test_split

from finsent.data.schema import COLUMNS, LABEL2ID, validate

# Liens : avec schéma (https://t.co/...) ou sans (www.site.com)
URL = re.compile(r"https?://\S+|www\.\S+")


def clean_text(text: str) -> str:
    """Couche 1 : retire le bruit technique (encodage cassé â€¦ -> …, URL, espaces multiples).

    Ne touche pas au langage (majuscules, cashtags, mentions, emojis, apostrophes) :
    c'est le rôle de chaque modèle.
    """
    text = ftfy.fix_encoding(text)  # répare seulement l'encodage (pas fix_text, trop large)
    text = URL.sub(" ", text)
    return " ".join(text.split())


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Garde une seule fois chaque (phrase, label) identique."""
    return df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)


def remove_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les phrases présentes avec plusieurs labels différents."""
    conflicting = df["text"].duplicated(keep=False)
    return df[~conflicting].reset_index(drop=True)


def stratified_split(df: pd.DataFrame, ratios: dict, seed: int) -> pd.DataFrame:
    """Découpe en train / val / test en gardant la même proportion de classes partout."""
    train, rest = train_test_split(
        df, train_size=ratios["train"], stratify=df["label"], random_state=seed
    )
    # `rest` = val + test ; on le recoupe selon leurs proportions relatives
    val_share = ratios["val"] / (ratios["val"] + ratios["test"])
    val, test = train_test_split(
        rest, train_size=val_share, stratify=rest["label"], random_state=seed
    )
    return pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")],
        ignore_index=True,
    )


def to_schema(df: pd.DataFrame, lang: str, domain: str) -> pd.DataFrame:
    """Ajoute les colonnes du schéma commun et vérifie le résultat."""
    df = df.assign(label_id=df["label"].map(LABEL2ID), lang=lang, domain=domain)
    return validate(df[COLUMNS])
