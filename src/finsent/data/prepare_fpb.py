"""FinancialPhraseBank (EN, news) -> data/processed/fpb.parquet au schéma commun.

Étapes : lecture -> doublons -> conflits -> split stratifié -> schéma -> sauvegarde -> MLflow.

Lancement (depuis la racine du projet) :
    python -m finsent.data.prepare_fpb
"""

from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from finsent import ROOT, tracking
from finsent.data.schema import COLUMNS, LABEL2ID, LABELS, validate

AGREEMENT_FILES = {
    "50": "Sentences_50Agree.txt",
    "66": "Sentences_66Agree.txt",
    "75": "Sentences_75Agree.txt",
    "all": "Sentences_AllAgree.txt",
}


def load_fpb(path: Path) -> pd.DataFrame:
    """Lit un fichier FPB : une ligne = `phrase@label`, encodage latin-1."""
    rows = []
    for line in path.read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        text, _, label = line.rpartition("@")  # coupe sur le DERNIER @
        rows.append({"text": " ".join(text.split()), "label": label.strip()})
    return pd.DataFrame(rows)


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


def to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes du schéma commun et vérifie le résultat."""
    df = df.assign(label_id=df["label"].map(LABEL2ID), lang="en", domain="news")
    return validate(df[COLUMNS])


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))["fpb"]
    agreement = str(cfg["agreement"])
    raw_path = ROOT / cfg["raw_dir"] / AGREEMENT_FILES[agreement]
    out_path = ROOT / cfg["output"]

    # 1. Lecture
    raw = load_fpb(raw_path)
    # 2-3. Nettoyage (AVANT le split, pour éviter une même phrase en train ET en test)
    no_dup = remove_duplicates(raw)
    clean = remove_conflicts(no_dup)
    # 4. Split stratifié
    df = stratified_split(clean, cfg["split"], cfg["seed"])
    # 5. Schéma commun
    df = to_schema(df)
    # 6. Sauvegarde
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # Résumé
    counts = pd.crosstab(df["split"], df["label"]).reindex(index=["train", "val", "test"],
                                                           columns=LABELS)
    print(f"FPB {agreement}Agree : {len(raw)} lignes lues, "
          f"{len(raw) - len(no_dup)} doublons, {len(no_dup) - len(clean)} lignes en conflit "
          f"-> {len(df)} phrases")
    print(counts.to_string())
    print(f"-> {out_path.relative_to(ROOT)}")

    # 7. Traçabilité MLflow
    with tracking.start_run("data", run_name=f"fpb-{agreement}agree",
                            tags={"dataset": "fpb", "agreement": agreement}):
        mlflow.log_params({"agreement": agreement, "seed": cfg["seed"],
                           **{f"ratio_{k}": v for k, v in cfg["split"].items()}})
        mlflow.log_metrics({
            "n_raw": len(raw),
            "n_duplicates": len(raw) - len(no_dup),
            "n_conflicts": len(no_dup) - len(clean),
            **{f"n_{s}": int(counts.loc[s].sum()) for s in counts.index},
        })
        mlflow.log_text(counts.to_csv(), "label_distribution.csv")
        for s in counts.index:
            tracking.log_dataset(df[df["split"] == s], name=f"fpb-{agreement}agree-{s}",
                                 source=str(raw_path.relative_to(ROOT)), context=s)


if __name__ == "__main__":
    main()
