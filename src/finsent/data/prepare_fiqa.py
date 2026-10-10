"""FiQA-SA (EN, titres + microblogs) -> data/processed/fiqa.parquet au schéma commun.

Étapes : lecture -> bruit technique -> vides -> score -> 3 classes -> doublons -> conflits
-> contrôle de fuite -> schéma -> sauvegarde -> MLflow.

Décisions (notebook 02, partie D) :
- splits d'origine gardés (train / valid -> val / test) ;
- seuils -0.1 / +0.3 (GMM + k-means + lecture), lus dans configs/data.yaml ;
- une phrase notée pour plusieurs cibles : gardée une fois si toutes ses cibles donnent
  la même classe, supprimée sinon (conflit) ;
- les variantes de seuils (analyse de sensibilité) sont préparées dans des fichiers à part.

Lancement (depuis la racine du projet) :
    python -m finsent.data.prepare_fiqa
"""

import ftfy
import mlflow
import numpy as np
import pandas as pd
import yaml

from finsent import ROOT, tracking
from finsent.data.common import clean_text, remove_conflicts, remove_duplicates, to_schema
from finsent.data.schema import LABELS, SPLITS

# Fichiers d'origine -> nom de split du schéma. L'ORDRE compte : en cas de doublon,
# remove_duplicates garde la première occurrence, donc celle du train.
RAW_SPLITS = {"train": "train", "valid": "val", "test": "test"}


def load_fiqa(raw_dir) -> pd.DataFrame:
    """Lit les 3 fichiers parquet, dans l'ordre train -> val -> test."""
    parts = [pd.read_parquet(raw_dir / f"{name}.parquet").assign(split=split)
             for name, split in RAW_SPLITS.items()]
    raw = pd.concat(parts, ignore_index=True)
    return raw.rename(columns={"sentence": "text"})[["text", "score", "split"]]


def discretize(scores: pd.Series, neg: float, pos: float) -> pd.Series:
    """score < neg -> negative ; score > pos -> positive ; sinon neutral."""
    labels = np.select([scores < neg, scores > pos], ["negative", "positive"], "neutral")
    return pd.Series(labels, index=scores.index)


def check_no_leak(df: pd.DataFrame) -> None:
    """Erreur si un même texte se trouve dans plusieurs splits."""
    leaks = df.groupby("text")["split"].nunique().gt(1).sum()
    if leaks:
        raise ValueError(f"{leaks} textes présents dans plusieurs splits")


def prepare(raw: pd.DataFrame, neg: float, pos: float) -> tuple[pd.DataFrame, dict]:
    """Nettoie et discrétise FiQA avec les seuils (neg, pos) ; renvoie (données, statistiques)."""
    # 1. Bruit technique (les reposts au lien différent deviennent des doublons exacts)
    df = raw.assign(text=raw["text"].map(clean_text))
    # 2. Textes vides
    not_empty = df[df["text"] != ""].reset_index(drop=True)
    # 3. Score -> 3 classes
    labeled = not_empty.assign(label=discretize(not_empty["score"], neg, pos))
    # 4-5. Une phrase plusieurs fois (plusieurs cibles, reposts) : même classe -> gardée une fois,
    #      classes différentes -> conflit, supprimée. Fait sur TOUS les splits ensemble.
    no_dup = remove_duplicates(labeled)
    clean = remove_conflicts(no_dup)
    # 6. Contrôle : plus aucune phrase partagée entre train / val / test
    check_no_leak(clean)
    stats = {
        "n_raw": len(raw),
        "n_encoding_fixed": int((raw["text"].map(ftfy.fix_encoding) != raw["text"]).sum()),
        "n_empty": len(raw) - len(not_empty),
        "n_duplicates": len(not_empty) - len(no_dup),
        "n_conflicts": len(no_dup) - len(clean),
    }
    return to_schema(clean, lang="en", domain="fiqa"), stats


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))["fiqa"]
    raw_dir = ROOT / cfg["raw_dir"]
    out_path = ROOT / cfg["output"]
    raw = load_fiqa(raw_dir)

    # Version principale + variantes de sensibilité (fichiers fiqa_<variante>.parquet)
    t = cfg["thresholds"]
    variants = {"main": (t["negative"], t["positive"]),
                **{name: tuple(bounds) for name, bounds in cfg["sensitivity"].items()}}

    for name, (neg, pos) in variants.items():
        path = out_path if name == "main" else out_path.with_stem(f"{out_path.stem}_{name}")
        df, stats = prepare(raw, neg, pos)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

        counts = pd.crosstab(df["split"], df["label"]).reindex(index=SPLITS, columns=LABELS)
        print(f"\nFiQA [{name}] seuils {neg:+} / {pos:+} : {stats['n_raw']} lignes lues, "
              f"{stats['n_encoding_fixed']} encodages réparés, {stats['n_empty']} vides, "
              f"{stats['n_duplicates']} doublons, {stats['n_conflicts']} lignes en conflit "
              f"-> {len(df)} phrases")
        print(counts.to_string())
        print(f"-> {path.relative_to(ROOT)}")

        run_name = "fiqa" if name == "main" else f"fiqa-{name}"
        with tracking.start_run("data", run_name=run_name,
                                tags={"dataset": "fiqa", "variant": name}):
            mlflow.log_params({"variant": name, "threshold_negative": neg,
                               "threshold_positive": pos})
            mlflow.log_metrics({**stats,
                                **{f"n_{s}": int(counts.loc[s].sum()) for s in counts.index}})
            mlflow.log_text(counts.to_csv(), "label_distribution.csv")
            for s in counts.index:
                tracking.log_dataset(df[df["split"] == s], name=f"{run_name}-{s}",
                                     source=cfg["raw_dir"], context=s)


if __name__ == "__main__":
    main()
