"""Twitter Financial News (EN, tweets) -> data/processed/tweets.parquet au schéma commun.

Étapes : lecture + fusion -> labels -> nettoyage du bruit technique -> vides -> doublons
-> conflits -> split stratifié -> schéma -> sauvegarde -> MLflow.

Décisions (notebook 02, partie D) :
- option B : train et valid d'origine fusionnés puis redécoupés (fuite + pas de test) ;
- couche 1 seulement : on retire le bruit technique (URL, encodage cassé), on garde le langage
  (cashtags, mentions, hashtags, emojis, majuscules). Le reste se fait dans chaque modèle.

Lancement (depuis la racine du projet) :
    python -m finsent.data.prepare_tweets
"""

import re

import ftfy
import mlflow
import pandas as pd
import yaml

from finsent import ROOT, tracking
from finsent.data.common import remove_conflicts, remove_duplicates, stratified_split, to_schema
from finsent.data.schema import LABELS

# Labels du jeu d'origine (README) : 0 = Bearish, 1 = Bullish, 2 = Neutral.
# Attention : l'ordre n'est PAS celui de notre schéma, on traduit donc par NOM.
RAW_NAMES = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
NAME2LABEL = {"Bearish": "negative", "Bullish": "positive", "Neutral": "neutral"}

URL = re.compile(r"https?://\S+")


def load_tweets(raw_dir, files: list[str]) -> pd.DataFrame:
    """Lit et fusionne les fichiers CSV (colonnes text, label)."""
    return pd.concat([pd.read_csv(raw_dir / f) for f in files], ignore_index=True)


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Chiffre d'origine -> nom du README -> label du schéma ; erreur si chiffre inconnu."""
    unknown = set(df["label"]) - set(RAW_NAMES)
    if unknown:
        raise ValueError(f"labels d'origine inconnus : {unknown}")
    return df.assign(label=df["label"].map(RAW_NAMES).map(NAME2LABEL))


def clean_text(text: str) -> str:
    """Retire le bruit technique : encodage cassé (â€¦ -> …), URL, espaces multiples."""
    text = ftfy.fix_encoding(text)  # répare seulement l'encodage, ne touche pas au reste
    text = URL.sub(" ", text)
    return " ".join(text.split())


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))["tweets"]
    raw_dir = ROOT / cfg["raw_dir"]
    out_path = ROOT / cfg["output"]

    # 1. Lecture + fusion (option B)
    raw = load_tweets(raw_dir, cfg["files"])
    # 2. Labels traduits par nom
    df = map_labels(raw)
    # 3. Bruit technique (les doublons « cachés » par l'URL deviennent des doublons exacts)
    n_with_url = int(df["text"].str.contains(URL).sum())
    cleaned = df["text"].map(clean_text)
    n_encoding_fixed = int((df["text"].map(ftfy.fix_encoding) != df["text"]).sum())
    df = df.assign(text=cleaned)
    # 4. Tweets vides (ils ne contenaient qu'une URL)
    not_empty = df[df["text"] != ""].reset_index(drop=True)
    # 5-6. Doublons puis conflits, AVANT le split
    no_dup = remove_duplicates(not_empty)
    clean = remove_conflicts(no_dup)
    # 7. Split stratifié
    df = stratified_split(clean, cfg["split"], cfg["seed"])
    # 8. Schéma commun
    df = to_schema(df, lang="en", domain="tweets")
    # 9. Sauvegarde
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # Résumé
    counts = pd.crosstab(df["split"], df["label"]).reindex(index=["train", "val", "test"],
                                                           columns=LABELS)
    stats = {
        "n_raw": len(raw),
        "n_with_url": n_with_url,
        "n_encoding_fixed": n_encoding_fixed,
        "n_empty": len(raw) - len(not_empty),
        "n_duplicates": len(not_empty) - len(no_dup),
        "n_conflicts": len(no_dup) - len(clean),
    }
    print(f"Tweets : {stats['n_raw']} lus ({stats['n_with_url']} avec URL, "
          f"{stats['n_encoding_fixed']} encodages réparés), {stats['n_empty']} vides, "
          f"{stats['n_duplicates']} doublons, {stats['n_conflicts']} lignes en conflit "
          f"-> {len(df)} tweets")
    print(counts.to_string())
    print(f"-> {out_path.relative_to(ROOT)}")

    # 10. Traçabilité MLflow
    with tracking.start_run("data", run_name="tweets", tags={"dataset": "tweets"}):
        mlflow.log_params({"files": ",".join(cfg["files"]), "seed": cfg["seed"],
                           **{f"ratio_{k}": v for k, v in cfg["split"].items()}})
        mlflow.log_metrics({**stats, **{f"n_{s}": int(counts.loc[s].sum()) for s in counts.index}})
        mlflow.log_text(counts.to_csv(), "label_distribution.csv")
        for s in counts.index:
            tracking.log_dataset(df[df["split"] == s], name=f"tweets-{s}",
                                 source=cfg["raw_dir"], context=s)


if __name__ == "__main__":
    main()
