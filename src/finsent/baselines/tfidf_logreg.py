"""Baseline TF-IDF (mots + n-grammes de caractères) + Régression Logistique.

Étapes : config -> données -> pipeline -> entraînement (train) -> évaluation (val, test) -> MLflow.

Lancement (depuis la racine du projet) :
    python -m finsent.baselines.tfidf_logreg
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from finsent import ROOT, tracking
from finsent.data.schema import SPLITS
from finsent.eval.evaluate import evaluate


def build_pipeline(cfg: dict) -> Pipeline:
    """TF-IDF mots + TF-IDF caractères, concaténés, puis régression logistique."""
    word, char, lr = cfg["tfidf"]["word"], cfg["tfidf"]["char"], cfg["logreg"]
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=tuple(word["ngram_range"]),
                                 min_df=word["min_df"], sublinear_tf=word["sublinear_tf"])),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(char["ngram_range"]),
                                 min_df=char["min_df"], sublinear_tf=char["sublinear_tf"])),
    ])
    classifier = LogisticRegression(C=lr["C"], class_weight=lr["class_weight"],
                                    max_iter=lr["max_iter"], random_state=cfg["seed"])
    return Pipeline([("features", features), ("clf", classifier)])


def flat_params(cfg: dict) -> dict:
    """Aplatit la config pour MLflow : {"tfidf": {"word": {"min_df": 2}}} -> {"word_min_df": 2}."""
    params = {"data": cfg["data"], "seed": cfg["seed"]}
    for kind in ("word", "char"):
        params.update({f"{kind}_{k}": v for k, v in cfg["tfidf"][kind].items()})
    params.update({f"logreg_{k}": v for k, v in cfg["logreg"].items()})
    return params


def single_value(df: pd.DataFrame, column: str) -> str:
    """Valeur unique d'une colonne (ex. lang="en") ; erreur si le jeu mélange plusieurs valeurs."""
    values = df[column].unique()
    if len(values) != 1:
        raise ValueError(f"colonne {column!r} non homogène : {list(values)}")
    return values[0]


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
    dataset = Path(cfg["data"]).stem  # "fpb"

    # 1. Données (déjà nettoyées et découpées par prepare_fpb)
    df = pd.read_parquet(ROOT / cfg["data"])
    parts = {s: df[df["split"] == s] for s in SPLITS}
    lang, domain = single_value(df, "lang"), single_value(df, "domain")

    tags = {
        "model_family": "tfidf_logreg",
        "model_name": "tfidf-word12-char25-logreg",
        "train_lang": lang, "train_domain": domain,
        "eval_lang": lang, "eval_domain": domain,  # même jeu en entraînement et en test
        "seed": cfg["seed"],
        "dataset": dataset,
    }

    with tracking.start_run("baseline", run_name=f"tfidf-logreg-{dataset}", tags=tags):
        mlflow.log_params(flat_params(cfg))
        for s, part in parts.items():
            tracking.log_dataset(part, name=f"{dataset}-{s}", source=cfg["data"], context=s)

        # 2. Entraînement : le TF-IDF (vocabulaire + IDF) n'apprend QUE sur le train
        pipe = build_pipeline(cfg)
        pipe.fit(parts["train"]["text"], parts["train"]["label_id"])
        n_features = len(pipe.named_steps["features"].get_feature_names_out())
        mlflow.log_metric("n_features", n_features)

        # 3. Évaluation : val pour régler les choix, test pour le score final
        print(f"Baseline TF-IDF + LogReg sur {dataset} ({n_features} features)")
        for s in ("val", "test"):
            metrics = evaluate(parts[s]["label_id"], pipe.predict(parts[s]["text"]), prefix=s)
            print(f"  {s:5s} macro-F1 = {metrics['macro_f1']:.3f}   "
                  f"accuracy = {metrics['accuracy']:.3f}")

        # 4. Modèle complet (TF-IDF + classifieur) sauvegardé, réutilisable tel quel
        mlflow.sklearn.log_model(pipe, name="model")


if __name__ == "__main__":
    main()
