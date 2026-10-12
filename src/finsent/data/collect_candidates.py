"""Propose des phrases candidates pour le jeu de test FR / AR tunisien (cases C et D).

Le script AIDE la collecte manuelle, il ne la remplace pas :
1. il lit les derniers articles des sources de configs/collect.yaml (flux RSS ou page de rubrique) ;
2. il extrait le texte principal de chaque article (trafilatura) et le découpe en phrases ;
3. il garde les phrases qui ressemblent à une « bonne phrase » (guide de collecte §4) ;
4. il écrit data/annotation/candidats/candidats_AAAA-MM-JJ.csv, au format de collecte.csv
   avec une colonne `keep` en plus.

Ensuite, à la main : keep = 1 sur les phrases à garder (au plus 3 par article), et compléter
company / sector. AUCUN label n'est produit : l'annotation se fait plus tard, à froid.

Politesse envers les sites : robots.txt respecté (y compris Crawl-delay), une page à la fois,
une pause entre deux pages d'un même site, et les articles déjà lus sont ignorés ensuite.

Lancement (depuis la racine du projet) :
    python -m finsent.data.collect_candidates
    python -m finsent.data.collect_candidates --sources ilboursa lapresse --max-articles 10
"""

import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date

import ftfy
import pandas as pd
import requests
import trafilatura
import yaml

from finsent import ROOT

COLUMNS = ["id", "text", "lang", "source", "url", "date", "company", "sector",
           "collector_note", "keep"]
ID_PREFIX = {"fr": "FR", "ar": "AR", "aeb": "TN"}

URL_OR_MAIL = re.compile(r"https?://|www\.|\S+@\S+")
# Fin de phrase : . ! ? ؟ … éventuellement suivis de guillemets ou parenthèse fermants
SENTENCE_END = re.compile(r"[.!?؟…][»\"”’)]*$")
# Abréviations qui finissent par un point sans finir la phrase (M. Ben Salah, S.A., J. Dupont)
ABBREVIATION = re.compile(r"(?:\b(?:M|MM|Mme|Mmes|Mlle|Dr|Pr|Me|St|Ste|Cie|av|bd|S\.A|"
                          r"S\.A\.R\.L|Inc|Ltd|Co|n°|No|vol|p)|\b[A-Z])\.$")


# ---------------------------------------------------------------- accès aux sites
class PoliteFetcher:
    """Télécharge des pages en respectant robots.txt et une pause minimale par site."""

    def __init__(self, user_agent: str, min_delay: float):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.user_agent = user_agent
        self.min_delay = min_delay
        self.robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self.last_request: dict[str, float] = {}

    def _robots(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        host = urllib.parse.urlsplit(url).netloc
        if host not in self.robots:
            parser = urllib.robotparser.RobotFileParser()
            try:
                r = self.session.get(f"https://{host}/robots.txt", timeout=20)
                # pas de robots.txt (404...) = pas de restriction ; sinon on applique ses règles
                parser.parse(r.text.splitlines() if r.status_code == 200 else [])
            except requests.RequestException:
                parser.parse([])
            self.robots[host] = parser
        return self.robots[host]

    def get(self, url: str) -> requests.Response | None:
        """Renvoie la réponse, ou None si robots.txt l'interdit ou si la requête échoue."""
        robots = self._robots(url)
        if not robots.can_fetch(self.user_agent, url):
            print(f"    interdit par robots.txt : {url}")
            return None
        host = urllib.parse.urlsplit(url).netloc
        delay = max(self.min_delay, robots.crawl_delay(self.user_agent) or 0)
        wait = self.last_request.get(host, 0) + delay - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self.last_request[host] = time.monotonic()
        try:
            r = self.session.get(url, timeout=30)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"    échec ({e.__class__.__name__}) : {url}")
            return None


def article_links(fetcher: PoliteFetcher, source: dict) -> list[str]:
    """Liens des derniers articles d'une source (flux RSS ou page de rubrique), sans doublon."""
    r = fetcher.get(source["url"])
    if r is None:
        return []
    if source["type"] == "feed":
        root = ET.fromstring(r.content)
        links = [item.findtext("link") for item in root.iter("item")]  # RSS
        links += [link.get("href") for entry in root.iter("{http://www.w3.org/2005/Atom}entry")
                  for link in entry.iter("{http://www.w3.org/2005/Atom}link")]  # Atom
    else:
        pattern = re.compile(source["link_pattern"])
        hrefs = [html.unescape(h) for h in re.findall(r'href="([^"]+)"', r.text)]
        links = [urllib.parse.urljoin(source["url"], h) for h in hrefs if pattern.search(h)]
    return list(dict.fromkeys(link.strip() for link in links if link))


def extract_article(fetcher: PoliteFetcher, url: str) -> dict | None:
    """Texte principal, titre et date d'un article (None si rien d'exploitable)."""
    r = fetcher.get(url)
    if r is None:
        return None
    # On donne les octets bruts : trafilatura détecte lui-même l'encodage de la page
    result = trafilatura.extract(r.content, url=url, output_format="json", with_metadata=True,
                                 include_comments=False, include_tables=False)
    if not result:
        return None
    data = json.loads(result)
    return {"text": data.get("text") or "", "title": data.get("title") or "",
            "date": data.get("date") or ""}


# ---------------------------------------------------------------- phrases
def split_sentences(text: str) -> list[str]:
    """Découpe un texte en phrases (un paragraphe par ligne chez trafilatura)."""
    sentences = []
    for paragraph in text.splitlines():
        paragraph = " ".join(ftfy.fix_encoding(paragraph).split())
        pieces = re.split(r"(?<=[.!?؟…»\"”’)])\s+", paragraph)
        buffer = ""
        for piece in pieces:
            buffer = f"{buffer} {piece}".strip()
            # on ne coupe que sur une vraie fin de phrase, pas après « M. » ou « S.A. »
            if SENTENCE_END.search(buffer) and not ABBREVIATION.search(buffer):
                sentences.append(buffer)
                buffer = ""
        if buffer:
            sentences.append(buffer)  # fin de paragraphe sans ponctuation : filtrée ensuite
    return sentences


def arabic_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("؀" <= c <= "ۿ" for c in letters) / len(letters) if letters else 0.0


def rejection_reason(sentence: str, lang: str, keywords: list[str], cfg: dict) -> str | None:
    """None si la phrase est une bonne candidate, sinon la raison du rejet (guide §4)."""
    n_words = len(sentence.split())
    if not cfg["min_words"] <= n_words <= cfg["max_words"]:
        return "longueur"
    if not SENTENCE_END.search(sentence) or sentence.endswith(("...", "…")):
        return "phrase incomplète"
    if URL_OR_MAIL.search(sentence):
        return "lien ou e-mail"
    share = arabic_share(sentence)
    if (lang == "ar" and share < 0.8) or (lang == "fr" and share > 0.05):
        return "mauvaise langue"
    if lang == "fr":
        if not (sentence[0].isupper() or sentence[0].isdigit() or sentence[0] in "«\"“"):
            return "phrase incomplète"
        if re.match(r"(Il|Elle|Ils|Elles)\b", sentence):
            return "pronom en tête (incompréhensible seule)"
    lowered = sentence.lower()
    if not (re.search(r"\d", sentence) or any(k in lowered for k in keywords)):
        return "pas de fait économique"
    return None


def dedup_key(text: str) -> str:
    """Clé de comparaison : minuscules, sans ponctuation ni espaces multiples."""
    return " ".join(re.sub(r"[^\w]+", " ", text.lower()).split())


# ---------------------------------------------------------------- fichiers existants
def read_csv_any(path) -> pd.DataFrame:
    """Lit un CSV avec virgules ou points-virgules, avec ou sans marque UTF-8 (Excel)."""
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str,
                       keep_default_na=False)


def load_history(collecte_path, output_dir) -> tuple[pd.DataFrame, set[str]]:
    """Phrases déjà collectées ou proposées (pour les id et les doublons) + URLs déjà lues."""
    frames = [read_csv_any(p) for p in [collecte_path, *sorted(output_dir.glob("candidats_*.csv"))]
              if p.exists()]
    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    seen_file = output_dir / "urls_lues.txt"
    seen = set(seen_file.read_text(encoding="utf-8").split()) if seen_file.exists() else set()
    return history, seen


def next_numbers(history: pd.DataFrame) -> Counter:
    """Dernier numéro utilisé par préfixe (FR-0042 -> FR: 42) : les id ne sont jamais réutilisés."""
    last = Counter()
    for id_ in history.get("id", []):
        match = re.fullmatch(r"([A-Z]+)-(\d+)", str(id_))
        if match:
            last[match[1]] = max(last[match[1]], int(match[2]))
    return last


# ---------------------------------------------------------------- programme principal
def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "collect.yaml").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", nargs="*", help="noms des sources (défaut : toutes)")
    parser.add_argument("--max-articles", type=int, default=cfg["max_articles"])
    args = parser.parse_args()

    output_dir = ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    history, seen_urls = load_history(ROOT / cfg["collecte"], output_dir)
    known = {dedup_key(t) for t in history.get("text", [])}
    last_number = next_numbers(history)

    fetcher = PoliteFetcher(cfg["user_agent"], cfg["min_delay"])
    sources = [s for s in cfg["sources"] if not args.sources or s["name"] in args.sources]
    rows, rejected = [], Counter()

    for source in sources:
        lang = source["lang"]
        keywords = [k.lower() for k in cfg["keywords"][lang]]
        links = [u for u in article_links(fetcher, source) if u not in seen_urls]
        links = links[:args.max_articles]
        print(f"\n[{source['name']}] {len(links)} nouveaux articles")
        n_source = 0
        for url in links:
            seen_urls.add(url)
            article = extract_article(fetcher, url)
            if article is None:
                continue
            if article["date"] and article["date"] < cfg["min_date"]:
                rejected["article trop ancien"] += 1
                continue
            n_article = 0
            for sentence in split_sentences(article["text"]):
                reason = rejection_reason(sentence, lang, keywords, cfg["sentence"])
                if reason is None and dedup_key(sentence) in known:
                    reason = "doublon"
                if reason:
                    rejected[reason] += 1
                    continue
                known.add(dedup_key(sentence))
                prefix = ID_PREFIX[lang]
                last_number[prefix] += 1
                is_title = dedup_key(sentence) == dedup_key(article["title"])
                rows.append({"id": f"{prefix}-{last_number[prefix]:04d}", "text": sentence,
                             "lang": lang, "source": source["name"], "url": url,
                             "date": article["date"], "company": "", "sector": "",
                             "collector_note": "titre" if is_title else "", "keep": ""})
                n_article += 1
                if n_article >= cfg["max_candidates"]:
                    break
            n_source += n_article
        print(f"[{source['name']}] {n_source} phrases candidates")

    # URLs lues : sauvegardées même si aucune phrase n'a été gardée (pas de relecture inutile)
    (output_dir / "urls_lues.txt").write_text("\n".join(sorted(seen_urls)) + "\n",
                                              encoding="utf-8")
    print("\nPhrases écartées :", dict(rejected.most_common()) or "aucune")
    if not rows:
        print("Aucune nouvelle phrase candidate.")
        return
    out_path = output_dir / f"candidats_{date.today().isoformat()}.csv"
    if out_path.exists():  # deuxième lancement le même jour : on ajoute à la suite
        rows = read_csv_any(out_path).to_dict("records") + rows
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"{len(rows)} phrases -> {out_path.relative_to(ROOT)}")
    print("À toi : keep = 1 sur les phrases à garder (au plus 3 par article), "
          "puis company / sector. Aucun label à ce stade.")


if __name__ == "__main__":
    main()
