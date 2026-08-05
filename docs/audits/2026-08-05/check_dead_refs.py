"""
Recompte les references mortes vers experiments/scratch/*.py dans la doc versionnee.

Constat du crible du 2026-08-05 : la doc cite des scripts producteurs qui vivent
dans `experiments/scratch/`, dossier gitignore (`.gitignore:85`). Un clone propre
ne les a pas. Les 7 producteurs de CLAIMS ont ete reloges le 31/07 ; les scripts de
FIGURES du README, eux, sont restes dedans.

Pourquoi un script et pas un chiffre dans un rapport : une valeur figee dans une
phrase est une dette contractee au moment ou on l'ecrit (lecon du 02/08). Ce compte
doit se recalculer, pas se recopier.

Usage (depuis n'importe ou) :
    python docs/audits/2026-08-05/check_dead_refs.py

Sortie : liste des references, MORTE (absente du depot) ou RELOGEE (le fichier
existe ailleurs sous un autre chemin), puis le total. Exit 1 s'il reste des mortes.
"""

import re
import subprocess
import sys
from pathlib import Path

# La racine du depot, demandee a git -- surtout PAS deduite de __file__ :
# ce script vit dans docs/audits/<date>/ et un `parent.parent` se casserait
# au premier rangement (motif du 30/07 et du 02/08).
ROOT = Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True, text=True, check=True,
        encoding="utf-8", errors="replace",
    ).stdout.strip()
)

# Les fichiers de doc qui engagent la reproductibilite pour un tiers.
#
# experiments/FOLDER_SUMMARY.md ajoute le 2026-08-05 (demande de Julien) : il catalogue
# les scripts du dossier experiments/ et en cite des dizaines, mais il manquait a cette
# liste. Consequence mesuree le jour meme : event_phase_transition.py et son rerun y
# etaient cites tout en vivant dans scratch/, et ils n'ont jamais compte dans le total.
# Un compteur de dettes qui omet un fichier de doc ne mesure pas moins de dette : il
# mesure moins bien, ce qui est pire, parce que son chiffre a l'air complet.
#
# NE SONT PAS SCANNES, et c'est deliberé : PROJECT_HISTORY.md, AUDIT_LOG.md, sessions/*
# et les CHANGELOG sont des JOURNAUX DATES. Un script cite dans un journal decrit l'etat
# du depot ce jour-la ; ce n'est pas une promesse faite au lecteur d'aujourd'hui. On
# n'edite pas un journal, donc on ne compte pas ses references comme des dettes.
DOC_FILES = [
    "README.md",
    "REPRODUCE_RESULTS.md",
    "docs/CLAIMS_REGISTER.md",
    "experiments/FOLDER_SUMMARY.md",
]

# Les deux dettes sont comptees SEPAREMENT, et c'est le coeur de ce fichier.
#
#   PROMESSE  : un CHEMIN complet (experiments/scratch/x.py). Le document dit au lecteur
#               « lance ceci pour reproduire cette figure ». Si le fichier n'est pas
#               versionne, la promesse est intenable -> BLOQUANT (exit 1).
#   CATALOGUE : un NOM NU entre backticks (`x.py`), forme employee par FOLDER_SUMMARY,
#               qui inventorie le dossier experiments/. C'est une description, pas une
#               promesse -> INFORMATIF.
#
# Les fusionner donnerait un chiffre plus gros et moins utile : 69 entrees de catalogue
# noieraient 17 promesses intenables, et on ne saurait plus laquelle traiter. Meme
# principe que N4 dans tex_guardian.py, demarre en observation plutot qu'en bloquant.
REF_RE = re.compile(r"experiments/scratch/[A-Za-z0-9_]+\.py")
NAME_RE = re.compile(r"`([A-Za-z0-9_]+\.py)`")


def git(*args: str) -> str:
    # encoding explicite : sans lui, subprocess decode en cp1252 sous Windows et
    # meurt sur le premier caractere accentue de la doc (piege console du 02/08).
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True, text=True, check=False,
        encoding="utf-8", errors="replace",
    ).stdout or ""


def main() -> int:
    tracked = set(git("ls-files").splitlines())
    by_basename: dict[str, list[str]] = {}
    for path in tracked:
        by_basename.setdefault(path.rsplit("/", 1)[-1], []).append(path)

    refs: set[str] = set()
    names: dict[str, set[str]] = {}
    for doc in DOC_FILES:
        # On lit le WORKING TREE, pas `git show HEAD:` — sinon l'outil ne peut pas
        # servir a verifier une correction AVANT de la commiter, ce qui est
        # precisement le moment ou on en a besoin (constate le 05/08 : 20 references
        # repointees, compteur toujours a 17). La question « un clone l'aurait-il ? »
        # reste posee au bon endroit : la PRESENCE des fichiers vient de `git ls-files`.
        content = (ROOT / doc).read_text(encoding="utf-8", errors="replace")
        refs.update(REF_RE.findall(content))
        for n in NAME_RE.findall(content):
            names.setdefault(n, set()).add(doc)

    dead, relocated = [], []
    for ref in sorted(refs):
        basename = ref.rsplit("/", 1)[-1]
        hits = by_basename.get(basename, [])
        (relocated if hits else dead).append((ref, hits) if hits else ref)

    # Catalogue : noms nus introuvables. On retire ceux deja comptes comme promesses,
    # pour ne jamais faire porter la meme dette par les deux compteurs.
    deja = {r.rsplit("/", 1)[-1] for r in refs}
    orphelins = sorted(n for n in names if n not in by_basename and n not in deja)

    print("=== PROMESSES (bloquant) — chemins experiments/scratch/*.py cites ===")
    print(f"  fichiers scannes : {', '.join(DOC_FILES)}")
    print(f"  citees : {len(refs)}\n")
    for ref, hits in relocated:
        print(f"  RELOGEE : {ref}  ->  {', '.join(hits)}")
    for ref in dead:
        print(f"  MORTE   : {ref}")
    print(f"\n  TOTAL MORTES : {len(dead)} / {len(refs)}")
    if dead:
        print("  Un clone propre ne peut pas rejouer ces scripts.")

    print(f"\n=== CATALOGUE (informatif) — noms nus `x.py` introuvables : {len(orphelins)} ===")
    print("  Un document versionne les nomme comme s'ils existaient. Ce n'est pas une")
    print("  promesse de reproductibilite : c'est un inventaire perime par le rangement")
    print("  du 14/07, qui a deplace le contenu d'experiments/ vers scratch/.")
    for n in orphelins:
        print(f"  ABSENT : {n}   (cite dans {', '.join(sorted(names[n]))})")

    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())
