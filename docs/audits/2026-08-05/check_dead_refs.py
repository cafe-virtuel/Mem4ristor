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
DOC_FILES = ["README.md", "REPRODUCE_RESULTS.md", "docs/CLAIMS_REGISTER.md"]

REF_RE = re.compile(r"experiments/scratch/[A-Za-z0-9_]+\.py")


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
    for doc in DOC_FILES:
        content = git("show", f"HEAD:{doc}")
        refs.update(REF_RE.findall(content))

    dead, relocated = [], []
    for ref in sorted(refs):
        basename = ref.rsplit("/", 1)[-1]
        hits = by_basename.get(basename, [])
        if hits:
            relocated.append((ref, hits))
        else:
            dead.append(ref)

    print(f"References `experiments/scratch/*.py` citees : {len(refs)}")
    print(f"  fichiers scannes : {', '.join(DOC_FILES)}\n")

    for ref, hits in relocated:
        print(f"RELOGEE : {ref}  ->  {', '.join(hits)}")
    for ref in dead:
        print(f"MORTE   : {ref}")

    print(f"\nTOTAL MORTES : {len(dead)} / {len(refs)}")
    if dead:
        print("\nUn clone propre ne peut pas rejouer ces scripts.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
