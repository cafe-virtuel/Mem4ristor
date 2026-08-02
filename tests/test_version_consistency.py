"""Coherence des metadonnees de version et de citation (02/08/2026, audit externe B8).

Contexte : le 02/08/2026, huit marqueurs de version divergeaient dans le depot, et le badge
DOI du README pointait vers `zenodo.19700749` -- c'est-a-dire la version **v3.2.0 du 22 avril
2026**, quatre versions en arriere de la V6 que le README annoncait, sous un tout autre titre
(« Frustrated Synchronization... »).

Verifie contre l'API Zenodo le jour meme : le concept `10.5281/zenodo.18620596` compte
**8 versions**, la plus recente est **V4.0.0 du 2026-05-02** (DOI de version
`10.5281/zenodo.19986042`). **Ni V5 ni V6 n'ont ete deposees** ; le code de ce depot est en
V6.0.0. C'est pourquoi la version du CODE et la version CITABLE different volontairement :
ce n'est pas une incoherence a « corriger » en les egalisant.

Ces tests gravent DEUX choses, pour qu'elles cessent de deriver en silence :

  (a) les trois marqueurs de version du CODE (`VERSION`, `pyproject.toml`, `__version__`)
      disent la meme chose ;
  (b) aucun fichier de surface ne cite un DOI Zenodo hors de la liste autorisee.

(b) est volontairement une allow-list, et non un simple test d'unicite : un DOI obsolete est
un DOI parfaitement **valide** qui resout vers une vraie page -- rien dans sa forme ne le
trahit. La seule facon de l'attraper est de nommer ceux qui sont acceptes.

>>> Quand une nouvelle version sera deposee sur Zenodo, ce test ECHOUERA tant que
>>> DOIS_AUTORISES n'aura pas ete mis a jour. C'est voulu : c'est le rappel.
"""

import re
from pathlib import Path

import pytest

import mem4ristor

ROOT = Path(__file__).resolve().parent.parent

# --- DOI Zenodo autorises dans les fichiers de surface -----------------------------------
# 18620596 : CONCEPT DOI (resout toujours vers la derniere version deposee)
# 19986042 : DOI de la version V4.0.0 (2026-05-02), derniere deposee au 02/08/2026
DOIS_AUTORISES = {"18620596", "19986042"}

# Fichiers lus par un tiers (relecteur, GitHub, Zenodo) avant tout le reste.
FICHIERS_DE_SURFACE = [
    "README.md",
    "CITATION.cff",
    "CONTEXT.md",
    "PROJECT_STATUS.md",
    "REPRODUCE_IN_5_MINUTES.md",
    "REPRODUCE_RESULTS.md",
    "RESULTS_INDEX.json",
    "docs/compendium/COMPENDIUM.md",
    "docs/compendium/COMPENDIUM.tex",
]

_RE_SEMVER = re.compile(r"[Vv]?(\d+\.\d+\.\d+)")
_RE_DOI_ZENODO = re.compile(r"zenodo\.(\d+)")


def _version_de_pyproject() -> str:
    texte = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    trouve = re.search(r'^version\s*=\s*"([^"]+)"', texte, re.MULTILINE)
    assert trouve, "pas de `version = \"...\"` en debut de ligne dans pyproject.toml"
    return trouve.group(1)


def _version_du_fichier_VERSION() -> str:
    texte = (ROOT / "VERSION").read_text(encoding="utf-8")
    trouve = _RE_SEMVER.search(texte)
    assert trouve, f"aucun numero de version lisible dans VERSION : {texte!r}"
    return trouve.group(1)


# --- (a) les trois marqueurs de version du CODE ------------------------------------------

def test_version_paquet_alignee_sur_pyproject():
    """`mem4ristor.__version__` doit suivre pyproject.toml.

    Le 02/08/2026 il valait encore "4.0.0" alors que pyproject declarait 6.0.0 : un
    utilisateur qui interrogeait le paquet obtenait la mauvaise version depuis deux versions.
    """
    assert mem4ristor.__version__ == _version_de_pyproject()


def test_fichier_VERSION_aligne_sur_pyproject():
    assert _version_du_fichier_VERSION() == _version_de_pyproject()


# --- (b) les DOI cites en surface ---------------------------------------------------------

@pytest.mark.parametrize("chemin_relatif", FICHIERS_DE_SURFACE)
def test_aucun_doi_zenodo_obsolete_en_surface(chemin_relatif):
    """Aucun fichier de surface ne cite un DOI Zenodo hors allow-list.

    Attrape precisement le defaut du 02/08/2026 : `zenodo.19700749` (v3.2.0, 22/04/2026)
    presente comme etant la V6 dans le badge et le bibtex du README.
    """
    chemin = ROOT / chemin_relatif
    if not chemin.exists():
        pytest.skip(f"{chemin_relatif} absent du depot")

    trouves = set(_RE_DOI_ZENODO.findall(chemin.read_text(encoding="utf-8")))
    interdits = sorted(trouves - DOIS_AUTORISES)
    assert not interdits, (
        f"{chemin_relatif} cite un ou des DOI Zenodo non autorises : {interdits}. "
        f"Autorises : {sorted(DOIS_AUTORISES)}. Si une nouvelle version vient d'etre "
        f"deposee, ajouter son DOI a DOIS_AUTORISES ; sinon, c'est un DOI obsolete."
    )


def test_citation_cff_decrit_la_version_deposee_pas_le_code():
    """CITATION.cff doit decrire la derniere version DEPOSEE, pas la version du code.

    Ce test verrouille une distinction, pas une egalite : au 02/08/2026 le code est en
    V6.0.0 et la derniere version citable est V4.0.0. Egaliser les deux ferait pointer une
    citation vers un depot qui n'existe pas -- c'est exactement ce que l'audit externe
    recommandait de faire, et qui aurait aggrave le probleme.
    """
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

    version_cff = re.search(r"^version:\s*(\S+)", cff, re.MULTILINE)
    assert version_cff, "CITATION.cff n'a pas de champ `version:`"

    doi_cff = _RE_DOI_ZENODO.search(cff)
    assert doi_cff, "CITATION.cff n'a pas de DOI Zenodo"
    assert doi_cff.group(1) in DOIS_AUTORISES

    # La version citee ne peut pas depasser celle du code : on ne cite pas du futur.
    def _tuple(v):
        return tuple(int(x) for x in v.lstrip("Vv").split("."))

    assert _tuple(version_cff.group(1)) <= _tuple(_version_de_pyproject()), (
        "CITATION.cff annonce une version superieure a celle du code."
    )
