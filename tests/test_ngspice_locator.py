"""Le locator ne doit JAMAIS rendre un build silencieux si une option existe.

Ces tests sont hermetiques : ils ne dependent ni de ngspice etant installe, ni
d'un emplacement reel. Tout est fabrique dans `tmp_path`, et le PATH comme la
variable NGSPICE sont monkeypatches.

Le piege qu'ils verrouillent, mesure le 2026-10-03 sur ngspice-46_64 :
`ngspice.exe` (build graphique) rend 0 octet sur stdout en mode batch avec le
code de sortie 0, pendant que `ngspice_con.exe` rend 2 434 octets. Les six
scripts SPICE ne testent que le code de retour : avec le mauvais binaire, ils
croiraient avoir obtenu un resultat qui n'existe pas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import ngspice_locator as loc  # noqa: E402


def _faux_arbre(dossier: Path, noms: list[str]) -> Path:
    """Cree un dossier avec des binaires vides, et retourne ce dossier."""
    dossier.mkdir(parents=True, exist_ok=True)
    for nom in noms:
        (dossier / nom).write_bytes(b"")
    return dossier


@pytest.fixture(autouse=True)
def _environnement_propre(monkeypatch, tmp_path):
    """Neutralise la machine : aucun NGSPICE, aucun emplacement connu reel."""
    monkeypatch.delenv("NGSPICE", raising=False)
    faux = tmp_path / "connus"
    faux.mkdir()
    monkeypatch.setattr(loc, "EMPLACEMENTS_CONNUS", [faux / "ngspice_con.exe"])


# ---------------------------------------------------------------- Windows

def test_constat_de_base_le_build_graphique_est_silencieux():
    """Constat mesure, pas une opinion : sur Windows, le nom `ngspice` suffit."""
    if loc.os.name != "nt":
        pytest.skip("comportement propre a Windows")
    assert loc.est_build_silencieux(Path("C:/x/ngspice.exe")) is True
    assert loc.est_build_silencieux(Path("C:/x/ngspice_con.exe")) is False


def test_le_locator_prefere_le_build_console_du_path(monkeypatch, tmp_path):
    """Les deux dans le PATH : le console gagne. C'est le cas d'un installeur normal."""
    if loc.os.name != "nt":
        pytest.skip("comportement propre a Windows")
    dossier = _faux_arbre(tmp_path / "bin", ["ngspice.exe", "ngspice_con.exe"])

    def faux_which(nom):
        return str(dossier / nom) if (dossier / nom).exists() else None

    monkeypatch.setattr(loc.shutil, "which", faux_which)
    assert loc.trouver_ngspice().name.lower() == "ngspice_con.exe"


def test_le_locator_refuse_le_build_graphique_si_mieux_existe(monkeypatch, tmp_path):
    """LE PIEGE : seul le graphique est dans le PATH, mais un emplacement connu existe.

    Avant le correctif, le locator rendait le build muet. Maintenant il continue
    a chercher, et trouve l'emplacement connu.
    """
    if loc.os.name != "nt":
        pytest.skip("comportement propre a Windows")
    dossier = _faux_arbre(tmp_path / "bin", ["ngspice.exe"])
    connu = loc.EMPLACEMENTS_CONNUS[0]
    connu.parent.mkdir(parents=True, exist_ok=True)
    connu.write_bytes(b"")

    def faux_which(nom):
        return str(dossier / nom) if (dossier / nom).exists() else None

    monkeypatch.setattr(loc.shutil, "which", faux_which)
    rendu = loc.trouver_ngspice()
    assert rendu == connu, "le build silencieux a ete rendu alors qu'une option meilleure existait"
    assert not loc.est_build_silencieux(rendu)


def test_ngspice_env_mal_pointe_ne_poisonne_pas(monkeypatch, tmp_path):
    """NGSPICE pointe sur le build muet : on ne s'y fie pas, on cherche mieux."""
    if loc.os.name != "nt":
        pytest.skip("comportement propre a Windows")
    graphique = _faux_arbre(tmp_path / "env", ["ngspice.exe"]) / "ngspice.exe"
    connu = loc.EMPLACEMENTS_CONNUS[0]
    connu.parent.mkdir(parents=True, exist_ok=True)
    connu.write_bytes(b"")

    monkeypatch.setenv("NGSPICE", str(graphique))
    monkeypatch.setattr(loc.shutil, "which", lambda nom: None)
    rendu = loc.trouver_ngspice()
    assert rendu == connu, "une variable d'environnement mal pointee doit etre ecartee"


def test_ngspice_env_bien_pointe_est_respecte(monkeypatch):
    """Un NGSPICE correct reste prioritaire : pas de comportement surprise."""
    cible = Path("C:/outils/ngspice_con.exe")
    monkeypatch.setenv("NGSPICE", str(cible))
    monkeypatch.setattr(loc.shutil, "which", lambda nom: None)
    assert loc.trouver_ngspice() == cible


def test_en_dernier_recours_le_build_muet_est_rendu_mais_signale(monkeypatch, tmp_path):
    """Rien d'autre n'existe : on rend le build muet, MAIS on avertit.

    Refuser categoriquement casserait un tiers qui n'a que lui. Le rendre en
    silence reproduirait la panne. L'avertissement est la seule position tenable.
    """
    if loc.os.name != "nt":
        pytest.skip("comportement propre a Windows")
    graphique = _faux_arbre(tmp_path / "seul", ["ngspice.exe"]) / "ngspice.exe"
    monkeypatch.setenv("NGSPICE", str(graphique))
    monkeypatch.setattr(loc.shutil, "which", lambda nom: None)
    loc.EMPLACEMENTS_CONNUS[0].parent.mkdir(parents=True, exist_ok=True)

    with pytest.warns(RuntimeWarning, match="build GRAPHIQUE"):
        rendu = loc.trouver_ngspice()
    assert rendu == graphique


def test_aucun_avertissement_quand_le_chemin_est_sain(monkeypatch, tmp_path):
    """Un run normal ne doit pas polluer la sortie d'un avertissement."""
    if loc.os.name != "nt":
        pytest.skip("comportement propre a Windows")
    dossier = _faux_arbre(tmp_path / "ok", ["ngspice_con.exe"])
    monkeypatch.delenv("NGSPICE", raising=False)
    monkeypatch.setattr(loc.shutil, "which", lambda nom: str(dossier / nom) if (dossier / nom).exists() else None)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # tout avertissement = echec
        loc.trouver_ngspice()


# ------------------------------------------------------------------- Unix

def test_sur_unix_le_nom_ngspice_est_accepte(monkeypatch, tmp_path):
    """Sous Linux/macOS, `ngspice` est le BON nom : il ne doit pas etre ecarte.

    Une regle Windows appliquee partout casserait la portabilite du depot.

    On ne maquille pas `os.name` : sur une vraie machine Windows, cela ferait
    basculer `pathlib` en `PosixPath` et le test echouerait pour une raison
    fausse. On neutralise l'indirection exposee par le module, qui est le seul
    endroit ou la plateforme est lue.
    """
    dossier = _faux_arbre(tmp_path / "unix", ["ngspice"])
    monkeypatch.setattr(loc, "_est_windows", lambda: False)
    monkeypatch.setattr(loc.shutil, "which", lambda nom: str(dossier / nom) if (dossier / nom).exists() else None)
    monkeypatch.setattr(loc, "NOMS_CONSOLE", ("ngspice_con",))
    rendu = loc.trouver_ngspice()
    assert rendu.name == "ngspice", "sur Unix, ngspice est le build console"
    assert not loc.est_build_silencieux(rendu)


def test_sur_windows_ngspice_est_bien_considere_comme_silencieux(monkeypatch):
    """L'autre moitie du contrat : la regle ne doit pas etre un no-op."""
    monkeypatch.setattr(loc, "_est_windows", lambda: True)
    assert loc.est_build_silencieux(Path("C:/x/ngspice.exe")) is True
    assert loc.est_build_silencieux(Path("C:/x/ngspice_con.exe")) is False
