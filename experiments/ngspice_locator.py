"""Localisation de l'executable ngspice — source unique pour tous les scripts SPICE.

POURQUOI CE FICHIER EXISTE (2026-08-05).

Six scripts SPICE portaient chacun leur propre constante :

    NGSPICE = Path("D:/ANTIGRAVITY/ngspice-46_64/Spice64/bin/ngspice_con.exe")

Cet emplacement n'existait plus — l'installation avait ete deplacee sans que rien ne le
note. Consequence : plus aucun de ces scripts ne pouvait tourner, ni chez un tiers ni en
local, et le claim C11 restait pourtant affiche « verifie » par le Guardian, qui lit le
CSV et non le producteur.

Le 05/08, la resolution a d'abord ete corrigee dans spice_art_kirchhoff.py SEULEMENT.
La passe soustractive de cloture a retrouve les cinq autres : un correctif local qui ne
mesure pas sa propagation, exactement le defaut que ce projet outille depuis le 30/07.
D'ou ce module : UN endroit ou le chemin se resout, six scripts qui l'appellent.

Ordre de recherche, du plus explicite au plus implicite :
  1. la variable d'environnement NGSPICE (chemin complet de l'executable) ;
  2. le PATH du systeme (`ngspice_con` sous Windows, `ngspice` ailleurs) ;
  3. les emplacements connus des machines de developpement, en dernier recours.

Regle : un chemin machine n'est pas une dependance, c'est une panne differee. Les
etapes 1 et 2 existent pour que l'etape 3 ne soit jamais necessaire.

LE BUILD SILENCIEUX (ajoute le 2026-10-03)
------------------------------------------
Sous Windows, l'installateur livre DEUX binaires dans le meme dossier :

    ngspice.exe      build graphique — en mode batch, il rend la main SANS RIEN
                     ECRIRE : code de sortie 0, stdout vide. Mesure le 2026-10-03
                     sur ngspice-46_64, sur un circuit de 2 resistors :
                     0 octet, contre 2 434 pour le build console.
    ngspice_con.exe  build console — renvoie les resultats.

Les six scripts appelants ne testent que le code de retour. Avec le build
graphique, ils obtiennent donc un "succes" vide : aucun resultat, aucune erreur,
et le script continue comme si la simulation avait tourne. C'est la pire panne
possible — silencieuse, et destructive pour la reproductibilite sans laisser de
trace.

Ce module refuse donc de rendre un build silencieux **quand une autre option
existe**. Il ne le refuse jamais categoriquement : sous Linux et macOS le
binaire s'appelle justement `ngspice` et fonctionne, la regle est donc propre a
Windows. Et s'il ne reste que lui, il est rendu avec un avertissement explicite
plutot que dans le silence.
"""
from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

#: Emplacements observes, du plus recent au plus ancien. Ajouter en TETE.
EMPLACEMENTS_CONNUS = [
    Path("D:/Autres programmes/ngspice-46_64/Spice64/bin/ngspice_con.exe"),
    Path("D:/ANTIGRAVITY/ngspice-46_64/Spice64/bin/ngspice_con.exe"),  # mort depuis ~08/2026
]

#: Nom du build console, prefere partout ou il est disponible.
NOMS_CONSOLE = ("ngspice_con", "ngspice_con.exe")


def _est_windows() -> bool:
    """Indirection volontaire : c'est le SEUL endroit ou la plateforme est lue.

    Elle rend la regle « Windows seulement » testable sans maquiller `os.name` —
    ce qui ferait basculer `pathlib` en `PosixPath` sur une vraie machine
    Windows, et ferait echouer le test pour une raison fausse.
    """
    return os.name == "nt"


def est_build_silencieux(chemin: Path) -> bool:
    """Sous Windows, `ngspice.exe` rend la main sans rien ecrire. Vrai ?

    Sous Linux et macOS, le binaire s'appelle `ngspice` et fonctionne : la
    fonction renvoie toujours `False` la, et c'est voulu.
    """
    return _est_windows() and chemin.stem.lower() == "ngspice"


def _avertissement_silencieux(chemin: Path) -> str:
    return (
        f"ATTENTION : ngspice a ete localise sur {chemin}, qui est le build GRAPHIQUE. "
        "Sous Windows il rend la main sans ecrire rien en mode batch (code 0, "
        "stdout vide) : les scripts SPICE d'ici croiront avoir obtenu un "
        "resultat sans qu'il existe.\n"
        "  - le build console attendu est `ngspice_con.exe`, dans le meme dossier ;\n"
        "  - pour l'utiliser sans toucher au PATH :\n"
        "      set NGSPICE=C:\\chemin\\vers\\ngspice_con.exe"
    )


def trouver_ngspice() -> Path:
    """Rend le chemin de l'executable ngspice. Ne verifie PAS qu'il existe.

    L'appelant doit tester `.exists()` et expliquer quoi installer — voir
    `message_absence()`.

    Sous Windows, un build graphique n'est rendu qu'en dernier recours, et
    avec un avertissement : voir `est_build_silencieux`.
    """
    # Ce qu'on n'a accepte qu'a contrecour, et qu'on ne rend que si rien d'autre.
    defaillance: Path | None = None

    depuis_env = os.environ.get("NGSPICE")
    if depuis_env:
        candidat = Path(depuis_env)
        if est_build_silencieux(candidat):
            # Une variable d'environnement mal pointee ne doit pas empoisonner
            # toute la session : on cherche mieux, et on garde celle-ci en
            # dernier recours.
            defaillance = candidat
        else:
            return candidat

    for nom in NOMS_CONSOLE:
        trouve = shutil.which(nom)
        if trouve:
            candidat = Path(trouve)
            if est_build_silencieux(candidat):
                defaillance = candidat
                continue  # Windows : on cherche encore, un build console reste possible
            return candidat

    # Sur Unix le binaire s'appelle `ngspice` : dernier nom a tester.
    if not _est_windows():
        trouve = shutil.which("ngspice")
        if trouve:
            return Path(trouve)

    for candidat in EMPLACEMENTS_CONNUS:
        if candidat.exists():
            return candidat

    if defaillance is not None:
        warnings.warn(_avertissement_silencieux(defaillance), RuntimeWarning, stacklevel=2)
        return defaillance

    return EMPLACEMENTS_CONNUS[0]


def message_absence(chemin: Path, quoi: str = "Ce script") -> str:
    """Message d'erreur qui dit quoi faire, pas seulement ce qui manque."""
    return (
        f"ngspice introuvable a : {chemin}\n"
        f"{quoi} a besoin de ngspice pour tourner.\n"
        "  - installer ngspice et le mettre dans le PATH, ou\n"
        "  - pointer la variable d'environnement NGSPICE sur l'executable :\n"
        "      Windows  set NGSPICE=C:\\chemin\\vers\\ngspice_con.exe\n"
        "      Linux    export NGSPICE=/usr/bin/ngspice"
    )
