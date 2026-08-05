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
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

#: Emplacements observes, du plus recent au plus ancien. Ajouter en TETE.
EMPLACEMENTS_CONNUS = [
    Path("D:/Autres programmes/ngspice-46_64/Spice64/bin/ngspice_con.exe"),
    Path("D:/ANTIGRAVITY/ngspice-46_64/Spice64/bin/ngspice_con.exe"),  # mort depuis ~08/2026
]


def trouver_ngspice() -> Path:
    """Rend le chemin de l'executable ngspice. Ne verifie PAS qu'il existe.

    L'appelant doit tester `.exists()` et expliquer quoi installer — voir
    `message_absence()`.
    """
    depuis_env = os.environ.get("NGSPICE")
    if depuis_env:
        return Path(depuis_env)
    for nom in ("ngspice_con", "ngspice"):
        trouve = shutil.which(nom)
        if trouve:
            return Path(trouve)
    for candidat in EMPLACEMENTS_CONNUS:
        if candidat.exists():
            return candidat
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
