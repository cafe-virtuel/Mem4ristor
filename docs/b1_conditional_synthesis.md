# Synthese Volet B1 -- valeur conditionnelle du doute

> Genere par `experiments/b1_conditional_synthesis.py` depuis les CSV des 3 consolidations (deterministe). IC95 bootstrap.

| Topologie | Tache LOYALE (doute-conv) | Tache TROMPEUSE (doute-conv) | Watchdog (valid-hasard) |
|---|---|---|---|
| LATTICE | -0.06 [-0.09,-0.02] | +0.67 [+0.43,+0.87] | +0.73 [+0.67,+0.79] |
| BA_m3 | -0.48 [-0.56,-0.39] | +0.35 [+0.02,+0.65] | +0.15 [+0.02,+0.29] |
| ER_p06 | -0.25 [-0.30,-0.21] | +0.63 [+0.40,+0.83] | +0.74 [+0.69,+0.79] |

**Lecture.** Le doute n'ajoute rien quand converger tot suffit (tache loyale, gain <=0) et paie quand converger tot est un piege (tache trompeuse, gain >0). Robuste aux seeds et a la topologie ; **BA scale-free est le cas le plus faible partout** (les hubs empechent le desaccord laplacien de retomber), coherent avec la reformulation degre/champ-moyen du preprint.
