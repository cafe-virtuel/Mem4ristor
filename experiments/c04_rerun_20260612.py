#!/usr/bin/env python3
"""
C04 re-run documenté — 2026-06-12 (Claude Code / Fable, session autonome)

CONTEXTE : question ouverte bloquante avant soumission (SYNAPSE 2026-06-11,
claims_mapping.json C04 note). Deux valeurs concurrentes pour sync_mean FULL
(ablation sigma_social, BA m=3 N=100, I_stim=0.5) :

  - 0.0673 : CSV commité figures/p2_sigma_social_ablation.csv (TEST_HERMES,
             source unique) + preprint (sync_FULL=0.067). Produit ~2026-04-25.
  - 0.0072 : re-run Hermès 2026-06-03 (AUDIT-022), exécuté dans
             GITHUB_REPOSITORY (AVANT réconciliation des repos du 7 juin) —
             donc potentiellement avec un code source divergent.

MÉTHODE : on importe run_one() du script ORIGINAL commité
(experiments/p2_sigma_social_ablation.py) sans le modifier, et on le relance
dans TEST_HERMES (source unique) avec le code source actuel :
  - conditions FULL et FROZEN_U (celles qui définissent C04 : +985%)
  - les 5 seeds canoniques [42, 123, 777, 456, 999]
Sortie : figures/c04_rerun_20260612.csv (NOUVEAU fichier — on n'écrase JAMAIS
le CSV commité, leçon C08 du 10 juin).

INTERPRÉTATION ATTENDUE :
  - sync_FULL ≈ 0.067  -> le code actuel reproduit le preprint ; le 0.0072
    d'Hermès venait du code divergent de GITHUB_REPOSITORY. C04 tranché.
  - sync_FULL ≈ 0.007  -> le code actuel NE reproduit PAS le preprint ;
    la valeur publiée dépend d'une version de code disparue. ALERTE à
    documenter avant soumission.
"""
import csv
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # pour importer le script original
sys.path.insert(0, str(HERE.parent / 'src'))

import p2_sigma_social_ablation as base  # noqa: E402  (script original, non modifié)
from mem4ristor.graph_utils import make_ba  # noqa: E402

SEEDS = [42, 123, 777, 456, 999]         # seeds canoniques du script original
CONDITIONS = ['FULL', 'FROZEN_U']

if __name__ == '__main__':
    print("=" * 78)
    print("C04 re-run documenté — 2026-06-12 — TEST_HERMES (source unique)")
    print(f"run_one() importé de : {base.__file__}")
    print(f"I_STIM={base.I_STIM} STEPS={base.STEPS} WARM_UP={base.WARM_UP} N={base.N}")
    print("=" * 78)

    t0 = time.time()
    adj = make_ba(base.N, 3, seed=42)    # identique au script original
    rows = []

    for condition in CONDITIONS:
        print(f"\nCondition : {condition}")
        print(f"  {'seed':>6}  {'H_cog':>7}  {'H_cont':>7}  {'sync':>7}  {'f_dom':>7}")
        sync_l, hcog_l, hcont_l, fdom_l = [], [], [], []
        for seed in SEEDS:
            hcog, hcont, sync, fdom, _ = base.run_one(adj, condition, seed)
            hcog_l.append(hcog); hcont_l.append(hcont)
            sync_l.append(sync); fdom_l.append(fdom)
            print(f"  {seed:>6}  {hcog:>7.4f}  {hcont:>7.4f}  {sync:>7.4f}  {fdom:>7.4f}")
            rows.append({'condition': condition, 'seed': seed,
                         'h_cog': hcog, 'h_cont': hcont, 'sync': sync, 'f_dom': fdom})
        print(f"  {'MEAN':>6}  {np.mean(hcog_l):>7.4f}  {np.mean(hcont_l):>7.4f}  "
              f"{np.mean(sync_l):>7.4f}  {np.mean(fdom_l):>7.4f}")
        rows.append({'condition': condition, 'seed': 'MEAN',
                     'h_cog': float(np.mean(hcog_l)), 'h_cont': float(np.mean(hcont_l)),
                     'sync': float(np.mean(sync_l)), 'f_dom': float(np.mean(fdom_l))})

    # Verdict C04
    full_sync = next(r['sync'] for r in rows
                     if r['condition'] == 'FULL' and r['seed'] == 'MEAN')
    frozen_sync = next(r['sync'] for r in rows
                       if r['condition'] == 'FROZEN_U' and r['seed'] == 'MEAN')
    print("\n--- VERDICT C04 ---")
    print(f"sync FULL   = {full_sync:.4f}  (preprint/CSV commité : 0.0673 | Hermès GITHUB_REPO : 0.0072)")
    print(f"sync FROZEN = {frozen_sync:.4f} (preprint : 0.730)")
    if abs(full_sync - 0.0673) <= 0.025:
        print("=> Le code actuel de TEST_HERMES REPRODUIT le preprint (0.067).")
        print("   Le 0.0072 d'Hermès venait du code divergent de GITHUB_REPOSITORY.")
    elif abs(full_sync - 0.0072) <= 0.01:
        print("=> ALERTE : le code actuel reproduit 0.0072, PAS le preprint (0.067).")
        print("   La valeur publiée dépend d'une version de code disparue.")
    else:
        print("=> Valeur intermédiaire/nouvelle : investigation supplémentaire requise.")

    out = HERE.parent / 'figures' / 'c04_rerun_20260612.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nCSV : {out}")
    print(f"Elapsed : {time.time() - t0:.1f}s")
