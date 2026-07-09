#!/usr/bin/env python3
"""
B2 — Ancrage physique du dispositif : mapping dimensionnel 3 familles — 2026-07-09
Claude Code (Opus 4.8) / Julien Chauvin — Volet B, fond (B2/B3/B5/B6)

CE QUE FAIT CE SCRIPT (et ce qu'il ne fait PAS) :
  Ce n'est PAS une simulation physique (pas de LLG micromagnétique, pas de VTEAM
  résolu, pas de modèle GST optique intégré). C'est un calcul dimensionnel de
  premier ordre, reproductible, qui ancre le temps de pas dt (sans dimension)
  du modèle Mem4ristor sur la période caractéristique RÉELLE de 3 dispositifs
  candidats (documentée dans la littérature, voir sources ci-dessous), et qui
  en déduit une énergie/opération. Utile pour cadrer B3 (énergie) et B6
  (signature falsifiable) — PAS pour valider la physique du dispositif, qui
  resterait un travail de plusieurs semaines par famille (cf. B2 dans
  docs/FUTURE_WORK.md).

ANCRAGE TEMPOREL — le nœud isolé, pas un choix arbitraire :
  `experiments/reviewer2_linear_stability.py` (1er mai 2026) a mesuré la
  linéarisation du nœud FHN isolé à alpha=0.15 (valeur du modèle) :
  v* = -1.285, lambda = -0.0473 +/- 0.2824i (spirale stable, sub-Hopf).
  Im(lambda) donne la pulsation propre du nœud -> période naturelle
  T_node = 2*pi / Im(lambda) en UNITÉS DE TEMPS DU MODÈLE (pas des pas dt).
  On ancre T_node sur la période caractéristique réelle du dispositif candidat :
  1 unité de temps modèle = T_node_physique / T_node_modele.

DISPOSITIFS ET SOURCES (vérifiées par recherche web le 2026-07-09) :
  - PHOTONIQUE (GST) : réponse d'amorphisation/cristallisation ~100-200 ns
    (pulses UV nanoseconde, cristallisation pleine >180ns).
    Source : "Structural Transitions in Ge2Sb2Te5 Phase Change Memory Thin
    Films Induced by Nanosecond UV Optical Pulses", PMC7254329.
    Rôle candidat : `u` (état multi-niveau, cf. PHOTONIC_PATHWAY.md §5).
  - SPINTRONIQUE (STNO à vortex, type Torrejon 2017 / Romera 2018) : temps de
    réponse ~ns (large gamme selon conception), input power ~qq mW pour un
    STNO à vortex (les linéaires descendent à ~138 uW, certains designs à ~1uW).
    Sources : Grollier group / Nature Torrejon et al. 2017 review ("fast
    response time ~ns"); recherche input power STNO vortex "few mW" vs
    linéaire "138 uW" vs design bas-bruit "1 uW".
    Rôle candidat : `v` (oscillateur, cf. SPINTRONIC_PATHWAY.md).
  - ÉLECTRIQUE — neuristor Mott NbO2 (Pickett et al. 2013, Nature Materials
    12:114-117) : "3 ordres de grandeur plus rapide, 1% de l'énergie" qu'un
    neurone biologique (spike bio ~1-10 nJ, ~1-2 ms, ordres de grandeur
    couramment cités, PAS mesurés directement ici -> réserve explicite).
    Rôle candidat : `v` (alternative électrique au STNO pour l'oscillateur).
  - ÉLECTRIQUE — RRAM/VTEAM (HfOx, filamentary) : commutation ~ns, énergie
    ~10-50 fJ/bit (cellule 10x10nm2 HfO2, cas optimiste scalé) à qq nJ (cellules
    plus lentes/plus grandes). Source : littérature RRAM filamentaire HfOx
    (recherche web 2026-07-09, plusieurs papiers convergents sur l'ordre de
    grandeur ns / 10-50fJ pour les cellules les plus scalées).
    Rôle candidat : poids de couplage D_eff (crossbar analogique STATIQUE,
    PAS l'oscillateur -- une cellule RRAM filamentaire ne se prête pas à un
    cycle limite FHN ; c'est le neuristor Mott qui joue ce rôle électrique).

  Référence CMOS/neuromorphique (recherche web 2026-07-09) :
  Loihi ~24 pJ/synaptic op (SNNTorch benchmark) ; TrueNorth ~26 pJ/synaptic
  event (chiffre historique IBM). Point de comparaison pour B3.

RÉSERVE MAJEURE (à ne jamais perdre de vue) : ce script ne prouve AUCUNE
compatibilité physique réelle. Il montre que les ordres de grandeur ne sont
PAS absurdes (ni 10 ordres de grandeur d'écart, ni un mismatch de vitesse
disqualifiant) -- c'est un scoping, pas une validation.
"""
import csv
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

# --- Constantes du modèle (dynamics.py + reviewer2_linear_stability.csv) ---
DT_MODEL = 0.05          # pas d'intégration, unités de temps modèle
TAU_U = 10.0             # constante de temps du doute, unités de temps modèle
EPSILON_W = 0.08         # dynamics.py default -> tau_w intrinsèque ~ 1/epsilon
TAU_W = 1.0 / EPSILON_W
IM_LAMBDA_ALPHA015 = 0.28235030455309174   # reviewer2_linear_stability.csv, alpha=0.15075 (le plus proche de 0.15 du défaut)
RE_LAMBDA_ALPHA015 = -0.04731750861602678
T_NODE_MODEL = 2 * np.pi / IM_LAMBDA_ALPHA015   # période propre du nœud, unités modèle

STEPS_CAMPAIGN = 4000    # WARM_UP=1000 + STEPS=3000, convention des POCs photoniques

E_PHOTON_1550NM = 1.28e-19   # J, hc/lambda a 1550nm (deja utilise dans PHOTONIC_PATHWAY.md)
LAMBDA_PHOTONS_PER_STEP = 10  # budget nominal retenu (photonic_transduction_poc.py, Lambda_min ~ 10)


def anchor(t_node_physical_s, label):
    """Ancre 1 unite de temps modele sur T_node_physical (secondes)."""
    unit_s = t_node_physical_s / T_NODE_MODEL
    dt_physical = DT_MODEL * unit_s
    tau_u_physical = TAU_U * unit_s
    tau_w_physical = TAU_W * unit_s
    campaign_physical = STEPS_CAMPAIGN * dt_physical
    return {
        'device': label,
        't_node_physical_s': t_node_physical_s,
        'unit_model_time_s': unit_s,
        'dt_physical_s': dt_physical,
        'tau_u_physical_s': tau_u_physical,
        'tau_w_physical_s': tau_w_physical,
        'campaign_4000steps_physical_s': campaign_physical,
    }


def main():
    print("=" * 78)
    print("Ancrage temporel du noeud FHN isole (reviewer2_linear_stability.csv)")
    print("=" * 78)
    print(f"alpha=0.15 (defaut modele) : Re(lambda)={RE_LAMBDA_ALPHA015:.5f}  "
          f"Im(lambda)={IM_LAMBDA_ALPHA015:.5f}")
    print(f"T_node (unites de temps modele) = 2*pi/Im(lambda) = {T_NODE_MODEL:.3f}")
    print(f"soit {T_NODE_MODEL/DT_MODEL:.1f} pas d'integration (dt={DT_MODEL})")
    print(f"tau_u/T_node = {TAU_U/T_NODE_MODEL:.3f}  |  tau_w/T_node = {TAU_W/T_NODE_MODEL:.3f}")
    print()

    rows = []

    # --- PHOTONIQUE (GST) : role candidat = u ---
    for t_ns, tag in [(100, 'GST_low'), (200, 'GST_high')]:
        r = anchor(t_ns * 1e-9, f'photonic_GST_{tag}')
        r['role_candidat'] = 'u (etat multi-niveau)'
        rows.append(r)

    # --- SPINTRONIQUE (STNO vortex) : role candidat = v ---
    for t_ns, tag in [(1, 'STNO_fast'), (10, 'STNO_slow')]:
        r = anchor(t_ns * 1e-9, f'spintronic_STNO_{tag}')
        r['role_candidat'] = 'v (oscillateur)'
        rows.append(r)

    # --- ELECTRIQUE (NbO2 Mott neuristor, Pickett 2013) : role candidat = v ---
    for t_us, tag in [(1, 'NbO2_fast'), (2, 'NbO2_slow')]:
        r = anchor(t_us * 1e-6, f'electrical_NbO2neuristor_{tag}')
        r['role_candidat'] = 'v (oscillateur, alternative electrique au STNO)'
        rows.append(r)

    for r in rows:
        print(f"{r['device']:38s} T_phys={r['t_node_physical_s']*1e9:8.2f} ns  "
              f"dt_phys={r['dt_physical_s']*1e12:9.3f} ps  "
              f"tau_u_phys={r['tau_u_physical_s']*1e9:9.4f} ns  "
              f"campagne(4000 pas)={r['campaign_4000steps_physical_s']*1e6:9.4f} us")

    # --- Energie / operation ---
    print()
    print("=" * 78)
    print("Energie par pas d'integration (ordre de grandeur, PAS mesure)")
    print("=" * 78)

    energy_rows = []

    # Photonique : energie de signal seule (budget de photons), independante de dt
    e_photon_step = LAMBDA_PHOTONS_PER_STEP * E_PHOTON_1550NM
    for r in rows:
        if 'photonic' in r['device']:
            power_w = e_photon_step / r['dt_physical_s']
            energy_rows.append({
                'device': r['device'], 'mechanism': 'signal photonique (budget seul, hors pertes/detecteur/laser)',
                'energy_per_step_J': e_photon_step, 'power_W': power_w,
            })

    # Spintronique : puissance continue (mW a qq uW), energie = P * dt
    for r in rows:
        if 'spintronic' in r['device']:
            for p_w, p_tag in [(3e-3, 'vortex_typique_qqmW'), (138e-6, 'lineaire_138uW')]:
                e_step = p_w * r['dt_physical_s']
                energy_rows.append({
                    'device': r['device'] + f'_{p_tag}', 'mechanism': 'puissance continue (oscillateur libre)',
                    'energy_per_step_J': e_step, 'power_W': p_w,
                })

    # Electrique NbO2 : energie de spike Pickett (relative, 1% bio ~1-10nJ) / pas par cycle T_node
    for r in rows:
        if 'NbO2' in r['device']:
            steps_per_cycle = r['t_node_physical_s'] / r['dt_physical_s']
            for e_spike, tag in [(10e-12, 'bas_1pct_de_1nJ'), (100e-12, 'haut_1pct_de_10nJ')]:
                e_step = e_spike / steps_per_cycle
                energy_rows.append({
                    'device': r['device'] + f'_{tag}', 'mechanism': 'spike Mott (Pickett, relatif au bio, non mesure ici)',
                    'energy_per_step_J': e_step, 'power_W': e_step / r['dt_physical_s'],
                })

    # RRAM : role = poids de couplage STATIQUE. Energie payee UNE FOIS (ecriture),
    # pas par pas -- different des 3 elements dynamiques ci-dessus.
    for e_fj, tag in [(10, 'HfO2_10x10nm_optimiste'), (50, 'HfO2_10x10nm_haut')]:
        energy_rows.append({
            'device': f'electrical_RRAM_coupling_weight_{tag}', 'mechanism': 'ecriture UNIQUE du poids (statique, pas par pas)',
            'energy_per_step_J': None, 'power_W': None, 'energy_per_write_J': e_fj * 1e-15,
        })

    print(f"{'Dispositif':45s} {'E/pas (J)':>14s} {'Puissance (W)':>14s}")
    for e in energy_rows:
        if e.get('energy_per_step_J') is not None:
            print(f"{e['device']:45s} {e['energy_per_step_J']:14.3e} {e['power_W']:14.3e}   [{e['mechanism']}]")
        else:
            print(f"{e['device']:45s} {'--':>14s} {'--':>14s}   E/ecriture={e['energy_per_write_J']:.2e} J [{e['mechanism']}]")

    print()
    print("Reference CMOS/neuromorphique (recherche web 2026-07-09) :")
    print("  Loihi     ~24 pJ / synaptic op (benchmark SNNTorch)")
    print("  TrueNorth ~26 pJ / synaptic event (chiffre historique IBM)")

    fig_dir = HERE.parent / 'figures'
    fig_dir.mkdir(exist_ok=True)
    out_path = fig_dir / 'b2_device_physics_mapping.csv'
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    for e in energy_rows:
        all_keys.update(e.keys())
    all_keys = sorted(all_keys)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        for e in energy_rows:
            w.writerow(e)
    print(f"\nCSV : {out_path}")


if __name__ == '__main__':
    main()
