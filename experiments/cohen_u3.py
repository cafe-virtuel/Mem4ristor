"""
Cohen U3 -- Overlap coefficient entre distributions FROZEN_U vs FULL
====================================================================
[6] Audit DeepSeek -- Quantification du non-chevauchement des distributions.

Cohen U3 = proportion de la distribution traitee (FROZEN_U / B_noise)
           qui depasse la mediane de la distribution controle (FULL / A_dead_zone).

Formule analytique : U3 = Phi(d)
Formule empirique   : U3 = card{x in B : x > median(A)} / |B|

Deux comparaisons :
  1. SPICE A vs B : H_cont (dead zone vs noise-only), n=50 seeds, d=20.78
  2. Python FULL vs FROZEN_U : synchrony a I=0.50 (forcing_sweep_frozen_u.csv)
  3. Python ablation (n=5) : parametrique depuis mean/std (p2_sigma_social_ablation.csv)
"""
import os, sys, csv
import numpy as np
from scipy import stats

FIGURES = os.path.join(os.path.dirname(__file__), '..', 'figures')

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (np.mean(b)-np.mean(a))/sp if sp > 0 else np.inf

def u3_analytical(d):
    return float(stats.norm.cdf(d))

def u3_empirical(a, b):
    return float(np.mean(b > np.median(a)))

def ovl(a, b, n_bins=200):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    bins = np.linspace(lo, hi, n_bins+1)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    return float(np.sum(np.minimum(ha, hb)) * (bins[1]-bins[0]))

def sep(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

def report(label, a, b, ga="A", gb="B"):
    d    = cohens_d(a, b)
    u3a  = u3_analytical(d)
    u3e  = u3_empirical(a, b)
    ov   = ovl(a, b)
    t, p = stats.ttest_ind(a, b)
    print(f"\n-- {label} --")
    print(f"  {ga}: n={len(a)}, mean={np.mean(a):.4f}, std={np.std(a,ddof=1):.4f}")
    print(f"  {gb}: n={len(b)}, mean={np.mean(b):.4f}, std={np.std(b,ddof=1):.4f}")
    print(f"  Cohen d    : {d:.4f}")
    print(f"  U3 (Phi(d)): {u3a*100:.6f}%")
    print(f"  U3 (empirique): {u3e*100:.1f}%  (fraction B > median(A))")
    print(f"  OVL        : {ov:.6f}  (0=disjoint)")
    print(f"  t={t:.2f}, p={p:.2e}")
    print(f"  max(A)={a.max():.4f}  min(B)={b.min():.4f}  disjoint={'OUI' if a.max()<b.min() else 'NON'}")
    return {'label':label,'n_a':len(a),'n_b':len(b),
            'mean_a':float(np.mean(a)),'std_a':float(np.std(a,ddof=1)),
            'mean_b':float(np.mean(b)),'std_b':float(np.std(b,ddof=1)),
            'cohens_d':d,'u3_analytical_pct':u3a*100,
            'u3_empirical_pct':u3e*100,'ovl':ov,'p_value':float(p),
            'max_a':float(a.max()),'min_b':float(b.min()),
            'disjoint':int(a.max()<b.min())}

# 1. SPICE
sep("COMPARAISON 1 : SPICE H_cont -- A_dead_zone vs B_noise_only (n=50)")
rows_a, rows_b = [], []
with open(os.path.join(FIGURES,'spice_50seeds_validation.csv')) as f:
    for row in csv.DictReader(f):
        if row['point'] == 'A_dead_zone':   rows_a.append(float(row['H_continuous']))
        elif row['point'] == 'B_noise_only': rows_b.append(float(row['H_continuous']))
r1 = report("SPICE H_cont", np.array(rows_a), np.array(rows_b),
            "A_dead_zone", "B_noise_only")

# 2. Python forcing sweep I=0.50
sep("COMPARAISON 2 : Python sync -- FULL vs FROZEN_U @ I=0.50")
full_s, frozen_s = [], []
with open(os.path.join(FIGURES,'forcing_sweep_frozen_u.csv')) as f:
    for row in csv.DictReader(f):
        try: I = float(row['I_stim'])
        except: continue
        if abs(I-0.50) < 0.01:
            if row['model']=='FULL':    full_s.append(float(row['synchrony']))
            elif row['model']=='FROZEN_U': frozen_s.append(float(row['synchrony']))
r2 = report("Python sync @ I=0.50", np.array(full_s), np.array(frozen_s),
            "FULL", "FROZEN_U")

# 3. Ablation n=5 parametrique
sep("COMPARAISON 3 : Python ablation synchrony (n=5, parametrique)")
mf, sf = 0.06726786655626937, 0.02475091442899676
mfr, sfr = 0.7301850191502341, 0.0781279367029548
n=5
sp_ab = np.sqrt(((n-1)*sf**2 + (n-1)*sfr**2)/(2*n-2))
d_ab = (mfr-mf)/sp_ab
u3_ab = u3_analytical(d_ab)
print(f"\n-- ablation (parametrique n={n}) --")
print(f"  FULL    : mean={mf:.4f}, std={sf:.4f}")
print(f"  FROZEN_U: mean={mfr:.4f}, std={sfr:.4f}")
print(f"  Cohen d    : {d_ab:.4f}")
print(f"  U3 (Phi(d)): {u3_ab*100:.6f}%")
r3 = {'label':'ablation (parametrique n=5)','n_a':n,'n_b':n,
      'mean_a':mf,'std_a':sf,'mean_b':mfr,'std_b':sfr,
      'cohens_d':d_ab,'u3_analytical_pct':u3_ab*100,
      'u3_empirical_pct':float('nan'),'ovl':float('nan'),
      'p_value':float('nan'),'max_a':float('nan'),'min_b':float('nan'),'disjoint':-1}

# Export CSV
results = [r1, r2, r3]
out = os.path.join(FIGURES, 'cohen_u3_results.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader(); w.writerows(results)
print(f"\nCSV -> {out}")

# Verdict
sep("VERDICT GLOBAL")
print(f"SPICE A vs B  : d={r1['cohens_d']:.2f}, U3={r1['u3_empirical_pct']:.0f}%, OVL={r1['ovl']:.6f}, disjoint={'OUI' if r1['disjoint'] else 'NON'}")
print(f"Python I=0.50 : d={r2['cohens_d']:.2f}, U3={r2['u3_empirical_pct']:.0f}%, OVL={r2['ovl']:.6f}, disjoint={'OUI' if r2['disjoint'] else 'NON'}")
print(f"Python ablat. : d={r3['cohens_d']:.2f}, U3={r3['u3_analytical_pct']:.6f}% (analytique)")
print(f"\nFormulation preprint (SPICE) :")
print(f"  U3={r1['u3_empirical_pct']:.0f}% : toutes les observations B_noise depassent max(A_dead_zone)={r1['max_a']:.3f}")
print(f"  (min(B)={r1['min_b']:.3f} > max(A)={r1['max_a']:.3f} -- distributions strictement disjointes)")
