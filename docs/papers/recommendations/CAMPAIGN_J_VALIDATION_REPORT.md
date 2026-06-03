# Campaign J Figures Validation Report

**Date:** May 31, 2026  
**Source data:** `figures/campaign_j_agg.csv` (99 rows), `figures/campaign_j_raw.csv` (1800 rows)  
**Campaign:** 1800 simulations: N=100/200/400, c in [0,15], 40 seeds, joint U4 + LZ + H_stable measurements

---

## Summary

| Claim | Status | Key Numbers |
|-------|--------|-------------|
| (1) U4 flat at 2/3, no min at lambda2=2.31 | **CONFIRMED** | U4(2.1-2.5) = 0.6655-0.6664; global min elsewhere |
| (2) (2/3-U4)*N approx const | **CONFIRMED with drift** | ~0.16-0.18 across N, std ~0.03-0.09 |
| (3) LZ decreases with lambda2 | **CONFIRMED (overall trend)** | 0.827 -> 0.664 (N=100); local increases exist |
| (4) LZ < 0.85 at lambda2=7-8 (structured) | **CONFIRMED** | LZ = 0.70-0.72, well below 0.85 |

---

## 1. U4 Flat at 2/3 — No Minimum at lambda2=2.31

**Data source:** agg + raw CSV

### U4 at lambda2 ≈ 2.31 (from agg):

| N | lambda2_mean | U4_mean | U4_std |
|---|---|---|---|
| 100 | 2.254 | 0.665501 | 0.000472 |
| 100 | 2.492 | 0.665457 | 0.000447 |
| 200 | 2.272 | 0.666064 | 0.000222 |
| 200 | 2.493 | 0.666008 | 0.000264 |
| 400 | 2.279 | 0.666352 | 0.000109 |
| 400 | 2.491 | 0.666371 | 0.000090 |

**Interpretation:** At lambda2 ≈ 2.3, U4 ≈ 0.6655-0.6664, very close to 2/3=0.666667. No anomalous drop.

### U4 Global Minimum Location (from raw):

| N | lambda2_of_min | U4_min |
|---|---|---|
| 100 | 2.898 | 0.663020 |
| 200 | 8.216 | 0.664466 |
| 400 | 7.734 | 0.665681 |

**Interpretation:** The U4 minimum for N=100 is at lambda2=2.898, NOT at 2.31. For N=200/400, the minimum is at lambda2~7.7-8.2 (dense regime). There is NO special feature at lambda2=2.31.

**VERDICT: CONFIRMED.** U4 is flat at 2/3 across all lambda2. No minimum at lambda2=2.31.

---

## 2. (2/3 - U4)*N ≈ Constant (1/N Correction, Not Phase Transition)

**Data source:** agg CSV

| N | mean (2/3-U4)*N | std | min | max |
|---|---|---|---|---|
| 100 | 0.1628 | ~0.027 | 0.1135 | 0.2203 |
| 200 | 0.1674 | ~0.037 | 0.0961 | 0.2432 |
| 400 | 0.1743 | ~0.033 | 0.1113 | 0.2414 |

**Interpretation:** The correction term is roughly constant across N (around 0.16-0.18), confirming that (2/3-U4)*N scales as 1/N rather than indicating a phase transition. The spread (max-min range of ~0.10-0.15) reflects real variation across lambda2 bins, but the mean value is stable across system sizes.

**VERDICT: CONFIRMED.** The finite-size correction (2/3-U4)*N is approximately constant across N=100/200/400, consistent with 1/N scaling rather than a phase transition.

---

## 3. LZ Decreases with lambda2

**Data source:** agg + raw CSV

### From agg (binned data):

| N | LZ_min | LZ_max | Local increases in LZ |
|---|---|---|---|
| 100 | 0.6682 | 0.8287 | 8 |
| 200 | 0.6965 | 0.8308 | 10 |
| 400 | 0.6975 | 0.8288 | 8 |

Note: LZ is not strictly monotonic — there are 8-10 local increases across the lambda2 range. These are small wiggles, not systematic increases.

### From raw (binned into 20 bins, overall trend):

**N=100:** LZ drops from 0.828 (lam=1.41) to 0.664 (lam=10.73). Binned increases: 0/19.

**N=200:** LZ drops from 0.827 (lam=1.47) to 0.673 (lam=10.28). Binned increases: 1/19.

**N=400:** LZ drops from 0.826 (lam=1.42) to 0.682 (lam=9.19). Binned increases: 3/19.

### Key transition values:

| lambda2 range | LZ (N=100) | LZ (N=200) | LZ (N=400) |
|---|---|---|---|
| 1.2 - 1.5 | 0.827-0.830 | 0.823-0.831 | 0.821-0.829 |
| 4.5 - 5.5 | 0.758-0.783 | 0.750-0.779 | 0.742-0.774 |
| 7.0 - 8.5 | 0.689-0.729 | 0.698-0.730 | 0.698-0.725 |

**VERDICT: CONFIRMED (overall trend).** LZ decreases from ~0.83 at low lambda2 to ~0.67-0.70 at high lambda2. There are minor local increases (8-10 per curve), but the dominant trend is clear decrease. The claim "LZ decreases with lambda2" holds in aggregate.

---

## 4. At lambda2=7-8, LZ < 0.85 (STRUCTURED, Not CHAOS)

**Data source:** agg + raw CSV

### From agg (lambda2=7.0 to 8.5):

| N | lambda2_mean | U4_mean | LZ_mean | LZ_std |
|---|---|---|---|---|
| 100 | 7.031 | 0.664890 | 0.7291 | 0.0198 |
| 100 | 7.247 | 0.665009 | 0.7234 | 0.0270 |
| 100 | 7.527 | 0.664923 | 0.7227 | 0.0455 |
| 100 | 7.732 | 0.664464 | 0.7186 | 0.0425 |
| 100 | 7.986 | 0.664718 | 0.7033 | 0.0232 |
| 100 | 8.251 | 0.664941 | 0.7043 | 0.0244 |
| 100 | 8.470 | 0.664676 | 0.6893 | 0.0155 |
| 200 | 7.023 | 0.665587 | 0.7297 | 0.0241 |
| 200 | 7.252 | 0.665660 | 0.7171 | 0.0217 |
| 200 | 7.454 | 0.665630 | 0.7181 | 0.0226 |
| 200 | 7.782 | 0.665620 | 0.7177 | 0.0197 |
| 200 | 7.988 | 0.665589 | 0.7188 | 0.0256 |
| 200 | 8.226 | 0.665451 | 0.6980 | 0.0184 |
| 200 | 8.487 | 0.665715 | 0.7040 | 0.0218 |
| 400 | 7.050 | 0.666083 | 0.7254 | 0.0195 |
| 400 | 7.242 | 0.666095 | 0.7108 | 0.0098 |
| 400 | 7.427 | 0.666124 | 0.7087 | 0.0160 |
| 400 | 7.735 | 0.666063 | 0.7145 | 0.0168 |
| 400 | 7.996 | 0.666065 | 0.7120 | 0.0108 |
| 400 | 8.219 | 0.666138 | 0.6975 | 0.0233 |

### From raw (7.0 <= lambda2 <= 8.5):

| N | count | LZ_mean | LZ_std | LZ_range | U4_mean |
|---|---|---|---|---|---|
| 100 | 85 | 0.7125 | 0.0748 | [0.644, 0.794] | 0.664850 |
| 200 | 92 | 0.7168 | 0.0594 | [0.660, 0.778] | 0.665608 |
| 400 | 66 | 0.7119 | 0.0408 | [0.670, 0.752] | 0.666091 |

**VERDICT: CONFIRMED.** All LZ values in the lambda2=7-8 range are well below 0.85. Mean LZ is 0.71-0.72 across all N values. The system is firmly in the STRUCTURED regime (LZ < 0.85), not the chaotic regime (LZ > 0.85).

---

## Overall Assessment

All four key claims from AUDIT-017/018 are supported by the CSV data:

1. **U4 is flat at 2/3 everywhere** — confirmed with quantitative evidence
2. **(2/3-U4)*N ≈ const** — confirmed, ~0.16-0.18 across N  
3. **LZ decreases with lambda2** — confirmed as overall trend (not strict monotonicity)
4. **LZ < 0.85 at lambda2=7-8 (structured)** — confirmed, LZ ≈ 0.70-0.72

The figures (campaign_j_binder_and_lz.png, campaign_j_entropy_sync.png, campaign_j_u4_vs_lz.png) are consistent with the underlying CSV data.