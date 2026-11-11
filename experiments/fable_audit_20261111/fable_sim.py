#!/usr/bin/env python3
"""
AUDIT FABLE 2026-11-11 -- simulateur autonome pour E1 (doute sur les NOEUDS vs sur les LIENS).

Ne touche pas au coeur. Reprend ses equations FHN a l'identique (memes constantes, meme
ordre des operations, meme bruit Euler-Maruyama) et ne fait varier QU'UNE chose : la loi
de couplage.

  none         : pas de couplage
  fixed        : couplage attractif fige, g = g(sigma_baseline)  (= l'ablation FROZEN_U)
  node         : le coeur actuel -- u_i par NOEUD, pilote par |somme_j (v_j - v_i)|
  edge         : u_ij par LIEN, pilote par |v_i - v_j|  (le dipole : un memristor se
                 trouve ENTRE deux noeuds et voit la tension A SES BORNES)
  edge_noflip  : idem, conductance bornee a >= 0 (un memristor passif ne devient pas negatif)
  edge_frozen  : u_ij fournis et figes (aucune dynamique du doute)
  oracle       : couplage fige intra-groupe, nul inter-groupe (connait la verite)

Simplification commune a tous les bras, declaree : la plasticite d'inhibition du coeur
(lambda_intrinsic) est mise a 0 ; la fuite -w/tau_plasticity est conservee. Le gate de
e0_gate_and_ficurve.py verifie que le mode 'node' reproduit alors le coeur reel.
"""
from __future__ import annotations
import numpy as np

A_, B_, EPS, ALPHA, CUB, DT = 0.7, 0.8, 0.08, 0.15, 5.0, 0.05
TAU_PL = 1000.0
EPS_U, K_U, SIG0, TAU_U, ALPHA_S, S_CAP = 0.02, 1.0, 0.05, 10.0, 2.0, 5.0
SIGMA_V = 0.05
STEEP, LEAK = np.pi, 0.01


def g_of_u(u):
    return np.tanh(STEEP * (0.5 - u)) + LEAK


def simulate(adj, I, D, mode, seed, steps, rec_from=None, rec_every=4,
             u_edge=None, groups=None, rng=None, v0=None, w0=None, noflip=False):
    adj = np.asarray(adj, dtype=float)
    N = adj.shape[0]
    if rng is None:
        rng = np.random.RandomState(seed)
    v = rng.uniform(-1.5, 1.5, N) if v0 is None else np.array(v0, dtype=float)
    w = rng.uniform(0.0, 1.0, N) if w0 is None else np.array(w0, dtype=float)
    I = np.full(N, float(I)) if np.ndim(I) == 0 else np.asarray(I, dtype=float)
    D_eff = D / np.sqrt(N)
    L = np.diag(adj.sum(axis=1)) - adj
    iu, ju = np.where(np.triu(adj, 1) > 0)
    E = len(iu)
    g0 = float(g_of_u(SIG0))
    u_n = np.full(N, SIG0)
    u_e = None
    if mode in ("edge", "edge_noflip"):
        u_e = np.full(E, SIG0)
    elif mode == "edge_frozen":
        u_e = np.array(u_edge, dtype=float)
        ge_const = np.maximum(g_of_u(u_e), 0.0) if noflip else g_of_u(u_e)
    elif mode == "oracle":
        ge_const = np.where(groups[iu] == groups[ju], g0, 0.0)
    sq = np.sqrt(DT)
    rec = []
    lap = None
    for t in range(steps):
        eta = rng.normal(0, SIGMA_V, N) / sq
        if mode == "none":
            I_coup = 0.0
        elif mode == "fixed":
            lap = -(L @ v)
            I_coup = D_eff * g0 * lap
        elif mode == "node":
            lap = -(L @ v)
            u_filter = np.tanh(STEEP * (0.5 - u_n)) + LEAK
            I_coup = D_eff * u_filter * lap
        else:
            dvE = v[ju] - v[iu]
            if mode == "edge":
                ge = g_of_u(u_e)
            elif mode == "edge_noflip":
                ge = np.maximum(g_of_u(u_e), 0.0)
            else:
                ge = ge_const
            flux = ge * dvE
            I_coup = D_eff * (np.bincount(iu, weights=flux, minlength=N)
                              - np.bincount(ju, weights=flux, minlength=N))
        I_ext = I + I_coup
        dv = (v - (v ** 3) / CUB - w + I_ext - ALPHA * np.tanh(v) + eta)
        dw = EPS * (v + A_ - B_ * w)
        dw_learning = 0.0 - (w / TAU_PL)
        if mode == "node":
            sig = np.abs(lap)
            e_ad = EPS_U * np.clip(1.0 + ALPHA_S * np.clip(sig, 0.0, 100.0), 1.0, S_CAP)
            du = (e_ad * (K_U * sig + SIG0 - u_n)) / TAU_U
        elif mode in ("edge", "edge_noflip"):
            sigE = np.abs(dvE)
            e_ad = EPS_U * np.clip(1.0 + ALPHA_S * np.clip(sigE, 0.0, 100.0), 1.0, S_CAP)
            duE = (e_ad * (K_U * sigE + SIG0 - u_e)) / TAU_U
        v = v + dv * DT
        w = w + (dw + dw_learning) * DT
        if mode == "node":
            u_n = np.clip(u_n + du * DT, 0.0, 1.0)
        elif mode in ("edge", "edge_noflip"):
            u_e = np.clip(u_e + duE * DT, 0.0, 1.0)
        if rec_from is not None and t >= rec_from and (t - rec_from) % rec_every == 0:
            rec.append(v.copy())
    return dict(v=v, w=w, u_n=u_n, u_e=u_e, rec=np.array(rec), iu=iu, ju=ju)


# ----------------------------------------------------------------- lectures ("langages")
def freq_readout(rec):
    """Nombre de passages montants par v=0, par noeud (lecture frequentielle)."""
    return ((rec[:-1] < 0) & (rec[1:] >= 0)).sum(axis=0).astype(float)


def kmeans1d(x, K, iters=100):
    x = np.asarray(x, dtype=float)
    c = np.quantile(x, (np.arange(K) + 0.5) / K)
    lab = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        lab = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
        new = np.array([x[lab == k].mean() if np.any(lab == k) else c[k] for k in range(K)])
        if np.allclose(new, c):
            break
        c = new
    return lab


def ari(a, b):
    """Adjusted Rand Index entre deux partitions (1 = identiques, ~0 = hasard)."""
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    M = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(M, (a, b), 1)
    c2 = lambda x: x * (x - 1) / 2.0
    s = c2(M).sum(); sa = c2(M.sum(axis=1)).sum(); sb = c2(M.sum(axis=0)).sum()
    tot = c2(len(a))
    exp = sa * sb / tot
    mx = 0.5 * (sa + sb)
    return 0.0 if mx == exp else float((s - exp) / (mx - exp))


def components(N, iu, ju, keep):
    """Composantes connexes du graphe restreint aux liens 'keep' (union-find)."""
    par = np.arange(N)
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in zip(iu[keep], ju[keep]):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    return np.array([find(i) for i in range(N)])


def cognitive_bins(v):
    """Les 5 classes du preprint (frontieres -1.2, -0.4, 0.4, 1.2)."""
    return np.digitize(v, [-1.2, -0.4, 0.4, 1.2])
