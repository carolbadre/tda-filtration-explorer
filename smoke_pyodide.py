"""Smoke tests: tda_core (pure Python) vs server.py (GUDHI) parity.

Run with:  python3 smoke_pyodide.py
"""

import numpy as np

import tda_core
import server  # GUDHI-backed reference


def betti_signature(pairs, max_dim):
    """Count essential classes per dim. Robust to per-pair ordering."""
    sig = [0] * (max_dim + 1)
    for p in pairs:
        if p['death'] is None and p['dim'] <= max_dim:
            sig[p['dim']] += 1
    return tuple(sig)


def persistence_signature(pairs, tol=1e-6):
    """Multiset of (dim, round(birth), round(death-or-inf)) for diff."""
    out = []
    for p in pairs:
        d = round(p['death'], 6) if p['death'] is not None else None
        out.append((p['dim'], round(p['birth'], 6), d))
    return sorted(out, key=lambda x: (x[0], x[1], x[2] is None, x[2] or 0.0))


def compare_samplers():
    print("=== Samplers (deterministic with fixed seed) ===")
    for key in ['S1', 'S2', 'T2', 'RP2', 'klein', 'mobius', 'lens_5_2']:
        a = tda_core.sample(key, 30, 0.0, 'uniform', 7)
        b = server.sample_points(key, 30, 0.0, 'uniform', 7)
        ours = np.asarray(a['points'])
        ref  = np.asarray(b)
        same = ours.shape == ref.shape and np.allclose(ours, ref)
        print(f"  {key:10s} ours={ours.shape} ref={ref.shape} match={same}")


def compare_compute(key, N=40, max_r=1.6, max_dim=2, coeff=11, seed=3):
    print(f"\n=== compute({key}, N={N}, max_r={max_r}, max_dim={max_dim}, p={coeff}) ===")
    pts = server.sample_points(key, N, 0.0, 'uniform', seed)

    # GUDHI reference
    ref_r = server.persistence_pairs(
        server.build_rips_complex(pts, max_r, max_dim), max_dim, coeff)
    ref_c = server.persistence_pairs(
        server.build_cech_complex(pts, max_r, max_dim), max_dim, coeff)
    ref_a = server.persistence_pairs(
        server.build_alpha_complex(pts, max_r, max_dim), max_dim, coeff)

    # Ours
    ours = tda_core.compute(pts.tolist(), max_r, max_r, max_r,
                            max_dim, coeff, include_simplices=False)

    for name, ref_pers, our_pers in [
        ('rips',  ref_r, ours['rips']['persistence']),
        ('cech',  ref_c, ours['cech']['persistence']),
        ('alpha', ref_a, ours['alpha']['persistence']),
    ]:
        ref_betti = betti_signature(ref_pers, max_dim)
        our_betti = betti_signature(our_pers, max_dim)
        n_ref = len(ref_pers)
        n_ours = len(our_pers)
        flag = "OK" if ref_betti == our_betti else "BETTI MISMATCH"
        print(f"  {name:5s}  pairs ref={n_ref:4d} ours={n_ours:4d}  "
              f"β ref={ref_betti} ours={our_betti}  {flag}")


if __name__ == '__main__':
    compare_samplers()
    compare_compute('S1', N=40, max_r=1.5, max_dim=1, seed=3)
    compare_compute('S2', N=60, max_r=1.5, max_dim=2, seed=3)
    compare_compute('T2', N=80, max_r=1.0, max_dim=2, seed=3)
    compare_compute('klein', N=60, max_r=1.5, max_dim=2, coeff=2, seed=3)
