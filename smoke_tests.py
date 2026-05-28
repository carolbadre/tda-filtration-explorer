"""
TDA Filtration Explorer — phase 3 smoke tests
=============================================

Two criteria are checked per manifold:

  (A) BETTI-AT-SOME-r  : sweep r over the filtration, find any r where β
                          matches the manifold's expected Betti vector.
                          This is what the UI badge tests.

  (B) LONG-BARS-BY-DIM : take the top-K longest persistence bars (where K
                          = sum of expected β), check the count by dim
                          matches expected.

Criterion (A) is the "ideal" — it means the badge will fire cleanly at some
slider position. Criterion (B) is more lenient and tracks whether the
RIGHT topology is at least represented in the persistence diagram, even if
no single r value isolates it cleanly. (B) is what users will see in the
persistence diagram. Both criteria are reported separately per manifold.

KNOWN LIMITS:
  - Genus-g surfaces use the tube-around-loop construction (g circles
    meeting at origin). The meridian bars are SHORTER than longitude bars
    because tube_r < R (tube cross-section is smaller than loop radius),
    so the meridians may not show up among the top-K longest bars.
    Topology is correct asymptotically; user-visible recovery may be
    partial.
  - CP^3 is in the manifold zoo but is not smoke-tested: ambient R^16 with
    tractable N (~12) is too sparse for topology recovery via any
    filtration. Documented as a known limit.
  - Recovery is generally weaker for higher-ambient manifolds at small N
    (Carol's chosen N_max values are tuned for UI responsiveness, not for
    perfect topology recovery).

Run with:  python3 smoke_tests.py
"""

import sys
import time
import numpy as np

from server import (
    sample_by_key,
    build_alpha_complex,
    build_rips_complex,
    persistence_pairs,
)


def betti_at_r(pers, r, max_dim):
    b = [0] * (max_dim + 1)
    for p in pers:
        if p['birth'] <= r and (p['death'] is None or p['death'] > r):
            if p['dim'] <= max_dim:
                b[p['dim']] += 1
    return b


def criterion_A(pers, expected, max_dim, max_r):
    """True iff there's some r where β matches expected exactly."""
    rs = np.linspace(0.01, max_r, 400)
    for r in rs:
        b = betti_at_r(pers, r, max_dim)
        if b[:len(expected)] == list(expected):
            return True, r, b
    # Closest r by L1
    best_r, best_b, best_d = 0.0, None, 1e9
    for r in rs:
        b = betti_at_r(pers, r, max_dim)
        d = sum(abs(b[i] - expected[i]) for i in range(len(expected)))
        if d < best_d:
            best_r, best_b, best_d = r, b, d
    return False, best_r, best_b


def criterion_B(pers, expected, max_dim, max_r):
    """True iff the top-K longest persistence bars (K = Σ expected β) have
    the right distribution by dim."""
    K = sum(expected)
    bars = []
    for p in pers:
        d = p['death'] if p['death'] is not None else max_r
        length = d - p['birth']
        bars.append((p['dim'], length))
    bars.sort(key=lambda x: -x[1])
    top = bars[:K]
    by_dim = [0] * (max_dim + 1)
    for dim, _ in top:
        if dim < len(by_dim):
            by_dim[dim] += 1
    return by_dim[:len(expected)] == list(expected), by_dim


def run_test(key, N, max_dim, coeff_field, expected_betti, seed=42,
             max_r=3.0, builder='alpha'):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    pts = sample_by_key(key, N, rng)
    ambient = pts.shape[1]
    t1 = time.time()
    try:
        if builder == 'rips':
            diam = float(np.max(np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=-1)))
            actual_max_r = diam * 1.05
            simp = build_rips_complex(pts, actual_max_r, max_dim)
        else:
            actual_max_r = max_r
            simp = build_alpha_complex(pts, actual_max_r, max_dim)
    except Exception as e:
        return {'key': key, 'A': False, 'B': False, 'reason': f'{builder} error: {e}'}
    t2 = time.time()
    try:
        pers = persistence_pairs(simp, max_dim, coeff_field)
    except Exception as e:
        return {'key': key, 'A': False, 'B': False, 'reason': f'persistence error: {e}'}
    t3 = time.time()

    A_pass, A_r, A_b = criterion_A(pers, expected_betti, max_dim, actual_max_r)
    B_pass, B_dims = criterion_B(pers, expected_betti, max_dim, actual_max_r)

    return {
        'key': key, 'ambient': ambient, 'N': N, 'coef': coeff_field,
        'expected': list(expected_betti),
        'timing': f"{t1-t0:.2f}+{t2-t1:.2f}+{t3-t2:.2f}s",
        'A': A_pass, 'A_r': A_r, 'A_betti': A_b,
        'B': B_pass, 'B_dims': B_dims,
        'builder': builder,
    }


TESTS = [
    # Sphere / Torus sanity
    ('S1',          30, 2, 11, (1, 1),                 'alpha'),
    ('S2',          35, 3, 11, (1, 0, 1),              'alpha'),
    ('T2',          50, 3, 11, (1, 2, 1),              'alpha'),

    # RP^n over ℤ/2
    ('RP1',         30, 2,  2, (1, 1),                 'alpha'),
    ('RP2',         45, 3,  2, (1, 1, 1),              'alpha'),
    ('RP3',         22, 4,  2, (1, 1, 1, 1),           'rips'),

    # CP^2 (CP^3 skipped — see file docstring)
    ('CP2',         22, 5, 11, (1, 0, 1, 0, 1),        'rips'),

    # Surfaces
    ('mobius',      45, 2, 11, (1, 1, 0),              'alpha'),
    ('klein',       60, 3,  2, (1, 2, 1),              'alpha'),
    ('genus2',     150, 3, 11, (1, 4, 1),              'alpha'),
    ('genus3',     200, 3, 11, (1, 6, 1),              'alpha'),
    ('genus4',     250, 3, 11, (1, 8, 1),              'alpha'),

    # Lens spaces — ambient R^7
    ('lens_3_1',   30, 3, 3, (1, 1, 1, 1),             'rips'),
    ('lens_5_1',   30, 3, 5, (1, 1, 1, 1),             'rips'),
    ('lens_5_2',   30, 3, 5, (1, 1, 1, 1),             'rips'),
    ('lens_7_1',   30, 3, 7, (1, 1, 1, 1),             'rips'),
    ('lens_7_2',   30, 3, 7, (1, 1, 1, 1),             'rips'),

    # Products
    ('prod_S2_S1',    30, 4, 11, (1, 1, 1, 1),         'rips'),
    ('prod_S2_S2',    30, 5, 11, (1, 0, 2, 0, 1),      'rips'),
    ('prod_S1_RP2',   30, 4,  2, (1, 2, 2, 1),         'rips'),
    ('prod_S1_klein', 30, 4,  2, (1, 3, 3, 1),         'rips'),
    ('prod_RP2_RP2',  25, 5,  2, (1, 2, 3, 2, 1),      'rips'),
    ('prod_S2_T2',    30, 5, 11, (1, 2, 2, 2, 1),      'rips'),
]


def main():
    rows = []
    print(f"{'manifold':18s} {'ambient':>8s} {'N':>4s} {'A':>2s} {'B':>2s}    {'extra'}")
    print('-' * 100)
    for key, N, md, coef, expected, builder in TESTS:
        r = run_test(key, N, md, coef, expected, max_r=3.0, builder=builder)
        rows.append(r)
        amb = f"R^{r.get('ambient', '?')}"
        if 'reason' in r:
            extra = f"FAIL: {r['reason']}"
            print(f"{key:18s} {amb:>8s} {N:>4d} ✗  ✗     {extra}")
            continue
        a_mark = '✓' if r['A'] else '✗'
        b_mark = '✓' if r['B'] else '✗'
        if r['A']:
            a_msg = f"A: β at r={r['A_r']:.2f} = {r['A_betti'][:len(r['expected'])]}"
        else:
            a_msg = f"A: best={r['A_betti'][:len(r['expected'])]} (need {r['expected']})"
        if r['B']:
            b_msg = ""
        else:
            b_msg = f"B: top-{sum(r['expected'])} bars by dim = {r['B_dims']} (need {r['expected']})"
        extra = a_msg + ("   " + b_msg if b_msg else "")
        print(f"{key:18s} {amb:>8s} {N:>4d} {a_mark}  {b_mark}     {extra}")

    print()
    n_A = sum(1 for r in rows if r.get('A'))
    n_B = sum(1 for r in rows if r.get('B'))
    print(f"=== A (β-at-r): {n_A}/{len(rows)}    B (long bars by dim): {n_B}/{len(rows)} ===")
    return 0


if __name__ == '__main__':
    sys.exit(main())
