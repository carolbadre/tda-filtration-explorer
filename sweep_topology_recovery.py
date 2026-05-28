"""
sweep_topology_recovery.py
==========================

For each preset manifold in MANIFOLD_SETTINGS (as defined in index.html),
determine the smallest N at which *at least one* of (Vietoris-Rips, Čech,
Alpha) has some radius r where its β(r) vector matches the manifold's
expected Betti numbers.

Output is a JSON report (sweep_results.json) with one entry per manifold:
  {
    "key", "expected", "intrinsic_dim", "ambient_dim",
    "current_N_max",
    "current_N_recovers": bool,    # does current N_max already recover?
    "smallest_N_recovered": int|null,
    "recovered_filtration": "alpha"|"rips"|"cech"|null,
    "recovered_r_window": [float, float]|null,
    "trials": [ ... per (N, filt, seed) ... ],
    "browser_feasible_at_smallest_N": bool,
    "notes": [str]
  }

Design decisions
----------------
- Per-filtration r_max is *geometry-aware*: derived from the actual point-
  cloud diameter rather than the slider's static default. (The static
  defaults of 2.0/1.5/1.5 are calibrated to unit-radius manifolds and are
  too small for T^4, T^5, prod_S2_S2, etc. which have diameter ~2√d.)
- Filtration whitelist per ambient dim:
    ambient ≤ 5:  Alpha → Rips      (Alpha fast, Rips OK)
    6 ≤ amb ≤ 10: Alpha → Rips      (Rips capped at small N)
    ambient > 10: Alpha only        (Rips would explode at high max_dim)
  (Čech is skipped — same persistent homology as Alpha by Delaunay-Čech
  nerve theorem, redundant for "does some filtration recover?".)
- N ladder per intrinsic dim (caps chosen so each trial < ~30 s wall):
    intrinsic 1-2: [N_max, 2x, 3x, 4x]      cap 200
    intrinsic 3:   [N_max, 1.5x, 2x, 3x]    cap 150
    intrinsic 4:   [N_max, 1.5x, 2x, 2.5x]  cap 100
    intrinsic 5-6: [N_max, 1.3x, 1.6x, 2x]  cap  60
- TRIALS_PER_N seeds per combo. Recovery at any seed counts as "possible
  at this N" — we want existence, not guarantee.

Browser-feasibility heuristic
-----------------------------
The browser's existing N_max values give us empirical thresholds for
what's tolerable as an interactive /compute call. We treat as "browser
feasible":
  ambient ≤ 5  and  intrinsic ≤ 3:  N ≤ 100
  ambient ≤ 8  and  intrinsic = 4:  N ≤ 60
  ambient ≤ 10 and  intrinsic = 5:  N ≤ 30
  intrinsic = 6:                    N ≤ 20
(Above these, Rips would block the UI for several seconds; precompute is
preferred.) These are *suggestions* — actual /compute timing depends on
how many top-dim simplices the chosen filtration enumerates.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from server import (
    sample_by_key,
    build_alpha_complex,
    build_rips_complex,
    build_cech_complex,
    persistence_pairs,
)

# ============================================================================
# Manifold table
# ============================================================================

MANIFOLDS = [
    # Spheres
    {"key": "S1", "expected": [1, 1],             "intrinsic_dim": 1, "coeff": 11, "N_max":  40},
    {"key": "S2", "expected": [1, 0, 1],          "intrinsic_dim": 2, "coeff": 11, "N_max":  35},
    {"key": "S3", "expected": [1, 0, 0, 1],       "intrinsic_dim": 3, "coeff": 11, "N_max":  25},
    {"key": "S4", "expected": [1, 0, 0, 0, 1],    "intrinsic_dim": 4, "coeff": 11, "N_max":  18},
    {"key": "S5", "expected": [1, 0, 0, 0, 0, 1], "intrinsic_dim": 5, "coeff": 11, "N_max":  14},
    # Tori
    {"key": "T2", "expected": [1, 2, 1],          "intrinsic_dim": 2, "coeff": 11, "N_max":  35},
    {"key": "T3", "expected": [1, 3, 3, 1],       "intrinsic_dim": 3, "coeff": 11, "N_max":  25},
    {"key": "T4", "expected": [1, 4, 6, 4, 1],    "intrinsic_dim": 4, "coeff": 11, "N_max":  18},
    {"key": "T5", "expected": [1, 5, 10, 10, 5, 1], "intrinsic_dim": 5, "coeff": 11, "N_max":  14},
    # RP^n
    {"key": "RP1", "expected": [1, 1],            "intrinsic_dim": 1, "coeff": 2,  "N_max":  35},
    {"key": "RP2", "expected": [1, 1, 1],         "intrinsic_dim": 2, "coeff": 2,  "N_max":  25},
    {"key": "RP3", "expected": [1, 1, 1, 1],      "intrinsic_dim": 3, "coeff": 2,  "N_max":  18},
    # CP^n
    {"key": "CP2", "expected": [1, 0, 1, 0, 1],   "intrinsic_dim": 4, "coeff": 11, "N_max":  18},
    {"key": "CP3", "expected": [1, 0, 1, 0, 1, 0, 1], "intrinsic_dim": 6, "coeff": 11, "N_max":  12},
    # Surfaces
    {"key": "mobius",  "expected": [1, 1],        "intrinsic_dim": 2, "coeff": 11, "N_max":  40},
    {"key": "klein",   "expected": [1, 2, 1],     "intrinsic_dim": 2, "coeff": 2,  "N_max":  35},
    {"key": "genus2",  "expected": [1, 4, 1],     "intrinsic_dim": 2, "coeff": 11, "N_max":  80},
    {"key": "genus3",  "expected": [1, 6, 1],     "intrinsic_dim": 2, "coeff": 11, "N_max":  80},
    {"key": "genus4",  "expected": [1, 8, 1],     "intrinsic_dim": 2, "coeff": 11, "N_max":  80},
    # Lens
    {"key": "lens_3_1", "expected": [1, 1, 1, 1], "intrinsic_dim": 3, "coeff": 3,  "N_max":  20},
    {"key": "lens_5_1", "expected": [1, 1, 1, 1], "intrinsic_dim": 3, "coeff": 5,  "N_max":  20},
    {"key": "lens_5_2", "expected": [1, 1, 1, 1], "intrinsic_dim": 3, "coeff": 5,  "N_max":  20},
    {"key": "lens_7_1", "expected": [1, 1, 1, 1], "intrinsic_dim": 3, "coeff": 7,  "N_max":  20},
    {"key": "lens_7_2", "expected": [1, 1, 1, 1], "intrinsic_dim": 3, "coeff": 7,  "N_max":  20},
    # Products
    {"key": "prod_S2_S1",    "expected": [1, 1, 1, 1],    "intrinsic_dim": 3, "coeff": 11, "N_max":  25},
    {"key": "prod_S2_S2",    "expected": [1, 0, 2, 0, 1], "intrinsic_dim": 4, "coeff": 11, "N_max":  22},
    {"key": "prod_S1_RP2",   "expected": [1, 2, 2, 1],    "intrinsic_dim": 3, "coeff": 2,  "N_max":  22},
    {"key": "prod_S1_klein", "expected": [1, 3, 3, 1],    "intrinsic_dim": 3, "coeff": 2,  "N_max":  25},
    {"key": "prod_RP2_RP2",  "expected": [1, 2, 3, 2, 1], "intrinsic_dim": 4, "coeff": 2,  "N_max":  18},
    {"key": "prod_S2_T2",    "expected": [1, 2, 2, 2, 1], "intrinsic_dim": 4, "coeff": 11, "N_max":  22},
]

# ============================================================================
# Sweep parameters
# ============================================================================

BUDGET_S       = 25.0
TRIALS_PER_N   = 3
R_GRID_SIZE    = 400

# N ladders per intrinsic dim (multiplicative steps, capped).
# Each entry: (multipliers, cap).
N_LADDER_BY_DIM = {
    1: ([1.0, 2.0, 3.0, 4.0], 200),
    2: ([1.0, 2.0, 3.0, 4.0], 200),
    3: ([1.0, 1.5, 2.0, 3.0], 150),
    4: ([1.0, 1.5, 2.0, 2.5], 100),
    5: ([1.0, 1.3, 1.6, 2.0],  60),
    6: ([1.0, 1.3, 1.6, 2.0],  40),
}

# Browser-feasibility thresholds (used after the sweep to label which N's
# would be slow to /compute in the browser).
BROWSER_FEASIBLE_MAX_N = {
    # (ambient_band, intrinsic_dim) → max N considered interactive
    (5,  3):  100,
    (8,  3):   80,
    (8,  4):   60,
    (10, 4):   40,
    (10, 5):   30,
    (99, 6):   20,
}

def is_browser_feasible(N, ambient, intrinsic):
    """Approximate: is this (N, ambient, intrinsic) tractable for the
    browser's interactive /compute call?"""
    if intrinsic <= 2 and ambient <= 6:
        return N <= 150
    if intrinsic == 3 and ambient <= 8:
        return N <= 80
    if intrinsic == 4 and ambient <= 8:
        return N <= 60
    if intrinsic == 4:
        return N <= 40
    if intrinsic == 5:
        return N <= 30
    if intrinsic == 6:
        return N <= 20
    return N <= 100

# ============================================================================
# Persistence + recovery helpers
# ============================================================================

def betti_curve(persistence, max_dim, r_grid):
    out = np.zeros((len(r_grid), max_dim + 1), dtype=int)
    for p in persistence:
        d = p["dim"]
        if d > max_dim:
            continue
        b = p["birth"]
        x = math.inf if p["death"] is None else p["death"]
        lo = np.searchsorted(r_grid, b, side="left")
        hi = np.searchsorted(r_grid, x, side="left")
        out[lo:hi, d] += 1
    return out

def first_recovery_window(curve, expected):
    K1 = curve.shape[1]
    target = np.zeros(K1, dtype=int)
    for i in range(min(len(expected), K1)):
        target[i] = expected[i]
    match = np.all(curve == target[None, :], axis=1)
    if not np.any(match):
        return None
    best = None
    i = 0
    R = len(match)
    while i < R:
        if not match[i]:
            i += 1
            continue
        j = i
        while j < R and match[j]:
            j += 1
        if best is None or (j - i) > best[2]:
            best = (i, j - 1, j - i)
        i = j
    return best

def diameter_estimate(points):
    """Cheap estimate of pairwise diameter: 2 × max distance from centroid."""
    centered = points - points.mean(axis=0)
    return float(2.0 * np.linalg.norm(centered, axis=1).max())

def r_max_for(filt, diameter):
    """Pick a generous r_max per filtration based on observed diameter.
    Alpha/Čech use radius scale, Rips uses edge-length (diameter) scale."""
    if filt == "rips":
        # Rips edge length: needs to be at least diameter to fully connect.
        # Cap to avoid producing absurd numbers of high-dim simplices.
        return max(2.0, min(diameter, 4.0))
    # Alpha / Cech: radius. diameter/2 covers convex hulls; bump slightly.
    return max(1.5, min(diameter * 0.6, 2.5))

# ============================================================================
# One trial
# ============================================================================

def evaluate_recovery(points, filt_name, max_dim, coeff, expected, r_max, budget_s):
    t0 = time.perf_counter()
    try:
        if filt_name == "alpha":
            simp = build_alpha_complex(points, r_max, max_dim)
        elif filt_name == "rips":
            simp = build_rips_complex(points, r_max, max_dim)
        elif filt_name == "cech":
            simp = build_cech_complex(points, r_max, max_dim)
        else:
            raise ValueError(filt_name)
    except Exception as e:
        return {"recovered": False, "wall_s": time.perf_counter() - t0,
                "error": f"build: {type(e).__name__}: {e}",
                "r_window": None, "best_betti": None, "n_simplices": 0}
    n_simp = len(simp)
    if time.perf_counter() - t0 > budget_s:
        return {"recovered": False, "wall_s": time.perf_counter() - t0,
                "error": f"build exceeded budget ({budget_s}s)",
                "r_window": None, "best_betti": None, "n_simplices": n_simp}
    try:
        pers = persistence_pairs(simp, max_dim, coeff)
    except Exception as e:
        return {"recovered": False, "wall_s": time.perf_counter() - t0,
                "error": f"pers: {type(e).__name__}: {e}",
                "r_window": None, "best_betti": None, "n_simplices": n_simp}
    wall = time.perf_counter() - t0
    r_grid = np.linspace(0.0, r_max, R_GRID_SIZE)
    curve = betti_curve(pers, max_dim, r_grid)
    win = first_recovery_window(curve, expected)
    if win is None:
        target = np.zeros(curve.shape[1], dtype=int)
        for i in range(min(len(expected), curve.shape[1])):
            target[i] = expected[i]
        diffs = np.abs(curve - target[None, :]).sum(axis=1)
        i_best = int(diffs.argmin())
        return {"recovered": False, "wall_s": wall, "error": None,
                "r_window": None,
                "best_betti": curve[i_best].tolist(),
                "best_r": float(r_grid[i_best]),
                "n_simplices": n_simp}
    lo, hi, length = win
    return {"recovered": True, "wall_s": wall, "error": None,
            "r_window": [float(r_grid[lo]), float(r_grid[hi])],
            "window_length": int(length),
            "best_betti": curve[lo].tolist(),
            "best_r": float(r_grid[lo]),
            "n_simplices": n_simp}

# ============================================================================
# Per-manifold sweep
# ============================================================================

def n_ladder(N_start, intrinsic):
    mult, cap = N_LADDER_BY_DIM.get(intrinsic, ([1.0, 2.0, 3.0], 100))
    out = []
    for m in mult:
        n = int(round(N_start * m))
        if n > cap:
            n = cap
        if not out or n != out[-1]:
            out.append(n)
        if n == cap:
            break
    return out

def filtration_order(ambient_dim):
    """Which filtrations to try, in order. Alpha always first when feasible.
    Rips skipped at very high ambient dim because max_dim+1-simplex
    enumeration explodes."""
    if ambient_dim <= 5:
        return ["alpha", "rips"]
    if ambient_dim <= 10:
        return ["alpha", "rips"]   # Rips at higher ambient but small N
    return ["alpha"]               # ambient > 10: Alpha only

def sweep_one(m, verbose=True):
    key = m["key"]
    expected = m["expected"]
    intrinsic = m["intrinsic_dim"]
    coeff = m["coeff"]
    N_max = m["N_max"]
    sweep_max_dim = intrinsic + 1   # match index.html; preserves top-dim essentials

    pts_probe = sample_by_key(key, max(N_max, 20), np.random.default_rng(0))
    ambient = int(pts_probe.shape[1])
    diameter = diameter_estimate(pts_probe)

    if verbose:
        print(f"\n=== {key} intrinsic={intrinsic} ambient={ambient} "
              f"diam≈{diameter:.2f} expected={expected} N_max={N_max} ===",
              flush=True)

    ladder = n_ladder(N_max, intrinsic)
    filt_order = filtration_order(ambient)
    if verbose:
        print(f"  N ladder={ladder}  filt order={filt_order}", flush=True)

    trials = []
    smallest = None
    rec_filt = None
    rec_window = None
    rec_betti = None
    rec_r = None

    # First, evaluate current_N_max with all filtrations across seeds, to
    # report whether the manifold already recovers at the explorer default.
    current_recovers = False

    for N in ladder:
        # Cap N to a sane budget for Rips at high intrinsic dim:
        # if Rips would enumerate > ~5M simplices we skip it (Alpha only).
        per_n_recovered = False
        for filt in filt_order:
            r_max = r_max_for(filt, diameter)
            for seed in range(TRIALS_PER_N):
                rng = np.random.default_rng(seed * 1000 + N)
                pts = sample_by_key(key, N, rng)
                res = evaluate_recovery(pts, filt, sweep_max_dim, coeff,
                                        expected, r_max, BUDGET_S)
                res.update({"N": N, "filtration": filt, "seed": seed,
                            "max_dim": sweep_max_dim, "ambient_dim": ambient,
                            "r_max": r_max})
                trials.append(res)
                if verbose:
                    msg = f"  N={N:>4} {filt:5s} seed={seed} simp={res['n_simplices']:>7d} "
                    if res.get("error"):
                        msg += f"ERR {res['error'][:80]}"
                    elif res["recovered"]:
                        msg += (f"RECOVERED β=expected at r∈[{res['r_window'][0]:.3f},"
                                f"{res['r_window'][1]:.3f}] ({res['wall_s']:.1f}s)")
                    else:
                        msg += (f"miss best β={res['best_betti']} at r={res['best_r']:.3f}"
                                f" ({res['wall_s']:.1f}s)")
                    print(msg, flush=True)
                if res["recovered"]:
                    per_n_recovered = True
                    if N == N_max:
                        current_recovers = True
                    if smallest is None:
                        smallest = N
                        rec_filt = filt
                        rec_window = res["r_window"]
                        rec_betti = res["best_betti"]
                        rec_r = res["best_r"]
                    break  # done with seeds for this filt
                # If first seed of a filt exceeded budget hard, don't try more
                # seeds of the same filt at the same N.
                if res["wall_s"] > BUDGET_S * 1.2:
                    break
            if per_n_recovered:
                break  # done with filts for this N
        # If smallest is found AND we are past N_max, we can stop early
        # only if we also know whether current N_max recovers — we already
        # processed N_max first (it's ladder[0]).
        if smallest is not None and N > N_max:
            break

    notes = []
    feasible = (smallest is not None and is_browser_feasible(smallest, ambient, intrinsic))
    if smallest is not None and not feasible:
        notes.append(f"smallest N ({smallest}) likely too slow for live /compute — recommend precompute")
    if smallest is None:
        notes.append("no recovery within the sweep N ladder + r ranges tested")

    return {
        "key": key,
        "expected": expected,
        "intrinsic_dim": intrinsic,
        "ambient_dim": ambient,
        "diameter": diameter,
        "coeff_field": coeff,
        "current_N_max": N_max,
        "current_N_recovers": current_recovers,
        "smallest_N_recovered": smallest,
        "recovered_filtration": rec_filt,
        "recovered_r_window": rec_window,
        "recovered_betti": rec_betti,
        "recovered_r_first": rec_r,
        "browser_feasible_at_smallest_N": feasible,
        "trials": trials,
        "notes": notes,
    }

# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--out", default=str(THIS_DIR / "sweep_results.json"))
    args = ap.parse_args()

    targets = MANIFOLDS
    if args.keys:
        wanted = set(args.keys)
        targets = [m for m in MANIFOLDS if m["key"] in wanted]

    results = []
    for m in targets:
        try:
            r = sweep_one(m)
        except Exception as e:
            print(f"FAILED {m['key']}: {e}", flush=True)
            r = {"key": m["key"], "fatal_error": f"{type(e).__name__}: {e}"}
        results.append(r)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} entries to {args.out}", flush=True)

if __name__ == "__main__":
    main()
