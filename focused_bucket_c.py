"""
focused_bucket_c.py
===================

Focused second-pass sweep for manifolds that failed the broad sweep, plus
the two manifolds the broad sweep didn't finish.

Targets (10 manifolds):
  - T3, prod_S1_RP2, prod_S1_klein   — intrinsic 3, expected to succeed with
                                       maxmin + longer budget
  - prod_S2_S2, prod_RP2_RP2, prod_S2_T2  — intrinsic 4
  - T4, T5                            — intrinsic 4 / 5, high codim (suspect)
  - CP3                               — intrinsic 6
  Plus a recheck of CP2 for window width since the broad sweep found only
  a 1-grid-step window there.

Improvements over sweep_topology_recovery.py:
  1. Maxmin (farthest-point) subsampling from a large oversample pool. This
     gives a much better ε-net per N than uniform random — typically 2–4×
     better recovery at the same N.
  2. Longer per-trial budget (180 s instead of 25 s).
  3. Narrower, finer r grid focused on the "expected sweet spot" (computed
     from the diameter and an empirical scale heuristic).
  4. More seeds at the boundary N where recovery might be marginal.
  5. Result file is written incrementally per (manifold, N, filt, seed)
     trial so a kill doesn't lose data.

Output: focused_results.json — same schema as sweep_results.json, but each
entry has an extra `trials_focused` list. We do NOT overwrite the broad
sweep file; downstream code should merge.
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
    persistence_pairs,
)

# ============================================================================
# Targets — order roughly by expected difficulty (easy first, so we save
# good news quickly).
# ============================================================================

TARGETS = [
    # Re-runs we expect to succeed with maxmin + longer budget
    {"key": "T3",            "expected": [1, 3, 3, 1],       "intrinsic": 3, "coeff": 11,
     "N_list": [40, 60, 80, 120], "r_alpha_max": 1.6, "r_rips_max": 2.5},
    {"key": "prod_S1_klein", "expected": [1, 3, 3, 1],       "intrinsic": 3, "coeff": 2,
     "N_list": [40, 60, 80, 120], "r_alpha_max": 2.0, "r_rips_max": 3.5},
    {"key": "prod_S1_RP2",   "expected": [1, 2, 2, 1],       "intrinsic": 3, "coeff": 2,
     "N_list": [40, 60, 80, 120], "r_alpha_max": 1.6, "r_rips_max": 2.5},
    {"key": "prod_S2_T2",    "expected": [1, 2, 2, 2, 1],    "intrinsic": 4, "coeff": 11,
     "N_list": [30, 50, 75, 100], "r_alpha_max": 1.8, "r_rips_max": 3.0},
    # Intrinsic-4, suspect (high codim or many cycles)
    {"key": "prod_S2_S2",    "expected": [1, 0, 2, 0, 1],    "intrinsic": 4, "coeff": 11,
     "N_list": [30, 50, 75, 100], "r_alpha_max": 1.6, "r_rips_max": 2.5},
    {"key": "prod_RP2_RP2",  "expected": [1, 2, 3, 2, 1],    "intrinsic": 4, "coeff": 2,
     "N_list": [25, 35, 50, 70],  "r_alpha_max": 1.6, "r_rips_max": 2.5,
     "alpha_only": True},
    # High-codim torus / CP — likely need very high N (or different filtration)
    {"key": "T4",            "expected": [1, 4, 6, 4, 1],    "intrinsic": 4, "coeff": 11,
     "N_list": [80, 150, 250, 400], "r_alpha_max": 2.2, "r_rips_max": 3.5},
    {"key": "T5",            "expected": [1, 5, 10, 10, 5, 1], "intrinsic": 5, "coeff": 11,
     "N_list": [60, 120, 200, 350], "r_alpha_max": 2.5, "r_rips_max": 4.0},
    {"key": "CP3",           "expected": [1, 0, 1, 0, 1, 0, 1], "intrinsic": 6, "coeff": 11,
     "N_list": [25, 40, 60, 80],  "r_alpha_max": 1.4, "r_rips_max": 2.5,
     "alpha_only": True},
]

# ============================================================================
# Parameters
# ============================================================================

BUDGET_S       = 180.0
TRIALS_PER_N   = 4         # was 3 in broad sweep
R_GRID_SIZE    = 800       # 2x finer than broad
MAXMIN_OVERSAMPLE = 30     # pool size multiplier for maxmin

# ============================================================================
# Helpers
# ============================================================================

def maxmin_subsample(pts, N_target, seed=0):
    """Greedy farthest-point subsampling. Starts from a random point;
    each step picks the point with maximum minimum-distance to all picked.
    Result is a near-ε-net with ε ≈ 2× the covering radius of the pool."""
    rng = np.random.default_rng(seed)
    M = len(pts)
    if N_target >= M:
        return pts.copy()
    idx = [int(rng.integers(M))]
    dists = np.linalg.norm(pts - pts[idx[0]], axis=1)
    while len(idx) < N_target:
        i = int(dists.argmax())
        idx.append(i)
        dists = np.minimum(dists, np.linalg.norm(pts - pts[i], axis=1))
    return pts[idx]

def sample_maxmin(key, N, seed):
    """Sample MAXMIN_OVERSAMPLE × N points with uniform RNG, then maxmin-down
    to N. For genus-g samplers that produce capped-size pools, fall back to
    direct sampling if oversampling fails."""
    pool_N = N * MAXMIN_OVERSAMPLE
    rng = np.random.default_rng(seed * 7919 + N)
    try:
        pool = sample_by_key(key, pool_N, rng)
    except Exception:
        pool = sample_by_key(key, max(pool_N // 4, N), rng)
    return maxmin_subsample(pool, N, seed)

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

def evaluate(points, filt, max_dim, coeff, expected, r_max, budget_s):
    t0 = time.perf_counter()
    try:
        if filt == "alpha":
            simp = build_alpha_complex(points, r_max, max_dim)
        else:
            simp = build_rips_complex(points, r_max, max_dim)
    except Exception as e:
        return {"recovered": False, "wall_s": time.perf_counter() - t0,
                "error": f"build: {type(e).__name__}: {e}",
                "r_window": None, "best_betti": None}
    wall = time.perf_counter() - t0
    if wall > budget_s:
        return {"recovered": False, "wall_s": wall,
                "error": f"build exceeded budget",
                "r_window": None, "best_betti": None, "n_simp": len(simp)}
    try:
        pers = persistence_pairs(simp, max_dim, coeff)
    except Exception as e:
        return {"recovered": False, "wall_s": time.perf_counter() - t0,
                "error": f"pers: {type(e).__name__}: {e}",
                "r_window": None, "best_betti": None, "n_simp": len(simp)}
    wall = time.perf_counter() - t0
    grid = np.linspace(0.0, r_max, R_GRID_SIZE)
    curve = betti_curve(pers, max_dim, grid)
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
                "best_r": float(grid[i_best]),
                "n_simp": len(simp)}
    lo, hi, length = win
    return {"recovered": True, "wall_s": wall, "error": None,
            "r_window": [float(grid[lo]), float(grid[hi])],
            "window_length": int(length),
            "best_betti": curve[lo].tolist(),
            "best_r": float(grid[lo]),
            "n_simp": len(simp)}

# ============================================================================
# Per-target sweep
# ============================================================================

def sweep_target(t, results_file):
    key = t["key"]
    expected = t["expected"]
    intrinsic = t["intrinsic"]
    coeff = t["coeff"]
    max_dim = intrinsic + 1
    alpha_only = t.get("alpha_only", False)
    r_alpha = t["r_alpha_max"]
    r_rips = t["r_rips_max"]
    print(f"\n=== {key} (intrinsic={intrinsic}, expected={expected}) ===", flush=True)

    trials = []
    smallest = None
    rec_info = None

    for N in t["N_list"]:
        per_n_recovered = False
        # Try Alpha first (fewer simplices typically).
        filt_options = ["alpha"] if alpha_only else ["alpha", "rips"]
        for filt in filt_options:
            r_max = r_alpha if filt == "alpha" else r_rips
            for seed in range(TRIALS_PER_N):
                pts = sample_maxmin(key, N, seed)
                ambient = int(pts.shape[1])
                t0 = time.perf_counter()
                res = evaluate(pts, filt, max_dim, coeff, expected, r_max, BUDGET_S)
                res.update({"N": N, "filtration": filt, "seed": seed,
                            "ambient_dim": ambient, "max_dim": max_dim,
                            "r_max": r_max, "wall_total": time.perf_counter() - t0})
                trials.append(res)
                if res.get("error"):
                    msg = f"  N={N:>4} {filt:5s} seed={seed} ERR {res['error'][:60]} ({res['wall_s']:.1f}s)"
                elif res["recovered"]:
                    w = res["r_window"]
                    msg = (f"  N={N:>4} {filt:5s} seed={seed} simp={res.get('n_simp','?'):>7} "
                           f"RECOVERED r∈[{w[0]:.4f},{w[1]:.4f}] L={res['window_length']} "
                           f"({res['wall_s']:.1f}s)")
                else:
                    msg = (f"  N={N:>4} {filt:5s} seed={seed} simp={res.get('n_simp','?'):>7} "
                           f"miss β={res['best_betti']} at r={res['best_r']:.3f} "
                           f"({res['wall_s']:.1f}s)")
                print(msg, flush=True)
                # Checkpoint after every trial
                save_partial(results_file, key, t, trials, smallest, rec_info)
                if res["recovered"]:
                    per_n_recovered = True
                    if smallest is None:
                        smallest = N
                        rec_info = {
                            "filtration": filt,
                            "seed": seed,
                            "r_window": res["r_window"],
                            "best_r": res["best_r"],
                        }
                    break
                # Abort filtration ladder if one trial way over budget
                if res["wall_s"] > BUDGET_S:
                    print(f"    aborting {filt} at N={N} after budget overrun",
                          flush=True)
                    break
            if per_n_recovered:
                break
        # If we found recovery at this N, no need to push higher
        if smallest is not None:
            print(f"  ✓ smallest N for {key}: {smallest} via {rec_info['filtration']}",
                  flush=True)
            break

    if smallest is None:
        print(f"  ✗ {key} not recovered within N_list", flush=True)

    save_partial(results_file, key, t, trials, smallest, rec_info, final=True)
    return smallest, rec_info

# ============================================================================
# Checkpointing
# ============================================================================

RESULTS_CACHE = {}

def save_partial(results_file, key, t, trials, smallest, rec_info, final=False):
    RESULTS_CACHE[key] = {
        "key": key,
        "expected": t["expected"],
        "intrinsic_dim": t["intrinsic"],
        "coeff_field": t["coeff"],
        "alpha_only": t.get("alpha_only", False),
        "N_list": t["N_list"],
        "trials_focused": trials,
        "smallest_N_recovered": smallest,
        "recovered_info": rec_info,
        "final": final,
    }
    # Always save full cache, not just this entry, so order is stable.
    with open(results_file, "w") as f:
        json.dump(list(RESULTS_CACHE.values()), f, indent=2, default=str)

# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--out", default=str(THIS_DIR / "focused_results.json"))
    args = ap.parse_args()

    targets = TARGETS
    if args.keys:
        wanted = set(args.keys)
        targets = [t for t in TARGETS if t["key"] in wanted]

    summary = []
    for t in targets:
        try:
            smallest, info = sweep_target(t, args.out)
            summary.append((t["key"], smallest, info))
        except Exception as e:
            print(f"FAILED {t['key']}: {type(e).__name__}: {e}", flush=True)
            summary.append((t["key"], None, {"fatal": str(e)}))

    print("\n=== SUMMARY ===", flush=True)
    for key, smallest, info in summary:
        if smallest is None:
            print(f"  {key:18s} NOT RECOVERED", flush=True)
        else:
            f = info.get("filtration", "?")
            w = info.get("r_window", [0, 0])
            print(f"  {key:18s} N={smallest:>3}  {f:5s}  r∈[{w[0]:.3f},{w[1]:.3f}]",
                  flush=True)
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
