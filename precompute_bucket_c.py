"""
precompute_bucket_c.py
======================

For each Bucket-C manifold (the ones that don't recover at any browser-
feasible N), run a much higher-N Alpha persistence and save the result as
a precomputed JSON the frontend can load via a "prebuilt version" button.

Output layout
-------------
One file per manifold:  precomputed/<key>_data.json

JSON schema mirrors the live /compute + /sample responses combined, so the
frontend can ingest it through the same rendering pipeline:

{
  "key":            "T3",
  "label":          "T³",
  "expected_betti": [1, 3, 3, 1],
  "intrinsic_dim":  3,
  "ambient_dim":    6,
  "coeff_field":    11,
  "N":              300,
  "seed":           7,
  "max_dim":        4,
  "sampling":       "maxmin from 30N uniform pool",
  "recovered":      true,
  "recovered_betti": [1, 3, 3, 1, 0],
  "recovered_r":    1.213,
  "recovered_window": [1.208, 1.221],
  "filtration_used": "alpha",
  "wall_seconds":   274.3,
  "points":         [[..ambient..], ...],            # length N
  "alpha": {
    "simplices":    [{"vertices":[...], "filtration": f, "dim": d}, ...],
    "persistence":  [{"dim": k, "birth": ..., "death": ...|null,
                      "birth_simplex":[...]}, ...],
    "max_r":        recovery_r_max,
    "error":        null
  }
}

For failure cases we still write a stub:
{
  "key": "...", "recovered": false, "best_betti": [...], "notes": "..."
}

Strategy per manifold
---------------------
Per the soft-recovery probe at N=80, the right number of long-ish bars per
dim exists but separation is ≈1.0. To push separation past 2× (where the
strict β-match window opens) we need N roughly 3-5× higher.

  key             N_try         filtration   r_alpha_max  budget_s
  T3              200, 300, 400  alpha        1.6          900
  prod_S1_klein   200, 300, 400  alpha        2.0          900
  prod_S1_RP2     150, 200, 300  alpha        1.6          1800
  prod_S2_S2      200, 250, 350  alpha        1.6          900
  prod_S2_T2      150, 200, 300  alpha        1.8          1800
  prod_RP2_RP2     60,  90, 130  alpha        1.6          1800
  T4              200, 300, 500  alpha        2.2          3600   (uncertain)
  T5              120, 200       alpha        2.5          3600   (uncertain)
  CP3              30,  50       alpha        1.4          3600   (uncertain)

For each (manifold, N): try seeds 0..4, take first that strictly recovers
expected β at some r. If none in those seeds, escalate N. If no N works,
write the failure stub.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "precomputed"
OUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(THIS_DIR))

from server import sample_by_key, build_alpha_complex, persistence_pairs

# ============================================================================
# Plan
# ============================================================================

PLAN = [
    # Densification pass: re-run the 3 successful survivors at higher N to
    # widen the recovery window and give a denser cloud for visualization.
    {"key": "T3",            "label": "T³",
     "expected": [1, 3, 3, 1], "intrinsic": 3, "coeff": 11,
     "N_try": [400, 500],     "r_max": 1.8, "budget_s": 1800, "seeds": 3},

    {"key": "prod_S1_klein", "label": "S¹ × Klein",
     "expected": [1, 3, 3, 1], "intrinsic": 3, "coeff": 2,
     "N_try": [600, 800],     "r_max": 2.2, "budget_s": 1800, "seeds": 3},

    {"key": "prod_S2_S2",    "label": "S² × S²",
     "expected": [1, 0, 2, 0, 1], "intrinsic": 4, "coeff": 11,
     "N_try": [300, 400],     "r_max": 1.8, "budget_s": 1800, "seeds": 3},

    {"key": "prod_S1_RP2",   "label": "S¹ × RP²",
     "expected": [1, 2, 2, 1], "intrinsic": 3, "coeff": 2,
     "N_try": [60, 80, 100],   "r_max": 1.8, "budget_s": 600, "seeds": 3},

    {"key": "prod_S2_T2",    "label": "S² × T²",
     "expected": [1, 2, 2, 2, 1], "intrinsic": 4, "coeff": 11,
     "N_try": [80, 120, 160],  "r_max": 2.0, "budget_s": 600, "seeds": 3},

    {"key": "prod_RP2_RP2",  "label": "RP² × RP²",
     "expected": [1, 2, 3, 2, 1], "intrinsic": 4, "coeff": 2,
     "N_try": [30, 45, 60],    "r_max": 1.6, "budget_s": 600, "seeds": 3},

    # Hard cases — one careful attempt each, abort fast.
    {"key": "T4",            "label": "T⁴",
     "expected": [1, 4, 6, 4, 1], "intrinsic": 4, "coeff": 11,
     "N_try": [60, 100, 150],  "r_max": 2.4, "budget_s": 600, "seeds": 3},

    {"key": "T5",            "label": "T⁵",
     "expected": [1, 5, 10, 10, 5, 1], "intrinsic": 5, "coeff": 11,
     "N_try": [40, 60, 80],    "r_max": 2.8, "budget_s": 600, "seeds": 2},

    {"key": "CP3",           "label": "CP³",
     "expected": [1, 0, 1, 0, 1, 0, 1], "intrinsic": 6, "coeff": 11,
     "N_try": [20, 30, 40],    "r_max": 1.4, "budget_s": 600, "seeds": 2},
]

R_GRID_SIZE       = 1500       # very fine, since recovery windows may be tiny
MAXMIN_OVERSAMPLE = 30
WRITE_SIMPLICES   = True       # set False to drop simplex list (saves disk)
SIMPLEX_CAP       = 500_000    # if simplex list exceeds this, drop it (too big)

# ============================================================================
# Helpers (reuse logic from focused_bucket_c)
# ============================================================================

def maxmin_subsample(pts, N_target, seed=0):
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
    rng = np.random.default_rng(seed * 7919 + N)
    try:
        pool = sample_by_key(key, N * MAXMIN_OVERSAMPLE, rng)
    except Exception:
        pool = sample_by_key(key, max(N * 5, N), rng)
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
            i += 1; continue
        j = i
        while j < R and match[j]:
            j += 1
        if best is None or (j - i) > best[2]:
            best = (i, j - 1, j - i)
        i = j
    return best

# ============================================================================
# One precompute attempt
# ============================================================================

def try_recover(p, N, seed):
    """Run Alpha persistence for (manifold, N, seed). Return a result dict
    suitable for JSON serialization (no numpy types in leaves)."""
    expected = p["expected"]
    intrinsic = p["intrinsic"]
    coeff = p["coeff"]
    max_dim = intrinsic + 1
    r_max = p["r_max"]
    budget_s = p["budget_s"]

    t0 = time.perf_counter()
    pts = sample_maxmin(p["key"], N, seed)
    ambient = int(pts.shape[1])

    try:
        simp = build_alpha_complex(pts, r_max, max_dim)
    except Exception as e:
        return {"ok": False, "error": f"alpha build: {type(e).__name__}: {e}",
                "wall": time.perf_counter() - t0, "N": N, "seed": seed,
                "ambient": ambient}
    wall_build = time.perf_counter() - t0
    if wall_build > budget_s:
        return {"ok": False, "error": f"alpha build overran budget {budget_s}s",
                "wall": wall_build, "N": N, "seed": seed,
                "ambient": ambient, "n_simp": len(simp)}

    try:
        pers = persistence_pairs(simp, max_dim, coeff)
    except Exception as e:
        return {"ok": False, "error": f"persistence: {type(e).__name__}: {e}",
                "wall": time.perf_counter() - t0, "N": N, "seed": seed,
                "ambient": ambient, "n_simp": len(simp)}

    wall_total = time.perf_counter() - t0
    grid = np.linspace(0.0, r_max, R_GRID_SIZE)
    curve = betti_curve(pers, max_dim, grid)
    win = first_recovery_window(curve, expected)

    if win is None:
        # Closest match for diagnostics
        target = np.zeros(curve.shape[1], dtype=int)
        for i in range(min(len(expected), curve.shape[1])):
            target[i] = expected[i]
        diffs = np.abs(curve - target[None, :]).sum(axis=1)
        i_best = int(diffs.argmin())
        return {"ok": False, "error": None, "recovered": False,
                "wall": wall_total, "N": N, "seed": seed,
                "ambient": ambient, "n_simp": len(simp),
                "best_betti": curve[i_best].tolist(),
                "best_r": float(grid[i_best])}

    lo, hi, length = win
    return {
        "ok": True, "recovered": True,
        "wall": wall_total, "N": N, "seed": seed,
        "ambient": ambient, "n_simp": len(simp),
        "max_dim": max_dim,
        "r_window": [float(grid[lo]), float(grid[hi])],
        "best_r": float(grid[lo]),
        "best_betti": curve[lo].tolist(),
        "window_length": int(length),
        "points": pts.tolist(),
        "simplices": [
            {"vertices": list(s), "filtration": float(f), "dim": len(s) - 1}
            for s, f in simp
        ] if (WRITE_SIMPLICES and len(simp) <= SIMPLEX_CAP) else [],
        "simplices_dropped": (len(simp) > SIMPLEX_CAP),
        "persistence": pers,
    }

def run_one(p):
    print(f"\n=== {p['key']} ({p['label']}) expected={p['expected']} "
          f"intrinsic={p['intrinsic']} ===", flush=True)
    out_path = OUT_DIR / f"{p['key']}_data.json"
    attempts = []
    found = None
    abort_manifold = False
    consecutive_overruns = 0
    for N in p["N_try"]:
        if abort_manifold:
            print(f"  abort: {p['key']} skipping N={N} after consecutive overruns",
                  flush=True)
            break
        N_had_overrun = False
        for seed in range(p["seeds"]):
            print(f"  trying N={N} seed={seed} ... ", end="", flush=True)
            res = try_recover(p, N, seed)
            wall = res.get("wall", 0) or 0
            is_overrun = (res.get("error") and "overran budget" in res["error"]) \
                         or wall > p["budget_s"] * 1.5
            attempts.append({"N": N, "seed": seed,
                             "ok": res.get("ok", False),
                             "recovered": res.get("recovered", False),
                             "error": res.get("error"),
                             "wall": res.get("wall"),
                             "best_betti": res.get("best_betti")})
            if res.get("recovered"):
                w = res["r_window"]
                print(f"RECOVERED r∈[{w[0]:.4f},{w[1]:.4f}] β={res['best_betti']} "
                      f"({wall:.0f}s, {res['n_simp']} simp)", flush=True)
                found = res
                break
            elif res.get("error"):
                print(f"ERR {res['error'][:80]} ({wall:.0f}s)", flush=True)
            else:
                print(f"miss β={res['best_betti']} at r={res['best_r']:.3f} "
                      f"({wall:.0f}s, {res['n_simp']} simp)", flush=True)
            save_attempts(p, attempts, found, out_path)
            # ABORT-ON-OVERRUN: after a budget overrun at this (N), stop trying
            # more seeds at this N (they'll only be slower) and treat the N
            # as an overrun for the consecutive-N abort.
            if is_overrun:
                N_had_overrun = True
                print(f"  abort: {p['key']} N={N} overran budget, skipping "
                      f"remaining seeds at this N", flush=True)
                break
        if found is not None:
            break
        if N_had_overrun:
            consecutive_overruns += 1
            if consecutive_overruns >= 2:
                print(f"  abort: {p['key']} 2 consecutive Ns budget-overran, "
                      f"declaring manifold unrecoverable", flush=True)
                abort_manifold = True
        else:
            consecutive_overruns = 0
    save_attempts(p, attempts, found, out_path)
    return found

def save_attempts(p, attempts, found, out_path):
    payload = {
        "key": p["key"],
        "label": p["label"],
        "expected_betti": p["expected"],
        "intrinsic_dim": p["intrinsic"],
        "coeff_field": p["coeff"],
        "sampling": f"maxmin from {MAXMIN_OVERSAMPLE}N uniform pool",
        "filtration_used": "alpha",
        "attempts": attempts,
    }
    if found is None:
        payload["recovered"] = False
        payload["note"] = "no N / seed in the plan produced a strict β-match window"
    else:
        payload.update({
            "recovered":         True,
            "N":                 found["N"],
            "seed":              found["seed"],
            "ambient_dim":       found["ambient"],
            "max_dim":           found["max_dim"],
            "recovered_betti":   found["best_betti"],
            "recovered_r":       found["best_r"],
            "recovered_window":  found["r_window"],
            "window_length":     found["window_length"],
            "wall_seconds":      found["wall"],
            "n_simplices":       found["n_simp"],
            "points":            found["points"],
            "alpha": {
                "simplices":   found["simplices"],
                "persistence": found["persistence"],
                "max_r":       float(found["r_window"][1]),
                "error":       None,
                "simplices_dropped": found["simplices_dropped"],
            },
        })
    with open(out_path, "w") as f:
        json.dump(payload, f)

# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", default=None)
    args = ap.parse_args()

    plan = PLAN
    if args.keys:
        wanted = set(args.keys)
        plan = [p for p in PLAN if p["key"] in wanted]

    summary = []
    for p in plan:
        try:
            found = run_one(p)
            summary.append((p["key"], bool(found),
                            found.get("N") if found else None,
                            found.get("r_window") if found else None))
        except KeyboardInterrupt:
            print("\nInterrupted — partial results saved per-manifold.", flush=True)
            raise
        except Exception as e:
            print(f"FATAL on {p['key']}: {type(e).__name__}: {e}", flush=True)
            summary.append((p["key"], False, None, None))

    print("\n=== PRECOMPUTE SUMMARY ===", flush=True)
    for key, ok, N, win in summary:
        if ok:
            print(f"  {key:18s} ✓ N={N}  r∈[{win[0]:.3f},{win[1]:.3f}]", flush=True)
        else:
            print(f"  {key:18s} ✗ not recovered", flush=True)

if __name__ == "__main__":
    main()
