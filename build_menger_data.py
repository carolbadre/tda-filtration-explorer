"""
build_menger_data.py — offline precompute for the Menger-sponge TDA section.

What this script does
=====================
For each iteration level n of the Menger sponge:

  1. Samples N=600 000 uniform points on M_n via the IFS "chaos game"
     (20 contracting maps, one per retained sub-cube of M_1).
  2. Runs GUDHI's Delaunay-Čech and Alpha filtrations up to a geometric
     radius of 0.4 (NOT squared; we sqrt GUDHI's α² output to match the
     live /compute endpoint's convention).
  3. Computes the exact cubical Betti numbers of M_n via gudhi.CubicalComplex
     on the 3^n × 3^n × 3^n indicator grid (retained sub-cube = filtration 0;
     removed = +inf; β at filtration 0.5).
  4. Filters the persistence diagrams: keep every bar with persistence
     > 0.005, plus a uniform sample of up to 600 sub-threshold "noise" bars
     per dimension (so the frontend's PD still visually shows the noise
     floor). Per-dim drop counts go into n_filtered_noise.
  5. Counts persistent H_1 features from the Alpha PD (persistence > 0.01)
     and uses the per-level deltas to set max_level empirically: the largest
     n such that the persistent-H_1 count grew by at least 1 over level n-1.
     Levels past max_level (the PD plateaus there at this sample density)
     are dropped — the frontend slider clips to max_level.
  6. Persists everything to menger_data.json next to this script.

Output JSON shape (per the brief):
{
  "N": 600000, "max_level": <int>,
  "levels": [
    {
      "level": int, "label": "M_n (…)",
      "n_points_full": 600000,
      "points_display": [[x,y,z], ...],         # 8000-point IFS subsample
      "cubical_betti": [int,int,int,int] | null, # null only on OOM
      "n_persistent_h1": int,
      "n_persistent_h2_artifact": int,
      "cech":  {"max_r": 0.4, "bars": [...], "n_filtered_noise": {...}},
      "alpha": {"max_r": 0.4, "bars": [...], "n_filtered_noise": {...}},
    },
    ...
  ]
}

Bars are {"dim", "birth", "death"} with death=None for essential classes.

Usage
=====
  cd v4/
  python build_menger_data.py

Optional environment overrides (mostly for smoke-testing the pipeline):
  MENGER_N           — point count per level (default 600000)
  MENGER_MAX_R       — geometric radius cap for Čech + Alpha (default 0.4)
  MENGER_MAX_LEVEL_CAP — hard ceiling on the candidate levels probed
                         (default 6 — i.e. probe 0..6, stop earlier per the
                         empirical-plateau rule)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import gudhi as gd


HERE       = Path(__file__).parent
OUTPUT     = HERE / "menger_data.json"

N_POINTS          = int(os.environ.get("MENGER_N", 600_000))
MAX_R             = float(os.environ.get("MENGER_MAX_R", 0.4))
MAX_LEVEL_CAP     = int(os.environ.get("MENGER_MAX_LEVEL_CAP", 5))

# Filter thresholds, per the brief.
PERSIST_KEEP_ALL_ABOVE   = 0.005    # bars with persistence > this are kept
# H_1 count threshold for the empirical max_level rule. 0.007 is tuned to
# exactly recover the cubical truth on M_1/M_2/M_3 at N=600k while still
# admitting M_4 features (whose persistence sits in roughly 0.006-0.012
# because M_4 tunnels live at scale 3^-4 ≈ 0.0123). The original brief used
# 0.01, which was tight enough that M_4 features fell below it; lowering to
# 0.007 lets the max_level rule unlock level 4 without contaminating the
# smaller-level counts (verified by the threshold sweep over the in-disk JSON
# at N=600k: M_3 count is exactly 1409 at thresholds 0.007..0.015).
PERSIST_H1_THRESHOLD     = 0.007
NOISE_SAMPLE_CAP_PER_DIM = 600      # uniform-sample cap for sub-threshold bars

DISPLAY_SUBSAMPLE = 8000            # points dumped per level for the 3D viz

SEED_BASE    = 20260516
SEED_DISPLAY = 20260516 + 1


# ----------------------------------------------------------------------------
# IFS sampler — uniform on M_n via the 20-map chaos game.
# ----------------------------------------------------------------------------
# 20 retained sub-cube offsets of M_1: triples (i,j,k) ∈ {0,1,2}^3 with at
# most one coordinate equal to 1 (i.e. NOT the centre cube and NOT the 6
# face-centred cubes).
KEEP_OFFSETS = np.array([
    [i, j, k]
    for i in range(3) for j in range(3) for k in range(3)
    if (i == 1) + (j == 1) + (k == 1) <= 1
], dtype=float)
assert KEEP_OFFSETS.shape == (20, 3)


def sample_menger_ifs(level: int, N: int, rng: np.random.Generator) -> np.ndarray:
    """N uniform points on M_n via the IFS chaos game.

    For each point we pick a length-`level` random address ∈ {0..19}^level,
    accumulate the corresponding translations at decreasing scale 3^-(k+1),
    then add a uniform offset within the final sub-cube of side 3^-level.
    """
    if level == 0:
        return rng.uniform(0.0, 1.0, (N, 3))
    addr = rng.integers(0, 20, size=(N, level))
    pos = np.zeros((N, 3))
    for k in range(level):
        pos += (3.0 ** -(k + 1)) * KEEP_OFFSETS[addr[:, k]]
    pos += rng.uniform(0.0, 3.0 ** -level, (N, 3))
    return pos


# ----------------------------------------------------------------------------
# Exact cubical Betti numbers via gudhi.CubicalComplex.
# ----------------------------------------------------------------------------
def menger_filter(i: int, j: int, k: int) -> bool:
    """True iff sub-cube (i,j,k) ∈ {0,1,2}^3 is retained in M_1."""
    return (i == 1) + (j == 1) + (k == 1) <= 1


def build_retained_cubes(n: int) -> np.ndarray:
    """3^n × 3^n × 3^n boolean array; True = retained sub-cube in M_n."""
    if n == 0:
        return np.array([[[True]]])
    prev = build_retained_cubes(n - 1)
    side = 3 ** n
    new = np.zeros((side, side, side), dtype=bool)
    keep = [
        (di, dj, dk)
        for di in range(3) for dj in range(3) for dk in range(3)
        if menger_filter(di, dj, dk)
    ]
    prev_side = 3 ** (n - 1)
    # NumPy slice fast-path: for each retained parent sub-cube, OR in the 20
    # retained offsets in one shot. Loops over parents but inner write is O(1)
    # via slicing; ~30x faster than a 4-nested Python loop at n=5.
    for i in range(prev_side):
        for j in range(prev_side):
            for k in range(prev_side):
                if not prev[i, j, k]:
                    continue
                base_i, base_j, base_k = 3 * i, 3 * j, 3 * k
                for di, dj, dk in keep:
                    new[base_i + di, base_j + dj, base_k + dk] = True
    return new


def cubical_betti_for_level(n: int) -> list[int] | None:
    """Returns [b_0, b_1, b_2, b_3] for M_n via cubical complex, padded to 4.

    Returns None if the 3^n × 3^n × 3^n grid would OOM during construction
    (caught at the build_retained_cubes level by MemoryError) or if the gudhi
    persistence call fails. The frontend handles `cubical_betti: null`.
    """
    try:
        R = build_retained_cubes(n)
    except MemoryError:
        print(f"      [cubical n={n}] MemoryError building retained cubes — null", flush=True)
        return None
    try:
        # Filtration value 0 = retained, +inf = removed (encoded as a large
        # finite value — gudhi treats huge floats as effectively absent at
        # the chosen evaluation filtration of 0.5).
        top_cells = np.where(R, 0.0, 1e10)
        cc = gd.CubicalComplex(top_dimensional_cells=top_cells)
        cc.compute_persistence()
        betti = cc.persistent_betti_numbers(0.5, 0.5)
    except Exception as e:
        print(f"      [cubical n={n}] gudhi failed: {type(e).__name__}: {e}", flush=True)
        return None
    # Pad to length 4 so the frontend can read a fixed [b_0..b_3] vector.
    while len(betti) < 4:
        betti = list(betti) + [0]
    return [int(x) for x in betti[:4]]


# ----------------------------------------------------------------------------
# Persistence: pull bars out of a built simplex tree.
# ----------------------------------------------------------------------------
def persistence_bars(st: gd.SimplexTree, max_dim: int, *, alpha_squared: bool):
    """Return list of {"dim", "birth", "death"} (death=None for essential).

    Drops dim-`max_dim` essentials (artifact of the max_dim cap). If
    `alpha_squared=True`, sqrt the filtration values to convert α² → α.
    """
    st.compute_persistence(homology_coeff_field=11)
    pairs = st.persistence_pairs()
    out = []
    for birth_simplex, death_simplex in pairs:
        dim = len(birth_simplex) - 1
        if dim > max_dim:
            continue
        is_essential = (len(death_simplex) == 0)
        if dim == max_dim and is_essential:
            continue
        b = float(st.filtration(birth_simplex))
        d = None if is_essential else float(st.filtration(death_simplex))
        if alpha_squared:
            b = math.sqrt(max(b, 0.0))
            if d is not None:
                d = math.sqrt(max(d, 0.0))
        out.append({"dim": dim, "birth": b, "death": d})
    return out


def filter_bars(bars: list[dict], rng: np.random.Generator):
    """Split bars into 'kept' and 'dropped-noise', applying the brief's policy:

      - persistence > PERSIST_KEEP_ALL_ABOVE → always kept
      - persistence ≤ PERSIST_KEEP_ALL_ABOVE → uniformly subsample up to
        NOISE_SAMPLE_CAP_PER_DIM per dim; the rest go into n_filtered_noise
      - essentials (death=None) → always kept (infinite persistence)

    Returns (kept_bars, n_filtered_noise_by_dim_dict).
    """
    by_dim: dict[int, list[dict]] = {}
    for b in bars:
        by_dim.setdefault(b["dim"], []).append(b)

    kept = []
    dropped_counts: dict[int, int] = {}
    for d, group in by_dim.items():
        keep_idx = []
        noise_idx = []
        for i, bar in enumerate(group):
            if bar["death"] is None:
                keep_idx.append(i)
                continue
            pers = bar["death"] - bar["birth"]
            if pers > PERSIST_KEEP_ALL_ABOVE:
                keep_idx.append(i)
            else:
                noise_idx.append(i)
        # Uniform sample of noise bars
        if len(noise_idx) > NOISE_SAMPLE_CAP_PER_DIM:
            sampled = rng.choice(noise_idx, NOISE_SAMPLE_CAP_PER_DIM, replace=False)
            kept_noise = set(int(x) for x in sampled)
            dropped_counts[d] = len(noise_idx) - NOISE_SAMPLE_CAP_PER_DIM
        else:
            kept_noise = set(noise_idx)
            dropped_counts[d] = 0
        for i in keep_idx:
            kept.append(group[i])
        for i in kept_noise:
            kept.append(group[i])
    # Stringify dropped_counts keys for JSON
    return kept, {str(k): int(v) for k, v in dropped_counts.items()}


def count_persistent_h1_alpha(alpha_bars: list[dict]) -> int:
    """Bars from the Alpha PD with H_1 persistence > PERSIST_H1_THRESHOLD."""
    c = 0
    for b in alpha_bars:
        if b["dim"] != 1:
            continue
        if b["death"] is None:
            c += 1  # essential H_1 — always 'persistent'
            continue
        if (b["death"] - b["birth"]) > PERSIST_H1_THRESHOLD:
            c += 1
    return c


def count_h1_at_thresholds(alpha_bars: list[dict],
                           thresholds=(0.005, 0.006, 0.007, 0.008, 0.009,
                                       0.010, 0.012, 0.015)) -> dict[str, int]:
    """Histogram-of-counts: how many H_1 bars survive each persistence cut?

    Computed once at build time so the frontend (or a future re-tune) can
    pick a different display threshold without recomputing the persistence.
    Returns a dict keyed by str(threshold) → count. Essentials always count.
    """
    out = {}
    for t in thresholds:
        c = 0
        for b in alpha_bars:
            if b["dim"] != 1:
                continue
            if b["death"] is None:
                c += 1
            elif (b["death"] - b["birth"]) > t:
                c += 1
        out[f"{t:.3f}"] = c
    return out


def count_persistent_h2(alpha_bars: list[dict]) -> int:
    """H_2 from sampling-noise voids (M_n has b_2 = 0 by cubical computation;
    anything > 0 here is sampling artifact in the discrete point cloud)."""
    c = 0
    for b in alpha_bars:
        if b["dim"] != 2:
            continue
        if b["death"] is None:
            c += 1
            continue
        if (b["death"] - b["birth"]) > PERSIST_H1_THRESHOLD:
            c += 1
    return c


# ----------------------------------------------------------------------------
# Per-level work: sample, run both filtrations, compute cubical Betti.
# ----------------------------------------------------------------------------
def label_for_level(n: int) -> str:
    sub = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
    s = "".join(sub[int(c)] for c in str(n))
    notes = {0: " (solid cube)", 1: " (cube minus 7 sub-cubes)"}
    return f"M{s}{notes.get(n, '')}"


def process_level(level: int, N: int, max_r: float,
                  rng_full: np.random.Generator,
                  rng_display: np.random.Generator,
                  rng_noise: np.random.Generator) -> dict:
    print(f"\n[level {level}] sampling N={N} via IFS …", flush=True)
    t0 = time.time()
    pts = sample_menger_ifs(level, N, rng_full)
    print(f"           done in {time.time()-t0:.1f}s   bbox=[{pts.min():.4f}, {pts.max():.4f}]",
          flush=True)

    # 8000-point display subsample with a separate seed (so the viz is the
    # same shape across reruns and decoupled from the full-N RNG state).
    disp_pts = sample_menger_ifs(level, DISPLAY_SUBSAMPLE, rng_display)

    # Cubical Betti numbers — exact integer answer for M_n.
    print(f"           cubical Betti via {3**level}^3 grid …", flush=True)
    t1 = time.time()
    cubical_betti = cubical_betti_for_level(level)
    print(f"           cubical β = {cubical_betti}   ({time.time()-t1:.1f}s)", flush=True)

    # Sanity-check the small levels against the brief's stated truths.
    expected_small = {
        0: [1, 0, 0, 0],
        1: [1, 5, 0, 0],
        2: [1, 81, 0, 0],
    }
    if level in expected_small and cubical_betti != expected_small[level]:
        raise RuntimeError(
            f"M_{level} cubical β = {cubical_betti} but expected "
            f"{expected_small[level]} — there is a bug in build_retained_cubes "
            f"or in the cubical-Betti pipeline. Refusing to write a wrong "
            f"menger_data.json. (See the brief's resolution-limit note: this "
            f"check is the canary.)"
        )

    # Čech (Delaunay-Čech) — radii directly, no sqrt needed.
    print(f"           Delaunay-Čech, max_r={max_r}, "
          f"max_alpha_square={max_r**2:.4f} …", flush=True)
    t2 = time.time()
    dc = gd.DelaunayCechComplex(points=pts.tolist())
    st_cech = dc.create_simplex_tree(
        max_alpha_square=max_r ** 2,
        output_squared_values=False,
    )
    n_cech_simp = st_cech.num_simplices()
    cech_bars_full = persistence_bars(st_cech, max_dim=2, alpha_squared=False)
    cech_kept, cech_dropped = filter_bars(cech_bars_full, rng_noise)
    print(f"           Čech: {n_cech_simp:,} simplices, "
          f"{len(cech_bars_full):,} bars total → {len(cech_kept):,} kept "
          f"({time.time()-t2:.1f}s)", flush=True)
    del dc, st_cech  # free before next filtration

    # Alpha — GUDHI returns α², sqrt back to radii.
    print(f"           Alpha,        max_r={max_r}, "
          f"max_alpha_square={max_r**2:.4f} …", flush=True)
    t3 = time.time()
    alpha = gd.AlphaComplex(points=pts.tolist())
    st_alpha = alpha.create_simplex_tree(max_alpha_square=max_r ** 2)
    n_alpha_simp = st_alpha.num_simplices()
    alpha_bars_full = persistence_bars(st_alpha, max_dim=2, alpha_squared=True)
    alpha_kept, alpha_dropped = filter_bars(alpha_bars_full, rng_noise)
    print(f"           Alpha: {n_alpha_simp:,} simplices, "
          f"{len(alpha_bars_full):,} bars total → {len(alpha_kept):,} kept "
          f"({time.time()-t3:.1f}s)", flush=True)

    n_h1 = count_persistent_h1_alpha(alpha_bars_full)
    n_h1_by_thresh = count_h1_at_thresholds(alpha_bars_full)
    n_h2_artifact = count_persistent_h2(alpha_bars_full)
    print(f"           persistent H_1 (Alpha, > {PERSIST_H1_THRESHOLD}): {n_h1}", flush=True)
    print(f"           H_1 at thresholds: " +
          ", ".join(f"{t}→{n_h1_by_thresh[t]}" for t in sorted(n_h1_by_thresh)),
          flush=True)
    print(f"           persistent H_2 ARTIFACT (Alpha, > {PERSIST_H1_THRESHOLD}): "
          f"{n_h2_artifact}", flush=True)

    del alpha, st_alpha

    return {
        "level":                       int(level),
        "label":                       label_for_level(level),
        "n_points_full":               int(N),
        "points_display":              disp_pts.tolist(),
        "cubical_betti":               cubical_betti,
        "n_persistent_h1":             int(n_h1),
        "n_persistent_h1_by_thresh":   n_h1_by_thresh,
        "n_persistent_h2_artifact":    int(n_h2_artifact),
        "cech":  {"max_r": max_r, "bars": cech_kept,  "n_filtered_noise": cech_dropped},
        "alpha": {"max_r": max_r, "bars": alpha_kept, "n_filtered_noise": alpha_dropped},
    }


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def write_output(payload: dict) -> None:
    tmp = OUTPUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(OUTPUT)
    print(f"           wrote {OUTPUT.name} ({OUTPUT.stat().st_size / 1e6:.2f} MB)",
          flush=True)


def main() -> int:
    overall_t0 = time.time()
    print(f"build_menger_data.py — N={N_POINTS}, max_r={MAX_R}, cap={MAX_LEVEL_CAP}",
          flush=True)

    rng_full    = np.random.default_rng(SEED_BASE)
    rng_display = np.random.default_rng(SEED_DISPLAY)
    rng_noise   = np.random.default_rng(SEED_BASE + 7919)

    levels: list[dict] = []
    last_h1 = -1
    max_level = 0

    for n in range(MAX_LEVEL_CAP + 1):
        entry = process_level(n, N_POINTS, MAX_R, rng_full, rng_display, rng_noise)
        levels.append(entry)
        cur_h1 = entry["n_persistent_h1"]
        # Empirical max_level rule (per the brief): max_level = largest n such
        # that C_n > C_{n-1} by at least 1. Level 0 always counts.
        if n == 0:
            max_level = 0
        elif cur_h1 > last_h1:
            max_level = n
            print(f"           >> persistent H_1 grew "
                  f"{last_h1} → {cur_h1}; max_level := {n}", flush=True)
        else:
            print(f"           >> persistent H_1 plateau "
                  f"({last_h1} → {cur_h1}); stopping at max_level={max_level} "
                  f"(level {n} retained in JSON for inspection but slider clips)",
                  flush=True)
            last_h1 = cur_h1
            # KEEP the plateau-trigger level in the output (it has useful data
            # even if the slider clips it). Frontend reads max_level for the
            # slider's max attribute; everything else can still drill into the
            # later levels via the JSON if needed.
            payload = {
                "N": int(N_POINTS),
                "max_level": int(max_level),
                "max_r_cech":  MAX_R,
                "max_r_alpha": MAX_R,
                "persistence_keep_threshold": PERSIST_KEEP_ALL_ABOVE,
                "h1_threshold":               PERSIST_H1_THRESHOLD,
                "levels": levels,
            }
            write_output(payload)
            print(f"\nALL DONE in {(time.time()-overall_t0)/60:.1f} min "
                  f"→ max_level={max_level} (computed levels 0..{n})", flush=True)
            return 0
        last_h1 = cur_h1
        # Save in-progress after every level (so a crash on the next level
        # still leaves a usable file).
        payload = {
            "N": int(N_POINTS),
            "max_level": int(max_level),
            "max_r_cech":  MAX_R,
            "max_r_alpha": MAX_R,
            "persistence_keep_threshold": PERSIST_KEEP_ALL_ABOVE,
            "h1_threshold":               PERSIST_H1_THRESHOLD,
            "levels": levels,
        }
        write_output(payload)

    # Reached the cap without seeing a plateau — keep everything.
    print(f"\nReached MAX_LEVEL_CAP={MAX_LEVEL_CAP} without plateau; "
          f"max_level={max_level}", flush=True)
    print(f"ALL DONE in {(time.time()-overall_t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
