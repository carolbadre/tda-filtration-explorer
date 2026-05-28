"""
build_sierpinski_data.py — offline precompute for the Sierpinski-triangle
TDA section.  Companion to build_menger_data.py; see that file's docstring
for the architectural pattern.

Sierpinski-specific deltas vs Menger
====================================
  - 2D ambient (points in R²; output JSON has 2-vectors, not 3-vectors)
  - 3-map IFS (corner contractions at scale 1/2), not the 20-map Menger one
  - max_dim = 1 (no H_2 in 2D — Delaunay/Čech 2-simplices kill H_1, and
    there are no 3-simplices to give H_2 bars).  Persistence_bars keeps
    H_1 essentials, which is the right thing in 2D where H_1 is the
    top-measured dimension (in contrast to Menger's H_2-essential drop,
    which discards bars right-censored by the max_alpha_square cap).
  - β computed from the closed form
        b_0(T_n) = 1,  b_1(T_n) = (3^n − 1)/2,  b_2(T_n) = 0
    — no cubical-complex computation.
  - max_level chosen empirically by comparing the observed Alpha-H_1
    count C_n to the closed-form expectations E_n, E_{n-1}: the largest
    n with |C_n − E_n| < |C_n − E_{n-1}| is the deepest level that
    resolves cleanly at this sample density.

JSON shape (cf. menger_data.json):
{
  "N": 200000, "max_level": <int>, "ambient_dim": 2,
  "max_r_cech": 0.4, "max_r_alpha": 0.4,
  "persistence_keep_threshold": 0.005, "h1_threshold": 0.01,
  "levels": [
    {
      "level": int, "label": "T_n (…)",
      "n_points_full": 200000,
      "points_display": [[x, y], ...],         # 8000-point IFS subsample
      "exact_betti": [1, (3^n-1)//2, 0],
      "n_persistent_h1": int,
      "cech":  {"max_r": 0.4, "bars": [...], "n_filtered_noise": {...}},
      "alpha": {"max_r": 0.4, "bars": [...], "n_filtered_noise": {...}},
    },
    ...
  ]
}

Bars are {"dim", "birth", "death"} with death=None for essential classes.
No dim=2 bars anywhere (max_dim filter).

Usage
=====
  cd v4/
  python build_sierpinski_data.py

Optional environment overrides:
  SIERPINSKI_N             — points per level (default 200000)
  SIERPINSKI_MAX_R         — geometric radius cap for Čech + Alpha (default 0.4)
  SIERPINSKI_MAX_LEVEL_CAP — hard ceiling on the candidate levels probed
                              (default 11; the empirical stop normally fires
                              around 9 at N=200k)
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


HERE   = Path(__file__).parent
OUTPUT = HERE / "sierpinski_data.json"

N_POINTS      = int(os.environ.get("SIERPINSKI_N", 200_000))
MAX_R         = float(os.environ.get("SIERPINSKI_MAX_R", 0.4))
MAX_LEVEL_CAP = int(os.environ.get("SIERPINSKI_MAX_LEVEL_CAP", 11))

PERSIST_KEEP_ALL_ABOVE   = 0.005
PERSIST_H1_THRESHOLD     = 0.01
NOISE_SAMPLE_CAP_PER_DIM = 600

DISPLAY_SUBSAMPLE = 8000

SEED_BASE    = 20260516
SEED_DISPLAY = SEED_BASE + 1
SEED_NOISE   = SEED_BASE + 7919


# ----------------------------------------------------------------------------
# IFS sampler — uniform on T_n via the chaos game on the 3 corner maps.
# ----------------------------------------------------------------------------
TRIANGLE_VERTICES = np.array([
    [0.0,             0.0           ],
    [1.0,             0.0           ],
    [0.5, math.sqrt(3) / 2          ],
], dtype=float)


def _uniform_in_triangle(N: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform sample in the filled triangle T_0 via the standard barycentric
    trick: with r1, r2 ~ U(0,1), the point
        (1 - sqrt(r1)) v0 + sqrt(r1)(1 - r2) v1 + sqrt(r1) r2 v2
    is uniformly distributed in the triangle (v0, v1, v2).
    """
    r1 = rng.uniform(0.0, 1.0, N)
    r2 = rng.uniform(0.0, 1.0, N)
    s1 = np.sqrt(r1)
    return ((1.0 - s1)[:, None]      * TRIANGLE_VERTICES[0]
            + (s1 * (1.0 - r2))[:, None] * TRIANGLE_VERTICES[1]
            + (s1 * r2)[:, None]      * TRIANGLE_VERTICES[2])


def sample_sierpinski_ifs(level: int, N: int, rng: np.random.Generator) -> np.ndarray:
    """N uniform points on T_n via the chaos game.

    At each level, every retained sub-triangle has equal area and is the
    image of T_0 under a length-`level` random address ∈ {0,1,2}^level of
    corner contractions x → (x + v_i)/2.  Picking the address uniformly
    therefore picks the sub-triangle uniformly, and applying the address
    to a uniform-in-T_0 starting point gives a uniform-in-(sub-triangle)
    point — i.e. uniform on T_n.
    """
    pts = _uniform_in_triangle(N, rng)
    if level == 0:
        return pts
    addr = rng.integers(0, 3, size=(N, level))
    for k in range(level):
        v = TRIANGLE_VERTICES[addr[:, k]]
        pts = (pts + v) / 2.0
    return pts


# ----------------------------------------------------------------------------
# Closed-form Betti numbers for T_n.
# ----------------------------------------------------------------------------
def exact_betti_for_level(n: int) -> list[int]:
    """[b_0, b_1, b_2] for the Sierpinski-triangle iterate T_n."""
    return [1, (3 ** n - 1) // 2, 0]


# ----------------------------------------------------------------------------
# Persistence: read bars out of a built simplex tree.
# ----------------------------------------------------------------------------
def persistence_bars(st: gd.SimplexTree, max_dim: int, *, alpha_squared: bool):
    """Return list of {"dim", "birth", "death"} (death=None for essential).

    Differs from the Menger version in one key place: we do NOT drop
    dim==max_dim essentials.  In 2D ambient, the alpha/Čech complex DOES
    include 2-simplices (Delaunay triangles), so H_1 bars die naturally
    when their filling triangle arrives — any H_1 essential here is a
    genuine "hole still open at max_r", which is information we want to
    keep (and which the persistence-bar plot can show as a vertical death
    line at the top of the panel).

    `alpha_squared=True` sqrts the filtration values (GUDHI's Alpha
    complex stores α²; we want α radii in the output JSON).
    """
    st.compute_persistence(homology_coeff_field=11)
    pairs = st.persistence_pairs()
    out = []
    for birth_simplex, death_simplex in pairs:
        dim = len(birth_simplex) - 1
        if dim > max_dim:
            continue
        is_essential = (len(death_simplex) == 0)
        b = float(st.filtration(birth_simplex))
        d = None if is_essential else float(st.filtration(death_simplex))
        if alpha_squared:
            b = math.sqrt(max(b, 0.0))
            if d is not None:
                d = math.sqrt(max(d, 0.0))
        out.append({"dim": dim, "birth": b, "death": d})
    return out


def filter_bars(bars: list[dict], rng: np.random.Generator):
    """Filter / subsample noise bars (same policy as the Menger build)."""
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
    return kept, {str(k): int(v) for k, v in dropped_counts.items()}


def count_persistent_h1(alpha_bars: list[dict]) -> int:
    """Alpha-PD H_1 bars with persistence > PERSIST_H1_THRESHOLD.  Essentials
    count (infinite persistence)."""
    c = 0
    for b in alpha_bars:
        if b["dim"] != 1:
            continue
        if b["death"] is None:
            c += 1
            continue
        if (b["death"] - b["birth"]) > PERSIST_H1_THRESHOLD:
            c += 1
    return c


# ----------------------------------------------------------------------------
# Per-level processing.
# ----------------------------------------------------------------------------
def label_for_level(n: int) -> str:
    sub = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
    s = "".join(sub[int(c)] for c in str(n))
    if n == 0:
        suffix = " (filled triangle)"
    elif n == 1:
        suffix = " (triangle minus centre)"
    else:
        suffix = ""
    return f"T{s}{suffix}"


def process_level(level: int, N: int, max_r: float,
                  rng_full: np.random.Generator,
                  rng_display: np.random.Generator,
                  rng_noise: np.random.Generator) -> dict:
    print(f"\n[level {level}] sampling N={N} via IFS …", flush=True)
    t0 = time.time()
    pts = sample_sierpinski_ifs(level, N, rng_full)
    print(f"           done in {time.time()-t0:.1f}s   "
          f"x∈[{pts[:,0].min():.4f}, {pts[:,0].max():.4f}]   "
          f"y∈[{pts[:,1].min():.4f}, {pts[:,1].max():.4f}]", flush=True)

    disp_pts = sample_sierpinski_ifs(level, DISPLAY_SUBSAMPLE, rng_display)

    betti = exact_betti_for_level(level)
    print(f"           closed-form β = {betti}", flush=True)

    # Sanity-check the formula against the brief's stated values.
    small_expected = {
        0: [1, 0, 0],
        1: [1, 1, 0],
        2: [1, 4, 0],
        3: [1, 13, 0],
    }
    if level in small_expected and betti != small_expected[level]:
        raise RuntimeError(
            f"exact_betti_for_level({level}) = {betti} ≠ expected "
            f"{small_expected[level]} — closed-form Betti has a bug."
        )

    # Delaunay-Čech.
    print(f"           Delaunay-Čech, max_r={max_r}, "
          f"max_alpha_square={max_r**2:.4f} …", flush=True)
    t2 = time.time()
    dc = gd.DelaunayCechComplex(points=pts.tolist())
    st_cech = dc.create_simplex_tree(
        max_alpha_square=max_r ** 2,
        output_squared_values=False,
    )
    n_cech_simp = st_cech.num_simplices()
    cech_bars_full = persistence_bars(st_cech, max_dim=1, alpha_squared=False)
    cech_kept, cech_dropped = filter_bars(cech_bars_full, rng_noise)
    print(f"           Čech: {n_cech_simp:,} simplices, "
          f"{len(cech_bars_full):,} bars total → {len(cech_kept):,} kept "
          f"({time.time()-t2:.1f}s)", flush=True)
    del dc, st_cech

    # Alpha.
    print(f"           Alpha,        max_r={max_r}, "
          f"max_alpha_square={max_r**2:.4f} …", flush=True)
    t3 = time.time()
    alpha = gd.AlphaComplex(points=pts.tolist())
    st_alpha = alpha.create_simplex_tree(max_alpha_square=max_r ** 2)
    n_alpha_simp = st_alpha.num_simplices()
    alpha_bars_full = persistence_bars(st_alpha, max_dim=1, alpha_squared=True)
    alpha_kept, alpha_dropped = filter_bars(alpha_bars_full, rng_noise)
    print(f"           Alpha: {n_alpha_simp:,} simplices, "
          f"{len(alpha_bars_full):,} bars total → {len(alpha_kept):,} kept "
          f"({time.time()-t3:.1f}s)", flush=True)

    n_h1 = count_persistent_h1(alpha_bars_full)
    print(f"           persistent H_1 (Alpha, > {PERSIST_H1_THRESHOLD}): {n_h1}   "
          f"(expected exact b_1 = {betti[1]})", flush=True)

    del alpha, st_alpha

    return {
        "level":            int(level),
        "label":            label_for_level(level),
        "n_points_full":    int(N),
        "points_display":   disp_pts.tolist(),
        "exact_betti":      betti,
        "n_persistent_h1":  int(n_h1),
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
    print(f"build_sierpinski_data.py — N={N_POINTS}, max_r={MAX_R}, "
          f"cap={MAX_LEVEL_CAP}", flush=True)

    rng_full    = np.random.default_rng(SEED_BASE)
    rng_display = np.random.default_rng(SEED_DISPLAY)
    rng_noise   = np.random.default_rng(SEED_NOISE)

    levels: list[dict] = []
    max_level = 0

    for n in range(MAX_LEVEL_CAP + 1):
        entry = process_level(n, N_POINTS, MAX_R, rng_full, rng_display, rng_noise)
        levels.append(entry)

        # Empirical max_level rule (per the brief): max_level = largest n
        # with |C_n − E_n| < |C_n − E_{n-1}|, where E_n = (3^n - 1)/2 is
        # the closed-form expectation and C_n is the observed persistent
        # H_1 count.  Level 0 always counts (E_0 = 0, C_0 ≈ 0).
        C = entry["n_persistent_h1"]
        E_n     = (3 ** n - 1) // 2
        E_prev  = (3 ** (n - 1) - 1) // 2 if n >= 1 else -1
        if n == 0:
            max_level = 0
        elif abs(C - E_n) < abs(C - E_prev):
            max_level = n
            print(f"           >> C_{n}={C}, E_{n}={E_n}, E_{n-1}={E_prev} — "
                  f"resolved; max_level := {n}", flush=True)
        else:
            print(f"           >> C_{n}={C} is closer to E_{n-1}={E_prev} than "
                  f"E_{n}={E_n} — stopping; max_level={max_level}", flush=True)
            payload = {
                "N":             int(N_POINTS),
                "max_level":     int(max_level),
                "ambient_dim":   2,
                "max_r_cech":    MAX_R,
                "max_r_alpha":   MAX_R,
                "persistence_keep_threshold": PERSIST_KEEP_ALL_ABOVE,
                "h1_threshold":               PERSIST_H1_THRESHOLD,
                "levels": [L for L in levels if L["level"] <= max_level],
            }
            write_output(payload)
            print(f"\nALL DONE in {(time.time()-overall_t0)/60:.1f} min "
                  f"→ max_level={max_level}", flush=True)
            return 0

        # Save in-progress after every level.
        payload = {
            "N":             int(N_POINTS),
            "max_level":     int(max_level),
            "ambient_dim":   2,
            "max_r_cech":    MAX_R,
            "max_r_alpha":   MAX_R,
            "persistence_keep_threshold": PERSIST_KEEP_ALL_ABOVE,
            "h1_threshold":               PERSIST_H1_THRESHOLD,
            "levels": levels,
        }
        write_output(payload)

    print(f"\nReached MAX_LEVEL_CAP={MAX_LEVEL_CAP} without plateau; "
          f"max_level={max_level}", flush=True)
    print(f"ALL DONE in {(time.time()-overall_t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
