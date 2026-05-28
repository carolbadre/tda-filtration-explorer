"""
build_packing_data.py — offline TDA precompute for a random sphere packing.

Companion section to the Menger / Sierpinski / HK97 chainmail blocks, but
the headline character here is H_2: every interstitial void in a dense
random sphere packing is an essential or high-persistence H_2 class in
the Alpha complex of the sphere centres.

What this script does
=====================
1. Generates a 3D random packing of N equal spheres via a
   Lubachevsky-Stillinger-style growth algorithm (start with small spheres,
   grow them iteratively while pushing apart overlapping pairs). Runs until
   a target volume fraction or a max iteration budget.
2. Runs the GUDHI Alpha complex on the sphere centres up to a generous
   max_alpha_square, with max_dim=3 so 3-simplices kill the noise-floor
   H_2 bars cleanly.
3. Saves the points + persistence + run config to packing_data.json,
   structured the same way as hk97_data.json so the frontend can render
   the standard PD + barcode panels.

JSON shape:
  {
    "source": "lubachevsky-stillinger N=... vf=...",
    "n_spheres": int,
    "box_L": float,
    "sphere_r": float,
    "volume_fraction": float,
    "points": [[x,y,z], ...],
    "alpha": {"max_r": float, "persistence": [...]},
    "compute_log": [...],
    "config": {...}
  }

Bars are {"dim", "birth", "death", "birth_simplex"?} with death=None for
essentials. dim=3 essentials are dropped (artifact of the dim cap).

Usage:
  python build_packing_data.py

Optional env overrides (mostly for smoke-testing):
  PACK_N             — sphere count (default 3000)
  PACK_BOX_L         — box edge length (default 10.0)
  PACK_TARGET_VF     — stop when volume fraction reaches this (default 0.50)
  PACK_MAX_ITERS     — hard ceiling on growth iterations (default 600)
  PACK_MAX_R         — Alpha max_alpha (default = 0.6 × box_L for big voids)
  PACK_SEED          — RNG seed (default 20260517)
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
from scipy.spatial import cKDTree


HERE   = Path(__file__).parent
OUTPUT = HERE / "packing_data.json"

N_SPHERES   = int  (os.environ.get("PACK_N", 3000))
BOX_L       = float(os.environ.get("PACK_BOX_L", 10.0))
TARGET_VF   = float(os.environ.get("PACK_TARGET_VF", 0.50))
MAX_ITERS   = int  (os.environ.get("PACK_MAX_ITERS", 600))
ALPHA_MAX_R = float(os.environ.get("PACK_MAX_R", 0.6 * 10.0))   # 6 by default
SEED        = int  (os.environ.get("PACK_SEED", 20260517))

# Filter thresholds for the saved persistence: keep all bars with persistence
# > KEEP_THRESH; uniformly subsample sub-threshold bars per dim to NOISE_CAP
# so the JSON stays compact but the noise floor is still visible.
PERSIST_KEEP_THRESH      = 0.02
NOISE_SAMPLE_CAP_PER_DIM = 800


# ----------------------------------------------------------------------------
# Lubachevsky-Stillinger sphere packing
# ----------------------------------------------------------------------------
def grow_packing(N: int, L: float, target_vf: float, max_iters: int,
                 seed: int) -> tuple[np.ndarray, float, float]:
    """Place N small spheres uniformly at random; iteratively grow radius +
    resolve overlaps by pushing pairs apart along their separation vector.

    Stops at target_vf or max_iters, whichever first. Walls are reflecting
    (a sphere whose centre would leave the box gets clipped back in).

    Returns (centres, final radius, achieved volume fraction).
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, L, (N, 3))
    # Initial r small enough to fit even on lattice
    r = (L / N**(1/3)) * 0.10
    print(f"   start r={r:.4f}, target vf={target_vf}")

    growth_step = r * 0.04   # 4% per iteration — quick climb
    push_passes = 8          # overlap-resolution sub-passes per growth step

    for it in range(max_iters):
        r_new = r + growth_step
        # Resolve all overlapping pairs.
        for _ in range(push_passes):
            tree = cKDTree(pts)
            pairs = tree.query_pairs(r=2.0 * r_new, output_type='ndarray')
            if len(pairs) == 0:
                break
            # Vectorised push: for each pair, push half the overlap apart.
            i = pairs[:, 0]
            j = pairs[:, 1]
            diff = pts[i] - pts[j]
            dn = np.linalg.norm(diff, axis=1)
            # Tiny perturbation when two spheres are exactly coincident.
            zero = dn < 1e-12
            if zero.any():
                diff[zero] = rng.normal(0, 1e-4, (zero.sum(), 3))
                dn[zero] = np.linalg.norm(diff[zero], axis=1)
            overlap = 2.0 * r_new - dn
            shift = (diff / dn[:, None]) * (overlap[:, None] * 0.5)
            # Multiple pairs may touch the same sphere — use np.add.at for
            # atomic accumulation (vs in-place += which would drop updates).
            np.add.at(pts,  i,  shift)
            np.add.at(pts,  j, -shift)
        # Clip to box (reflecting walls — straightforward clamp keeps centres
        # at least r_new from each face).
        pts = np.clip(pts, r_new, L - r_new)
        r = r_new
        vf = N * (4.0 / 3.0 * math.pi * r**3) / (L**3)
        if (it + 1) % 25 == 0:
            print(f"   iter {it+1:4d}: r={r:.4f}, vf={vf:.4f}, "
                  f"overlapping pairs={len(pairs):,}")
        if vf >= target_vf:
            print(f"   reached target vf {target_vf} at iter {it+1}, r={r:.4f}")
            return pts, r, vf
    print(f"   max iters reached: r={r:.4f}, vf={vf:.4f}")
    return pts, r, vf


# ----------------------------------------------------------------------------
# GUDHI persistence on the sphere centres
# ----------------------------------------------------------------------------
def run_alpha(pts: np.ndarray, max_r: float, max_dim: int = 3):
    """Alpha complex with max_dim=3 to kill H_2 bars cleanly (3-simplices
    are the natural co-faces for H_2 in 3-D). Returns the list of bars
    in the standard {dim, birth, death, birth_simplex} format AND flat
    edge + triangle filtration arrays (sorted by filtration) for the
    live-α viz on the frontend.
    """
    print(f"   alpha complex on {len(pts)} points, max_r={max_r}, max_dim={max_dim}")
    t0 = time.time()
    ac = gd.AlphaComplex(points=pts.tolist())
    st = ac.create_simplex_tree(max_alpha_square=max_r ** 2)
    print(f"     simplex tree: {st.num_simplices():,} simplices ({time.time()-t0:.1f}s)")

    # Extract edges + triangles for the browser viz, sorted by birth filtration
    # (sqrt of GUDHI's internal α²). Frontend BufferGeometry.drawRange clips
    # to the slider's α. EDGE_CAP / TRI_CAP keep the JSON compact.
    t_ext = time.time()
    EDGE_CAP, TRI_CAP = 120_000, 200_000
    edges, tris = [], []
    for simplex, f in st.get_filtration():
        dim = len(simplex) - 1
        if dim == 1:
            edges.append((int(simplex[0]), int(simplex[1]),
                          math.sqrt(max(float(f), 0.0))))
        elif dim == 2:
            tris.append((int(simplex[0]), int(simplex[1]), int(simplex[2]),
                         math.sqrt(max(float(f), 0.0))))
    edges.sort(key=lambda e: e[2])
    tris.sort(key=lambda x: x[3])
    if len(edges) > EDGE_CAP:
        print(f"     capping {len(edges):,} edges to first {EDGE_CAP:,} by birth")
        edges = edges[:EDGE_CAP]
    if len(tris) > TRI_CAP:
        print(f"     capping {len(tris):,} triangles to first {TRI_CAP:,} by birth")
        tris = tris[:TRI_CAP]
    edges_flat = []
    for i, j, b in edges:
        edges_flat.extend([i, j, round(b, 5)])
    tris_flat = []
    for i, j, k, b in tris:
        tris_flat.extend([i, j, k, round(b, 5)])
    print(f"     extracted {len(edges):,} edges + {len(tris):,} triangles "
          f"({time.time()-t_ext:.1f}s)")

    t1 = time.time()
    st.compute_persistence(homology_coeff_field=11)
    raw_pairs = st.persistence_pairs()
    print(f"     persistence: {len(raw_pairs):,} pairs ({time.time()-t1:.1f}s)")

    out = []
    for birth_simplex, death_simplex in raw_pairs:
        dim = len(birth_simplex) - 1
        if dim > max_dim - 1:    # max_dim=3 → drop dim≥3 (boundary artifact)
            continue
        is_essential = (len(death_simplex) == 0)
        if dim == max_dim - 1 and is_essential:
            # dim==2 essentials are spurious if no 3-simplex caps them within
            # max_alpha; drop. dim==1 essentials are legit (rings still alive).
            continue
        b = math.sqrt(max(float(st.filtration(birth_simplex)), 0.0))
        de = None if is_essential \
             else math.sqrt(max(float(st.filtration(death_simplex)), 0.0))
        entry = {"dim": dim, "birth": b, "death": de}
        if dim >= 1:
            entry["birth_simplex"] = [int(v) for v in birth_simplex]
        out.append(entry)
    return out, edges_flat, tris_flat, time.time() - t0


def filter_persistence(bars: list[dict], rng: np.random.Generator):
    """Keep all bars with persistence > PERSIST_KEEP_THRESH; uniformly
    subsample sub-threshold bars per dim to NOISE_SAMPLE_CAP_PER_DIM.
    Essentials (death=None) always kept.
    """
    by_dim: dict[int, list[dict]] = {}
    for b in bars:
        by_dim.setdefault(b["dim"], []).append(b)

    kept = []
    dropped: dict[int, int] = {}
    for d, group in by_dim.items():
        keep_idx, noise_idx = [], []
        for i, bar in enumerate(group):
            if bar["death"] is None:
                keep_idx.append(i); continue
            if (bar["death"] - bar["birth"]) > PERSIST_KEEP_THRESH:
                keep_idx.append(i)
            else:
                noise_idx.append(i)
        if len(noise_idx) > NOISE_SAMPLE_CAP_PER_DIM:
            sampled = rng.choice(noise_idx, NOISE_SAMPLE_CAP_PER_DIM, replace=False)
            kept_noise = set(int(x) for x in sampled)
            dropped[d] = len(noise_idx) - NOISE_SAMPLE_CAP_PER_DIM
        else:
            kept_noise = set(noise_idx)
            dropped[d] = 0
        for i in keep_idx:
            kept.append(group[i])
        for i in kept_noise:
            kept.append(group[i])
    return kept, {str(k): int(v) for k, v in dropped.items()}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    overall_t0 = time.time()
    print(f"build_packing_data.py — N={N_SPHERES}, box L={BOX_L}, "
          f"target vf={TARGET_VF}, alpha max_r={ALPHA_MAX_R}")

    # Step 1: pack
    print("\n[1/3] sphere packing (Lubachevsky-Stillinger growth)")
    t1 = time.time()
    pts, r_final, vf_final = grow_packing(N_SPHERES, BOX_L, TARGET_VF,
                                          MAX_ITERS, SEED)
    print(f"   done in {time.time()-t1:.1f}s. "
          f"final r={r_final:.4f}, vf={vf_final:.4f}")

    # Step 2: Alpha persistence
    print("\n[2/3] Alpha persistence on sphere centres")
    bars, edges_flat, tris_flat, alpha_elapsed = run_alpha(pts, ALPHA_MAX_R)
    rng = np.random.default_rng(SEED + 7919)
    bars_kept, dropped = filter_persistence(bars, rng)
    print(f"   kept {len(bars_kept):,} of {len(bars):,} bars after "
          f"persistence-threshold + noise-subsample")
    print(f"   noise-bars dropped per dim: {dropped}")

    # Count by dim for the log
    by_dim_count = {}
    for b in bars_kept:
        by_dim_count[b["dim"]] = by_dim_count.get(b["dim"], 0) + 1
    print(f"   bars by dim (kept): {by_dim_count}")

    # Persistent H_2 (the headline character)
    h2_finite = [b for b in bars_kept
                 if b["dim"] == 2 and b["death"] is not None]
    h2_pers = sorted([b["death"] - b["birth"] for b in h2_finite], reverse=True)
    print(f"   top 10 H_2 persistences: {[round(p, 4) for p in h2_pers[:10]]}")
    print(f"   H_2 bars by threshold:  "
          f">0.05: {sum(1 for p in h2_pers if p > 0.05)}, "
          f">0.10: {sum(1 for p in h2_pers if p > 0.10)}, "
          f">0.20: {sum(1 for p in h2_pers if p > 0.20)}")

    # Step 3: write
    print("\n[3/3] writing packing_data.json")
    payload = {
        "source":             f"lubachevsky-stillinger N={N_SPHERES} vf={vf_final:.3f}",
        "n_spheres":          int(N_SPHERES),
        "box_L":              float(BOX_L),
        "sphere_r":           float(r_final),
        "volume_fraction":    float(vf_final),
        "points":             pts.tolist(),
        "alpha": {
            "max_r":            ALPHA_MAX_R,
            "persistence":      bars_kept,
            "n_filtered_noise": dropped,
            # Flat edge + triangle arrays sorted by birth filtration, for the
            # frontend live-α complex viz. edges_flat = [i, j, birth, …];
            # tris_flat = [i, j, k, birth, …]. The browser uses these with
            # BufferGeometry.drawRange to slide through α.
            "edges_flat":       edges_flat,
            "tris_flat":        tris_flat,
            "n_edges":          len(edges_flat) // 3,
            "n_tris":           len(tris_flat) // 4,
        },
        "compute_log": [
            {"phase": "pack",  "elapsed_s": round(time.time() - t1 - alpha_elapsed, 1)},
            {"phase": "alpha", "elapsed_s": round(alpha_elapsed, 1)},
        ],
        "config": {
            "N":          N_SPHERES,
            "box_L":      BOX_L,
            "target_vf":  TARGET_VF,
            "max_iters":  MAX_ITERS,
            "alpha_max_r": ALPHA_MAX_R,
            "seed":       SEED,
            "persist_keep_thresh": PERSIST_KEEP_THRESH,
            "noise_sample_cap":    NOISE_SAMPLE_CAP_PER_DIM,
        },
    }
    tmp = OUTPUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(OUTPUT)
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"   wrote {OUTPUT.name} ({size_mb:.2f} MB)")

    print(f"\nALL DONE in {(time.time() - overall_t0):.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
