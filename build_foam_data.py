"""
build_foam_data.py — offline TDA precompute for the foam (Weaire-Phelan)
                     persistent-homology section.

What this script does
=====================
1. Constructs the Weaire-Phelan cell-centre lattice (A15 / Pm-3n positions)
   tiled to fill an (nx_outer × ny_outer × nz_outer) cube. The OUTER ring of
   unit cubes exists only to give the inner (nx × ny × nz) cubes proper
   bounded Voronoi neighbourhoods; only inner cells are sampled.
2. Computes the 3D Voronoi tessellation of all centres (scipy.spatial.Voronoi
   → Qhull). The Voronoi cells are flat-faced polyhedra; this is the
   piecewise-flat homotopy equivalent of true minimal-surface Weaire-Phelan
   foam, and is what every TDA experiment on foams I've seen uses.
3. Collects every Voronoi face (ridge) incident to at least one inner cell,
   deduplicates, triangulates each face by fan, and samples N points
   uniformly proportional to triangle area. These face points are the
   bubble-surface membrane samples — NOT bubble centres.
4. Runs GUDHI Alpha on the face point cloud with max_dim=2, max_alpha_square
   large enough to see the bubble voids close (≈ half the bubble
   characteristic length).
5. Writes foam_data.json with the schema specified in the brief.

Why face samples (not centres)
==============================
Bubble centres give a sparse point cloud with NO H_2 — their Alpha complex
is the Delaunay tessellation itself, which is contractible. To see each
bubble's enclosed VOID as a persistent H_2 class you need to sample its
bounding SURFACE densely enough that the surface closes (under Alpha) below
the radius at which the void fills in. The face point cloud does this:
each inner bubble is enclosed by a triangulated polyhedral surface, and the
void inside it is detected as a persistent H_2 class.

Expected ground truth
=====================
β_2 ≈ n_inner_bubbles. β_1 is the count of independent 1-cycles in the
surface complex (windows/edges between adjacent face triangulations) —
constrained by Plateau's laws but harder to predict without simulation.

Honesty caveats
===============
- Recovering β_2 ≈ n_bubbles proves "1000 enclosed voids", NOT "this is
  Weaire-Phelan foam" or "Plateau's laws hold". A scrambled collection of
  1000 random closed polyhedra would give the same H_2 count.
- Plateau geometry (F=4 per vertex, E=120° dihedrals) is NOT measured here.
- Alpha uses flat-face Voronoi cells; true soap-film minimal surfaces are
  slightly curved. Topologically identical, geometrically different.

Output (foam_data.json):
{
  "source":         "weaire-phelan 5x5x5",
  "n_bubbles":      1000,
  "n_points":       int,
  "points":         [[x,y,z], ...],
  "alpha":          {"max_r": float, "persistence": [...]},
  "foam_geometry":  {                  # so the frontend can draw the cells
    "vertices":     [[x,y,z], ...],    # Voronoi vertex pool (shared)
    "inner_faces":  [{"v": [idx,...], "k": "AD"|"AA"|"DD"}, ...],
    "inner_cells":  [{"c": [x,y,z], "k": "A"|"D"}, ...],
  },
  "alpha_subsample": {                 # small enough to render live
    "n":            int (~500),
    "points":       [[x,y,z], ...],
    "max_r":        float,
    "simplices":    [{"d": dim, "v": [...], "f": birth_radius}, ...],
    "persistence":  [...],             # same shape as alpha.persistence
  },
  "compute_log":    [{"phase": "...", "elapsed_s": ...}, ...],
  "config":         {...},
}
Persistence entries: {"dim", "birth", "death", "birth_simplex"} with
death=null for essential classes. Drops dim > MAX_DIM (= 2).

Usage
=====
  cd v4/
  python build_foam_data.py

Optional env overrides:
  FOAM_NX_INNER     inner-cube count per axis (default 5  → 5^3 × 8 = 1000)
  FOAM_PAD_LAYERS   padding cubes on each side  (default 1)
  FOAM_N_POINTS     points sampled on face membranes (default 30000)
  FOAM_MAX_R        alpha radius cap (default 0.4 — unit cell side L=1)
  FOAM_PERTURB      tiny perturbation on centres to break Qhull degeneracies
                    (default 1e-6 — A15 is highly symmetric)
  FOAM_SUBSAMPLE_N  point count for the live α-complex subsample
                    (default 600 — keeps the simplex tree under ~15k entries
                    for tractable browser rendering)
  FOAM_SUBSAMPLE_K  how many CENTRAL bubbles the subsample focuses on
                    (default 8 — one A15 unit cube near the centroid: 2
                    dodecahedra + 6 14-hedra. We sample the FULL bounding
                    surfaces of just these 8 cells, so the live α-complex
                    closes 8 bubble voids cleanly as α grows — that's the
                    pedagogical hook the random-over-1000 sampler couldn't
                    give at small N.)
  FOAM_SUBSAMPLE_MAX_R alpha radius cap for the subsample (default 0.30 —
                    enough headroom to watch all 8 voids fill in)
  FOAM_SEED         RNG seed (default 20260517)
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
from scipy.spatial import Voronoi


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE   = Path(__file__).parent
OUTPUT = HERE / "foam_data.json"

NX_INNER         = int  (os.environ.get("FOAM_NX_INNER",         5))
PAD_LAYERS       = int  (os.environ.get("FOAM_PAD_LAYERS",       1))
N_POINTS         = int  (os.environ.get("FOAM_N_POINTS",         30_000))
MAX_R            = float(os.environ.get("FOAM_MAX_R",            0.4))
PERTURB          = float(os.environ.get("FOAM_PERTURB",          1e-6))
SUBSAMPLE_N      = int  (os.environ.get("FOAM_SUBSAMPLE_N",      600))
SUBSAMPLE_K      = int  (os.environ.get("FOAM_SUBSAMPLE_K",      8))
SUBSAMPLE_MAX_R  = float(os.environ.get("FOAM_SUBSAMPLE_MAX_R",  0.30))
SEED             = int  (os.environ.get("FOAM_SEED",             20260517))

# Perturbation-strength slider levels. Each value is the half-width of a
# uniform-displacement [-p/2, +p/2]^3 noise added to each centre, expressed
# as a fraction of L_CELL. Level 0 (with PERTURB=1e-6 baseline) is the
# crystal-perfect Weaire-Phelan; the higher levels approach Poisson-Voronoi
# random foam (~30-50% noise is the regime where soap-froth comparisons
# kick in).
PERTURB_LEVELS = [float(x) for x in
    os.environ.get("FOAM_PERTURB_LEVELS", "0.0,0.08,0.16,0.25,0.40").split(",")
]

MAX_DIM     = 2
L_CELL      = 1.0   # unit-cube side length (defines all coordinate scales)


# A15 / Pm-3n / Weaire-Phelan unit-cube site positions
# ----------------------------------------------------
# 8 cells per cube:
#   2 A-sites (irregular dodecahedra) at the BCC positions
#   6 D-sites (tetrakaidecahedra / 14-hedra) at the (1/4, 1/2, 0) crystal
#     positions on each of the three coordinate-plane faces
# Reference: Weaire & Phelan, Phil. Mag. Lett. 1994.
A15_SITES = np.array([
    [0.00, 0.00, 0.00],
    [0.50, 0.50, 0.50],
    [0.25, 0.50, 0.00],
    [0.75, 0.50, 0.00],
    [0.00, 0.25, 0.50],
    [0.00, 0.75, 0.50],
    [0.50, 0.00, 0.25],
    [0.50, 0.00, 0.75],
], dtype=float)
SITE_KIND = np.array(["A", "A", "D", "D", "D", "D", "D", "D"])  # for reporting


def tile_a15(nx: int, ny: int, nz: int, L: float = L_CELL) -> tuple[np.ndarray, np.ndarray]:
    """Tile the 8-site A15 unit cube nx×ny×nz times.

    Returns (centres, cube_ids) where:
      centres   : (nx*ny*nz*8, 3) site positions
      cube_ids  : (nx*ny*nz*8, 4) per-site (i, j, k, s) — cube indices + site
    """
    coords = []
    ids    = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shift = np.array([i * L, j * L, k * L])
                coords.append(A15_SITES * L + shift)
                for s in range(8):
                    ids.append((i, j, k, s))
    return np.vstack(coords), np.array(ids, dtype=int)


# ----------------------------------------------------------------------------
# Face sampling
# ----------------------------------------------------------------------------
def collect_inner_faces(vor: Voronoi, inner_mask: np.ndarray) -> list[np.ndarray]:
    """Return list of (k_i, 3) vertex arrays for every Voronoi ridge that
    bounds at least one inner cell, skipping unbounded ridges (vertex -1).

    Each face is included ONCE (scipy stores each ridge once, between its
    two ridge_points). De-duplication is therefore automatic via that
    storage convention.
    """
    polys = []
    rp = vor.ridge_points          # (n_ridges, 2)
    rv = vor.ridge_vertices        # list of lists
    for (a, b), verts in zip(rp, rv):
        if any(v < 0 for v in verts):
            continue
        if not (inner_mask[a] or inner_mask[b]):
            continue
        if len(verts) < 3:
            continue
        polys.append(vor.vertices[verts])
    return polys


def extract_inner_geometry(vor: Voronoi, inner_mask: np.ndarray,
                           centres: np.ndarray, cube_ids: np.ndarray) -> dict:
    """Build the foam-cell geometry block for the frontend.

    Returns {vertices, inner_faces, inner_cells}:
      vertices    — shared vertex pool from vor.vertices, but pruned to
                    only the indices actually referenced by inner faces
                    (keeps the JSON tight) and re-indexed.
      inner_faces — [{"v":[idx,...], "k":"AD"|"AA"|"DD"}] one per inner ridge
                    (k = pair of A15 site kinds — A=dodecahedron, D=14-hedron)
      inner_cells — [{"c":[x,y,z], "k":"A"|"D"}] for each of the n_inner sites

    Bounded faces only. (a, b)-shared faces appear once.
    """
    rp = vor.ridge_points
    rv = vor.ridge_vertices

    # First pass: which vertex indices are referenced?
    used = set()
    raw_faces = []  # (vert_indices, kind)
    for (a, b), verts in zip(rp, rv):
        if any(v < 0 for v in verts):
            continue
        if not (inner_mask[a] or inner_mask[b]):
            continue
        if len(verts) < 3:
            continue
        ka = SITE_KIND[cube_ids[a, 3]]
        kb = SITE_KIND[cube_ids[b, 3]]
        kind = "".join(sorted([ka, kb]))   # 'AD', 'AA', or 'DD'
        for v in verts:
            used.add(int(v))
        raw_faces.append(([int(v) for v in verts], kind))

    # Build a compact vertex pool + index remap
    old_to_new = {old: new for new, old in enumerate(sorted(used))}
    pruned_vertices = vor.vertices[sorted(used)]

    inner_faces = [
        {"v": [old_to_new[v] for v in verts], "k": kind}
        for verts, kind in raw_faces
    ]

    inner_cells = []
    for idx in np.where(inner_mask)[0]:
        site_s = int(cube_ids[idx, 3])
        inner_cells.append({
            "c": [round(float(x), 6) for x in centres[idx]],
            "k": SITE_KIND[site_s],
        })

    return {
        "vertices":    [[round(float(x), 6) for x in v] for v in pruned_vertices],
        "inner_faces": inner_faces,
        "inner_cells": inner_cells,
    }


# ----------------------------------------------------------------------------
# Small-N subsample for the live α-complex viz
# ----------------------------------------------------------------------------
def alpha_subsample(vor: Voronoi, inner_mask: np.ndarray,
                    cube_ids: np.ndarray, centres: np.ndarray,
                    k_central: int, target_n: int, max_r: float,
                    rng: np.random.Generator) -> dict:
    """Build a 'central-cluster' subsample for the live α-complex viz.

    We pick the k_central inner bubbles whose centres lie closest to the
    centroid of the inner region (with k_central = 8, that's typically one
    A15 unit cube near the middle: 2 dodecahedra + 6 14-hedra). We collect
    every Voronoi face that bounds at least one of those bubbles, triangulate
    by fan, and sample target_n points area-weighted across that face set.

    The result is a small point cloud densely sampled on the FULL bounding
    surfaces of just those k_central bubbles. Run Alpha on it: as α grows,
    the bubble surfaces close (births of H_2 classes) and then the bubble
    voids fill in (deaths). β_2 should peak at k_central inside the
    appropriate α window.

    Returns {n, k_central, points, focus_cells, max_r, simplices, persistence}.
    """
    inner_indices = np.where(inner_mask)[0]
    inner_centres = centres[inner_indices]
    centroid = inner_centres.mean(axis=0)
    dists = np.linalg.norm(inner_centres - centroid, axis=1)
    chosen_local = inner_indices[np.argsort(dists)[:k_central]]
    chosen_set = set(int(i) for i in chosen_local)

    polys = []
    for (a, b), verts in zip(vor.ridge_points, vor.ridge_vertices):
        if any(v < 0 for v in verts):
            continue
        if not (int(a) in chosen_set or int(b) in chosen_set):
            continue
        if len(verts) < 3:
            continue
        polys.append(vor.vertices[verts])

    pts, total_area, n_tris = triangulate_and_sample(polys, target_n, rng)
    n_sub = int(len(pts))

    alpha = gd.AlphaComplex(points=pts.tolist())
    st = alpha.create_simplex_tree(max_alpha_square=max_r ** 2)

    simplices = []
    for simplex, f_sq in st.get_filtration():
        dim = len(simplex) - 1
        if dim > 3:
            continue                     # tets kept for completeness (we render up to triangles)
        f = math.sqrt(max(f_sq, 0.0))
        simplices.append({
            "d": dim,
            "v": [int(v) for v in simplex],
            "f": f,
        })

    st.compute_persistence(homology_coeff_field=11)
    raw = st.persistence_pairs()
    persistence = []
    for birth_simplex, death_simplex in raw:
        dim = len(birth_simplex) - 1
        if dim > 2:
            continue
        is_essential = (len(death_simplex) == 0)
        if dim == 2 and is_essential:
            continue
        b = math.sqrt(max(float(st.filtration(birth_simplex)), 0.0))
        d = (None if is_essential
             else math.sqrt(max(float(st.filtration(death_simplex)), 0.0)))
        persistence.append({
            "dim":   dim,
            "birth": b,
            "death": d,
        })

    focus_cells = []
    for idx in chosen_local:
        site_s = int(cube_ids[idx, 3])
        focus_cells.append({
            "c": [round(float(x), 6) for x in centres[idx]],
            "k": SITE_KIND[site_s],
        })

    return {
        "n":            n_sub,
        "k_central":    int(k_central),
        "points":       [[round(float(x), 6) for x in p] for p in pts],
        "focus_cells":  focus_cells,
        "n_focus_faces": int(len(polys)),
        "max_r":        float(max_r),
        "simplices":    simplices,
        "persistence":  persistence,
    }


def triangulate_and_sample(polys: list[np.ndarray], n_total: int,
                           rng: np.random.Generator) -> np.ndarray:
    """Fan-triangulate every polygon, sample points uniformly proportional
    to triangle area until we have n_total points.

    Uses the classical (1-√r1, √r1(1-r2), √r1·r2) barycentric uniform
    sampler on each chosen triangle.
    """
    tris = []     # (T, 3, 3): T triangles, 3 vertices each, in R^3
    areas = []
    for poly in polys:
        p0 = poly[0]
        for i in range(1, len(poly) - 1):
            p1 = poly[i]
            p2 = poly[i + 1]
            tri = np.stack([p0, p1, p2])
            area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            if area > 0:
                tris.append(tri)
                areas.append(area)
    tris   = np.stack(tris)                          # (T, 3, 3)
    areas  = np.asarray(areas, dtype=float)
    weights = areas / areas.sum()

    # Pick which triangle each sample comes from (multinomial)
    idx = rng.choice(len(tris), size=n_total, replace=True, p=weights)
    chosen = tris[idx]                                # (n_total, 3, 3)

    r1 = rng.uniform(0.0, 1.0, n_total)
    r2 = rng.uniform(0.0, 1.0, n_total)
    s1 = np.sqrt(r1)
    b0 = (1.0 - s1)
    b1 = s1 * (1.0 - r2)
    b2 = s1 * r2
    pts = (b0[:, None] * chosen[:, 0]
         + b1[:, None] * chosen[:, 1]
         + b2[:, None] * chosen[:, 2])

    return pts, float(areas.sum()), int(len(tris))


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------
def alpha_persistence(points: np.ndarray, max_r: float, max_dim: int) -> list[dict]:
    """GUDHI Alpha → list of {"dim", "birth", "death", "birth_simplex"}.

    α values are reported as RADII (sqrt of GUDHI's internal α²).
    Essentials get death=None. Drops dim > max_dim. Drops dim==max_dim
    essentials (guaranteed artifact of max_dim cap), matching the
    server.py / build_hk97_data.py convention.
    """
    alpha = gd.AlphaComplex(points=points.tolist())
    st = alpha.create_simplex_tree(max_alpha_square=max_r ** 2)
    print(f"           simplex tree: {st.num_simplices():,} simplices",
          flush=True)
    st.compute_persistence(homology_coeff_field=11)
    raw = st.persistence_pairs()
    out = []
    for birth_simplex, death_simplex in raw:
        dim = len(birth_simplex) - 1
        if dim > max_dim:
            continue
        is_essential = (len(death_simplex) == 0)
        if dim == max_dim and is_essential:
            continue
        b_sq = float(st.filtration(birth_simplex))
        d_sq = None if is_essential else float(st.filtration(death_simplex))
        b = math.sqrt(max(b_sq, 0.0))
        d = None if d_sq is None else math.sqrt(max(d_sq, 0.0))
        out.append({
            "dim":           dim,
            "birth":         b,
            "death":         d,
            "birth_simplex": [int(v) for v in birth_simplex],
        })
    return out


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------
def write_output(payload: dict) -> None:
    tmp = OUTPUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(OUTPUT)
    print(f"           wrote {OUTPUT.name} "
          f"({OUTPUT.stat().st_size / 1e6:.2f} MB)", flush=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
HIGH_H2_THRESH = 0.05   # finite-H_2 persistence threshold for "bubble bar"


def process_level(perturb_pct: float, level_idx: int,
                  base_centres: np.ndarray, cube_ids: np.ndarray,
                  inner_mask: np.ndarray, nx: int,
                  do_subsample: bool, rng: np.random.Generator) -> dict:
    """Run the full pipeline at a single perturbation level.

    perturb_pct is the half-width of the uniform displacement noise added to
    every centre, expressed as a fraction of L_CELL. perturb_pct=0 falls
    back to the PERTURB symmetry-breaking baseline (1e-6) so Qhull does not
    choke on the exact A15 lattice.

    Returns a dict with foam_geometry, summary, top-H_2-persistence, plus
    (only when do_subsample=True) the full persistence + α-subsample blocks.
    """
    label = f"[level {level_idx}] perturb={perturb_pct*100:.1f}%"
    print(f"\n{label}", flush=True)
    t_lvl = time.time()

    # Effective perturbation: max(perturb_pct * L, 1e-6) to keep Qhull happy.
    eff_perturb = max(perturb_pct * L_CELL, 1e-6)
    centres = base_centres + rng.uniform(-eff_perturb, +eff_perturb,
                                         base_centres.shape)

    vor = Voronoi(centres)
    print(f"  Voronoi: verts={len(vor.vertices)}, "
          f"ridges={len(vor.ridge_vertices)} ({time.time() - t_lvl:.1f}s)",
          flush=True)

    polys = collect_inner_faces(vor, inner_mask)
    foam_geom = extract_inner_geometry(vor, inner_mask, centres, cube_ids)
    kind_counts: dict[str, int] = {}
    for f in foam_geom["inner_faces"]:
        kind_counts[f["k"]] = kind_counts.get(f["k"], 0) + 1
    a_cells = sum(1 for c in foam_geom["inner_cells"] if c["k"] == "A")
    d_cells = sum(1 for c in foam_geom["inner_cells"] if c["k"] == "D")
    print(f"  foam_geometry: {len(foam_geom['vertices']):,} verts, "
          f"{len(foam_geom['inner_faces']):,} faces ({kind_counts}), "
          f"cells A={a_cells}/D={d_cells}", flush=True)

    pts, total_area, n_tris = triangulate_and_sample(polys, N_POINTS, rng)
    print(f"  sampled {len(pts):,} face points, total area {total_area:.3f}",
          flush=True)

    t_alpha = time.time()
    persistence = alpha_persistence(pts, MAX_R, MAX_DIM)
    alpha_elapsed = time.time() - t_alpha

    h2_essential = sum(1 for p in persistence if p["dim"] == 2 and p["death"] is None)
    h2_finite    = [p for p in persistence if p["dim"] == 2 and p["death"] is not None]
    h2_finite_high = [p for p in h2_finite if (p["death"] - p["birth"]) > HIGH_H2_THRESH]
    h1_finite = [p for p in persistence if p["dim"] == 1 and p["death"] is not None]
    h1_essential = sum(1 for p in persistence if p["dim"] == 1 and p["death"] is None)
    h0 = sum(1 for p in persistence if p["dim"] == 0)

    print(f"  Alpha ({alpha_elapsed:.1f}s): "
          f"H_0={h0} · H_1 finite={len(h1_finite)} ess={h1_essential} · "
          f"H_2 finite={len(h2_finite)} (high>{HIGH_H2_THRESH}: "
          f"{len(h2_finite_high)}) ess={h2_essential}",
          flush=True)

    # Cell-size statistics for the prose callout: mean+std of Voronoi-cell
    # volume estimated as 1/n_neighbours-weighted face area sum is fiddly;
    # use the simple proxy of nearest-neighbour distance, which is exactly
    # 2 × (Voronoi-equivalent-sphere radius) for an isotropic cell.
    inner_centres = centres[inner_mask]
    from scipy.spatial import cKDTree
    tree = cKDTree(centres)
    dists, _ = tree.query(inner_centres, k=2)
    nn = dists[:, 1]    # nearest-neighbour distance
    nn_mean = float(np.mean(nn))
    nn_std  = float(np.std(nn))
    nn_cv   = float(nn_std / nn_mean) if nn_mean > 0 else 0.0
    print(f"  nearest-neighbour-distance: mean={nn_mean:.4f} "
          f"std={nn_std:.4f} CV={nn_cv:.3f}", flush=True)

    # Top H_2 persistence values (sorted desc), kept tight for the level
    # summary. The PD/barcode panels for non-zero levels will use this
    # sorted array; the full persistence list is shipped only for level 0.
    h2_top = sorted(((p["death"] - p["birth"]) for p in h2_finite),
                    reverse=True)
    h2_top = [round(float(x), 5) for x in h2_top[:2000]]

    summary = {
        "perturb_pct":       round(perturb_pct, 4),
        "n_h0":              h0,
        "n_h1_finite":       len(h1_finite),
        "n_h1_essential":    h1_essential,
        "n_h2_finite":       len(h2_finite),
        "n_h2_finite_high":  len(h2_finite_high),
        "n_h2_essential":    h2_essential,
        "h2_high_threshold": HIGH_H2_THRESH,
        "expected_n_bubbles": int(inner_mask.sum()),
        "nn_dist_mean":      round(nn_mean, 5),
        "nn_dist_std":       round(nn_std, 5),
        "nn_dist_cv":        round(nn_cv, 4),
        "face_kinds":        kind_counts,
        "n_a_cells":         a_cells,
        "n_d_cells":         d_cells,
    }

    out: dict = {
        "perturb_pct":   round(perturb_pct, 4),
        "foam_geometry": foam_geom,
        "summary":       summary,
        "h2_top_persistence": h2_top,
    }

    if do_subsample:
        # Full payload for the primary level (level 0): full persistence,
        # full points, α-subsample.
        out["persistence"]    = persistence
        out["points"]         = pts.tolist()
        out["n_points"]       = int(len(pts))
        out["n_inner_faces"]  = int(len(polys))
        out["n_triangles"]    = int(n_tris)
        out["total_face_area"] = total_area

        t_sub = time.time()
        sub_block = alpha_subsample(vor, inner_mask, cube_ids, centres,
                                    SUBSAMPLE_K, SUBSAMPLE_N, SUBSAMPLE_MAX_R,
                                    rng)
        sub_elapsed = time.time() - t_sub
        sub_focus_kinds = "".join(c["k"] for c in sub_block["focus_cells"])
        sub_h2_high = sum(1 for p in sub_block["persistence"]
                          if p["dim"] == 2 and p["death"] is not None
                          and (p["death"] - p["birth"]) > HIGH_H2_THRESH)
        print(f"  α-subsample: {sub_focus_kinds} cluster, "
              f"{len(sub_block['simplices']):,} simplices "
              f"({sub_elapsed:.1f}s), high-H_2 = {sub_h2_high}",
              flush=True)
        out["alpha_subsample"] = sub_block

    print(f"  level done in {time.time() - t_lvl:.1f}s", flush=True)
    return out


def main() -> int:
    overall_t0 = time.time()
    rng = np.random.default_rng(SEED)

    nx = ny = nz = NX_INNER + 2 * PAD_LAYERS
    print(f"build_foam_data.py — outer={nx}x{ny}x{nz} cubes "
          f"(inner={NX_INNER}^3, pad={PAD_LAYERS} per side), "
          f"N_POINTS={N_POINTS}, MAX_R={MAX_R}, "
          f"PERTURB_LEVELS={PERTURB_LEVELS}", flush=True)

    base_centres, cube_ids = tile_a15(nx, ny, nz, L=L_CELL)
    n_total_cells = len(base_centres)
    n_inner_bubbles = NX_INNER ** 3 * 8

    inner_mask = np.zeros(n_total_cells, dtype=bool)
    for idx, (i, j, k, _s) in enumerate(cube_ids):
        if PAD_LAYERS <= i < nx - PAD_LAYERS \
           and PAD_LAYERS <= j < ny - PAD_LAYERS \
           and PAD_LAYERS <= k < nz - PAD_LAYERS:
            inner_mask[idx] = True
    assert inner_mask.sum() == n_inner_bubbles, \
        f"inner_mask {inner_mask.sum()} != expected {n_inner_bubbles}"
    print(f"tiled A15: {n_total_cells} centres ({n_inner_bubbles} inner)",
          flush=True)

    # ---- per-level runs
    levels: list[dict] = []
    compute_log: list[dict] = []
    for li, p in enumerate(PERTURB_LEVELS):
        do_subsample = (li == 0)
        lvl_t = time.time()
        lvl = process_level(p, li, base_centres, cube_ids, inner_mask, nx,
                            do_subsample, rng)
        levels.append(lvl)
        compute_log.append({
            "phase": f"level_{li}_perturb_{p:.3f}",
            "elapsed_s": round(time.time() - lvl_t, 2),
        })

    # ---- assemble output. Level 0 fields are mirrored at the top level for
    # backwards compat (front-end without slider awareness gets exact-WP data).
    lvl0 = levels[0]
    payload = {
        "source":          f"weaire-phelan {NX_INNER}x{NX_INNER}x{NX_INNER}",
        "n_bubbles":       int(n_inner_bubbles),
        "n_points":        lvl0["n_points"],
        "points":          lvl0["points"],
        "alpha":           {"max_r": MAX_R, "persistence": lvl0["persistence"]},
        "foam_geometry":   lvl0["foam_geometry"],
        "alpha_subsample": lvl0["alpha_subsample"],
        "summary":         lvl0["summary"],
        "compute_log":     compute_log,
        "config": {
            "nx_inner":      NX_INNER,
            "pad_layers":    PAD_LAYERS,
            "n_outer":       nx,
            "n_points":      N_POINTS,
            "max_r":         MAX_R,
            "max_dim":       MAX_DIM,
            "perturb_levels": PERTURB_LEVELS,
            "seed":          SEED,
            "L_cell":        L_CELL,
            "n_inner_faces": lvl0["n_inner_faces"],
            "n_triangles":   lvl0["n_triangles"],
            "total_face_area": lvl0["total_face_area"],
            "subsample_n":     SUBSAMPLE_N,
            "subsample_max_r": SUBSAMPLE_MAX_R,
            "h2_high_threshold": HIGH_H2_THRESH,
        },
        # New: per-perturbation-level slices for the slider. Level 0 here
        # is a thin slice (no full points / no full persistence) because
        # those are already at the top level for level 0. Higher levels
        # ship geometry + summary + sorted H_2 persistences (for showing
        # the bubble cluster's distribution without the full PD).
        "foam_levels": [
            {
                "perturb_pct":        lvl["perturb_pct"],
                "foam_geometry":      lvl["foam_geometry"],
                "summary":            lvl["summary"],
                "h2_top_persistence": lvl["h2_top_persistence"],
            }
            for lvl in levels
        ],
    }
    write_output(payload)

    print(f"\nALL DONE in {(time.time() - overall_t0) / 60:.2f} min")
    print(f"β_2 high-persistence count by level:")
    for lvl in levels:
        s = lvl["summary"]
        print(f"  perturb={s['perturb_pct']*100:5.1f}%  "
              f"β_2(high) = {s['n_h2_finite_high']:>4}  "
              f"(of {n_inner_bubbles})  "
              f"nn-CV = {s['nn_dist_cv']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
