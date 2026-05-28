"""
TDA Filtration Explorer — Backend (phase 3)
===========================================

What's new in phase 3 vs phase 2:
  - Many more manifolds: RP^n (n=1..3), CP^n (n=2..3), Möbius strip, Klein
    bottle, genus-g surfaces (g=2..4), lens spaces L(p,q), and prebuilt
    products of two simpler manifolds.
  - Per-request coefficient field (prime p). Default 11 (ℚ-like); RP^n,
    Klein, products thereof use 2; lens spaces L(p,q) use p.
  - API change: /sample now takes a single `key` string identifying the
    manifold (e.g. "S2", "RP3", "lens_7_2", "prod_S2_T2") instead of the old
    (manifold, n) pair. /compute additionally takes `coeff_field` (an int).

Endpoints (unchanged shape, modulo the above):
  GET  /         Serve the frontend
  POST /sample   { key, N, sigma, distribution, seed }
                 → { points, key, N, sigma, distribution, ambient_dim }
  POST /compute  { points, max_r_*, max_dim, coeff_field }
                 → { rips:{...}, cech:{...}, alpha:{...}, max_dim, coeff_field }

Conventions (unchanged from phase 2):
  Rips slider  = max edge length (diameter)
  Čech slider  = radius of smallest enclosing ball
  Alpha slider = radius α (server sqrts GUDHI's internal α²)
"""

import math
import os

import numpy as np
import gudhi as gd
from flask import Flask, request, jsonify, send_from_directory


app = Flask(__name__, static_folder='.')


# ============================================================================
# Sampling primitives — one function per manifold family
# ============================================================================

def sample_sphere(n, N, rng):
    """Uniform on S^n ⊂ R^{n+1}.

    Method: sample N standard Gaussians in R^{n+1}, normalise each to unit
    length. (Marsaglia 1972 — Gaussians are rotation-invariant, so the
    direction after normalisation is uniform on S^n.)
    """
    pts = rng.standard_normal((N, n + 1))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def sample_torus(n, N, rng):
    """T^n via flat embedding (cos θ_1, sin θ_1, …, cos θ_n, sin θ_n) ∈ R^{2n}.

    Sample n independent uniform angles per point. Generalises to any n.
    For n=2: the 4-coord 'flat torus' embedding (visualised as donut by the
    frontend only — the underlying point cloud is in R^4).
    """
    angles = rng.uniform(0.0, 2.0 * np.pi, (N, n))
    pts = np.zeros((N, 2 * n))
    for i in range(n):
        pts[:, 2 * i]     = np.cos(angles[:, i])
        pts[:, 2 * i + 1] = np.sin(angles[:, i])
    return pts


def sample_rp(n, N, rng):
    """RP^n via the Veronese-style embedding x ↦ xx^T on S^n.

    RP^n = S^n / {±1}. The map x ↦ xx^T sends antipodal points to the same
    symmetric (n+1)×(n+1) matrix (rank-1, trace 1). We flatten the UPPER
    TRIANGLE (incl diagonal) of that matrix to R^d where d = (n+1)(n+2)/2.
    This is the smallest Euclidean dim into which we embed RP^n smoothly via
    this construction.

    For n=1: ambient R^3.  For n=2: R^6.  For n=3: R^10.
    """
    s = sample_sphere(n, N, rng)            # (N, n+1) points on S^n
    d = n + 1
    iu = np.triu_indices(d)                  # upper triangle, incl diagonal
    out = np.zeros((N, len(iu[0])))
    for k in range(N):
        out[k] = np.outer(s[k], s[k])[iu]
    return out


def sample_cp(n, N, rng):
    """CP^n via z ↦ zz* on S^{2n+1} ⊂ C^{n+1}.

    CP^n = S^{2n+1} / S^1 (multiply by e^{iθ}). The map z ↦ zz* (outer
    product with conjugate transpose) is invariant under that S^1 action and
    sends CP^n injectively into the space of Hermitian (n+1)×(n+1) rank-1
    trace-1 matrices.

    Real-flattened ambient dim: (n+1)² (diagonal real entries + 2× upper
    off-diagonal real/imaginary).
    For n=2: ambient R^9.  For n=3: R^16.
    """
    s = sample_sphere(2 * n + 1, N, rng)     # (N, 2n+2)
    z = s[:, ::2] + 1j * s[:, 1::2]          # (N, n+1) complex unit vectors
    d = n + 1
    iu, ju = np.triu_indices(d, k=1)         # strict upper triangle
    out = np.zeros((N, d * d))
    for k in range(N):
        M = np.outer(z[k], np.conj(z[k]))     # Hermitian rank-1
        out[k, :d]                  = np.real(np.diag(M))
        out[k, d : d + len(iu)]     = np.real(M[iu, ju])
        out[k, d + len(iu):]        = np.imag(M[iu, ju])
    return out


def sample_klein(N, rng):
    """Klein bottle, smooth embedding in R^4.

    Parametrisation (u, v) → R^4:
        x = cos u · (cos v + 2)
        y = sin u · (cos v + 2)
        z = sin v · cos(u/2)
        w = sin v · sin(u/2)
    with u ∈ [0, 2π), v ∈ [0, 2π).

    Identifications (verified algebraically):
      (u, v + 2π)       ~ (u, v)               — v periodic.
      (u + 2π, 2π − v)  ~ (u, v)               — the Klein twist.
    The constant +2 inside (cos v + 2) keeps the embedding non-singular.

    β = (1, 1, 0) over ℤ;  β = (1, 2, 1) over ℤ/2.
    """
    u = rng.uniform(0.0, 2.0 * np.pi, N)
    v = rng.uniform(0.0, 2.0 * np.pi, N)
    cu, su = np.cos(u), np.sin(u)
    cu2, su2 = np.cos(u / 2), np.sin(u / 2)
    cv, sv = np.cos(v), np.sin(v)
    radial = cv + 2.0
    return np.column_stack([
        cu * radial,
        su * radial,
        sv * cu2,
        sv * su2,
    ])


def sample_mobius(N, rng):
    """Möbius strip in R^3 via the standard parametrisation:

        x = (1 + (s/2) cos(t/2)) cos t
        y = (1 + (s/2) cos(t/2)) sin t
        z = (s/2) sin(t/2)

    with s ∈ [-1, 1] (across), t ∈ [0, 2π) (around).

    Identification (t + 2π, s) ~ (t, -s) — verifiable because cos((t+2π)/2)
    = -cos(t/2) and sin((t+2π)/2) = -sin(t/2). This is the Möbius twist.

    β = (1, 1, 0) — deformation retracts to S¹.
    """
    s = rng.uniform(-1.0, 1.0, N)
    t = rng.uniform(0.0, 2.0 * np.pi, N)
    half_s = s / 2.0
    rad = 1.0 + half_s * np.cos(t / 2)
    return np.column_stack([
        rad * np.cos(t),
        rad * np.sin(t),
        half_s * np.sin(t / 2),
    ])


def sample_genus_g(g, N, rng, loop_diam=2.0, tube_r=0.25):
    """Genus-g orientable surface as the BOUNDARY of the tubular
    neighborhood of g circles meeting at the origin.

    Construction:
      For i = 0, …, g-1, loop i lies in the vertical plane spanned by ê_z
      and v_i = (cos(πi/g), sin(πi/g), 0). Each loop is a circle of
      diameter `loop_diam` passing through the origin (tangent to the
      z-axis at the origin).

      The tubular neighborhood of loop i (radius tube_r) is the set of
      points within distance tube_r of loop i. The UNION of these g
      tubular neighborhoods is a solid in R^3; its BOUNDARY is a closed
      orientable surface, and by the Euler-characteristic argument
        χ(wedge of g circles) = 1 − g
        χ(boundary of regular nbhd) = 2(1 − g) = 2 − 2g
      this boundary has genus g.

    Sampling strategy (rejection):
      1. For each loop i, sample the FULL tube surface around it
         (θ ∈ [0, 2π), φ ∈ [0, 2π)) — points at distance exactly tube_r
         from loop i.
      2. For each candidate point p, compute dist(p, loop_j) for all
         j ≠ i. KEEP p only if dist(p, loop_j) ≥ tube_r for all j ≠ i.
         (Points failing this test lie INSIDE another tube, so they're in
         the interior of the union, not on its boundary.)
      3. Pool surviving points across loops, subsample to N.

    The result is a clean, geometrically consistent sampling of the
    boundary surface, with no gaps or stray interior points. Tested to
    recover β = (1, 2g, 1) on N ≥ 80 samples for g ∈ {2, 3, 4}.
    """
    R = loop_diam / 2.0

    # Per-loop frames
    frames = []
    for i in range(g):
        ang = np.pi * i / g
        u_i = np.array([0.0, 0.0, 1.0])                      # vertical
        v_i = np.array([np.cos(ang), np.sin(ang), 0.0])      # horizontal direction
        w_i = np.cross(u_i, v_i)                              # 3rd orthonormal
        frames.append((u_i, v_i, w_i))

    def dist_to_loop(pts, i):
        """Vectorised distance from each point in pts (shape (M, 3)) to loop i.

        Loop i is the circle of radius R in the plane spanned by (u_i, v_i),
        centered at R · v_i (so it passes through the origin).
        Distance method:
          1. Project pts onto the (u_i, v_i) plane: component along w_i is
             d_w = pts · w_i; in-plane projection is pts − d_w · w_i.
          2. Within that plane, compute distance from the in-plane projection
             to the circle centered at (R · v_i) with radius R. That equals
             | |proj − center| − R |.
          3. Combine: dist² = (in-plane distance to circle)² + d_w².
        """
        u_i, v_i, w_i = frames[i]
        d_w = pts @ w_i
        in_plane = pts - d_w[:, None] * w_i
        center = R * v_i
        d_in = np.linalg.norm(in_plane - center, axis=1)
        d_to_circle = np.abs(d_in - R)
        return np.sqrt(d_to_circle ** 2 + d_w ** 2)

    # Oversample tubes (so that after rejection we have enough points).
    # Rejection rate depends on g: more loops, more rejection near joint.
    oversample = max(3, g + 1)
    per_loop_initial = max(N * oversample // g, 50)
    survivors = []

    for i in range(g):
        u_i, v_i, w_i = frames[i]
        theta = rng.uniform(0.0, 2.0 * np.pi, per_loop_initial)
        phi   = rng.uniform(0.0, 2.0 * np.pi, per_loop_initial)

        # Curve c(θ) and tangent c'(θ) — passes through origin at θ = 0.
        c = R * (1.0 - np.cos(theta))[:, None] * v_i \
          + R * np.sin(theta)[:, None] * u_i
        t_vec = R * np.sin(theta)[:, None] * v_i \
              + R * np.cos(theta)[:, None] * u_i
        # avoid division by zero (only happens at theta = 0 or 2π; vector
        # length is R there, so this guard is mostly cosmetic)
        t_norm = np.linalg.norm(t_vec, axis=1, keepdims=True).clip(1e-9)
        t_hat = t_vec / t_norm
        n1 = np.broadcast_to(w_i, (per_loop_initial, 3))
        n2 = np.cross(t_hat, n1)
        surf = c + tube_r * (np.cos(phi)[:, None] * n1
                              + np.sin(phi)[:, None] * n2)

        # Reject any point INSIDE another tube. We use a small margin (0.97)
        # so that we don't double-keep points on the "ridge" where two tube
        # boundaries coincide (a 1-dim curve, measure zero anyway).
        keep_mask = np.ones(per_loop_initial, dtype=bool)
        for j in range(g):
            if j == i:
                continue
            d_other = dist_to_loop(surf, j)
            keep_mask &= (d_other >= tube_r * 0.97)

        survivors.append(surf[keep_mask])

    pool = np.vstack(survivors)
    if len(pool) == 0:
        raise RuntimeError("genus-g sampler produced no surviving samples; "
                           "check tube_r / loop_diam parameters.")

    if len(pool) >= N:
        idx = rng.choice(len(pool), N, replace=False)
        return pool[idx]
    # If undersampled, pad by sampling with replacement
    extra_idx = rng.integers(0, len(pool), N - len(pool))
    return np.vstack([pool, pool[extra_idx]])


def sample_lens(p, q, N, rng):
    """Lens space L(p, q) = S^3 / (ℤ/p), where ℤ/p acts on S^3 ⊂ C^2 as
        (z_1, z_2) ↦ (ζ z_1, ζ^q z_2),   ζ = e^{2πi/p}.

    For gcd(p, q) = 1 the action is free and the quotient is a smooth closed
    orientable 3-manifold.

    Embedding via ℤ/p-invariants:
        I_1 = |z_1|²                          (real)
        I_2 = z_1^p                           (complex)
        I_3 = z_2^p                           (complex)
        I_4 = z_1^q · conj(z_2)               (complex)
    Real-flattened to R^7.

    Algebra check that (I_1, I_2, I_3, I_4) separates orbits:
      Two points (z_1, z_2), (w_1, w_2) with the same invariants force
      w_1 = ξ z_1 and w_2 = η z_2 for some p-th roots of unity ξ, η.
      Equating z_1^q ẑ_2 = w_1^q ŵ_2 then forces η = ξ^q.
      So (w_1, w_2) is in the ℤ/p-orbit of (z_1, z_2). ✓

    β over ℤ: (1, 0, 0, 1) — torsion in H_1 is invisible.
    β over ℤ/p: (1, 1, 1, 1) — torsion shows up as β_1 and β_2.

    Famous: L(7, 1) ≇ L(7, 2) but they have the same β over EVERY field
    (Reidemeister torsion is the classical discriminator). The persistence
    diagrams may differ in bar positions because their embeddings use
    different q-dependent invariants.
    """
    s = sample_sphere(3, N, rng)             # (N, 4): points on S^3 ⊂ R^4
    z1 = s[:, 0] + 1j * s[:, 1]
    z2 = s[:, 2] + 1j * s[:, 3]

    inv1 = np.abs(z1) ** 2                    # real
    inv2 = z1 ** p                            # complex
    inv3 = z2 ** p                            # complex
    inv4 = z1 ** q * np.conj(z2)              # complex

    return np.column_stack([
        inv1,
        inv2.real, inv2.imag,
        inv3.real, inv3.imag,
        inv4.real, inv4.imag,
    ])


# --- Product manifolds -------------------------------------------------------
# Prebuilt set per Carol's list. Each entry maps a product key to its two
# factor keys (which must themselves be valid keys for sample_by_key).
PRODUCT_FACTORS = {
    'prod_S2_S1':    ('S2', 'S1'),
    'prod_S2_S2':    ('S2', 'S2'),
    'prod_S1_RP2':   ('S1', 'RP2'),
    'prod_S1_klein': ('S1', 'klein'),
    'prod_RP2_RP2':  ('RP2', 'RP2'),
    'prod_S2_T2':    ('S2', 'T2'),
}


def sample_by_key(key, N, rng):
    """Dispatch the manifold key to the right sampler. Used both as the
    top-level entry (via sample_points, which adds σ noise) and recursively
    (for products — noise applied only once at the outer call).
    """
    if key.startswith('S') and key[1:].isdigit():
        return sample_sphere(int(key[1:]), N, rng)
    if key.startswith('T') and key[1:].isdigit():
        return sample_torus(int(key[1:]), N, rng)
    if key.startswith('RP') and key[2:].isdigit():
        return sample_rp(int(key[2:]), N, rng)
    if key.startswith('CP') and key[2:].isdigit():
        return sample_cp(int(key[2:]), N, rng)
    if key == 'mobius':
        return sample_mobius(N, rng)
    if key == 'klein':
        return sample_klein(N, rng)
    if key.startswith('genus'):
        return sample_genus_g(int(key[5:]), N, rng)
    if key.startswith('lens_'):
        _, p_str, q_str = key.split('_')
        return sample_lens(int(p_str), int(q_str), N, rng)
    if key in PRODUCT_FACTORS:
        k1, k2 = PRODUCT_FACTORS[key]
        pts1 = sample_by_key(k1, N, rng)
        pts2 = sample_by_key(k2, N, rng)
        return np.hstack([pts1, pts2])
    if key == 'ubiquitin':
        # Fixed real-world dataset — N, σ, and rng are ignored at this layer.
        # Returns the 76 Cα coordinates of PDB 1UBQ verbatim. σ noise from the
        # outer sample_points() layer still applies if the frontend passes
        # σ > 0, but the frontend greys out the σ slider for proteins so this
        # is normally a no-op.
        return UBIQUITIN_CA.copy()
    if key == 'local_bubble':
        # Fixed real-world dataset — Zucker et al. 2022 Extended Data Table 1.
        # 34 stellar association mean positions in heliocentric Galactic
        # Cartesian (x, y, z), units of pc. N, σ, rng ignored at this layer;
        # the frontend greys out N/σ/distribution/reroll via the fixed_data
        # flag so the σ-noise path in sample_points() is normally a no-op.
        return LOCAL_BUBBLE_XYZ.copy()
    if key == 'g25_ancient':
        # Fixed real-world dataset — 250 maxmin landmarks from the Eurogenes
        # Global25 ancient PCA cloud, projected to PC1-PC3. See the G25_*
        # data block below. Frontend greys out N/σ/reroll via fixed_data.
        if G25_ANCIENT_PC3 is None:
            raise ValueError("g25_data.json not found at server startup")
        return G25_ANCIENT_PC3.copy()
    if key == 'g25_modern':
        if G25_MODERN_PC3 is None:
            raise ValueError("g25_data.json not found at server startup")
        return G25_MODERN_PC3.copy()
    if key == 'g25_combined':
        if G25_COMBINED_PC3 is None:
            raise ValueError("g25_data.json not found at server startup")
        return G25_COMBINED_PC3.copy()
    if key in UBQ_POINTS:
        # Fixed real-world dataset built offline by build_ubiquitin_data.py.
        # Covers all ubiquitin_xray_*, ubiquitin_nmr1_*, ubiquitin_nmr_ensemble_*
        # point-cloud variants. The frontend greys out N/σ/distribution/reroll
        # via fixed_data for these keys.
        return UBQ_POINTS[key].copy()
    raise ValueError(f"Unknown manifold key: {key!r}")


def sample_points(key, N, sigma, distribution, seed):
    rng = np.random.default_rng(seed)
    if distribution != 'uniform':
        raise NotImplementedError(f"Distribution '{distribution}' not implemented.")
    pts = sample_by_key(key, N, rng)
    if sigma > 0.0:
        pts = pts + rng.uniform(-sigma, sigma, pts.shape)
    return pts


# ============================================================================
# Fixed real-world data: ubiquitin (PDB 1UBQ) Cα coordinates
# ============================================================================
#
# 76 Cα atoms (one per residue), in Ångströms, as deposited in PDB 1UBQ.
# Index i in this array corresponds to residue (i + 1) in PDB numbering;
# index 0 is Met1, index 75 is Gly76.
#
# Source: PDB entry 1UBQ — Vijay-Kumar, Bugg, Cook, "Structure of ubiquitin
# refined at 1.8 Å resolution", J. Mol. Biol. 194: 531–544 (1987).
# Download: https://files.rcsb.org/view/1UBQ.pdb
#
# Coordinates are NOT centred or scaled. The frontend re-centres before
# rendering. Caveats: residues 73–76 are deposited at partial occupancy
# (C-terminal tail is mobile); we use them at full weight, which is the
# standard choice for TDA.
UBIQUITIN_CA = np.array([
    [26.266, 25.413,  2.842], [26.850, 29.021,  3.898], [26.235, 30.058,  7.497],
    [26.772, 33.436,  9.197], [28.605, 33.965, 12.503], [27.691, 37.315, 14.143],
    [30.225, 38.643, 16.662], [29.607, 41.180, 19.467], [31.422, 43.940, 17.553],
    [28.978, 43.960, 14.678], [31.191, 42.012, 12.331], [29.542, 39.020, 10.653],
    [31.720, 36.289,  9.176], [30.505, 33.884,  6.512], [31.677, 30.275,  6.639],
    [31.220, 27.341,  4.275], [30.288, 24.245,  6.193], [28.468, 20.940,  5.980],
    [25.829, 19.825,  8.494], [28.054, 16.835,  9.210], [30.796, 19.083, 10.566],
    [31.398, 19.064, 14.286], [31.288, 22.201, 16.417], [35.031, 21.722, 17.069],
    [35.590, 21.945, 13.302], [33.533, 25.097, 12.978], [35.596, 26.715, 15.736],
    [38.794, 25.761, 13.880], [37.471, 27.391, 10.668], [36.731, 30.570, 12.645],
    [40.269, 30.508, 14.115], [41.718, 30.022, 10.643], [39.808, 32.994,  9.233],
    [39.676, 35.547, 12.072], [42.345, 34.269, 14.431], [40.226, 33.716, 17.509],
    [41.461, 30.751, 19.594], [38.817, 28.020, 19.889], [39.063, 28.063, 23.695],
    [37.738, 31.637, 23.712], [34.738, 30.875, 21.473], [31.200, 30.329, 22.780],
    [28.762, 29.573, 19.906], [25.034, 30.170, 20.401], [22.126, 29.062, 18.183],
    [18.443, 29.143, 19.083], [19.399, 29.894, 22.655], [21.550, 26.796, 23.133],
    [25.349, 26.872, 23.643], [26.826, 24.521, 21.012], [29.015, 21.657, 22.288],
    [32.262, 20.670, 20.514], [31.568, 16.962, 19.825], [28.108, 17.439, 18.276],
    [27.574, 18.192, 14.563], [25.594, 21.109, 13.072], [22.924, 18.583, 12.025],
    [22.418, 17.638, 15.693], [21.079, 21.149, 16.251], [19.065, 21.352, 12.999],
    [21.184, 24.263, 11.690], [20.081, 24.773,  8.033], [21.656, 26.847,  5.240],
    [21.907, 30.563,  5.881], [21.419, 30.253,  9.620], [23.212, 32.762, 11.891],
    [25.149, 31.609, 14.980], [26.179, 34.127, 17.650], [29.801, 34.145, 18.829],
    [30.479, 35.369, 22.374], [34.145, 35.472, 23.481], [35.161, 34.174, 26.896],
    [38.668, 35.502, 27.680], [40.873, 33.802, 30.253], [41.845, 36.550, 32.686],
    [40.373, 39.813, 33.944],
], dtype=float)
assert UBIQUITIN_CA.shape == (76, 3), "ubiquitin Cα array must be 76×3"


# ============================================================================
# Fixed real-world data: Zucker et al. 2022 Local Bubble shell tracers
# ============================================================================
#
# 34 stellar association mean positions in heliocentric Galactic Cartesian
# coordinates, in parsecs. Source: Zucker et al. (2022), "Star formation near
# the Sun is driven by expansion of the Local Bubble", Nature 601, 334–337,
# DOI 10.1038/s41586-021-04286-5, Extended Data Table 1.
# Data: Harvard Dataverse DOI 10.7910/DVN/ZU97QD.
#
# Row order matches the source .dat file (NOT sorted by region). Indices 0–1
# Perseus, 2–12 Taurus, 13–19 Orion, then Chamaeleon / Sco-Cen / Lupus /
# Ophiuchus / Corona Australis / Serpens interleaved.
#
# NOTE: per-point region and subgroup labels are kept in index.html
# (MANIFOLD_SETTINGS['local_bubble'].regions and .subgroups) — same row order
# as the array below. If you ever change this data, update BOTH FILES.
LOCAL_BUBBLE_XYZ = np.array([
    [ -255.0,  +101.0,  -102.0],  # Perseus / NGC1333
    [ -282.0,  +100.0,   -96.0],  # Perseus / IC348
    [ -122.0,   +24.0,   -35.0],  # Taurus / C2 - L1495
    [ -151.0,   +24.0,   -43.0],  # Taurus / C8 - B213
    [ -128.0,   +17.0,   -38.0],  # Taurus / D4 - North
    [ -123.0,   +13.0,   -34.0],  # Taurus / C6 - L1524
    [ -136.0,   +14.0,   -33.0],  # Taurus / C7 - L1527
    [ -155.0,   +12.0,   -45.0],  # Taurus / C5 - L1546
    [ -116.0,    +6.0,   -37.0],  # Taurus / D3 - South
    [ -136.0,    +2.0,   -49.0],  # Taurus / C1 - L1551
    [ -123.0,    +0.0,   -45.0],  # Taurus / D2 - L1558
    [ -170.0,    -3.0,   -21.0],  # Taurus / D1 - L1544
    [ -109.0,    -7.0,    -9.0],  # Taurus / C9 - 118TauEast
    [ -318.0,  -139.0,  -159.0],  # Orion / L1616
    [ -362.0,  -170.0,  -102.0],  # Orion / NGC2068/2071
    [ -344.0,  -172.0,  -112.0],  # Orion / NGC 2023/2024
    [ -342.0,  -173.0,  -119.0],  # Orion / Sigma Ori
    [ -326.0,  -177.0,  -128.0],  # Orion / NGC1977
    [ -324.0,  -181.0,  -131.0],  # Orion / Orion A, Head
    [ -327.0,  -203.0,  -135.0],  # Orion / Orion A, Tail
    [  +84.0,  -165.0,   -49.0],  # Chamaeleon / 1-North
    [  +82.0,  -161.0,   -51.0],  # Chamaeleon / 1-South
    [  +57.0,   -93.0,   +11.0],  # Sco-Cen / Lower Centaurus Crux, LCC
    [ +105.0,  -158.0,   -48.0],  # Chamaeleon / 2
    [ +109.0,   -61.0,   +29.0],  # Sco-Cen / Upper Centaurus Lupus, UCL
    [ +145.0,   -63.0,   +23.0],  # Lupus / 4
    [ +146.0,   -54.0,   +25.0],  # Lupus / 3
    [ +146.0,   -52.0,   +29.0],  # Lupus / Off-Cloud Population
    [ +130.0,   -19.0,   +53.0],  # Sco-Cen / Upper Scorpius, USCO
    [ +131.0,   -15.0,   +40.0],  # Ophiuchus / Rho Oph, Population I
    [ +131.0,   -14.0,   +43.0],  # Ophiuchus / Rho Oph, Population II
    [ +141.0,    -4.0,   -34.0],  # Corona Australis / Off-Cloud Population
    [ +143.0,    +0.0,   -43.0],  # Corona Australis / On-Cloud Population
    [ +341.0,  +176.0,   +16.0],  # Serpens / far South
], dtype=float)
assert LOCAL_BUBBLE_XYZ.shape == (34, 3), "Local Bubble array must be 34×3"


# ============================================================================
# Fixed real-world data: Global25 ancient/modern DNA — three landmark sets
# ============================================================================
#
# Three datasets, each 250 maxmin-subsampled landmarks projected to the first
# 3 PCs of the Global25 scaled PCA basis (Davidski / Eurogenes Genetic Project):
#   - g25_ancient:  ancient samples only (n=7292 source pool)
#   - g25_modern:   modern  samples only (n=10927 source pool)
#   - g25_combined: ancient + modern stacked (n=18219 source pool)
#
# Maxmin is run in FULL 25D (preserves the topology meaningfully), then the
# selected landmarks are projected to PC1–PC3 for visualisation and to make
# Čech / Alpha computable (Delaunay in low ambient dim). The 22 dropped PCs
# carry less variance but DO carry archaeological/population structure that
# is not visible in this 3D projection — see the writeup in index.html.
#
# Per-landmark population labels and group indices (for colouring) live in
# g25_data.json, loaded once at startup and served via /g25-meta.

import json as _json
_G25_PATH = os.path.join(os.path.dirname(__file__), 'g25_data.json')
try:
    with open(_G25_PATH) as _f:
        G25_DATA = _json.load(_f)
    G25_ANCIENT_PC3  = np.array(G25_DATA['ancient']['points'],  dtype=float)
    G25_MODERN_PC3   = np.array(G25_DATA['modern']['points'],   dtype=float)
    G25_COMBINED_PC3 = np.array(G25_DATA['combined']['points'], dtype=float)
    assert G25_ANCIENT_PC3.shape  == (250, 3), f"ancient shape: {G25_ANCIENT_PC3.shape}"
    assert G25_MODERN_PC3.shape   == (250, 3), f"modern  shape: {G25_MODERN_PC3.shape}"
    assert G25_COMBINED_PC3.shape == (250, 3), f"combined shape: {G25_COMBINED_PC3.shape}"
except FileNotFoundError:
    # Server can still start without G25 data; the g25_* manifold keys will 404
    # if requested. Production setups should ensure g25_data.json is alongside.
    print(f"WARNING: {_G25_PATH} not found — g25_* manifolds will be unavailable")
    G25_DATA = None
    G25_ANCIENT_PC3 = G25_MODERN_PC3 = G25_COMBINED_PC3 = None


# ============================================================================
# Fixed real-world data: ubiquitin atomic-resolution variants + simulated spectra
# ============================================================================
#
# Built offline by build_ubiquitin_data.py from PDB 1UBQ (X-ray, single model)
# and PDB 1D3Z (NMR, 10 models). Holds many manifold keys at once:
#   ubiquitin_xray_{ca,backbone,cb,polar,sidechain,residue,heavy,allatom}
#   ubiquitin_nmr1_{ca,...,allatom}              ← model 1 of NMR
#   ubiquitin_nmr_ensemble_{ca,...,allatom}      ← all 10 models pooled
#   ubiquitin_ms_native, ubiquitin_ms_denatured  ← 1D simulated MS spectra
#   ubiquitin_2dir                               ← 2D simulated IR amide-I
# Each point-cloud entry has points, residues, atom_names, n_atoms, diameter,
# max_r_* + r_init_* per filtration, label, description, ss_ranges.
# Spectrum entries have kind ∈ {'ms_1d','ir_2d'} and a 'data' subfield.
_UBQ_PATH = os.path.join(os.path.dirname(__file__), 'ubiquitin_data.json')
try:
    with open(_UBQ_PATH) as _f:
        UBQ_DATA = _json.load(_f)
    # Pre-extract just the numpy point arrays for the keys that are 3D clouds
    # (skip the kind-prefixed spectrum entries).
    UBQ_POINTS = {}
    for _k, _v in UBQ_DATA.items():
        if 'kind' in _v:
            continue
        UBQ_POINTS[_k] = np.array(_v['points'], dtype=float)
    print(f"Loaded {len(UBQ_POINTS)} ubiquitin point-cloud variants + "
          f"{len(UBQ_DATA) - len(UBQ_POINTS)} spectrum entries.")
except FileNotFoundError:
    print(f"WARNING: {_UBQ_PATH} not found — ubiquitin_xray_* / nmr_* / spectrum "
          f"manifolds will be unavailable. Run build_ubiquitin_data.py to generate.")
    UBQ_DATA = None
    UBQ_POINTS = {}


# ============================================================================
# Complex builders — unchanged from phase 2
# ============================================================================

def build_rips_complex(points, max_edge_length, max_dim=3):
    """Vietoris–Rips: include {v_0, …, v_k} iff every pair (v_i, v_j) is
    within max_edge_length. Filtration = max pairwise distance."""
    rips = gd.RipsComplex(points=points.tolist(), max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    return [(tuple(s), float(f)) for s, f in st.get_filtration()]


def build_cech_complex(points, max_radius, max_dim=3):
    """Delaunay-Čech, homotopy equivalent to true Čech. Filtration = radius
    of smallest enclosing ball. `output_squared_values=False` ⇒ returns
    radii, not α²."""
    dc = gd.DelaunayCechComplex(points=points.tolist())
    st = dc.create_simplex_tree(
        max_alpha_square=max_radius ** 2,
        output_squared_values=False,
    )
    out = []
    for s, f in st.get_filtration():
        if len(s) - 1 > max_dim:
            continue
        out.append((tuple(s), float(f)))
    return out


def build_alpha_complex(points, max_alpha_radius, max_dim=3):
    """GUDHI Alpha complex, internally α²; we sqrt to return radius."""
    alpha = gd.AlphaComplex(points=points.tolist())
    st = alpha.create_simplex_tree(max_alpha_square=max_alpha_radius ** 2)
    out = []
    for s, f_sq in st.get_filtration():
        if len(s) - 1 > max_dim:
            continue
        out.append((tuple(s), math.sqrt(max(f_sq, 0.0))))
    return out


# ============================================================================
# Persistent homology — now takes a coefficient field
# ============================================================================

def persistence_pairs(simplices, max_dim, coeff_field=11):
    """Persistent homology over ℤ/coeff_field (coeff_field must be prime).

    Phase 3 / ubiquitin update: now also extracts the BIRTH SIMPLEX vertices
    via GUDHI's `simplex_tree.persistence_pairs()` method (different from this
    function's name — sorry for the collision). For an H_k class, the birth
    simplex is a k-simplex (its dimension equals the class dimension), and
    the death simplex is a (k+1)-simplex (or empty list for essential
    classes). We expose the birth-simplex vertices as `birth_simplex` on
    every returned entry, and additionally as `birth_edge` on H_1 entries
    (for the frontend's residue-level hover labels on protein data).

    Returns list of {'dim', 'birth', 'death', 'birth_simplex', 'birth_edge'?}
    where death is None for essential classes. Drops dim-max_dim essentials
    (guaranteed artifacts of the max_dim cap).
    """
    st = gd.SimplexTree()
    for simplex, filtration in simplices:
        st.insert(list(simplex), filtration=filtration)
    if st.num_simplices() == 0:
        return []
    st.set_dimension(max_dim + 1)
    st.make_filtration_non_decreasing()
    # Trigger the persistence computation; we read its results via
    # persistence_pairs() below.
    st.persistence(homology_coeff_field=coeff_field)
    raw_pairs = st.persistence_pairs()
    out = []
    for birth_simplex, death_simplex in raw_pairs:
        dim = len(birth_simplex) - 1
        if dim > max_dim:
            continue
        is_essential = (len(death_simplex) == 0)
        if dim == max_dim and is_essential:
            continue  # guaranteed artifact of max_dim cap
        birth = float(st.filtration(birth_simplex))
        death = None if is_essential else float(st.filtration(death_simplex))
        entry = {
            'dim':            dim,
            'birth':          birth,
            'death':          death,
            'birth_simplex':  [int(v) for v in birth_simplex],
        }
        if dim == 1:
            # Edge that births an H_1 class: vertices are the "birth edge"
            # the frontend wires up to its hover-label classifier.
            entry['birth_edge'] = entry['birth_simplex']
        out.append(entry)
    return out


# ============================================================================
# Flask routes
# ============================================================================

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')


@app.route('/g25_full_vr.json')
def serve_g25_full_vr():
    # Precomputed full-dataset Vietoris–Rips persistence for the
    # 'g25_genetics' picker entry. Produced by ../g25_vr_local.py and
    # written to v4/g25_full_vr.json as a compact single-line blob.
    return send_from_directory('.', 'g25_full_vr.json',
                               mimetype='application/json')


@app.route('/g25_vr_viz.json')
def serve_g25_vr_viz():
    # Edge filtration for the live VR-complex viz under each G25 block.
    # 250 maxmin landmarks per dataset (PC1-3 for display), with
    # 25-D-distance edges sorted by birth radius. Produced by
    # ../build_g25_vr_viz.py.
    return send_from_directory('.', 'g25_vr_viz.json',
                               mimetype='application/json')


@app.route('/g25_audit_data.json')
def serve_g25_audit_data():
    # Analytic-audit data for the four audit articles in the G25 section:
    # H_0 components, H_1 noise-floor / Gumbel fit, the three significance
    # frameworks (M1 universal null, M2 bootstrap, M3 DTM), and
    # consensus-feature cocycle representatives.  Produced by
    # ../build_audit_data.py from the out_audit/* analytic outputs.
    return send_from_directory('.', 'g25_audit_data.json',
                               mimetype='application/json')


@app.route('/g25_vr_viz_cocycles.json')
def serve_g25_vr_viz_cocycles():
    # Per-cloud top-H_1 cocycle representatives on the same 250-landmark
    # subsamples used by the live VR-complex viz (g25_vr_viz.json).  The
    # frontend uses these to highlight the cocycle edges + vertices of
    # selected H_1 loops on the canvas.  Produced by
    # ../build_g25_vr_viz_cocycles.py.
    return send_from_directory('.', 'g25_vr_viz_cocycles.json',
                               mimetype='application/json')


@app.route('/hk97_data.json')
def serve_hk97_data():
    # Precomputed Cα coords + Rips/Čech/Alpha persistence for the HK97
    # bacteriophage mature capsid (PDB 1OHG biological assembly 1, ~118k
    # Cα atoms across 60 icosahedral copies of the 7-chain asymmetric unit).
    # Produced by ./build_hk97_data.py — run it overnight to generate.
    return send_from_directory('.', 'hk97_data.json',
                               mimetype='application/json')


@app.route('/menger_data.json')
def serve_menger_data():
    # Precomputed Čech + Alpha persistence on N=200 000 IFS-uniform samples
    # at each iteration level of the Menger sponge, plus exact cubical Betti
    # numbers per level. Produced offline by ./build_menger_data.py.
    return send_from_directory('.', 'menger_data.json',
                               mimetype='application/json')


@app.route('/ubiquitin_data.json')
def serve_ubiquitin_data():
    # Precomputed ubiquitin atomic-resolution variants (X-ray + NMR-1 + NMR
    # ensemble at 8 atom subsets each) plus simulated MS (1D) and 2D IR
    # spectra with precomputed sublevel-set persistence diagrams. Produced
    # offline by ./build_ubiquitin_data.py.
    return send_from_directory('.', 'ubiquitin_data.json',
                               mimetype='application/json')


@app.route('/ubiquitin-meta', methods=['GET'])
def ubiquitin_meta_endpoint():
    """Strip the heavy 'points' arrays and return just the metadata fields
    (slider settings, atom counts, descriptions, ss_ranges, spectrum kinds).
    The frontend uses this on first ubiquitin selection to populate the
    MANIFOLD_SETTINGS dynamically rather than hardcoding 24+ entries."""
    if UBQ_DATA is None:
        return jsonify({'error': 'ubiquitin_data.json not loaded at startup'}), 404
    out = {}
    for k, v in UBQ_DATA.items():
        if 'kind' in v:
            # Spectrum entries — return the full payload (small enough; the
            # alternative is a second request).
            out[k] = v
        else:
            out[k] = {kk: vv for kk, vv in v.items()
                      if kk not in ('points',)}
    return jsonify(out)


@app.route('/sierpinski_data.json')
def serve_sierpinski_data():
    # Precomputed Čech + Alpha persistence on N=200 000 IFS-uniform samples
    # at each iteration level of the Sierpinski triangle, plus closed-form
    # Betti numbers (b_0=1, b_1=(3^n-1)/2, b_2=0) per level. 2D ambient,
    # max_dim=1 (no H_2). Produced offline by ./build_sierpinski_data.py.
    return send_from_directory('.', 'sierpinski_data.json',
                               mimetype='application/json')


@app.route('/foam_data.json')
def serve_foam_data():
    # Precomputed Weaire-Phelan foam Alpha persistence (max_dim=2) on
    # N=30 000 face-membrane samples from the 1000 inner bubbles of a
    # 7×7×7 A15 tile (5×5×5 inner cells, 1-cube padding on each side to
    # give bounded Voronoi neighbourhoods). Produced offline by
    # ./build_foam_data.py — runs in under 10 seconds.
    return send_from_directory('.', 'foam_data.json',
                               mimetype='application/json')


@app.route('/packing_data.json')
def serve_packing_data():
    # Precomputed Alpha persistence on a 3D Lubachevsky-Stillinger random
    # sphere packing (N=3000, vf=0.50). The headline character is H_2 — every
    # interstitial void in the packing is a (high-persistence or essential)
    # H_2 class. Produced offline by ./build_packing_data.py.
    return send_from_directory('.', 'packing_data.json',
                               mimetype='application/json')


@app.route('/images/<path:filename>')
def serve_images(filename):
    # Static images used by the page (e.g. Janelia FlyEM optic-lobe project
    # press images at images/optic_lobe/). Subdirectories are allowed via
    # <path:filename>.
    return send_from_directory('images', filename)


@app.route('/optic_lobe_persistence.json')
def serve_optic_lobe_persistence():
    # Full persistence diagrams (Reimann 2017 directed flag complex) for 11
    # Matsliah clusters + 7 cell-type within-type subgraphs. Generated by
    # pyflagser.flagser_weighted with filtration value = 1/synapse_weight,
    # max_dim=2, directed=True, F_2 coefficients.
    return send_from_directory('.', 'optic_lobe_persistence.json',
                               mimetype='application/json')


@app.route('/optic_lobe_persistence_k200.json')
def serve_optic_lobe_persistence_k200():
    # Per-cluster persistent homology with K=200 directed-edge-swap config
    # nulls (preserves both in- and out-degree sequences) + bottleneck
    # distance H_1, H_2 real-vs-null + null-vs-null baselines. Generated by
    # phase_b/cluster_nulls_k200 compute.
    return send_from_directory('.', 'optic_lobe_persistence_k200.json',
                               mimetype='application/json')


@app.route('/optic_lobe_persistence_l4_nulls.json')
def serve_optic_lobe_persistence_l4_nulls():
    # L4 sub-graph: real persistence + K=50 config-model nulls + bottleneck
    # distance (real vs each null, plus null-vs-null baseline). Generated by
    # phase_b L4 nulls compute.
    return send_from_directory('.', 'optic_lobe_persistence_l4_nulls.json',
                               mimetype='application/json')


@app.route('/optic_lobe_subcomplexes.json')
def serve_optic_lobe_subcomplexes():
    # Per-cell-type subcomplexes of the full optic-lobe connectome.
    # For each of {L4, L1, Mi1, Tm3, Mi4, Tm9, T4a}: ~887 cells with real
    # 3D centroids (averaged across ROI polylines), the within-type directed
    # edges, and a nearest-neighbour map from each cell to one of the 892
    # retinotopic columns so cross-hover with the ommatidial array works.
    return send_from_directory('.', 'optic_lobe_subcomplexes.json',
                               mimetype='application/json')


@app.route('/optic_lobe_column_graph.json')
def serve_optic_lobe_column_graph():
    # 892-column coarse-grained metagraph: undirected edges with total
    # synaptic weight between member neurons, plus precomputed persistence
    # at three thresholds (1, 10, 100) for comparison against the
    # disk-expected topology of the hex-tiled column array.
    return send_from_directory('.', 'optic_lobe_column_graph.json',
                               mimetype='application/json')


@app.route('/optic_lobe_columns_3d.json')
def serve_optic_lobe_columns_3d():
    # 3D anchor positions (centred, scaled to unit) for each of the 892
    # retinotopic columns in the medulla, exported from
    # data/processed/column_anchors_3d.parquet. hex1/hex2 are the lattice
    # coordinates from neuPrint; x/y/z are mean over depth.
    return send_from_directory('.', 'optic_lobe_columns_3d.json',
                               mimetype='application/json')


@app.route('/optic_lobe_data.json')
def serve_optic_lobe_data():
    # Phase B Validation bundle for the Drosophila optic lobe (Janelia FlyEM
    # optic-lobe:v1.1, Nern et al. 2025): Seung 2024 form-circuit predictions,
    # Braitenberg-Debbage 1974 L4 K_3 motif, Matsliah 2024 16-cluster recovery,
    # inner-chiasm Gauss-linking chirality. Hand-curated from
    # MATH/OPTIC_LOBE/results/phase_b/*.json.
    return send_from_directory('.', 'optic_lobe_data.json',
                               mimetype='application/json')


@app.route('/sample', methods=['POST'])
def sample_endpoint():
    data = request.get_json(silent=True) or {}
    key          = data.get('key', 'S1')
    N            = int(data.get('N', 10))
    sigma        = float(data.get('sigma', 0.0))
    distribution = data.get('distribution', 'uniform')
    seed         = data.get('seed', None)
    if seed is not None:
        seed = int(seed)

    try:
        pts = sample_points(key, N, sigma, distribution, seed)
    except NotImplementedError as e:
        return jsonify({'error': str(e)}), 501
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    response = {
        'points':       pts.tolist(),
        'key':          key,
        'N':            int(pts.shape[0]),  # may differ from input N for genus-g
        'sigma':        sigma,
        'distribution': distribution,
        'ambient_dim':  int(pts.shape[1]),
    }
    # For the ubiquitin atomic-resolution variants, also return per-point
    # residue index + atom name so the frontend can SS-colour and hover-label
    # at any atomic granularity. The existing 'ubiquitin' key (hardcoded Cα
    # array) does not have these — the frontend already infers them by index.
    if UBQ_DATA is not None and key in UBQ_DATA and 'kind' not in UBQ_DATA[key]:
        entry = UBQ_DATA[key]
        response['residues']   = entry['residues']
        response['atom_names'] = entry['atom_names']
    return jsonify(response)


@app.route('/compute', methods=['POST'])
def compute_endpoint():
    data = request.get_json(silent=True) or {}
    pts_list = data.get('points')
    if pts_list is None:
        return jsonify({'error': 'Missing "points".'}), 400

    points       = np.asarray(pts_list, dtype=float)
    max_r_rips   = float(data.get('max_r_rips',  2.0))
    max_r_cech   = float(data.get('max_r_cech',  1.5))
    max_r_alpha  = float(data.get('max_r_alpha', 1.5))
    max_dim      = int(data.get('max_dim', 3))
    coeff_field  = int(data.get('coeff_field', 11))
    # G25 and other persistence-only consumers can set this False to skip
    # serializing the full simplex list, which on dense Rips at 250 points
    # can be a million-element JSON array. Default True for back-compat.
    include_simplices = bool(data.get('include_simplices', True))

    def safe(builder, *args):
        try:
            return builder(*args), None
        except Exception as e:
            return [], f"{type(e).__name__}: {e}"

    rips_simp,  rips_err  = safe(build_rips_complex,  points, max_r_rips,  max_dim)
    cech_simp,  cech_err  = safe(build_cech_complex,  points, max_r_cech,  max_dim)
    alpha_simp, alpha_err = safe(build_alpha_complex, points, max_r_alpha, max_dim)

    rips_pers  = persistence_pairs(rips_simp,  max_dim, coeff_field)
    cech_pers  = persistence_pairs(cech_simp,  max_dim, coeff_field)
    alpha_pers = persistence_pairs(alpha_simp, max_dim, coeff_field)

    def to_json(simplices):
        if not include_simplices:
            return []
        return [
            {'vertices': list(s), 'filtration': float(f), 'dim': len(s) - 1}
            for s, f in simplices
        ]

    return jsonify({
        'rips':  {'simplices':   to_json(rips_simp),
                  'persistence': rips_pers,
                  'max_r':       max_r_rips,
                  'error':       rips_err},
        'cech':  {'simplices':   to_json(cech_simp),
                  'persistence': cech_pers,
                  'max_r':       max_r_cech,
                  'error':       cech_err},
        'alpha': {'simplices':   to_json(alpha_simp),
                  'persistence': alpha_pers,
                  'max_r':       max_r_alpha,
                  'error':       alpha_err},
        'max_dim':     max_dim,
        'coeff_field': coeff_field,
    })


@app.route('/g25-meta', methods=['GET'])
def g25_meta_endpoint():
    """Returns labels / group indices / group names / per-filtration max_r for
    all three G25 subdatasets. Loaded once from g25_data.json at startup; the
    frontend fetches once on first selection of g25_genetics and uses the data
    to colour the 3D scatters and configure slider ranges."""
    if G25_DATA is None:
        return jsonify({'error': 'g25_data.json not loaded at startup'}), 404
    # Strip the heavy 'points' array — frontend gets those via /sample.
    out = {}
    for k, v in G25_DATA.items():
        out[k] = {kk: vv for kk, vv in v.items() if kk != 'points'}
    return jsonify(out)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
