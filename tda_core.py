"""
TDA Filtration Explorer — Pyodide-compatible core
==================================================

Replaces server.py's compute-heavy endpoints (/sample, /compute) with
pure NumPy + SciPy + Python code that runs in the browser via Pyodide.

Two public entry points consumed from the frontend (index.html → JS):
  sample(key, N, sigma, distribution, seed)
      → dict matching the original /sample response shape
  compute(points, max_r_rips, max_r_cech, max_r_alpha, max_dim,
          coeff_field=11, include_simplices=True)
      → dict matching the original /compute response shape

Fixed real-world datasets that were embedded in server.py:
  - UBIQUITIN_CA and LOCAL_BUBBLE_XYZ are inlined here (small).
  - G25 and the ubiquitin atomic-resolution variants are loaded
    JSON-side and pushed in via set_fixed_data(key, points) so we don't
    duplicate megabytes of data into this module.

GUDHI is NOT available in Pyodide. The complex builders and persistence
reduction are pure-Python re-implementations validated against GUDHI on a
smoke fixture (see smoke_pyodide.py). They are slower than GUDHI's C++
core; sensible defaults for interactive use are N ≤ ~250 and max_dim ≤ 2.
"""

import json
import math
from itertools import combinations

import numpy as np


# ============================================================================
# Sampling primitives — ported verbatim from server.py
# ============================================================================

def sample_sphere(n, N, rng):
    pts = rng.standard_normal((N, n + 1))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def sample_torus(n, N, rng):
    angles = rng.uniform(0.0, 2.0 * np.pi, (N, n))
    pts = np.zeros((N, 2 * n))
    for i in range(n):
        pts[:, 2 * i]     = np.cos(angles[:, i])
        pts[:, 2 * i + 1] = np.sin(angles[:, i])
    return pts


def sample_rp(n, N, rng):
    s = sample_sphere(n, N, rng)
    d = n + 1
    iu = np.triu_indices(d)
    out = np.zeros((N, len(iu[0])))
    for k in range(N):
        out[k] = np.outer(s[k], s[k])[iu]
    return out


def sample_cp(n, N, rng):
    s = sample_sphere(2 * n + 1, N, rng)
    z = s[:, ::2] + 1j * s[:, 1::2]
    d = n + 1
    iu, ju = np.triu_indices(d, k=1)
    out = np.zeros((N, d * d))
    for k in range(N):
        M = np.outer(z[k], np.conj(z[k]))
        out[k, :d]                  = np.real(np.diag(M))
        out[k, d : d + len(iu)]     = np.real(M[iu, ju])
        out[k, d + len(iu):]        = np.imag(M[iu, ju])
    return out


def sample_klein(N, rng):
    u = rng.uniform(0.0, 2.0 * np.pi, N)
    v = rng.uniform(0.0, 2.0 * np.pi, N)
    cu, su = np.cos(u), np.sin(u)
    cu2, su2 = np.cos(u / 2), np.sin(u / 2)
    cv, sv = np.cos(v), np.sin(v)
    radial = cv + 2.0
    return np.column_stack([cu * radial, su * radial, sv * cu2, sv * su2])


def sample_mobius(N, rng):
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
    R = loop_diam / 2.0
    frames = []
    for i in range(g):
        ang = np.pi * i / g
        u_i = np.array([0.0, 0.0, 1.0])
        v_i = np.array([np.cos(ang), np.sin(ang), 0.0])
        w_i = np.cross(u_i, v_i)
        frames.append((u_i, v_i, w_i))

    def dist_to_loop(pts, i):
        u_i, v_i, w_i = frames[i]
        d_w = pts @ w_i
        in_plane = pts - d_w[:, None] * w_i
        center = R * v_i
        d_in = np.linalg.norm(in_plane - center, axis=1)
        d_to_circle = np.abs(d_in - R)
        return np.sqrt(d_to_circle ** 2 + d_w ** 2)

    oversample = max(3, g + 1)
    per_loop_initial = max(N * oversample // g, 50)
    survivors = []

    for i in range(g):
        u_i, v_i, w_i = frames[i]
        theta = rng.uniform(0.0, 2.0 * np.pi, per_loop_initial)
        phi   = rng.uniform(0.0, 2.0 * np.pi, per_loop_initial)
        c = R * (1.0 - np.cos(theta))[:, None] * v_i \
          + R * np.sin(theta)[:, None] * u_i
        t_vec = R * np.sin(theta)[:, None] * v_i \
              + R * np.cos(theta)[:, None] * u_i
        t_norm = np.linalg.norm(t_vec, axis=1, keepdims=True).clip(1e-9)
        t_hat = t_vec / t_norm
        n1 = np.broadcast_to(w_i, (per_loop_initial, 3))
        n2 = np.cross(t_hat, n1)
        surf = c + tube_r * (np.cos(phi)[:, None] * n1
                              + np.sin(phi)[:, None] * n2)

        keep_mask = np.ones(per_loop_initial, dtype=bool)
        for j in range(g):
            if j == i:
                continue
            d_other = dist_to_loop(surf, j)
            keep_mask &= (d_other >= tube_r * 0.97)
        survivors.append(surf[keep_mask])

    pool = np.vstack(survivors)
    if len(pool) == 0:
        raise RuntimeError("genus-g sampler produced no surviving samples")
    if len(pool) >= N:
        idx = rng.choice(len(pool), N, replace=False)
        return pool[idx]
    extra_idx = rng.integers(0, len(pool), N - len(pool))
    return np.vstack([pool, pool[extra_idx]])


def sample_lens(p, q, N, rng):
    s = sample_sphere(3, N, rng)
    z1 = s[:, 0] + 1j * s[:, 1]
    z2 = s[:, 2] + 1j * s[:, 3]
    inv1 = np.abs(z1) ** 2
    inv2 = z1 ** p
    inv3 = z2 ** p
    inv4 = z1 ** q * np.conj(z2)
    return np.column_stack([
        inv1, inv2.real, inv2.imag,
        inv3.real, inv3.imag, inv4.real, inv4.imag,
    ])


PRODUCT_FACTORS = {
    'prod_S2_S1':    ('S2', 'S1'),
    'prod_S2_S2':    ('S2', 'S2'),
    'prod_S1_RP2':   ('S1', 'RP2'),
    'prod_S1_klein': ('S1', 'klein'),
    'prod_RP2_RP2':  ('RP2', 'RP2'),
    'prod_S2_T2':    ('S2', 'T2'),
}


# ============================================================================
# Fixed datasets — small ones inlined, large ones injected from JS
# ============================================================================

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

LOCAL_BUBBLE_XYZ = np.array([
    [-255.0, +101.0, -102.0], [-282.0, +100.0,  -96.0], [-122.0,  +24.0,  -35.0],
    [-151.0,  +24.0,  -43.0], [-128.0,  +17.0,  -38.0], [-123.0,  +13.0,  -34.0],
    [-136.0,  +14.0,  -33.0], [-155.0,  +12.0,  -45.0], [-116.0,   +6.0,  -37.0],
    [-136.0,   +2.0,  -49.0], [-123.0,   +0.0,  -45.0], [-170.0,   -3.0,  -21.0],
    [-109.0,   -7.0,   -9.0], [-318.0, -139.0, -159.0], [-362.0, -170.0, -102.0],
    [-344.0, -172.0, -112.0], [-342.0, -173.0, -119.0], [-326.0, -177.0, -128.0],
    [-324.0, -181.0, -131.0], [-327.0, -203.0, -135.0], [ +84.0, -165.0,  -49.0],
    [ +82.0, -161.0,  -51.0], [ +57.0,  -93.0,  +11.0], [+105.0, -158.0,  -48.0],
    [+109.0,  -61.0,  +29.0], [+145.0,  -63.0,  +23.0], [+146.0,  -54.0,  +25.0],
    [+146.0,  -52.0,  +29.0], [+130.0,  -19.0,  +53.0], [+131.0,  -15.0,  +40.0],
    [+131.0,  -14.0,  +43.0], [+141.0,   -4.0,  -34.0], [+143.0,   +0.0,  -43.0],
    [+341.0, +176.0,  +16.0],
], dtype=float)


# Filled in lazily from the JS side via set_fixed_data(key, points).
# Frontend pushes: g25_ancient, g25_modern, g25_combined, and every
# ubiquitin_xray_/nmr1_/nmr_ensemble_* point-cloud variant on first need.
_FIXED_DATA = {}


def set_fixed_data(key, points):
    """JS calls this to hand a precomputed point cloud over to Python.
    Accepts a JS Array proxy (gets converted) or a Python list/array.
    Returns nothing — avoids a PyProxy leak on the JS side."""
    _FIXED_DATA[key] = np.asarray(points, dtype=float)


def has_fixed_data(key):
    return key in _FIXED_DATA


# ============================================================================
# Top-level samplers
# ============================================================================

def sample_by_key(key, N, rng):
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
        return np.hstack([sample_by_key(k1, N, rng), sample_by_key(k2, N, rng)])
    if key == 'ubiquitin':
        return UBIQUITIN_CA.copy()
    if key == 'local_bubble':
        return LOCAL_BUBBLE_XYZ.copy()
    if key in _FIXED_DATA:
        return _FIXED_DATA[key].copy()
    raise ValueError(f"Unknown manifold key: {key!r} "
                     f"(if it's a g25_*/ubiquitin_* variant, JS must call "
                     f"set_fixed_data first)")


def sample(key, N, sigma, distribution, seed):
    """Top-level entry. Returns the dict shape the frontend's doSample()
    expects from POST /sample."""
    if distribution != 'uniform':
        raise NotImplementedError(f"Distribution '{distribution}' not implemented")
    seed_arg = int(seed) if seed is not None else None
    rng = np.random.default_rng(seed_arg)
    pts = sample_by_key(key, int(N), rng)
    sigma = float(sigma)
    if sigma > 0.0:
        pts = pts + rng.uniform(-sigma, sigma, pts.shape)
    return {
        'points':       pts.tolist(),
        'key':          key,
        'N':            int(pts.shape[0]),
        'sigma':        sigma,
        'distribution': distribution,
        'ambient_dim':  int(pts.shape[1]),
    }


# ============================================================================
# Complex builders — pure Python (GUDHI replacement)
# ============================================================================

def _pairwise_distances(points):
    pts = np.asarray(points, dtype=float)
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff * diff).sum(axis=2))


def build_rips_complex(points, max_edge_length, max_dim=3):
    """Vietoris-Rips. Each k-simplex appears at filtration = max pairwise
    distance among its vertices. Incremental clique expansion: a (k+1)-simplex
    σ ∪ {w} is candidate iff w is a common neighbour of all vertices of σ
    in the 1-skeleton.

    Returns list of (tuple_of_vertex_indices, filtration) ordered exactly the
    way the frontend's chain-complex panel expects: vertices first, then by
    increasing filtration with dimension as tiebreak (we sort at the end).
    """
    pts = np.asarray(points, dtype=float)
    N = len(pts)
    D = _pairwise_distances(pts)

    out = [((i,), 0.0) for i in range(N)]

    # 1-skeleton: edges within max_edge_length
    neighbors = [set() for _ in range(N)]
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            d = D[i, j]
            if d <= max_edge_length:
                edges.append(((i, j), float(d)))
                neighbors[i].add(j)
                neighbors[j].add(i)
    out.extend(edges)
    if max_dim < 2:
        return _sort_simplices(out)

    # Higher dims: incremental clique expansion. current_cliques are tuples
    # sorted ascending; we extend each by a vertex strictly greater than its
    # max, that is a common neighbour of every vertex in the clique.
    current = [c for c, _ in edges]
    for k in range(2, max_dim + 1):
        next_cliques = []
        for clique in current:
            common = neighbors[clique[0]].copy()
            for v in clique[1:]:
                common &= neighbors[v]
            mx = clique[-1]
            for w in common:
                if w > mx:
                    new_clique = clique + (w,)
                    # filtration = max pairwise distance
                    f = 0.0
                    for a in range(len(new_clique)):
                        for b in range(a + 1, len(new_clique)):
                            d = D[new_clique[a], new_clique[b]]
                            if d > f:
                                f = d
                    if f <= max_edge_length:
                        out.append((new_clique, float(f)))
                        next_cliques.append(new_clique)
        current = next_cliques
        if not current:
            break
    return _sort_simplices(out)


def _sort_simplices(simplices):
    """Stable sort by (filtration, dim, vertex-tuple). Matches the order
    persistence reduction needs, and gives the frontend a clean filtration
    progression."""
    return sorted(simplices, key=lambda s: (s[1], len(s[0]), s[0]))


# --- Alpha + Čech (Delaunay-based) ------------------------------------------

def _smallest_enclosing_ball(pts):
    """Smallest enclosing ball of a small set of points (typically ≤ d+1).

    Flat brute-force over subsets: the miniball is the circumsphere of some
    affinely-independent subset whose circumsphere happens to enclose every
    other point. Cost is O(2^n · n · d), fine for n ≤ 5 (4-D Delaunay tops).
    Returns (center, radius).
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n == 1:
        return pts[0].copy(), 0.0
    if n == 2:
        c = 0.5 * (pts[0] + pts[1])
        return c, 0.5 * float(np.linalg.norm(pts[1] - pts[0]))
    best_c, best_r = None, math.inf
    for k in range(2, n + 1):
        for combo in combinations(range(n), k):
            try:
                cc, cr = _circumsphere(pts[list(combo)])
            except np.linalg.LinAlgError:
                continue
            if cr >= best_r:
                continue
            ds = np.linalg.norm(pts - cc, axis=1)
            if np.all(ds <= cr + 1e-9):
                best_c, best_r = cc, cr
    if best_c is None:
        c = pts.mean(axis=0)
        return c, float(np.linalg.norm(pts - c, axis=1).max())
    return best_c, best_r


def _circumsphere(pts):
    """Circumscribed sphere of a k-simplex (k+1 points) in R^d.
    Solves for the unique center equidistant from every vertex lying in the
    affine span. Returns (center, radius). Raises LinAlgError on degeneracy.
    """
    pts = np.asarray(pts, dtype=float)
    n, d = pts.shape
    if n == 1:
        return pts[0].copy(), 0.0
    if n == 2:
        c = 0.5 * (pts[0] + pts[1])
        return c, 0.5 * float(np.linalg.norm(pts[1] - pts[0]))
    # General case: center c minimises max ||c - p_i||² subject to lying in
    # affine span of {p_i}. With c = p_0 + V·t where V is the (k×d) basis of
    # (p_i - p_0), we solve 2·(V·t)·(p_j - p_0) = ||p_j - p_0||² for all j>0.
    p0 = pts[0]
    rel = pts[1:] - p0           # (n-1, d)
    A = 2.0 * rel @ rel.T        # (n-1, n-1)
    b = (rel * rel).sum(axis=1)  # (n-1,)
    t = np.linalg.solve(A, b)    # coefficients in the rel basis
    c = p0 + t @ rel
    r = float(np.linalg.norm(pts[0] - c))
    return c, r


def _ball_cached(pts_all, vertex_tuple, kind, cache):
    """Memoised (center, radius) for either 'alpha' (circumsphere) or 'cech'
    (smallest enclosing ball). cache is a dict keyed by (vertex_tuple, kind)."""
    key = (vertex_tuple, kind)
    hit = cache.get(key)
    if hit is not None:
        return hit
    sub = pts_all[list(vertex_tuple)]
    if kind == 'alpha':
        c, r = _circumsphere(sub)
    else:
        c, r = _smallest_enclosing_ball(sub)
    cache[key] = (c, r)
    return c, r


def _build_delaunay_filtration(points, max_r, max_dim, radius_fn):
    """Shared skeleton for Alpha and Delaunay-Čech.

    radius_fn(simplex_pts) -> r is the per-simplex "intrinsic" radius:
      - Alpha:  circumradius
      - Čech:   smallest-enclosing-ball radius

    GUDHI semantics:
      α(σ) = r(σ)                             if σ is "Gabriel" w.r.t. the
                                              Delaunay point set, i.e. the
                                              ball of radius r(σ) centred at
                                              σ's own ball-center contains no
                                              other Delaunay vertex.
      α(σ) = min{α(τ) : τ ⊃ σ, dim τ = dim σ + 1}   otherwise.
    Top-dim Delaunay simplices are always Gabriel (Delaunay property).
    """
    from scipy.spatial import Delaunay  # imported here so module import works
                                        # in environments without scipy yet.

    pts = np.asarray(points, dtype=float)
    N, D = pts.shape
    if N <= D:
        # Delaunay needs > D points in general position. Degenerate: only
        # vertices.
        return [((i,), 0.0) for i in range(N)]

    try:
        dt = Delaunay(pts)
    except Exception:
        return [((i,), 0.0) for i in range(N)]

    kind = 'alpha' if radius_fn is _circumradius else 'cech'
    ball_cache = {}

    top_simps = [tuple(sorted(int(v) for v in s)) for s in dt.simplices]

    # Enumerate every face and build the immediate-coface map (face → list of
    # simplices one dim higher that contain it).
    all_simps = set()
    for ts in top_simps:
        for k in range(1, D + 2):
            for combo in combinations(ts, k):
                all_simps.add(combo)
    coface_map = {s: [] for s in all_simps}
    for s in all_simps:
        if len(s) >= 2:
            for i in range(len(s)):
                face = s[:i] + s[i + 1:]
                coface_map[face].append(s)

    # Intrinsic per-simplex (center, radius) — memoised so the Gabriel test
    # below can reuse the center without recomputing.
    by_dim = {}  # dim -> list of simplex tuples
    for s in all_simps:
        by_dim.setdefault(len(s) - 1, []).append(s)

    alpha = {}
    # Top-dim cells: always Gabriel (Delaunay property) → take their own r.
    for ts in set(top_simps):
        try:
            _, r = _ball_cached(pts, ts, kind, ball_cache)
            alpha[ts] = float(r)
        except np.linalg.LinAlgError:
            alpha[ts] = math.inf

    # Bottom-up: every sub-simplex inherits min(cofaces); if its own ball is
    # smaller AND empty of other vertices, its intrinsic radius wins.
    for dim in range(D - 1, -1, -1):
        for s in by_dim.get(dim, ()):
            cofaces_alpha = [alpha[cs] for cs in coface_map[s] if cs in alpha]
            inherited = min(cofaces_alpha) if cofaces_alpha else math.inf
            if dim == 0:
                alpha[s] = 0.0
                continue
            try:
                center, own_r = _ball_cached(pts, s, kind, ball_cache)
            except np.linalg.LinAlgError:
                alpha[s] = inherited
                continue
            if own_r < inherited:
                ds = np.linalg.norm(pts - center, axis=1)
                # Empty-ball test: no Delaunay vertex outside s lies strictly
                # inside the ball (radius own_r).
                ok = True
                for v in range(N):
                    if v in s:
                        continue
                    if ds[v] < own_r - 1e-9:
                        ok = False
                        break
                alpha[s] = own_r if ok else inherited
            else:
                alpha[s] = inherited

    out = []
    for s, f in alpha.items():
        if len(s) - 1 > max_dim:
            continue
        if math.isfinite(f) and f <= max_r:
            out.append((s, float(f)))
    return _sort_simplices(out)


def _ball(pts, radius_fn):
    """Return (center, radius) for the simplex's ball under whichever
    convention radius_fn uses."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 1:
        return pts[0].copy(), 0.0
    if radius_fn is _circumradius:
        c, r = _circumsphere(pts)
    else:
        c, r = _smallest_enclosing_ball(pts)
    return c, r


def _circumradius(pts):
    _, r = _circumsphere(pts)
    return r


def _miniball_radius(pts):
    _, r = _smallest_enclosing_ball(pts)
    return r


def build_alpha_complex(points, max_alpha_radius, max_dim=3):
    """Alpha filtration value of σ = circumradius if σ is Gabriel, else
    min over cofaces. Returns radii (not α²) to match server.py's contract."""
    return _build_delaunay_filtration(points, max_alpha_radius, max_dim,
                                       _circumradius)


def build_cech_complex(points, max_radius, max_dim=3):
    """Delaunay-Čech: filtration value of σ = smallest-enclosing-ball radius
    if σ is Gabriel-for-miniball, else min over cofaces."""
    return _build_delaunay_filtration(points, max_radius, max_dim,
                                       _miniball_radius)


# ============================================================================
# Persistent homology — standard reduction over F_p
# ============================================================================

def persistence_pairs(simplices, max_dim, coeff_field=11):
    """Compute persistent homology of the given filtered complex over the
    finite field F_coeff_field (must be prime).

    Returns list of dicts:
      {'dim', 'birth', 'death', 'birth_simplex', 'birth_edge'?}
    death is None for essential classes. Drops dim==max_dim essentials
    (artifacts of capping the simplex tree at max_dim).
    """
    if not simplices:
        return []
    p = int(coeff_field)

    # Index by sorted (filtration, dim, vertices) order.
    sorted_simps = _sort_simplices(simplices)
    n = len(sorted_simps)
    vert_to_idx = {s[0]: i for i, s in enumerate(sorted_simps)}

    # Boundary columns as sparse {row: coeff} dicts.
    cols = [None] * n
    for j, (v, _) in enumerate(sorted_simps):
        dim = len(v) - 1
        if dim == 0:
            cols[j] = {}
            continue
        col = {}
        for i in range(len(v)):
            face = v[:i] + v[i + 1:]
            face_idx = vert_to_idx.get(face)
            if face_idx is None:
                # Should not happen for a valid filtered complex; skip safely.
                continue
            sign = 1 if (i % 2 == 0) else (p - 1)  # (-1)^i mod p
            col[face_idx] = sign
        cols[j] = col

    # Reduction. low_to_col[r] = j  means column j has its lowest 1 at row r.
    low_to_col = {}
    for j in range(n):
        if cols[j] is None:
            cols[j] = {}
        while cols[j]:
            low = max(cols[j])
            if low in low_to_col:
                k = low_to_col[low]
                a = cols[j][low]
                b = cols[k][low]
                # factor = a / b  in F_p
                b_inv = pow(b, p - 2, p)
                factor = (a * b_inv) % p
                # cols[j] -= factor * cols[k]
                for row, val in cols[k].items():
                    new_val = (cols[j].get(row, 0) - factor * val) % p
                    if new_val == 0:
                        cols[j].pop(row, None)
                    else:
                        cols[j][row] = new_val
            else:
                low_to_col[low] = j
                break

    # Extract pairs. Match GUDHI's default min_persistence=0 by dropping
    # zero-persistence (death == birth) finite pairs, but still tracking
    # them as "paired" so they don't reappear as essentials.
    paired_births = set()
    out = []
    for low_row, j in low_to_col.items():
        i = low_row
        s_i = sorted_simps[i]
        s_j = sorted_simps[j]
        dim = len(s_i[0]) - 1
        paired_births.add(i)
        if dim > max_dim:
            continue
        birth = float(s_i[1])
        death = float(s_j[1])
        if death <= birth:
            continue  # zero-persistence — GUDHI drops these too
        entry = {
            'dim':           dim,
            'birth':         birth,
            'death':         death,
            'birth_simplex': list(s_i[0]),
        }
        if dim == 1:
            entry['birth_edge'] = list(s_i[0])
        out.append(entry)

    # Essential classes: positive simplices whose column was reduced to zero
    # and which never appeared as low(k) for any later column.
    for i, (v, f) in enumerate(sorted_simps):
        if i in paired_births:
            continue
        if cols[i]:
            # column non-empty after reduction → "negative", paired upstream
            continue
        dim = len(v) - 1
        if dim > max_dim:
            continue
        if dim == max_dim:
            # essential at the capped dimension = artifact of max_dim cap
            continue
        entry = {
            'dim':           dim,
            'birth':         float(f),
            'death':         None,
            'birth_simplex': list(v),
        }
        if dim == 1:
            entry['birth_edge'] = list(v)
        out.append(entry)

    return out


# ============================================================================
# Top-level /compute entry
# ============================================================================

def compute(points, max_r_rips, max_r_cech, max_r_alpha,
            max_dim, coeff_field=11, include_simplices=True):
    """Returns the dict shape the frontend's doCompute() expects from
    POST /compute. Errors per-complex are caught and reported as the
    'error' field, matching server.py's behaviour."""
    pts = np.asarray(points, dtype=float)
    max_dim     = int(max_dim)
    coeff_field = int(coeff_field)
    max_r_rips  = float(max_r_rips)
    max_r_cech  = float(max_r_cech)
    max_r_alpha = float(max_r_alpha)

    def safe(builder, *args):
        try:
            return builder(*args), None
        except Exception as e:
            return [], f"{type(e).__name__}: {e}"

    rips_simp,  rips_err  = safe(build_rips_complex,  pts, max_r_rips,  max_dim)
    cech_simp,  cech_err  = safe(build_cech_complex,  pts, max_r_cech,  max_dim)
    alpha_simp, alpha_err = safe(build_alpha_complex, pts, max_r_alpha, max_dim)

    def safe_pers(simp):
        try:
            return persistence_pairs(simp, max_dim, coeff_field), None
        except Exception as e:
            return [], f"{type(e).__name__}: {e}"

    rips_pers, _e1  = safe_pers(rips_simp)
    cech_pers, _e2  = safe_pers(cech_simp)
    alpha_pers, _e3 = safe_pers(alpha_simp)

    rips_err  = rips_err  or _e1
    cech_err  = cech_err  or _e2
    alpha_err = alpha_err or _e3

    def to_json(simp):
        if not include_simplices:
            return []
        return [{'vertices': list(s), 'filtration': float(f), 'dim': len(s) - 1}
                for s, f in simp]

    return {
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
    }


# ============================================================================
# JSON wrappers — Pyodide entry points
# ============================================================================
# The JS bridge calls these instead of sample()/compute() directly because the
# default PyProxy → JS conversion turns nested Python dicts into JS Maps
# (where obj.death is undefined, not null). Going through json.dumps on this
# side + JSON.parse on the JS side yields plain nested JS objects with the
# field semantics the frontend expects.

def sample_json(key, N, sigma, distribution, seed):
    return json.dumps(sample(key, N, sigma, distribution, seed))


def compute_json(points, max_r_rips, max_r_cech, max_r_alpha,
                 max_dim, coeff_field=11, include_simplices=True):
    return json.dumps(compute(points, max_r_rips, max_r_cech, max_r_alpha,
                              max_dim, coeff_field, include_simplices))
