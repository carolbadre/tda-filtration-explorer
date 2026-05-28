"""
build_hk97_data.py — offline TDA precompute for the HK97 bacteriophage
                      mature capsid (PDB 1OHG, biological assembly 1).

What this script does
=====================
1. Downloads 1OHG.pdb1.gz from RCSB (the .pdb1 biological-assembly file
   contains the asymmetric unit *plus* the REMARK 350 BIOMT operators that
   expand it 60-fold by icosahedral symmetry).
2. Parses the file with a small hand-rolled PDB reader (no biopython needed):
   extracts the Cα coordinates of the asymmetric unit, plus the BIOMT
   rotation/translation operators.
3. Applies the 60 BIOMT operators to the 7-chain asymmetric unit, yielding
   ~118 400 Cα atoms (≈420 subunits, ≈282 residues each) — the full mature
   T=7 icosahedral capsid.
4. Runs three GUDHI filtrations:
     - Alpha     on the full point cloud, max_alpha_radius = 30 Å,  max_dim = 2.
     - Rips      on a 10 000-point maxmin landmark subset.
     - Delaunay-Čech on the same landmark subset.
   (Full-scale Rips and Čech on 118 k points would be memory-hostile; the
   maxmin subset preserves the global topology and lets all three filtrations
   finish overnight on a workstation.)
5. After every phase, rewrites hk97_data.json with whatever's been computed
   so far — so a crash on (say) Rips still leaves you a usable file with
   alpha persistence in it.

Output
======
v4/hk97_data.json with the following shape:

  {
    "pdb_id":         "1OHG",
    "n_subunits":     420,                    # subunits in the full capsid
    "n_atoms":        118440,                 # Cα atoms after BIOMT expansion
    "asu_n_chains":   7,                      # chains in the asymmetric unit
    "asu_n_atoms":    1974,
    "n_biomt":        60,                     # icosahedral expansion factor
    "landmark_k":     10000,
    "coords":         [[x,y,z], ...],         # full Cα coords (Å)
    "subunit_idx":    [int, ...],             # per-atom: which BIOMT op produced it
    "asu_chain_idx":  [int, ...],             # per-asu-atom: which chain in the asu
    "landmark_indices": [int, ...],           # indices into coords[] used for rips/cech
    "alpha":          {"max_r": 30.0, "persistence": [...]},
    "rips":           {"max_r": 30.0, "persistence": [...]},
    "cech":           {"max_r": 30.0, "persistence": [...]},
    "compute_log":    [{"phase": "alpha", "elapsed_s": 3214.7}, ...],
  }

Persistence entries match the live /compute endpoint's schema:
  {"dim": int, "birth": float, "death": float|null, "birth_edge"?: [i, j]}
(birth_edge is present on H_1 entries; non-H_1 entries omit the field.)

Usage
=====
  cd v4/
  python build_hk97_data.py            # writes hk97_data.json in cwd

Optional environment overrides (mostly for testing the script before
committing to the overnight run):
  HK97_LANDMARK_K   — landmark count (default 10000; try 500 for a smoke test)
  HK97_MAX_R_ALPHA  — alpha max radius in Å  (default 30.0)
  HK97_MAX_R_RIPS   — rips  max radius in Å  (default 30.0)
  HK97_MAX_R_CECH   — čech  max radius in Å  (default 30.0)
  HK97_SKIP_ALPHA   — set to "1" to skip the full-scale alpha computation
  HK97_SKIP_RIPS    — set to "1" to skip the landmark rips
  HK97_SKIP_CECH    — set to "1" to skip the landmark čech
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import gudhi as gd


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE         = Path(__file__).parent
PDB_URL      = "https://files.rcsb.org/download/1OHG.pdb1.gz"
LOCAL_PDB_GZ = HERE / "1OHG.pdb1.gz"
LOCAL_PDB    = HERE / "1OHG.pdb1"
OUTPUT       = HERE / "hk97_data.json"

MAX_DIM       = 2
ALPHA_MAX_R   = float(os.environ.get("HK97_MAX_R_ALPHA", 30.0))
RIPS_MAX_R    = float(os.environ.get("HK97_MAX_R_RIPS",  30.0))
CECH_MAX_R    = float(os.environ.get("HK97_MAX_R_CECH",  30.0))
LANDMARK_K    = int  (os.environ.get("HK97_LANDMARK_K",  10000))
SKIP_ALPHA    = os.environ.get("HK97_SKIP_ALPHA") == "1"
SKIP_RIPS     = os.environ.get("HK97_SKIP_RIPS")  == "1"
SKIP_CECH     = os.environ.get("HK97_SKIP_CECH")  == "1"
SKIP_CHAINMAIL = os.environ.get("HK97_SKIP_CHAINMAIL") == "1"

CHAINMAIL_MAX_R = float(os.environ.get("HK97_CHAINMAIL_MAX_R", 80.0))


# ----------------------------------------------------------------------------
# Step 1: download the PDB biological-assembly file
# ----------------------------------------------------------------------------
def download_pdb() -> None:
    """Fetch 1OHG.pdb1.gz from RCSB if we don't already have it locally."""
    if LOCAL_PDB.exists():
        print(f"[1/5] using existing {LOCAL_PDB.name}")
        return
    if not LOCAL_PDB_GZ.exists():
        print(f"[1/5] downloading {PDB_URL} ...")
        t0 = time.time()
        with urllib.request.urlopen(PDB_URL, timeout=120) as resp:
            data = resp.read()
        LOCAL_PDB_GZ.write_bytes(data)
        print(f"      saved {LOCAL_PDB_GZ.name} ({len(data)/1e6:.2f} MB) in {time.time()-t0:.1f}s")
    print(f"      decompressing ...")
    with gzip.open(LOCAL_PDB_GZ, "rb") as f:
        LOCAL_PDB.write_bytes(f.read())
    print(f"      wrote {LOCAL_PDB.name} ({LOCAL_PDB.stat().st_size/1e6:.2f} MB)")


# ----------------------------------------------------------------------------
# Step 2: parse the PDB biological-assembly file. Two layouts are supported:
#
#  (A) Multi-MODEL: the .pdb1 file is structured as 60 consecutive MODEL/ENDMDL
#      blocks, each containing the 7-chain asymmetric unit already positioned
#      by its symmetry operator. This is what RCSB serves for 1OHG.pdb1. The
#      total atom count is therefore 60 × asu_n_atoms; no BIOMT application is
#      needed.
#
#  (B) Single MODEL + REMARK 350 BIOMT: the file contains the asymmetric unit
#      once, plus 60 BIOMT rotation/translation operators in the REMARK 350
#      block. This is the older layout still served for some entries.
#
# REMARK 350 BIOMT line format (column-aligned but space-separated parses fine):
#   REMARK 350   BIOMT1  60 -0.500000  0.309017  0.809017       57.16095
#               ^^^^^^^ ^^^ ^^^^^^^^^  ^^^^^^^^  ^^^^^^^^      ^^^^^^^^^
#               row id  op# matrix row (3 floats)            translation
# Three consecutive rows (BIOMT1, BIOMT2, BIOMT3) with matching op# define one
# 3×3 rotation matrix and a 3-vector translation.
#
# ATOM column layout (PDB v3.3 spec):
#   cols  1- 6 record name            (e.g. "ATOM  ")
#   cols 13-16 atom name              (right-justified, e.g. " CA ")
#   col  22    chain identifier
#   cols 31-38 x coordinate (Å)
#   cols 39-46 y coordinate (Å)
#   cols 47-54 z coordinate (Å)
#
# We slice 0-indexed: line[0:6], line[12:16], line[21:22], line[30:38] etc.

def extract_full_assembly(path: Path):
    """Parse the PDB and return:
        full_coords      (Nfull, 3)  ndarray
        per_atom_subunit (Nfull,)    ndarray  — 0-based subunit/model index
        asu_coords       (Nasu, 3)   ndarray  — one subunit only
        asu_chain_idx    (Nasu,)     ndarray  — 0-based chain index inside the ASU
        n_subunits       int         — number of expanded copies
    Handles both layout (A) and layout (B) above.
    """
    print(f"[2/5] parsing {path.name} ...")
    t0 = time.time()

    # Per-model storage. Model index starts at 0; if no MODEL records appear,
    # everything stays in bucket 0 and we fall back to BIOMT expansion.
    by_model: dict[int, list[tuple[float, float, float, str]]] = {0: []}
    current_model = 0
    saw_model_record = False
    transforms: list[tuple[np.ndarray, np.ndarray]] = []
    rot_rows: list[list[float]] = []
    trans_components: list[float] = []

    with open(path) as fh:
        for line in fh:
            if line.startswith("MODEL"):
                saw_model_record = True
                try:
                    current_model = int(line[10:14]) - 1
                except (ValueError, IndexError):
                    current_model += 1
                by_model.setdefault(current_model, [])
            elif line.startswith("ENDMDL"):
                pass  # current_model is reassigned on next MODEL line
            elif line.startswith("REMARK 350") and "BIOMT" in line:
                parts = line.split()
                tag = parts[2] if len(parts) >= 3 else ""
                if not tag.startswith("BIOMT") or len(tag) != 6:
                    continue
                try:
                    row_id = int(tag[-1])  # 1, 2, or 3
                    r0, r1, r2 = (float(parts[4]), float(parts[5]), float(parts[6]))
                    t          = float(parts[7])
                except (ValueError, IndexError):
                    continue
                rot_rows.append([r0, r1, r2])
                trans_components.append(t)
                if row_id == 3:
                    R = np.array(rot_rows, dtype=float)
                    T = np.array(trans_components, dtype=float)
                    transforms.append((R, T))
                    rot_rows = []
                    trans_components = []
            elif line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                chain = line[21]
                by_model[current_model].append((x, y, z, chain))

    # Decide which layout
    if saw_model_record and len(by_model) > 1:
        # Layout A: multi-MODEL biological assembly.
        model_ids = sorted(by_model.keys())
        print(f"      layout: multi-MODEL ({len(model_ids)} copies in the file)")
        # The asymmetric unit is the first model.
        first = by_model[model_ids[0]]
        asu_coords = np.array([(a[0], a[1], a[2]) for a in first], dtype=float)
        asu_chains = [a[3] for a in first]
        # Pool everything.
        all_xyz, all_model = [], []
        for m_pos, m_id in enumerate(model_ids):
            atoms = by_model[m_id]
            for a in atoms:
                all_xyz.append((a[0], a[1], a[2]))
                all_model.append(m_pos)
        full_coords = np.array(all_xyz, dtype=float)
        per_atom_subunit = np.array(all_model, dtype=int)
        n_subunits = len(model_ids)
    else:
        # Layout B: single ASU + BIOMT expansion.
        asu_atoms = by_model[0]
        if not asu_atoms:
            raise RuntimeError(f"parsed 0 Cα atoms from {path}")
        if not transforms:
            raise RuntimeError(
                f"{path} has neither multiple MODEL records nor BIOMT operators — "
                "is this really a biological-assembly file? Try downloading "
                "https://files.rcsb.org/download/1OHG.pdb1.gz")
        print(f"      layout: ASU + {len(transforms)} BIOMT operators")
        asu_coords = np.array([(a[0], a[1], a[2]) for a in asu_atoms], dtype=float)
        asu_chains = [a[3] for a in asu_atoms]
        chunks, idx_chunks = [], []
        n = len(asu_coords)
        for k, (R, T) in enumerate(transforms):
            chunks.append(asu_coords @ R.T + T)
            idx_chunks.append(np.full(n, k, dtype=int))
        full_coords = np.vstack(chunks)
        per_atom_subunit = np.concatenate(idx_chunks)
        n_subunits = len(transforms)

    # Map ASU chain letters → 0-based indices preserving first-occurrence order.
    seen: dict[str, int] = {}
    asu_chain_idx = np.empty(len(asu_chains), dtype=int)
    for i, c in enumerate(asu_chains):
        if c not in seen:
            seen[c] = len(seen)
        asu_chain_idx[i] = seen[c]

    print(f"      asu: {len(asu_coords)} Cα atoms across {len(seen)} chains")
    print(f"      full assembly: {len(full_coords)} Cα atoms in {n_subunits} subunits")
    print(f"      parse complete in {time.time()-t0:.1f}s")
    return full_coords, per_atom_subunit, asu_coords, asu_chain_idx, n_subunits


# ----------------------------------------------------------------------------
# Step 4: maxmin landmark subsampling
# ----------------------------------------------------------------------------
# Greedy farthest-point sampling. Start with the point closest to centroid,
# then repeatedly pick the point maximally far from the current landmark set.
# Result preserves the cloud's overall shape (especially good for capturing
# the icosahedral curvature with a modest landmark budget).
def maxmin_subsample(coords: np.ndarray, k: int) -> np.ndarray:
    n = len(coords)
    if k >= n:
        return np.arange(n)
    print(f"[4/5] maxmin subsampling {n} → {k} landmarks ...")
    t0 = time.time()
    centroid = coords.mean(0)
    seed = int(np.argmin(np.linalg.norm(coords - centroid, axis=1)))
    landmarks = np.empty(k, dtype=int)
    landmarks[0] = seed
    # min-dist[i] = min over chosen landmarks of distance(i, landmark)
    min_d = np.linalg.norm(coords - coords[seed], axis=1)
    for j in range(1, k):
        i = int(np.argmax(min_d))
        landmarks[j] = i
        d = np.linalg.norm(coords - coords[i], axis=1)
        np.minimum(min_d, d, out=min_d)
        if j % 1000 == 0:
            print(f"      … {j}/{k}")
    print(f"      done in {time.time()-t0:.1f}s")
    return landmarks


# ----------------------------------------------------------------------------
# Step 5: persistence — three filtrations
# ----------------------------------------------------------------------------
def persistence_pairs_from_st(st, *, alpha_squared: bool):
    """Pull persistence pairs out of a fully-built GUDHI SimplexTree.

    If alpha_squared=True, take sqrt of filtration values to convert α² → α
    (matches the convention used by the live /compute endpoint, which
    returns Alpha as radius rather than squared radius).
    """
    st.compute_persistence()
    out = []
    for birth_simplex, death_simplex in st.persistence_pairs():
        dim = len(birth_simplex) - 1
        if dim > MAX_DIM:
            continue
        is_essential = len(death_simplex) == 0
        if dim == MAX_DIM and is_essential:
            continue  # guaranteed artifact of the max_dim cap
        b = float(st.filtration(birth_simplex))
        d = None if is_essential else float(st.filtration(death_simplex))
        if alpha_squared:
            b = math.sqrt(max(b, 0.0))
            if d is not None:
                d = math.sqrt(max(d, 0.0))
        entry = {"dim": dim, "birth": b, "death": d}
        if dim == 1:
            entry["birth_edge"] = [int(v) for v in birth_simplex]
        out.append(entry)
    return out


def run_alpha(coords: np.ndarray, max_r: float):
    print(f"[5/5a] alpha on full cloud, max_r={max_r} Å, {len(coords)} points ...")
    t0 = time.time()
    alpha = gd.AlphaComplex(points=coords.tolist())
    st = alpha.create_simplex_tree(max_alpha_square=max_r ** 2)
    print(f"       simplex tree: {st.num_simplices()} simplices ({time.time()-t0:.1f}s)")
    t1 = time.time()
    pers = persistence_pairs_from_st(st, alpha_squared=True)
    print(f"       persistence: {len(pers)} pairs ({time.time()-t1:.1f}s)")
    return pers, time.time() - t0


def run_rips(coords: np.ndarray, max_r: float):
    print(f"[5/5b] rips on landmarks, max_r={max_r} Å, {len(coords)} points ...")
    t0 = time.time()
    rips = gd.RipsComplex(points=coords.tolist(), max_edge_length=max_r)
    st = rips.create_simplex_tree(max_dimension=MAX_DIM)
    print(f"       simplex tree: {st.num_simplices()} simplices ({time.time()-t0:.1f}s)")
    # Extract the edge + triangle filtrations (1- and 2-simplices) for the
    # live VR viz: sorted by birth, returned as flat arrays so the frontend
    # can use BufferGeometry.drawRange to show only the prefix with birth ≤ r.
    # Caps keep the JSON size and browser draw cost reasonable.
    EDGE_CAP, TRI_CAP = 150000, 200000
    t1 = time.time()
    edges, triangles = [], []
    for s, f in st.get_filtration():
        if len(s) == 2:
            edges.append((int(s[0]), int(s[1]), float(f)))
        elif len(s) == 3:
            triangles.append((int(s[0]), int(s[1]), int(s[2]), float(f)))
    edges.sort(key=lambda e: e[2])
    triangles.sort(key=lambda t: t[3])
    if len(edges) > EDGE_CAP:
        print(f"       (capping {len(edges)} edges to first {EDGE_CAP} by birth)")
        edges = edges[:EDGE_CAP]
    if len(triangles) > TRI_CAP:
        print(f"       (capping {len(triangles)} triangles to first {TRI_CAP} by birth)")
        triangles = triangles[:TRI_CAP]
    edges_flat = []
    for i, j, b in edges:
        edges_flat.extend([i, j, round(b, 4)])
    tris_flat = []
    for i, j, k, b in triangles:
        tris_flat.extend([i, j, k, round(b, 4)])
    print(f"       edges: {len(edges)}, triangles: {len(triangles)} ({time.time()-t1:.1f}s)")
    t2 = time.time()
    pers = persistence_pairs_from_st(st, alpha_squared=False)
    print(f"       persistence: {len(pers)} pairs ({time.time()-t2:.1f}s)")
    return pers, edges_flat, tris_flat, time.time() - t0


def run_cech(coords: np.ndarray, max_r: float):
    print(f"[5/5c] delaunay-čech on landmarks, max_r={max_r} Å, {len(coords)} points ...")
    t0 = time.time()
    dc = gd.DelaunayCechComplex(points=coords.tolist())
    st = dc.create_simplex_tree(
        max_alpha_square=max_r ** 2,
        output_squared_values=False,
    )
    print(f"       simplex tree: {st.num_simplices()} simplices ({time.time()-t0:.1f}s)")
    t1 = time.time()
    pers = persistence_pairs_from_st(st, alpha_squared=False)
    print(f"       persistence: {len(pers)} pairs ({time.time()-t1:.1f}s)")
    return pers, time.time() - t0


def run_chainmail(coords: np.ndarray, biomt_idx: np.ndarray, asu_chain_idx: np.ndarray,
                  max_r: float):
    """Coarse-grain to one centroid per chain (BIOMT op × asu_chain), then run
    Rips on the resulting ~420 points. At this resolution the 60 hexamers + 12
    pentamers (= 72 capsomer rings) show up as a clean cluster of H_1 bars
    matching β_1 exactly — the chainmail topology that the atomic-resolution
    Rips on maxmin landmarks washes out under intra-capsomer noise.
    """
    print(f"[5/5d] chainmail centroid Rips, max_r={max_r} Å ...")
    t0 = time.time()
    # Tile asu_chain_idx across the BIOMT copies and compute per-atom chain id.
    n_biomt = int(biomt_idx.max() + 1)
    asu_n   = len(asu_chain_idx)
    full_chain = biomt_idx * 7 + np.tile(asu_chain_idx, n_biomt)
    n_chains = int(full_chain.max() + 1)
    centroids = np.zeros((n_chains, 3))
    chain_biomt = np.zeros(n_chains, dtype=int)
    chain_in_asu = np.zeros(n_chains, dtype=int)
    for c in range(n_chains):
        m = full_chain == c
        centroids[c] = coords[m].mean(axis=0)
        chain_biomt[c] = biomt_idx[m][0]
        chain_in_asu[c] = (c % 7)
    print(f"       {n_chains} chain centroids built ({time.time()-t0:.1f}s)")

    t1 = time.time()
    rips = gd.RipsComplex(points=centroids.tolist(), max_edge_length=max_r)
    st = rips.create_simplex_tree(max_dimension=MAX_DIM)
    print(f"       simplex tree: {st.num_simplices()} simplices ({time.time()-t1:.1f}s)")
    t2 = time.time()
    pers = persistence_pairs_from_st(st, alpha_squared=False)
    print(f"       persistence: {len(pers)} pairs ({time.time()-t2:.1f}s)")

    return {
        "max_r":        max_r,
        "n_chains":     n_chains,
        "centroids":    centroids.tolist(),
        "chain_biomt":  chain_biomt.tolist(),
        "chain_in_asu": chain_in_asu.tolist(),
        "persistence":  pers,
    }, time.time() - t0


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def write_output(payload: dict) -> None:
    """Atomic-ish JSON write: write to .tmp, then rename. Doesn't truncate the
    existing file if the JSON encode itself raises."""
    tmp = OUTPUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(OUTPUT)
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"       wrote {OUTPUT.name} ({size_mb:.2f} MB)")


def main() -> int:
    overall_t0 = time.time()

    download_pdb()
    try:
        full_coords, per_atom_subunit, asu_coords, asu_chain_idx, n_subunits = \
            extract_full_assembly(LOCAL_PDB)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if len(full_coords) == 0:
        print("ERROR: parsed 0 Cα atoms — is the file a valid PDB?", file=sys.stderr)
        return 2

    landmark_idx = maxmin_subsample(full_coords, LANDMARK_K)
    lm_coords    = full_coords[landmark_idx]

    payload = {
        "pdb_id":            "1OHG",
        "n_subunits":        int(n_subunits),
        "n_atoms":           int(len(full_coords)),
        "asu_n_chains":      int(len(set(asu_chain_idx.tolist()))),
        "asu_n_atoms":       int(len(asu_coords)),
        "n_biomt":           int(n_subunits),    # display alias — see comment
        "landmark_k":        int(len(landmark_idx)),
        "coords":            full_coords.tolist(),
        "subunit_idx":       per_atom_subunit.tolist(),
        "asu_chain_idx":     asu_chain_idx.tolist(),
        "landmark_indices":  landmark_idx.tolist(),
        "alpha":             None,
        "rips":              None,
        "cech":              None,
        "compute_log":       [],
        "config": {
            "max_dim":       MAX_DIM,
            "alpha_max_r":   ALPHA_MAX_R,
            "rips_max_r":    RIPS_MAX_R,
            "cech_max_r":    CECH_MAX_R,
        },
    }
    # Note on n_biomt: for the multi-MODEL layout there are no BIOMT operators
    # per se — the file ships the 60 copies explicitly. The frontend displays
    # "ASU × n_biomt" in the overview tag, so we set n_biomt = n_subunits to
    # keep the tag accurate in either layout.
    write_output(payload)   # baseline file with coords but no persistence yet

    if not SKIP_ALPHA:
        pers, elapsed = run_alpha(full_coords, ALPHA_MAX_R)
        payload["alpha"] = {"max_r": ALPHA_MAX_R, "persistence": pers}
        payload["compute_log"].append({"phase": "alpha", "elapsed_s": round(elapsed, 1)})
        write_output(payload)
    else:
        print("       (SKIP_ALPHA set; skipping)")

    if not SKIP_RIPS:
        pers, edges_flat, tris_flat, elapsed = run_rips(lm_coords, RIPS_MAX_R)
        payload["rips"] = {"max_r": RIPS_MAX_R, "persistence": pers}
        # Edge + triangle filtrations on the landmarks for the live VR viz.
        # edges_flat: [i, j, birth_r, …] sorted by birth_r.
        # tris_flat:  [i, j, k, birth_r, …] sorted by birth_r.
        # Frontend walks a prefix of each up to the slider's r.
        payload["rips_edges"] = {
            "max_r":   RIPS_MAX_R,
            "n_edges": len(edges_flat) // 3,
            "flat":    edges_flat,
        }
        payload["rips_triangles"] = {
            "max_r":       RIPS_MAX_R,
            "n_triangles": len(tris_flat) // 4,
            "flat":        tris_flat,
        }
        payload["compute_log"].append({"phase": "rips", "elapsed_s": round(elapsed, 1)})
        write_output(payload)
    else:
        print("       (SKIP_RIPS set; skipping)")

    if not SKIP_CECH:
        pers, elapsed = run_cech(lm_coords, CECH_MAX_R)
        payload["cech"] = {"max_r": CECH_MAX_R, "persistence": pers}
        payload["compute_log"].append({"phase": "cech", "elapsed_s": round(elapsed, 1)})
        write_output(payload)
    else:
        print("       (SKIP_CECH set; skipping)")

    if not SKIP_CHAINMAIL:
        chain_data, elapsed = run_chainmail(
            full_coords, per_atom_subunit, asu_chain_idx, CHAINMAIL_MAX_R)
        payload["chainmail"] = chain_data
        payload["compute_log"].append({"phase": "chainmail", "elapsed_s": round(elapsed, 1)})
        write_output(payload)
    else:
        print("       (SKIP_CHAINMAIL set; skipping)")

    total = time.time() - overall_t0
    print(f"\nALL DONE in {total/60:.1f} min  →  {OUTPUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
