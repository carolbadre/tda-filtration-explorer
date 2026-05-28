"""
build_ubiquitin_data.py — ubiquitin "fishing for results" data builder.

What this produces (-> ubiquitin_data.json):

  Atomic-resolution variants of two ubiquitin structures, each cut into many
  point-cloud subsets:

    X-ray  (1UBQ, single model, 1.8 Å resolution, no H deposited):
      ubiquitin_xray_ca               76 Cα  (the existing "ubiquitin" entry,
                                              kept here for parity / regression)
      ubiquitin_xray_backbone        304 N, Cα, C, O atoms
      ubiquitin_xray_cb              152 Cα + Cβ (Gly has no Cβ, see code)
      ubiquitin_xray_polar           ~230 polar atoms (N, O), H-bond network
      ubiquitin_xray_sidechain        76 residue side-chain centroids
      ubiquitin_xray_residue          76 residue centroids (all heavy atoms)
      ubiquitin_xray_heavy           602 all non-H atoms
      ubiquitin_xray_allatom         602 (no H deposited in 1UBQ)

    NMR (1D3Z, 10 deposited models, with H):
      Same 8 subsets as above for model 1 (..._nmr1_*)
      Same 8 subsets pooled across all 10 models (..._nmr_ensemble_*)
      ubiquitin_nmr1_allatom         ~1231 atoms (heavy + H) for model 1
      ubiquitin_nmr_ensemble_allatom ~12310 atoms pooled

  Experimental-comparison data (NOT topology-of-spectrum; rather, data shapes
  that ARE topology-comparable with the structural TDA output):
      __nmr                          NMR distance restraints parsed from
                                     1D3Z.mr (real experimental data):
                                       - NOEs aggregated per residue-pair
                                         (count + shortest distance)
                                       - H-bond restraints as a flat list of
                                         distinct β-sheet / α-helix contacts.
                                     This is the densest experimental
                                     residue-pair contact data available for
                                     ubiquitin and aligns most strongly with
                                     TDA H₁ birth edges.
      __contact_ref                  Cα-Cα distance matrix from the 1UBQ
                                     X-ray structure. Geometry derived from
                                     real coordinates, used as the contact-
                                     map backdrop. Not a simulation.

  NO simulated experimental data in this build. To add an XL-MS comparison
  later, parse a literature cross-link list (e.g. published BS3/DSS pair
  list from PRIDE / supplementary material) into __xlms_literature.

Each point-cloud entry stores:
    points    : (n, 3) Å coordinates
    residues  : (n,)   1-based residue index per atom  (-1 = pooled NMR model)
    atom_names: (n,)   atom-name strings (or 'centroid' / 'ensemble')
    diameter  : scalar
    max_r_*   : per-filtration slider caps (calibrated to diameter)
    r_init_*  : initial slider position (β-sheet pairings ~6 Å on Cα tracks)

Each spectrum entry stores:
    grid      : 1D x-axis (m/z or cm⁻¹) or 2D shape descriptor
    spectrum  : 1D intensities, or 2D intensity matrix (row-major)
    persistence_sub : sublevel-set persistence pairs computed via GUDHI
                      CubicalComplex on the (sign-flipped) spectrum
                      (so "births" are peaks)

Run:
    .venv/bin/python build_ubiquitin_data.py
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import gudhi as gd


HERE = Path(__file__).resolve().parent
XRAY = HERE / "1UBQ.pdb"
NMR  = HERE / "1D3Z.pdb"
OUT  = HERE / "ubiquitin_data.json"


# ============================================================================
# PDB parsing — fixed-column ATOM record reader, no biopython dependency
# ============================================================================
#
# PDB columns (1-indexed in spec; we slice 0-indexed):
#   13-16  atom name        line[12:16]
#   18-20  residue name     line[17:20]
#   22     chain id         line[21]
#   23-26  residue seq      line[22:26]
#   31-38  x                line[30:38]
#   39-46  y                line[38:46]
#   47-54  z                line[46:54]
#   77-78  element symbol   line[76:78]
# We DO NOT trust the element column blindly — some PDBs leave it blank;
# in that case the first non-digit char of the atom name is a fine proxy.

def parse_atom_line(line):
    name  = line[12:16].strip()
    res3  = line[17:20].strip()
    chain = line[21:22].strip()
    seq   = int(line[22:26])
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
    elem = line[76:78].strip()
    if not elem:
        # Fall back to first alpha char of the atom name with the leading digit dropped
        for c in name:
            if c.isalpha():
                elem = c.upper()
                break
    return {
        "name": name, "res3": res3, "chain": chain, "seq": seq,
        "xyz": (x, y, z), "element": elem,
    }


def parse_models(pdb_path, expect_models=None):
    """Parse a PDB file into a list of models. Each model is a list of atom
    dicts (as returned by parse_atom_line). If the file has no MODEL records
    (X-ray), the whole file is treated as one model.

    Only ATOM records (chain A, the protein) are kept — no HETATM (waters,
    ligands), no TER. Hydrogens are kept if present; downstream filters drop
    them as needed.
    """
    models = []
    current = None
    saw_model_record = False
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL "):
                saw_model_record = True
                current = []
                continue
            if line.startswith("ENDMDL"):
                if current is not None:
                    models.append(current)
                    current = None
                continue
            if line.startswith("ATOM "):
                rec = parse_atom_line(line)
                if rec["chain"] not in ("A", "", " "):
                    continue
                if current is None:
                    current = []
                current.append(rec)
    if not saw_model_record and current is not None:
        models.append(current)
    if expect_models is not None and len(models) != expect_models:
        raise RuntimeError(f"{pdb_path}: got {len(models)} models, expected {expect_models}")
    return models


# ============================================================================
# Atom-subset selectors — given a model (list of atom dicts), return a subset
# ============================================================================

def select_atoms(model, predicate):
    """Filter atoms by predicate(atom_dict) -> bool, return list of atoms."""
    return [a for a in model if predicate(a)]


def is_backbone(a):
    return a["name"] in ("N", "CA", "C", "O")


def is_ca(a):
    return a["name"] == "CA"


def is_heavy(a):
    return a["element"] != "H"


def is_polar(a):
    # Polar heavy atoms (H-bond donors/acceptors): all N and O, of any flavor.
    return a["element"] in ("N", "O")


def is_cb_or_ca(a):
    return a["name"] in ("CA", "CB")


def is_sidechain_heavy(a):
    # Side chain = everything that isn't backbone N/CA/C/O, excluding H.
    return is_heavy(a) and not is_backbone(a)


def residue_groups(model):
    """Group atoms by residue sequence number, preserving order."""
    groups = {}
    order = []
    for a in model:
        s = a["seq"]
        if s not in groups:
            groups[s] = []
            order.append(s)
        groups[s].append(a)
    return [(s, groups[s]) for s in order]


def centroid_per_residue(model, sub_predicate=None):
    """One point per residue: centroid of all atoms in that residue passing
    sub_predicate (default: all heavy atoms). Returns (points, residues, names).
    """
    if sub_predicate is None:
        sub_predicate = is_heavy
    pts, res, names = [], [], []
    for seq, atoms in residue_groups(model):
        relevant = [a for a in atoms if sub_predicate(a)]
        if not relevant:
            continue
        xyz = np.array([a["xyz"] for a in relevant], dtype=float)
        pts.append(xyz.mean(axis=0))
        res.append(seq)
        names.append("centroid")
    return np.array(pts, dtype=float), np.array(res, dtype=int), np.array(names)


def atoms_to_arrays(atoms):
    """Convert a list of atom dicts to parallel numpy arrays."""
    if not atoms:
        return np.zeros((0, 3)), np.zeros(0, dtype=int), np.array([], dtype=object)
    pts   = np.array([a["xyz"] for a in atoms], dtype=float)
    res   = np.array([a["seq"] for a in atoms], dtype=int)
    names = np.array([a["name"] for a in atoms], dtype=object)
    return pts, res, names


# ============================================================================
# NMR superposition — Kabsch alignment of each model onto model 1 by Cα
# ============================================================================

def kabsch(P, Q):
    """Return rotation R such that P @ R best aligns to Q (centered inputs).
    Standard formula: SVD of P^T Q, R = V D U^T with D = diag(1, 1, sign(det))."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return U @ D @ Vt


def superpose_to_model1(models):
    """Translate each model so its Cα centroid is at origin, then rotate it
    onto model 1's Cα set via Kabsch. Returns a list of new atom-dict lists
    with rotated/translated xyzs. The first model is left unchanged (centred
    at its own Cα centroid)."""
    ca_idx_per_model = []
    ca_xyz_per_model = []
    for m in models:
        ca_atoms = [a for a in m if a["name"] == "CA"]
        ca_idx_per_model.append([m.index(a) for a in ca_atoms])
        ca_xyz_per_model.append(np.array([a["xyz"] for a in ca_atoms], dtype=float))
    ref = ca_xyz_per_model[0]
    ref_centroid = ref.mean(axis=0)
    ref_centered = ref - ref_centroid
    out = []
    for k, m in enumerate(models):
        ca = ca_xyz_per_model[k]
        c = ca.mean(axis=0)
        cac = ca - c
        if k == 0:
            R = np.eye(3)
        else:
            R = kabsch(cac, ref_centered)
        new = []
        for a in m:
            x, y, z = a["xyz"]
            v = np.array([x, y, z]) - c
            v = v @ R + ref_centroid
            b = dict(a)
            b["xyz"] = (float(v[0]), float(v[1]), float(v[2]))
            new.append(b)
        out.append(new)
    return out


# ============================================================================
# Slider calibration — diameter-based, matches the G25 / Menger conventions
# ============================================================================

def slider_settings(pts):
    """Return (max_r_rips, max_r_cech, max_r_alpha, r_init_rips, r_init_cech,
    r_init_alpha, diameter) for a point cloud. Tighter than diameter so
    Rips doesn't fill out C(N,3) triangles."""
    if len(pts) < 2:
        return (1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0)
    # Use a sampled diameter estimate when N is large
    if len(pts) > 4000:
        idx = np.random.default_rng(0).choice(len(pts), 4000, replace=False)
        sub = pts[idx]
    else:
        sub = pts
    diffs = sub[None, :, :] - sub[:, None, :]
    pdist = np.linalg.norm(diffs, axis=-1)
    diameter = float(pdist.max())
    # Different defaults vs G25: on protein backbones we want to *see* up to
    # tertiary contacts (~30 Å for the longest non-bonded contact in
    # ubiquitin) without forcing the slider all the way to diameter.
    max_r_rips  = round(diameter * 0.50, 2)
    max_r_cech  = round(diameter * 0.40, 2)
    max_r_alpha = round(diameter * 0.40, 2)
    r_init_rips  = 6.0   # β-sheet pairings birth here on Cα tracks
    r_init_cech  = 3.0
    r_init_alpha = 3.0
    return (max_r_rips, max_r_cech, max_r_alpha,
            r_init_rips, r_init_cech, r_init_alpha, diameter)


def encode_cloud(pts, res, names, label, description, ss_ranges=None):
    """Standard JSON-serialisable cloud entry."""
    (max_r_rips, max_r_cech, max_r_alpha,
     r_init_rips, r_init_cech, r_init_alpha, diameter) = slider_settings(pts)
    out = {
        "points":       pts.tolist(),
        "residues":     [int(r) for r in res],
        "atom_names":   [str(n) for n in names],
        "n_atoms":      int(len(pts)),
        "diameter":     diameter,
        "max_r_rips":   max_r_rips,
        "max_r_cech":   max_r_cech,
        "max_r_alpha":  max_r_alpha,
        "r_init_rips":  r_init_rips,
        "r_init_cech":  r_init_cech,
        "r_init_alpha": r_init_alpha,
        "label":        label,
        "description":  description,
    }
    if ss_ranges is not None:
        out["ss_ranges"] = ss_ranges
    return out


# ============================================================================
# Ubiquitin secondary structure (from PDB 1UBQ HELIX/SHEET records)
# ============================================================================
# 0-based residue indices into Cα array. Used for SS colouring and the H₁
# classifier — and now for picking local frequencies in the 2D IR simulation.

SS_RANGES = [
    [ 0,  6, "β1"],
    [ 9, 16, "β2"],
    [22, 33, "α"],
    [39, 44, "β3"],
    [47, 49, "β4"],
    [55, 58, "3₁₀"],
    [63, 71, "β5"],
]


def ss_label(residue_idx_0):
    for lo, hi, lab in SS_RANGES:
        if lo <= residue_idx_0 <= hi:
            return lab
    return "coil"


# ============================================================================
# NMR restraints (real experimental data from PDB 1D3Z.mr)
# ============================================================================
#
# 1D3Z.mr is the CNS/X-PLOR-format NMR restraint file from PDB, containing
# the experimental data used to determine the 1D3Z solution structure.
# It has five sections:
#   A. NOE restraints      — 4767 lines, ~4700 unique atom-pair distance bounds
#   B. H-bond restraints   — 27 lines, β-sheet + α-helix backbone H-bonds
#   C. Dihedral restraints — 316 lines, φ/ψ backbone angle bounds
#   D. Dipolar restraints  — 5738 lines, residual dipolar couplings (orientations)
#   E. Scalar couplings    — not used in structure calc
#
# For alignment with TDA H₁ birth edges (which are residue-pair contacts) the
# topologically-comparable data is in sections A and B. We parse both,
# aggregate to per-residue-pair (i, j) contacts, and store separately:
#   noe_pairs    : list of {i, j, count, min_dist}
#                  count = number of NOE restraints contributing to this pair
#                  min_dist = shortest NOE distance among them
#   hbond_pairs  : list of {i, j, distance, donor_atom, acceptor_atom}
#                  one entry per H-bond restraint (these are unique)
# Plus a residue-pair NOE-count matrix for the contact-map heatmap underlay.

# CNS-format restraint lines look like:
#   assign ((resid X and name AT))  ((resid Y and name AT))  D L U !
# or (with disjunction over atoms):
#   assign ((resid X and name AT1) or (resid X and name AT2))
#          ((resid Y and name AT3))  D L U !
# A multi-line assign continues until the trailing "!"; we collect the whole
# block, then regex out all (resid N) tokens to find residue indices and the
# trailing three floats for the distances.

import re as _re

_RESID_RE = _re.compile(r'resid\s+(\d+)')
_NUM_RE   = _re.compile(r'(\d+\.\d+|\d+)')


def parse_assign_block(block_text):
    """Parse one CNS 'assign ...' restraint block. Returns:
        residues_left  : list of int resids in the first group
        residues_right : list of int resids in the second group
        d, l, u        : the three trailing floats (centre, lower, upper bounds)
    A block has exactly two parenthesised groups (left & right atoms) before
    the trailing numbers. The two groups are separated by ')) ((' in flat form,
    but multi-line blocks need group counting by parenthesis depth.
    Returns None if the line doesn't look like a valid distance restraint.
    """
    # Find the position separating the two groups: the last "))" before the
    # opening "((" of the second group. CNS always uses double parens around
    # each atom-set group.
    # Strategy: walk parenthesis depth, split where depth returns to zero
    # after the first group.
    depth = 0
    split_idx = None
    after_assign = block_text.find('assign')
    if after_assign < 0:
        return None
    text = block_text[after_assign + len('assign'):]
    for i, ch in enumerate(text):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                if split_idx is None:
                    split_idx = i
                else:
                    # second group end — we're done with both groups
                    second_end = i
                    break
    else:
        return None
    if split_idx is None:
        return None
    left_text  = text[:split_idx + 1]
    right_text = text[split_idx + 1:second_end + 1]
    after = text[second_end + 1:]
    left_resids  = sorted(set(int(m.group(1)) for m in _RESID_RE.finditer(left_text)))
    right_resids = sorted(set(int(m.group(1)) for m in _RESID_RE.finditer(right_text)))
    if not left_resids or not right_resids:
        return None
    # The three trailing floats — d, l, u
    nums = _NUM_RE.findall(after)
    if len(nums) < 3:
        return None
    d, l, u = float(nums[0]), float(nums[1]), float(nums[2])
    return left_resids, right_resids, d, l, u


def parse_mr_file(path):
    """Read a CNS/X-PLOR .mr file and return {'noe': [...], 'hbond': [...]}.
    Each list element: dict with i, j (1-based residue numbers), d, l, u.
    For ambiguous NOE restraints (multiple residues on a side), we emit one
    entry per (left_resid, right_resid) cross-product pair, but tag them as
    ambiguous so consumers can choose to weight them down.
    """
    with open(path) as f:
        lines = f.readlines()
    # Find section starts: lines beginning with letter + ". "
    section_idx = {}
    for i, line in enumerate(lines):
        m = _re.match(r'^([A-Z])\. ', line)
        if m:
            section_idx[m.group(1)] = i
    sections = {}
    keys = sorted(section_idx, key=lambda k: section_idx[k])
    for k_idx, k in enumerate(keys):
        start = section_idx[k]
        end = section_idx[keys[k_idx + 1]] if k_idx + 1 < len(keys) else len(lines)
        sections[k] = ''.join(lines[start:end])

    def parse_section(s):
        # Split into assign-blocks. Each block starts with 'assign' and ends
        # at the next 'assign' or at a section boundary. We don't trust line
        # boundaries (CNS allows multi-line restraints).
        if not s:
            return []
        parts = _re.split(r'(?=^assign\b)', s, flags=_re.MULTILINE)
        out = []
        for p in parts:
            if not p.strip().startswith('assign'):
                continue
            r = parse_assign_block(p)
            if r is None:
                continue
            L, R, d, l, u = r
            ambig = (len(L) > 1) or (len(R) > 1)
            for i_res in L:
                for j_res in R:
                    if i_res == j_res:
                        continue   # self-restraints aren't pair contacts
                    out.append({
                        "i": i_res, "j": j_res,
                        "d": d, "lower": l, "upper": u,
                        "ambiguous": ambig,
                    })
        return out

    return {
        "noe":   parse_section(sections.get('A', '')),
        "hbond": parse_section(sections.get('B', '')),
    }


def aggregate_pair_restraints(entries, drop_local=False):
    """Aggregate raw restraint entries to per-(i,j)-residue-pair summaries.
       drop_local=True drops |i-j| <= 1 pairs (sequential / same residue).
    """
    by_pair = {}
    for e in entries:
        i, j = sorted([e["i"], e["j"]])
        if drop_local and abs(i - j) <= 1:
            continue
        key = (i, j)
        if key not in by_pair:
            by_pair[key] = {"i": i, "j": j, "count": 0, "min_d": float('inf'),
                            "any_unambiguous": False}
        by_pair[key]["count"] += 1
        if e["d"] < by_pair[key]["min_d"]:
            by_pair[key]["min_d"] = e["d"]
        if not e["ambiguous"]:
            by_pair[key]["any_unambiguous"] = True
    return sorted(by_pair.values(), key=lambda p: (p["i"], p["j"]))


def build_nmr_restraints():
    """Parse 1D3Z.mr and produce the NMR-restraint data block for the
    alignment view. Returns dict with:
        noe_pairs   : aggregated NOE pair counts (long-range only, |i-j|>1)
        noe_all     : aggregated NOE pair counts (all, incl. sequential)
        hbond_pairs : H-bond restraint list (each one is a distinct contact)
        n_raw_noe, n_raw_hbond : raw assign-line counts
        n_residues  : 76
        pair_count_matrix : 76×76 matrix of NOE pair counts (long-range only)
    """
    mr_path = HERE / "1D3Z.mr"
    if not mr_path.exists():
        print(f"WARNING: {mr_path} not found — NMR restraints will be unavailable")
        return None
    parsed = parse_mr_file(mr_path)
    noe_all = aggregate_pair_restraints(parsed["noe"], drop_local=False)
    noe_lr  = aggregate_pair_restraints(parsed["noe"], drop_local=True)
    hbonds  = parsed["hbond"]
    # Aggregate H-bonds to one entry per (i,j) — they're already unique pairs
    # but we present them as a flat list of (i,j,d) with donor/acceptor info.
    hbond_pairs = []
    seen = set()
    for h in hbonds:
        key = (min(h["i"], h["j"]), max(h["i"], h["j"]))
        if key in seen:
            continue
        seen.add(key)
        hbond_pairs.append({"i": h["i"], "j": h["j"], "d": h["d"]})
    # Build NOE count matrix (76×76) for the heatmap underlay
    N = 76
    M = [[0] * N for _ in range(N)]
    for p in noe_lr:
        i_idx = p["i"] - 1
        j_idx = p["j"] - 1
        if 0 <= i_idx < N and 0 <= j_idx < N:
            M[i_idx][j_idx] = p["count"]
            M[j_idx][i_idx] = p["count"]
    return {
        "n_raw_noe":         len(parsed["noe"]),
        "n_raw_hbond":       len(parsed["hbond"]),
        "n_unique_noe_pairs":      len(noe_all),
        "n_unique_noe_pairs_long": len(noe_lr),
        "n_hbond_pairs":     len(hbond_pairs),
        "n_residues":        N,
        "noe_pairs_long":    noe_lr,        # |i-j| > 1, the topologically interesting ones
        "noe_pairs_all":     noe_all,       # for completeness
        "hbond_pairs":       hbond_pairs,
        "pair_count_matrix": M,
        "source":            "PDB 1D3Z.mr (CNS/X-PLOR format) · Cornilescu, Marquardt, Ottiger, Bax (1998)",
    }


# ============================================================================
# Cα-Cα distance matrix — derived from the real X-ray structure (no simulation)
# ============================================================================
# The contact-map view uses this 76×76 matrix as a backdrop so the experimental
# (NMR) signals + the TDA H₁ pairs can be read against actual spatial proximity.

def compute_ca_distance_matrix(model):
    """Cα-Cα distance matrix (76, 76) + 1-based residue indices, computed
    directly from the X-ray Cα coordinates. Not a simulation — just a
    derivation from real structural data."""
    ca_atoms = [a for a in model if a["name"] == "CA"]
    seqs = [a["seq"] for a in ca_atoms]
    coords = np.array([a["xyz"] for a in ca_atoms], dtype=float)
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return D, seqs


def build_contact_ref(model):
    """Backdrop data for the contact-map view: 76×76 Cα-Cα distance matrix
    + residue seqs. This is structural geometry, not a simulation."""
    D, seqs = compute_ca_distance_matrix(model)
    return {
        "ca_distance_matrix": D.tolist(),
        "residue_seqs":       seqs,
    }


# ============================================================================
# Main build
# ============================================================================

def build_subsets_for_model(model, prefix, label_prefix, description_template,
                            ss_ranges=None, single_model=True):
    """Construct the 8 atom-subset clouds for a single ubiquitin model.
    Returns a dict mapping manifold key -> cloud entry.
    """
    out = {}
    subsets = [
        ("ca",       "Cα only",           lambda m: atoms_to_arrays(select_atoms(m, is_ca))),
        ("backbone", "backbone N/Cα/C/O", lambda m: atoms_to_arrays(select_atoms(m, is_backbone))),
        ("cb",       "Cα + Cβ",           lambda m: atoms_to_arrays(select_atoms(m, is_cb_or_ca))),
        ("polar",    "polar N/O",         lambda m: atoms_to_arrays(select_atoms(m, lambda a: is_polar(a) and is_heavy(a)))),
        ("sidechain","side-chain centroids",
            lambda m: centroid_per_residue(m, sub_predicate=is_sidechain_heavy)),
        ("residue",  "residue centroids", lambda m: centroid_per_residue(m, sub_predicate=is_heavy)),
        ("heavy",    "all heavy atoms",   lambda m: atoms_to_arrays(select_atoms(m, is_heavy))),
        ("allatom",  "all atoms incl. H", lambda m: atoms_to_arrays(model)),
    ]
    for suffix, atom_desc, fn in subsets:
        pts, res, names = fn(model)
        n = len(pts)
        if n == 0:
            print(f"  [skip] {prefix}_{suffix}: no atoms")
            continue
        # All-atom and heavy: descriptions differ only when there are no H
        # atoms (X-ray case).
        desc = description_template.format(atom_desc=atom_desc, n=n)
        key = f"{prefix}_{suffix}"
        out[key] = encode_cloud(
            pts, res, names,
            label=f"{label_prefix} · {atom_desc} ({n})",
            description=desc,
            ss_ranges=ss_ranges,
        )
        print(f"  [{key}] n={n}, diameter={out[key]['diameter']:.2f} Å")
    return out


XRAY_DESC_TMPL = (
    "Ubiquitin X-ray crystal structure (PDB 1UBQ, 1.8 Å resolution, deposited 1987 by "
    "Vijay-Kumar, Bugg & Cook), restricted here to {atom_desc} — {n} points. "
    "Compare with the Cα-only baseline to see what richer atomic resolution buys you "
    "topologically: smaller r captures covalent + bond-angle geometry, mid-r captures "
    "side-chain packing and the β-sheet's hydrogen-bond network, and large r recovers "
    "the global β-grasp fold. The X-ray model is a single static snapshot at "
    "crystallographic temperature; for solution structure compare against the matched "
    "NMR ensemble (1D3Z)."
)

NMR1_DESC_TMPL = (
    "Ubiquitin solution NMR model 1 of PDB 1D3Z (Cornilescu, Marquardt, Ottiger & Bax, 1998), "
    "restricted to {atom_desc} — {n} points. This is one of 10 deposited solution-state "
    "conformers refined from RDC and NOE data; comparing its persistence diagram to the "
    "X-ray 1UBQ structure at the same atom subset isolates the X-ray-vs-NMR difference "
    "from the atom-resolution difference. Differences mostly show up in the flexible "
    "C-terminal tail (residues 73–76) and surface side chains."
)

NMR_ENS_DESC_TMPL = (
    "Ubiquitin solution NMR ensemble (PDB 1D3Z, all 10 models pooled after Cα-Kabsch "
    "superposition onto model 1), restricted to {atom_desc} — {n} points across 10 models. "
    "The cloud is a *thickened* version of model 1: each rigid region looks the same, while "
    "flexible regions (C-terminal tail, loops, exposed side chains) appear as small clusters "
    "of nearby points. Persistent homology of this pooled cloud reveals solution flexibility "
    "as additional low-persistence H₁ features at small radii — the famous NMR-ensemble "
    "'breathing' that the X-ray structure does not show."
)


def main():
    print(f"Reading X-ray:  {XRAY}")
    xray_models = parse_models(XRAY, expect_models=1)
    xray = xray_models[0]
    print(f"  X-ray atoms (chain A): {len(xray)}, residues: {len(set(a['seq'] for a in xray))}")

    print(f"\nReading NMR:    {NMR}")
    nmr_models = parse_models(NMR, expect_models=10)
    print(f"  NMR: {len(nmr_models)} models, {len(nmr_models[0])} atoms each "
          f"(includes H)")

    # X-ray subsets
    print("\n[X-ray subsets]")
    data = {}
    data.update(build_subsets_for_model(
        xray, "ubiquitin_xray", "Ubiquitin X-ray (1UBQ)",
        XRAY_DESC_TMPL, ss_ranges=SS_RANGES,
    ))

    # NMR model 1 subsets
    print("\n[NMR model 1 subsets]")
    data.update(build_subsets_for_model(
        nmr_models[0], "ubiquitin_nmr1", "Ubiquitin NMR model 1 (1D3Z m1)",
        NMR1_DESC_TMPL, ss_ranges=SS_RANGES,
    ))

    # NMR pooled ensemble subsets — superpose first
    print("\n[NMR ensemble subsets] superposing 10 models by Cα Kabsch onto model 1")
    aligned = superpose_to_model1(nmr_models)
    # Build a pseudo-model = concatenation of all aligned models' atoms
    pooled = []
    for m in aligned:
        pooled.extend(m)
    # The atom_name labels are the same per residue across models — that's fine.
    data.update(build_subsets_for_model(
        pooled, "ubiquitin_nmr_ensemble", "Ubiquitin NMR ensemble (1D3Z × 10)",
        NMR_ENS_DESC_TMPL, ss_ranges=SS_RANGES,
    ))

    # NMR restraints (real experimental data from 1D3Z.mr)
    print("\n[NMR restraints from 1D3Z.mr]")
    nmr = build_nmr_restraints()
    if nmr is not None:
        print(f"  raw NOE restraints:   {nmr['n_raw_noe']}")
        print(f"  unique NOE pairs:     {nmr['n_unique_noe_pairs']} total, {nmr['n_unique_noe_pairs_long']} long-range (|i-j|>1)")
        print(f"  H-bond restraints:    {nmr['n_hbond_pairs']}")
        print(f"  top-5 NOE long-range pairs by count:")
        for p in sorted(nmr['noe_pairs_long'], key=lambda x: -x['count'])[:5]:
            print(f"    ({p['i']}, {p['j']}): {p['count']} NOEs, min d={p['min_d']:.2f} Å")
        print(f"  H-bond pairs:")
        for p in nmr['hbond_pairs']:
            print(f"    N({p['i']}) → O({p['j']}): {p['d']:.2f} Å")
        data["__nmr"] = {
            "kind":  "nmr_restraints",
            "label": "Ubiquitin NMR restraints (1D3Z.mr)",
            "description": ("Real experimental NMR distance restraints used to "
                            "determine the 1D3Z solution structure (Cornilescu et al. "
                            "1998). NOEs are aggregated per residue-pair (count + "
                            "shortest distance); H-bond restraints are kept as a "
                            "list of distinct β-sheet and α-helix backbone H-bonds. "
                            "These (i, j) pair lists are directly topology-comparable "
                            "to TDA H₁ birth edges from the structural variants."),
            "data":  nmr,
        }

    # Contact-map backdrop — 76×76 Cα-Cα distance matrix from the real X-ray
    # structure. This is the geometric reference the experimental (NMR) and
    # TDA H₁ signals are read against. NOT a simulation; just a derivation
    # from the real coordinates.
    print("\n[Cα-Cα distance reference from X-ray]")
    contact_ref = build_contact_ref(xray)
    print(f"  matrix: {len(contact_ref['ca_distance_matrix'])}×"
          f"{len(contact_ref['ca_distance_matrix'][0])}")
    print(f"  residues: 1..{contact_ref['residue_seqs'][-1]}")
    data["__contact_ref"] = {
        "kind":        "contact_reference",
        "label":       "Ubiquitin Cα-Cα distance reference (PDB 1UBQ)",
        "description": ("76×76 Cα-Cα distance matrix derived from the 1UBQ X-ray "
                        "structure. Backdrop for the NMR-vs-TDA contact-map alignment "
                        "view. Real structural data — no simulation."),
        "data":        contact_ref,
    }

    print(f"\nWriting {OUT}")
    with open(OUT, "w") as f:
        json.dump(data, f)
    size_mb = OUT.stat().st_size / 1e6
    print(f"  size: {size_mb:.1f} MB")
    print(f"  entries: {len(data)}")
    for k in sorted(data):
        kind = data[k].get("kind")
        if kind:
            print(f"    {k}  [{kind}]")
        else:
            print(f"    {k}  n={data[k]['n_atoms']}, diam={data[k]['diameter']:.1f}")


if __name__ == "__main__":
    main()
