"""
build_g25_data.py — generates three G25 landmark datasets for the TDA explorer.

Input:
    /mnt/user-data/uploads/Global25_PCA_scaled.txt          (ancient, scaled)
    /mnt/user-data/uploads/Global25_PCA_modern_scaled.txt   (modern,  scaled)

Output:
    /home/claude/g25_data.json
        Contains three datasets — ancient, modern, combined — each with:
            points: 250 x 3 float array (PC1, PC2, PC3 of maxmin landmarks)
            labels: 250 strings (population/period for colouring)
            groups: 250 ints  (group index for the categorical colour scale)
            group_names: list[str] (legend entries, indexed by group)
            max_r_rips, max_r_cech, max_r_alpha: scalars calibrated to the cloud diameter

Each cloud is maxmin-subsampled in FULL 25D (not 3D) — preserves the topology
we care about — then projected to the first 3 PCs for display + Čech feasibility.
"""

import json
import numpy as np
from pathlib import Path

# ------------------------------------------------------------- I/O ---------
ANCIENT_PATH  = Path("/mnt/user-data/uploads/Global25_PCA_scaled.txt")
MODERN_PATH   = Path("/mnt/user-data/uploads/Global25_PCA_modern_scaled.txt")
OUTPUT_PATH   = Path("/home/claude/g25_data.json")

N_LANDMARKS   = 250
SEED          = 42

# ------------------------------------------------------------- parse -------
def load_g25(path):
    """Parse a Global25 file. First line is the header ',PC1,PC2,...,PC25'.
    Each data row is 'ID,c1,c2,...,c25'. ID is 'Population_Period:Sample'."""
    ids, coords = [], []
    with open(path) as f:
        first = True
        for line in f:
            line = line.strip()
            if not line: continue
            if first:
                first = False
                # Header line: starts with ',PC1' or 'PC1' — skip
                if line.startswith(",") or line.startswith("PC1") or "PC1" in line.split(",")[0:2]:
                    continue
            parts = line.split(",")
            ids.append(parts[0])
            coords.append([float(x) for x in parts[1:26]])
    return ids, np.array(coords, dtype=float)

print("Loading G25 data...")
ancient_ids, ancient_pcs = load_g25(ANCIENT_PATH)
modern_ids,  modern_pcs  = load_g25(MODERN_PATH)
print(f"  ancient: {len(ancient_ids)} samples x {ancient_pcs.shape[1]} PCs")
print(f"  modern:  {len(modern_ids)}  samples x {modern_pcs.shape[1]} PCs")

# ------------------------------------------------------------- maxmin ------
def maxmin_subsample(pts, n, seed=0):
    """Greedy maxmin: pick n landmarks that maximize the minimum pairwise distance.
    Memory-efficient: only carries the running min-distance vector, not full pdist.
    Returns indices into pts."""
    rng = np.random.default_rng(seed)
    N = len(pts)
    chosen = [int(rng.integers(N))]
    # min_dist[i] = min distance from pts[i] to any chosen landmark so far
    min_dist = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(min_dist))
        chosen.append(i)
        # update min_dist by taking element-wise min with distances to new landmark
        new_dist = np.linalg.norm(pts - pts[i], axis=1)
        min_dist = np.minimum(min_dist, new_dist)
    return np.array(chosen)

# ------------------------------------------------------------- labels ------
def parse_id(s):
    """'Albania_BA_IA:I14688' → ('Albania_BA_IA', 'I14688').
    Returns (population, sample). Some IDs lack the ':' — fall back to ID-as-population."""
    if ":" in s:
        pop, sample = s.split(":", 1)
        return pop, sample
    return s, ""

# Continental group inference for modern populations — coarse heuristic.
# Captures the dominant axis of G25 variation (continental separation).
MODERN_GROUPS = [
    ("Sub-Saharan African", ["Yoruba", "Mandenka", "Mende", "Esan", "Mbuti", "Biaka",
                              "Bantu", "San", "Khomani", "Dinka", "Luhya", "Maasai",
                              "Hadza", "Sandawe", "Luo", "Hutu", "Tutsi", "Igbo",
                              "Akan", "Ewe", "Fang", "Bakola", "Baka"]),
    ("North African / Levantine", ["Egyptian", "Saudi", "Yemen", "Jordan", "Lebanon",
                                    "Syrian", "Palestin", "Druze", "Bedouin", "Iraqi",
                                    "Algerian", "Tunisian", "Moroccan", "Libyan",
                                    "Berber", "Mozabite", "Sahar", "Maur", "Coptic",
                                    "Assyrian", "Samaritan"]),
    ("European", ["Sardinian", "Italian", "Spanish", "French", "Basque", "Portuguese",
                  "English", "Irish", "Scottish", "Welsh", "Dutch", "German", "Danish",
                  "Norwegian", "Swedish", "Finnish", "Polish", "Russian", "Ukrainian",
                  "Belarusian", "Lithuanian", "Latvian", "Estonian", "Greek", "Albanian",
                  "Bulgarian", "Romanian", "Hungarian", "Czech", "Slovak", "Croatian",
                  "Serbian", "Bosnian", "Slovenian", "Maltese", "Icelandic", "Cypriot",
                  "Sicilian", "Corsican", "Mordovian", "Saami"]),
    ("Caucasus / West Asian", ["Armenian", "Georgian", "Azeri", "Iranian", "Persian",
                                "Kurd", "Turk", "Turkmen", "Chechen", "Ossetian",
                                "Lezgin", "Avar", "Balkar", "Karachay", "Kabardian",
                                "Adygei", "Abkhaz", "Circassian", "Talysh", "Tat"]),
    ("Central / South Asian", ["Indian", "Pakistan", "Punjabi", "Bengali", "Tamil",
                                "Telugu", "Sinhalese", "Brahmin", "Gujarati",
                                "Kashmiri", "Pashtun", "Baloch", "Tajik", "Uzbek",
                                "Kazakh", "Kyrgyz", "Uyghur", "Hazara", "Burusho",
                                "Kalash", "Nepali", "Bhutanese", "Tibetan", "Sherpa"]),
    ("East / Southeast Asian", ["Han", "Chinese", "Japanese", "Korean", "Mongol",
                                 "Manchu", "Daur", "Hezhen", "Oroqen", "Xibo",
                                 "Tu", "Tujia", "Yi", "Naxi", "Lahu", "Miao",
                                 "She", "Cambodian", "Thai", "Vietnamese", "Lao",
                                 "Burmese", "Filipino", "Malay", "Indonesian",
                                 "Dai", "Hmong", "Karen"]),
    ("Siberian / Arctic", ["Buryat", "Yakut", "Even", "Evenk", "Yukaghir", "Chukchi",
                            "Koryak", "Nivkh", "Itelmen", "Nganasan", "Selkup",
                            "Ket", "Khanty", "Mansi", "Nenets", "Dolgan", "Tubalar",
                            "Altaian", "Tuvan", "Khakas", "Shor", "Eskimo", "Aleut",
                            "Inuit", "Greenland", "Saami"]),
    ("Native American", ["Maya", "Pima", "Mixe", "Mixtec", "Zapotec", "Karitiana",
                          "Surui", "Aymara", "Quechua", "Mapuche", "Toba", "Wichi",
                          "Wayuu", "Embera", "Waorani", "Yaghua", "Piapoco", "Cree",
                          "Ojibwa", "Chipewyan", "Tlingit", "Haida", "Algonquin"]),
    ("Oceanian", ["Papuan", "Bougainville", "Australian", "Aboriginal", "Maori",
                   "Tongan", "Samoan", "Fijian", "Hawaiian"]),
]

def modern_group(pop):
    """Match population name against MODERN_GROUPS keywords; fallback 'Other'."""
    pop_lower = pop.lower()
    for group_name, keywords in MODERN_GROUPS:
        for kw in keywords:
            if kw.lower() in pop_lower:
                return group_name
    return "Other"

# Ancient periods — match by suffix tokens after population stem.
# Order matters: more specific (UP, LBA_IA) before less specific (BA).
ANCIENT_PERIODS = [
    ("Upper Paleolithic / Mesolithic", ["UP_", "_UP", "Mesolithic", "_HG", "_EHG", "_WHG", "_CHG", "_ANE", "Aurignacian", "Gravettian", "Magdalenian", "Solutrean", "_LP", "Hoabinhian"]),
    ("Neolithic",                       ["_N_", "_LN", "_EN", "_MN", "Neolithic", "_Cardial", "_LBK", "_Funnel", "_TRB", "_PPN"]),
    ("Chalcolithic / Copper",           ["_CA", "_C_", "Chalcolithic", "Eneolithic", "_EBA_Cucu", "Copper"]),
    ("Bronze Age",                      ["_BA", "Bronze", "_EBA", "_MBA", "_LBA", "Yamnaya", "CordedWare", "Corded_Ware", "Sintashta", "Andronovo", "Afanasievo", "Bell_Beaker", "BellBeaker", "Unetice", "Tumulus", "Urnfield", "Catacomb", "Poltavka", "Srubnaya"]),
    ("Iron Age / Antiquity",            ["_IA", "Iron", "Scythian", "Sarmatian", "Hallstatt", "LaTene", "_Roman", "Hellenistic", "Etruscan", "Thracian", "Illyrian", "Han_", "_Greek", "_Hellen", "Saka"]),
    ("Medieval / Migration",            ["Medieval", "Migration", "_Viking", "_Saxon", "_Avar", "_Hun", "Conqueror", "Magyar", "_Slav", "Crusader", "_Pecheneg", "_Cuman", "Mongol", "Khitan", "_Tang", "_Song", "_Ming"]),
]

def ancient_period(pop):
    """Match population name against ANCIENT_PERIODS keywords; fallback 'Other / Unspecified'."""
    for period_name, keywords in ANCIENT_PERIODS:
        for kw in keywords:
            if kw in pop:
                return period_name
    return "Other / Unspecified"

# ------------------------------------------------------------- build -------
def build_dataset(ids, pcs, group_fn, name, seed=SEED):
    """Pick N_LANDMARKS via maxmin in 25D, project to first 3 PCs, assign groups."""
    print(f"\n[{name}] maxmin subsampling {N_LANDMARKS} from {len(ids)} in 25D...")
    idx = maxmin_subsample(pcs, N_LANDMARKS, seed=seed)

    landmark_ids = [ids[i] for i in idx]
    landmark_pcs = pcs[idx]
    landmark_pc3 = landmark_pcs[:, :3]   # first 3 PCs

    # Population names from IDs
    pops = [parse_id(s)[0] for s in landmark_ids]

    # Group assignment
    groups = [group_fn(p) for p in pops]

    # Deduplicate group names in order of first appearance
    group_names = []
    for g in groups:
        if g not in group_names:
            group_names.append(g)
    group_idx = [group_names.index(g) for g in groups]

    # Calibrate max_r based on the actual landmark cloud (PC3 projection).
    # Tighter than full diameter — at max_r = diameter, Rips fills out C(N,3)
    # ≈ 2.6M triangles for 250 points, blowing both compute time (~28s) and
    # JSON payload (~50MB). All interesting H_0/H_1 persistence pairs are
    # captured well before that radius — at 0.4 × diameter for Rips we get
    # the full pair count in ~7s. Čech and Alpha don't suffer the clique
    # blow-up the same way so they can extend further.
    diffs = landmark_pc3[None, :, :] - landmark_pc3[:, None, :]
    pdist = np.linalg.norm(diffs, axis=-1)
    diameter = float(pdist.max())
    max_r_rips  = round(diameter * 0.45, 3)
    max_r_cech  = round(diameter * 0.55, 3)
    max_r_alpha = round(diameter * 0.55, 3)
    # Initial slider positions — not used by the G25 path (no sliders) but
    # kept for hypothetical re-integration with the regular state machinery.
    r_init_rips  = round(diameter * 0.20, 3)
    r_init_cech  = round(diameter * 0.10, 3)
    r_init_alpha = round(diameter * 0.10, 3)

    print(f"  diameter (PC1-3 projection): {diameter:.3f}")
    print(f"  groups: {len(group_names)}")
    for g in group_names:
        count = groups.count(g)
        print(f"    {g}: {count}")

    return {
        "points":      landmark_pc3.tolist(),
        "labels":      pops,                  # population per landmark
        "groups":      group_idx,             # group index per landmark
        "group_names": group_names,           # unique group names in legend order
        "max_r_rips":  max_r_rips,
        "max_r_cech":  max_r_cech,
        "max_r_alpha": max_r_alpha,
        "r_init_rips":  r_init_rips,
        "r_init_cech":  r_init_cech,
        "r_init_alpha": r_init_alpha,
        "diameter":    diameter,
    }

# Ancient and modern: use their own samplings
ancient_data = build_dataset(ancient_ids, ancient_pcs, ancient_period, "ancient")
modern_data  = build_dataset(modern_ids,  modern_pcs,  modern_group,   "modern")

# Combined: stack ancient + modern, group by 'Ancient' vs 'Modern'
combined_ids  = ancient_ids + modern_ids
combined_pcs  = np.vstack([ancient_pcs, modern_pcs])
combined_origin = ["Ancient"] * len(ancient_ids) + ["Modern"] * len(modern_ids)
print(f"\n[combined] stacking ancient ({len(ancient_ids)}) + modern ({len(modern_ids)}) = {len(combined_ids)}")

def combined_group(pop, sample_idx, origin):
    """Combined cloud: group by ancient/modern (binary)."""
    return origin

# We need to track origin per-landmark, so build by hand
idx = maxmin_subsample(combined_pcs, N_LANDMARKS, seed=SEED)
landmark_ids = [combined_ids[i] for i in idx]
landmark_pcs = combined_pcs[idx]
landmark_pc3 = landmark_pcs[:, :3]
landmark_origins = [combined_origin[i] for i in idx]
pops = [parse_id(s)[0] for s in landmark_ids]
group_names = []
for g in landmark_origins:
    if g not in group_names:
        group_names.append(g)
group_idx = [group_names.index(g) for g in landmark_origins]

diffs = landmark_pc3[None, :, :] - landmark_pc3[:, None, :]
pdist = np.linalg.norm(diffs, axis=-1)
diameter = float(pdist.max())

print(f"  diameter (PC1-3 projection): {diameter:.3f}")
print(f"  ancient landmarks: {landmark_origins.count('Ancient')}")
print(f"  modern landmarks:  {landmark_origins.count('Modern')}")

combined_data = {
    "points":      landmark_pc3.tolist(),
    "labels":      pops,
    "groups":      group_idx,
    "group_names": group_names,
    "max_r_rips":  round(diameter * 0.45, 3),
    "max_r_cech":  round(diameter * 0.55, 3),
    "max_r_alpha": round(diameter * 0.55, 3),
    "r_init_rips":  round(diameter * 0.20, 3),
    "r_init_cech":  round(diameter * 0.10, 3),
    "r_init_alpha": round(diameter * 0.10, 3),
    "diameter":    diameter,
}

# ------------------------------------------------------------- write -------
output = {
    "ancient":  ancient_data,
    "modern":   modern_data,
    "combined": combined_data,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f)

print(f"\nWrote {OUTPUT_PATH}")
print(f"  size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
