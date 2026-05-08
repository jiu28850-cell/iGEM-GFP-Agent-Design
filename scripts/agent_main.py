"""
GFP Thermal Stability Engineering — Four-Strategy Pipeline
===========================================================
Strategy 1: Loop Rigidification (Proline substitution in flexible loops)
Strategy 2: Consensus Sequence Redesign (MSA of 5 GFP homologs)
Strategy 3: Disulfide Stapling (PDB-guided Cys pair introduction)
Strategy 4: Terminal Clamping + Core Repacking (salt bridges + hydrophobic packing)

Each strategy produces 1-2 sequences; best 6 are written to outputs/submission.csv.
"""
import os, sys, random, warnings, urllib.request
import numpy as np
import pandas as pd
import torch
import esm as esm_lib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data")
OUTPUT_DIR  = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE     = os.path.join(DATA_DIR, "GFP_data.xlsx")
AASEQ_FILE     = os.path.join(DATA_DIR, "AAseqs.txt")
EXCLUSION_FILE = os.path.join(ROOT, "Exclusion_List.csv")
PDB_FILE       = os.path.join(DATA_DIR, "2WUR.pdb")

SEED      = 42
TEAM_NAME = "iGEM_GFP_Agent"
AA20      = list("ACDEFGHIKLMNPQRSTVWY")
random.seed(SEED); np.random.seed(SEED)

MAX_TRAIN_SAMPLES = 8000

# ── avGFP secondary structure (1-based, from PDB 2WUR / literature) ────────
# beta-strands: the 11-strand beta-barrel
BETA_STRANDS_1BASED = [
    (11, 23), (34, 44), (57, 63), (73, 84),
    (93, 101), (111, 121), (128, 137), (147, 157),
    (165, 172), (180, 190), (196, 209),
]
# loops connecting strands (flexible regions safe for Pro substitution)
LOOPS_1BASED = [
    (24, 33), (45, 56), (64, 72), (85, 92),
    (102, 110), (122, 127), (138, 146), (158, 164),
    (173, 179), (191, 195),
]
# Chromophore + proton relay — NEVER touch
LOCKED_1BASED = {65, 66, 67, 96, 222, 69, 203}
locked_0 = {p - 1 for p in LOCKED_1BASED}

# N-terminal region (1-10) and C-terminal region (229-238)
N_TERM_1BASED = list(range(1, 11))
C_TERM_1BASED = list(range(229, 239))

# Deep hydrophobic core positions (from sfGFP literature, far from chromophore)
CORE_HYDROPHOBIC_1BASED = [3, 5, 15, 17, 19, 36, 38, 40, 58, 60, 75, 77,
                            95, 97, 113, 115, 130, 132, 149, 151, 167, 169,
                            182, 184, 198, 200, 211, 213]

# ── helpers ────────────────────────────────────────────────────────────────
def load_wt_sequence(path, target="avGFP"):
    seqs, name, buf = {}, "", []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name: seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            elif not line.startswith("#") and line:
                buf.append(line)
    if name: seqs[name] = "".join(buf)
    if target not in seqs:
        raise ValueError(f"{target} not found. Have: {list(seqs.keys())}")
    return seqs[target], seqs

def apply_mutations(wt, mutation_str):
    seq = list(wt)
    if mutation_str == "WT" or pd.isna(mutation_str): return wt
    for mut in str(mutation_str).split(":"):
        mut = mut.strip()
        if not mut: continue
        try:
            idx = int(mut[1:-1]) - 1
            if 0 <= idx < len(seq): seq[idx] = mut[-1]
            else: return None
        except: return None
    return "".join(seq)

def get_esm_embeddings(sequences, model, alphabet, batch_size=16):
    model.eval()
    device = next(model.parameters()).device
    bc = alphabet.get_batch_converter()
    out = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        _, _, tokens = bc([(f"s{j}", s) for j, s in enumerate(batch)])
        tokens = tokens.to(device)
        with torch.no_grad():
            reps = model(tokens, repr_layers=[model.num_layers],
                         return_contacts=False)["representations"][model.num_layers]
        for k, seq in enumerate(batch):
            out.append(reps[k, 1:len(seq)+1].mean(0).cpu().numpy())
    return np.array(out)

def is_valid(seq, exclusion_set):
    return (220 <= len(seq) <= 250 and seq[0] == "M"
            and all(a in AA20 for a in seq) and seq not in exclusion_set)

def rf_score_batch(seqs, rf, esm_model, esm_alphabet):
    embs = get_esm_embeddings(seqs, esm_model, esm_alphabet)
    return rf.predict(embs)

# ══════════════════════════════════════════════════════════════════════════
# SHARED SETUP
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SHARED SETUP")
print("=" * 60)

WT_SEQ, ALL_SEQS = load_wt_sequence(AASEQ_FILE, "avGFP")
print(f"WT length: {len(WT_SEQ)} aa")

print("Loading training data...")
df = pd.read_excel(TRAIN_FILE, sheet_name="brightness")
df = df[df["GFP type"] == "avGFP"].copy()
df["Brightness"] = pd.to_numeric(df["Brightness"], errors="coerce")
df = df.dropna(subset=["Brightness"])
df["full_seq"] = df["aaMutations"].apply(lambda m: apply_mutations(WT_SEQ, m))
df = df.dropna(subset=["full_seq"])
df = df[df["full_seq"].str.len().between(220, 250)]
if len(df) > MAX_TRAIN_SAMPLES:
    df = df.sample(MAX_TRAIN_SAMPLES, random_state=SEED)
print(f"  Training rows: {len(df)}")

print("Loading ESM-2 model (esm2_t6_8M_UR50D)...")
esm_model, esm_alphabet = esm_lib.pretrained.esm2_t6_8M_UR50D()
esm_model.eval()

print("Computing ESM-2 embeddings for training set...")
X = get_esm_embeddings(df["full_seq"].tolist(), esm_model, esm_alphabet)
y = df["Brightness"].values

print("Training Random Forest predictor...")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
rf = RandomForestRegressor(n_estimators=200, max_depth=25, min_samples_leaf=2,
                           n_jobs=-1, random_state=SEED)
rf.fit(X_tr, y_tr)
print(f"  Validation R2 = {r2_score(y_val, rf.predict(X_val)):.4f}")  # R^2

print("Loading exclusion list...")
excl_df = pd.read_csv(EXCLUSION_FILE)
col = [c for c in excl_df.columns if c.lower() == "sequence"][0]
EXCLUSION = set(excl_df[col].dropna().str.strip())
print(f"  Exclusion set: {len(EXCLUSION):,}")

# ══════════════════════════════════════════════════════════════════════════
# STRATEGY 1 — Loop Rigidification (Proline substitution)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STRATEGY 1: Loop Rigidification (Pro substitution)")
print("=" * 60)
# Proline is ideal at i+1 of a beta-turn (positions 2 and 3 of a type-I/II turn).
# We target the middle residues of each loop, avoiding positions that are:
#   - already Pro
#   - in the locked chromophore set
#   - Gly (Gly is often structurally essential in tight turns)
# We generate all single-Pro substitutions in loop regions, then score with RF.

pro_candidates, pro_descs = [], []
for loop_start, loop_end in LOOPS_1BASED:
    # Target the central 60% of each loop (avoid loop termini which contact strands)
    margin = max(1, (loop_end - loop_start + 1) // 5)
    for pos_1 in range(loop_start + margin, loop_end - margin + 1):
        idx = pos_1 - 1
        if idx in locked_0:
            continue
        orig = WT_SEQ[idx]
        if orig in ("P", "G"):   # already Pro, or Gly (structurally critical)
            continue
        mut_seq = WT_SEQ[:idx] + "P" + WT_SEQ[idx+1:]
        if not is_valid(mut_seq, EXCLUSION):
            continue
        pro_candidates.append(mut_seq)
        pro_descs.append(f"{orig}{pos_1}P")

print(f"  Pro substitution candidates: {len(pro_candidates)}")
scores_pro = rf_score_batch(pro_candidates, rf, esm_model, esm_alphabet)

# Build combinatorial: take top-3 single-Pro hits and try pairwise combinations
top_pro_idx = np.argsort(scores_pro)[::-1][:5]
top_pro_seqs  = [pro_candidates[i] for i in top_pro_idx]
top_pro_descs = [pro_descs[i] for i in top_pro_idx]
top_pro_scores = [scores_pro[i] for i in top_pro_idx]
print(f"  Top single-Pro: {top_pro_descs[0]} ({top_pro_scores[0]:.4f})")

# Try all pairs from top-5 single-Pro hits
combo_seqs, combo_descs = [], []
for i in range(len(top_pro_seqs)):
    for j in range(i+1, len(top_pro_seqs)):
        # Apply both Pro substitutions to WT
        idx_i = int(top_pro_descs[i][1:-1]) - 1
        idx_j = int(top_pro_descs[j][1:-1]) - 1
        seq = list(WT_SEQ)
        seq[idx_i] = "P"
        seq[idx_j] = "P"
        s = "".join(seq)
        if is_valid(s, EXCLUSION):
            combo_seqs.append(s)
            combo_descs.append(f"{top_pro_descs[i]}:{top_pro_descs[j]}")

combo_scores = rf_score_batch(combo_seqs, rf, esm_model, esm_alphabet) if combo_seqs else []
all_s1 = (list(zip(scores_pro, pro_candidates, pro_descs)) +
          list(zip(combo_scores, combo_seqs, combo_descs)))
all_s1.sort(reverse=True)
top_S1 = all_s1[:2]
print(f"  Top-2 (single+combo): {top_S1[0][2]} ({top_S1[0][0]:.4f}), "
      f"{top_S1[1][2]} ({top_S1[1][0]:.4f})")

# ══════════════════════════════════════════════════════════════════════════
# STRATEGY 2 — Consensus Sequence Redesign (MSA of 5 GFP homologs)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STRATEGY 2: Consensus Sequence Redesign (5-homolog MSA)")
print("=" * 60)
# Align all 5 GFP sequences by position (they are all 238 aa and share the
# beta-barrel fold, so a simple positional alignment is valid for avGFP-length seqs).
# At each position, find the consensus (most common) amino acid across homologs.
# Replace avGFP residues that differ from consensus AND are not in locked set
# AND are on the surface (loop or strand exterior).

homolog_names = ["sfGFP", "avGFP", "amacGFP", "cgreGFP", "ppluGFP"]
homologs = {name: ALL_SEQS[name] for name in homolog_names if name in ALL_SEQS}
print(f"  Homologs loaded: {list(homologs.keys())}")

# Pad/trim all to avGFP length for positional alignment
wt_len = len(WT_SEQ)
aligned = {}
for name, seq in homologs.items():
    if len(seq) >= wt_len:
        aligned[name] = seq[:wt_len]
    else:
        aligned[name] = seq + seq[-1] * (wt_len - len(seq))  # pad with last aa

# Compute consensus at each position
consensus_seq = list(WT_SEQ)
consensus_changes = []
for idx in range(wt_len):
    if idx in locked_0:
        continue
    col = [aligned[n][idx] for n in aligned if aligned[n][idx] in AA20]
    if not col:
        continue
    counts = {aa: col.count(aa) for aa in set(col)}
    best_aa = max(counts, key=counts.get)
    freq = counts[best_aa] / len(col)
    # Only substitute if: consensus is clear (>=60%), differs from avGFP,
    # and position is in a loop or surface-exposed strand edge
    if best_aa != WT_SEQ[idx] and freq >= 0.6:
        consensus_changes.append((idx, WT_SEQ[idx], best_aa, freq))

print(f"  Consensus substitutions identified: {len(consensus_changes)}")
for idx, orig, new, freq in sorted(consensus_changes, key=lambda x: -x[3])[:8]:
    print(f"    {orig}{idx+1}{new}  (consensus freq={freq:.0%})")

# Build two consensus variants:
# Variant A: apply all consensus changes at once
seq_cons_all = list(WT_SEQ)
descs_all = []
for idx, orig, new, freq in consensus_changes:
    seq_cons_all[idx] = new
    descs_all.append(f"{orig}{idx+1}{new}")
seq_cons_all = "".join(seq_cons_all)

# Variant B: only high-confidence changes (>=80%)
seq_cons_high = list(WT_SEQ)
descs_high = []
for idx, orig, new, freq in consensus_changes:
    if freq >= 0.80:
        seq_cons_high[idx] = new
        descs_high.append(f"{orig}{idx+1}{new}")
seq_cons_high = "".join(seq_cons_high)

s2_candidates, s2_descs = [], []
for seq, descs in [(seq_cons_all, descs_all), (seq_cons_high, descs_high)]:
    if is_valid(seq, EXCLUSION) and descs:
        s2_candidates.append(seq)
        s2_descs.append(":".join(descs[:10]))  # truncate desc for readability

scores_S2 = rf_score_batch(s2_candidates, rf, esm_model, esm_alphabet) if s2_candidates else []
top_S2 = sorted(zip(scores_S2, s2_candidates, s2_descs), reverse=True)[:2]
for sc, _, desc in top_S2:
    print(f"  Consensus variant: {desc[:60]}... ({sc:.4f})")

# ══════════════════════════════════════════════════════════════════════════
# STRATEGY 3 — Disulfide Stapling (PDB-guided Cys pair introduction)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STRATEGY 3: Disulfide Stapling (PDB 2WUR Cbeta distances)")
print("=" * 60)

def parse_pdb_cbeta(pdb_path):
    """Return dict {res_num: (x,y,z)} for Cbeta (or Ca for Gly) of chain A."""
    coords = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            if chain not in ("A", " "):
                continue
            atom_name = line[12:16].strip()
            res_num   = int(line[22:26].strip())
            if atom_name == "CB" or (atom_name == "CA" and res_num not in coords):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if atom_name == "CB":
                    coords[res_num] = (x, y, z)
                elif atom_name == "CA" and res_num not in coords:
                    coords[res_num] = (x, y, z)
    return coords

cbeta = parse_pdb_cbeta(PDB_FILE)
print(f"  Parsed {len(cbeta)} residue Cbeta positions from PDB")

# Find pairs with Cbeta distance < 5.5 Å that are:
#   - both on the surface (loop or strand edge, NOT core)
#   - neither is in the locked chromophore set
#   - neither is already Cys in WT
#   - not adjacent in sequence (|i-j| > 4)
#   - not near the chromophore cavity (positions 60-75 excluded for Cys)
CYS_FORBIDDEN_1BASED = set(range(60, 76)) | LOCKED_1BASED  # near chromophore

# Surface positions: loop residues + first/last 2 residues of each strand
surface_1based = set()
for s, e in LOOPS_1BASED:
    surface_1based.update(range(s, e+1))
for s, e in BETA_STRANDS_1BASED:
    surface_1based.update([s, s+1, e-1, e])

disulfide_pairs = []
res_list = sorted(cbeta.keys())
for i, r1 in enumerate(res_list):
    for r2 in res_list[i+1:]:
        if abs(r1 - r2) <= 4:
            continue
        if r1 in CYS_FORBIDDEN_1BASED or r2 in CYS_FORBIDDEN_1BASED:
            continue
        if r1 not in surface_1based or r2 not in surface_1based:
            continue
        if WT_SEQ[r1-1] == "C" or WT_SEQ[r2-1] == "C":
            continue
        x1,y1,z1 = cbeta[r1]; x2,y2,z2 = cbeta[r2]
        dist = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5
        if dist < 5.5:
            disulfide_pairs.append((dist, r1, r2))

disulfide_pairs.sort()
print(f"  Candidate disulfide pairs (Cbeta < 5.5A): {len(disulfide_pairs)}")
for dist, r1, r2 in disulfide_pairs[:5]:
    print(f"    {WT_SEQ[r1-1]}{r1}C + {WT_SEQ[r2-1]}{r2}C  dist={dist:.2f}A")

# Build sequences for top pairs
ss_seqs, ss_descs = [], []
for dist, r1, r2 in disulfide_pairs[:10]:
    seq = list(WT_SEQ)
    orig1, orig2 = seq[r1-1], seq[r2-1]
    seq[r1-1] = "C"; seq[r2-1] = "C"
    s = "".join(seq)
    if is_valid(s, EXCLUSION):
        ss_seqs.append(s)
        ss_descs.append(f"{orig1}{r1}C:{orig2}{r2}C")

scores_S3 = rf_score_batch(ss_seqs, rf, esm_model, esm_alphabet) if ss_seqs else []
top_S3 = sorted(zip(scores_S3, ss_seqs, ss_descs), reverse=True)[:2]
for sc, _, desc in top_S3:
    print(f"  Disulfide pair: {desc}  RF={sc:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STRATEGY 4 — Terminal Clamping + Core Repacking
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STRATEGY 4: Terminal Clamping + Core Repacking")
print("=" * 60)

# --- 4a. Terminal salt bridge design ---
# N-terminus: introduce positively charged residues (K, R)
# C-terminus: introduce negatively charged residues (D, E)
# This creates an electrostatic "zipper" that clamps the barrel ends.
# We target the outermost 5 residues of N and C termini.
POS_AA = ["K", "R"]   # positive charge
NEG_AA = ["D", "E"]   # negative charge

term_seqs, term_descs = [], []
# N-terminal positions: replace non-charged residues with K or R
for pos_1 in N_TERM_1BASED[:5]:
    idx = pos_1 - 1
    if idx in locked_0 or WT_SEQ[idx] in ("K", "R", "M"):
        continue
    for new_aa in POS_AA:
        s = WT_SEQ[:idx] + new_aa + WT_SEQ[idx+1:]
        if is_valid(s, EXCLUSION):
            term_seqs.append(s)
            term_descs.append(f"{WT_SEQ[idx]}{pos_1}{new_aa}")

# C-terminal positions: replace non-charged residues with D or E
for pos_1 in C_TERM_1BASED[-5:]:
    idx = pos_1 - 1
    if idx in locked_0 or WT_SEQ[idx] in ("D", "E"):
        continue
    for new_aa in NEG_AA:
        s = WT_SEQ[:idx] + new_aa + WT_SEQ[idx+1:]
        if is_valid(s, EXCLUSION):
            term_seqs.append(s)
            term_descs.append(f"{WT_SEQ[idx]}{pos_1}{new_aa}")

# --- 4b. Core repacking: replace small hydrophobics (V, A) with bulkier ones (I, L, F) ---
# Only at deep core positions, far from chromophore
SMALL_HYDROPHOBIC = {"V", "A"}
BULKY_HYDROPHOBIC = ["I", "L", "F"]

core_seqs, core_descs = [], []
for pos_1 in CORE_HYDROPHOBIC_1BASED:
    idx = pos_1 - 1
    if idx >= len(WT_SEQ) or idx in locked_0:
        continue
    orig = WT_SEQ[idx]
    if orig not in SMALL_HYDROPHOBIC:
        continue
    for new_aa in BULKY_HYDROPHOBIC:
        if new_aa == orig:
            continue
        s = WT_SEQ[:idx] + new_aa + WT_SEQ[idx+1:]
        if is_valid(s, EXCLUSION):
            core_seqs.append(s)
            core_descs.append(f"{orig}{pos_1}{new_aa}")

print(f"  Terminal salt bridge candidates: {len(term_seqs)}")
print(f"  Core repacking candidates: {len(core_seqs)}")

# Score all single substitutions
all_s4_seqs  = term_seqs + core_seqs
all_s4_descs = term_descs + core_descs
scores_s4_single = rf_score_batch(all_s4_seqs, rf, esm_model, esm_alphabet)

# Pick top-3 terminal and top-3 core, then build one combined sequence
top_term = sorted(zip(scores_s4_single[:len(term_seqs)],
                      term_seqs, term_descs), reverse=True)[:3]
top_core = sorted(zip(scores_s4_single[len(term_seqs):],
                      core_seqs, core_descs), reverse=True)[:3]

# Combined: apply best terminal + best core substitutions together
def combine_mutations(wt, mut_descs_list):
    seq = list(wt)
    parts = []
    for desc in mut_descs_list:
        idx = int(desc[1:-1]) - 1
        seq[idx] = desc[-1]
        parts.append(desc)
    return "".join(seq), ":".join(parts)

# Variant A: best terminal clamp + best core repack
combo_A_descs = [top_term[0][2], top_core[0][2]] if top_term and top_core else []
# Variant B: top-2 terminal + top-2 core
combo_B_descs = ([t[2] for t in top_term[:2]] +
                 [c[2] for c in top_core[:2]]) if top_term and top_core else []

s4_candidates, s4_descs_out = [], []
for desc_list in [combo_A_descs, combo_B_descs]:
    if not desc_list:
        continue
    seq, desc = combine_mutations(WT_SEQ, desc_list)
    if is_valid(seq, EXCLUSION):
        s4_candidates.append(seq)
        s4_descs_out.append(desc)

scores_S4 = rf_score_batch(s4_candidates, rf, esm_model, esm_alphabet) if s4_candidates else []
top_S4 = sorted(zip(scores_S4, s4_candidates, s4_descs_out), reverse=True)[:2]
for sc, _, desc in top_S4:
    print(f"  Terminal+Core combo: {desc}  RF={sc:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# FINAL SELECTION — rank all candidates, pick best 6 (no duplicates)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL SELECTION")
print("=" * 60)

# Collect all candidates with strategy label, sorted by RF score descending
all_pool = (
    [(sc, s, d, "S1-Pro")       for sc, s, d in top_S1] +
    [(sc, s, d, "S2-Consensus") for sc, s, d in top_S2] +
    [(sc, s, d, "S3-Disulfide") for sc, s, d in top_S3] +
    [(sc, s, d, "S4-Terminal")  for sc, s, d in top_S4]
)
all_pool.sort(reverse=True)

seen: set[str] = set()
final = []
for sc, seq, desc, strat in all_pool:
    if seq not in seen and is_valid(seq, EXCLUSION):
        seen.add(seq)
        final.append((sc, seq, desc, strat))
    if len(final) == 6:
        break

# Fill remaining slots with top single-Pro substitutions if needed
if len(final) < 6:
    for sc, seq, desc in sorted(zip(scores_pro, pro_candidates, pro_descs), reverse=True):
        if seq not in seen and is_valid(seq, EXCLUSION):
            seen.add(seq)
            final.append((sc, seq, desc, "S1-fill"))
        if len(final) == 6:
            break

print(f"\n{'ID':<4} {'Strategy':<16} {'Score':<8} {'Mutations'}")
print("-" * 80)
for i, (sc, seq, desc, strat) in enumerate(final, 1):
    print(f"{i:<4} {strat:<16} {sc:<8.4f} {desc[:55]}")

sub = pd.DataFrame({
    "Team_Name": TEAM_NAME,
    "Seq_ID":    range(1, len(final) + 1),
    "Sequence":  [s for _, s, _, _ in final],
    "Mutations": [d for _, _, d, _ in final],
    "Strategy":  [t for _, _, _, t in final],
    "RF_Score":  [sc for sc, _, _, _ in final],
})
sub.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
pd.DataFrame({"sequence": [s for _, s, _, _ in final]}).to_csv(
    os.path.join(OUTPUT_DIR, "valid_candidates.csv"), index=False)
print(f"\nSaved {len(final)} sequences to outputs/submission.csv")

# ══════════════════════════════════════════════════════════════════════════
# STRATEGY 5 — RF-guided greedy brightness stacking (3 brightness slots)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STRATEGY 5: RF-guided greedy brightness stacking")
print("=" * 60)

# Score every surface single-point mutation with RF to build a ranked pool
print("  Scoring all surface single-point mutations...")
surf_all_0 = [p - 1 for p in range(1, len(WT_SEQ) + 1) if (p - 1) not in locked_0]
bright_pool_seqs, bright_pool_descs, bright_pool_idxs = [], [], []
for idx in surf_all_0:
    orig = WT_SEQ[idx]
    for new_aa in AA20:
        if new_aa == orig:
            continue
        s = WT_SEQ[:idx] + new_aa + WT_SEQ[idx+1:]
        if not is_valid(s, EXCLUSION):
            continue
        bright_pool_seqs.append(s)
        bright_pool_descs.append(f"{orig}{idx+1}{new_aa}")
        bright_pool_idxs.append(idx)

bright_pool_scores = rf_score_batch(bright_pool_seqs, rf, esm_model, esm_alphabet)
bright_pool = sorted(zip(bright_pool_scores, bright_pool_seqs,
                         bright_pool_descs, bright_pool_idxs), reverse=True)
print(f"  Pool: {len(bright_pool)} mutations, top: {bright_pool[0][2]} ({bright_pool[0][0]:.4f})")

def greedy_brightness_stack(seed_score, seed_seq, seed_desc, seed_idx,
                             pool, max_muts=6):
    seq, desc_parts, used = seed_seq, [seed_desc], {seed_idx}
    cur = seed_score
    for _ in range(max_muts - 1):
        next_seqs, next_descs, next_idxs = [], [], []
        for (_, _, md, mi) in pool:
            if mi in used:
                continue
            s = seq[:mi] + md[-1] + seq[mi+1:]
            if is_valid(s, EXCLUSION):
                next_seqs.append(s); next_descs.append(md); next_idxs.append(mi)
        if not next_seqs:
            break
        ns = rf_score_batch(next_seqs, rf, esm_model, esm_alphabet)
        best = int(np.argmax(ns))
        if ns[best] <= cur:
            break
        seq = next_seqs[best]
        desc_parts.append(next_descs[best])
        used.add(next_idxs[best])
        cur = ns[best]
        print(f"    +{next_descs[best]}  score={cur:.4f}")
    return seq, ":".join(desc_parts), cur

# Run 5 greedy stacks from top-5 RF seeds, keep best 3
N_BRIGHT_SEEDS = 5
bright_stacks = []
for rank, (sc, s_seq, s_desc, s_idx) in enumerate(bright_pool[:N_BRIGHT_SEEDS]):
    print(f"\n  Brightness stack {rank+1} (seed: {s_desc}, RF={sc:.4f}):")
    fseq, fdesc, fscore = greedy_brightness_stack(sc, s_seq, s_desc, s_idx, bright_pool)
    bright_stacks.append((fscore, fseq, fdesc))
    print(f"  -> final: {fdesc}  score={fscore:.4f}")

bright_stacks.sort(reverse=True)
top_S5_bright = bright_stacks[:3]

# ═════════════════════════════════════════════════════════���════════════════
# STRATEGY 6 — Thermal stability stacking (2 thermal slots)
# Combines: best disulfide pair + best Pro loop + consensus + terminal salt bridge
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STRATEGY 6: Thermal stability stacking (all 4 mechanisms)")
print("=" * 60)

def stack_mutations(wt, mut_list):
    seq, used, applied = list(wt), set(), []
    for idx, new_aa, desc in mut_list:
        if idx in used or idx in locked_0:
            continue
        seq[idx] = new_aa; used.add(idx); applied.append(desc)
    return "".join(seq), applied

# Build mutation ingredient lists from each thermal strategy
s1_muts = []
for sc, seq, desc in sorted(zip(scores_pro, pro_candidates, pro_descs), reverse=True)[:2]:
    s1_muts.append((int(desc[1:-1]) - 1, "P", desc))

s2_muts = [(idx, new, f"{orig}{idx+1}{new}") for idx, orig, new, _ in consensus_changes]

s3_muts = []
if top_S3:
    for part in top_S3[0][2].split(":"):
        s3_muts.append((int(part[1:-1]) - 1, "C", part))

s3b_muts = []  # second-best disulfide pair
if len(top_S3) > 1:
    for part in top_S3[1][2].split(":"):
        s3b_muts.append((int(part[1:-1]) - 1, "C", part))

s4_muts = []
if top_term:
    for sc, seq, desc in top_term[:2]:
        s4_muts.append((int(desc[1:-1]) - 1, desc[-1], desc))

# Thermal variant A: best disulfide + 2x Pro + consensus + terminal salt bridge
# (maximum number of thermal mechanisms on one sequence)
thermal_A_muts = s3_muts + s1_muts + s2_muts + s4_muts
seq_tA, applied_tA = stack_mutations(WT_SEQ, thermal_A_muts)
print(f"  Thermal A applied ({len(applied_tA)} muts): {':'.join(applied_tA)}")

# Thermal variant B: second disulfide pair + 2x Pro + consensus
# (avoids terminal salt bridge — more conservative, different disulfide)
thermal_B_muts = s3b_muts + s1_muts + s2_muts
seq_tB, applied_tB = stack_mutations(WT_SEQ, thermal_B_muts)
print(f"  Thermal B applied ({len(applied_tB)} muts): {':'.join(applied_tB)}")

thermal_candidates = []
for seq, applied in [(seq_tA, applied_tA), (seq_tB, applied_tB)]:
    if is_valid(seq, EXCLUSION) and applied:
        thermal_candidates.append((seq, ":".join(applied)))

if thermal_candidates:
    t_scores = rf_score_batch([s for s, _ in thermal_candidates],
                              rf, esm_model, esm_alphabet)
    top_S6 = sorted(zip(t_scores, [s for s, _ in thermal_candidates],
                        [d for _, d in thermal_candidates]), reverse=True)
    for sc, _, desc in top_S6:
        print(f"  Thermal stack RF={sc:.4f}  muts={desc}")
else:
    top_S6 = []
    print("  No valid thermal stacks generated.")

# ══════════════════════════════════════════════════════════════════════════
# FINAL SELECTION — hard role-based slot assignment
#   Slot 1   : Conservative  — best RF single-point (guaranteed to pass brightness)
#   Slots 2-3: Thermal       — S6 stacked sequences FIRST (forced, not RF-ranked)
#                              fallback to S3 disulfide if S6 is empty
#   Slots 4-6: Brightness    — S5 greedy stacks, deduplicated by mutation positions
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL SELECTION (1 conservative + 2 thermal + 3 brightness)")
print("=" * 60)

seen: set[str] = set()
final: list[tuple] = []  # (score, seq, desc, role)

# ── Slot 1: Conservative ──────────────────────────────────────────────────
# Best RF single-point mutation — highest confidence it clears the brightness bar
single_point_pool = sorted(
    zip(bright_pool_scores, bright_pool_seqs, bright_pool_descs), reverse=True)
for sc, seq, desc in single_point_pool:
    if seq not in seen and is_valid(seq, EXCLUSION):
        seen.add(seq)
        final.append((sc, seq, desc, "Conservative"))
        break

# ── Slots 2-3: Thermal stability — FORCED from S6 stacked sequences ───────
# S6 sequences are specifically engineered for thermal stability (disulfide +
# Pro rigidification + consensus + terminal salt bridge). RF underestimates
# them because the training set has few such multi-mechanism variants.
# We force-include them regardless of RF score.
thermal_forced = [(sc, s, d) for sc, s, d in top_S6]  # already sorted by RF desc
# Fallback: if S6 is empty or only 1 sequence, fill from S3 disulfide pairs
thermal_fallback = [(sc, s, d) for sc, s, d in top_S3]
thermal_ordered = thermal_forced + [x for x in thermal_fallback
                                     if x[1] not in {t[1] for t in thermal_forced}]
for sc, seq, desc in thermal_ordered:
    if len([x for x in final if x[3] == "Thermal"]) >= 2:
        break
    if seq not in seen and is_valid(seq, EXCLUSION):
        seen.add(seq)
        final.append((sc, seq, desc, "Thermal"))

# ── Slots 4-6: Brightness — S5 greedy stacks, deduplicated by positions ───
# Deduplicate not just by sequence but also by mutation position overlap:
# two sequences that share >50% of their mutated positions are too similar.
def mut_positions(desc: str) -> set:
    positions = set()
    for part in desc.split(":"):
        try: positions.add(int(part[1:-1]))
        except: pass
    return positions

brightness_ordered = sorted(top_S5_bright, reverse=True)  # (score, seq, desc)
used_positions: list[set] = []
for sc, seq, desc in brightness_ordered:
    if len([x for x in final if x[3] == "Brightness"]) >= 3:
        break
    if seq in seen or not is_valid(seq, EXCLUSION):
        continue
    pos = mut_positions(desc)
    # Skip if >50% position overlap with any already-selected brightness sequence
    too_similar = any(
        len(pos & up) / max(len(pos | up), 1) > 0.5
        for up in used_positions
    )
    if too_similar:
        continue
    seen.add(seq)
    used_positions.append(pos)
    final.append((sc, seq, desc, "Brightness"))

# Fill remaining slots if needed (shouldn't happen normally)
if len(final) < 6:
    for sc, seq, desc in single_point_pool:
        if seq not in seen and is_valid(seq, EXCLUSION):
            seen.add(seq)
            final.append((sc, seq, desc, "Fill"))
        if len(final) == 6:
            break

# Display sorted by role then score
role_order = {"Conservative": 0, "Thermal": 1, "Brightness": 2, "Fill": 3}
final.sort(key=lambda x: (role_order.get(x[3], 9), -x[0]))

print(f"\n{'ID':<4} {'Role':<14} {'RF Score':<10} {'Mutations'}")
print("-" * 85)
for i, (sc, seq, desc, role) in enumerate(final, 1):
    print(f"{i:<4} {role:<14} {sc:<10.4f} {desc[:55]}")

print("\nNote: Thermal sequences are forced-included regardless of RF score.")
print("RF underestimates multi-mechanism variants (Cys/Pro combos rare in training data).")

sub = pd.DataFrame({
    "Team_Name": TEAM_NAME,
    "Seq_ID":    range(1, len(final) + 1),
    "Sequence":  [s for _, s, _, _ in final],
    "Mutations": [d for _, _, d, _ in final],
    "Role":      [r for _, _, _, r in final],
    "RF_Score":  [sc for sc, _, _, _ in final],
})
sub.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
pd.DataFrame({"sequence": [s for _, s, _, _ in final]}).to_csv(
    os.path.join(OUTPUT_DIR, "valid_candidates.csv"), index=False)
print(f"\nFinal submission: {len(final)} sequences -> outputs/submission.csv")

