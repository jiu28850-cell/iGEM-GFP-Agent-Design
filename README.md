# iGEM GFP Agent — Computational GFP Engineering Pipeline

A six-strategy pipeline for engineering avGFP variants with improved brightness and thermal stability,
using ESM-2 protein language model embeddings and a Random Forest brightness predictor.

## Requirements

- Python 3.9+
- Windows / Linux / macOS
- ~2 GB RAM (ESM-2 runs on CPU)

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 2. Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fair-esm scikit-learn pandas openpyxl numpy
```

## Data

Place the following files in `data/`:

| File | Description |
|------|-------------|
| `GFP_data.xlsx` | avGFP brightness dataset (sheet: "brightness") |
| `AAseqs.txt` | FASTA file with avGFP + 4 homolog sequences |
| `2WUR.pdb` | avGFP crystal structure (PDB ID: 2WUR) |

Place `Exclusion_List.csv` in the project root. It must contain a `sequence` column.

## Reproduction

```bash
# Windows (fixes GBK encoding on Chinese locale systems)
PYTHONIOENCODING=utf-8 venv/Scripts/python.exe -u scripts/agent_main.py

# Linux/macOS
python -u scripts/agent_main.py
```

Runtime: ~15–30 minutes on CPU (dominated by ESM-2 embedding of 8000 training sequences).

## Output

| File | Description |
|------|-------------|
| `outputs/submission.csv` | 6 engineered sequences with role, mutations, RF score |
| `outputs/final_submission.csv` | Same content, final submission copy |
| `outputs/valid_candidates.csv` | Sequence-only list for validation |
| `outputs/run_log.txt` | Full stdout log of the last run |

## Pipeline Overview

The script executes six strategies sequentially, then assembles a role-based final selection:

```
Strategy 1  Loop Rigidification     Proline substitution in flexible loops (reduces unfolding entropy)
Strategy 2  Consensus Redesign      MSA of 5 GFP homologs; substitutions at ≥60% consensus positions
Strategy 3  Disulfide Stapling      PDB Cβ-distance guided Cys pair introduction (< 5.5 Å pairs)
Strategy 4  Terminal Clamping       N/C-terminal salt bridge introduction
Strategy 5  Brightness Stacking     RF-guided greedy single-point mutation stacking (up to 6 muts)
Strategy 6  Thermal Stacking        Multi-mechanism combination: disulfide + Pro + consensus + salt bridge
```

Final slot assignment:
- **1 Conservative** — highest-RF single-point mutation (guaranteed brightness)
- **2 Thermal** — Strategy 6 stacked sequences (forced-included; RF underestimates multi-Cys variants)
- **3 Brightness** — Strategy 5 greedy stacks, deduplicated by mutation position overlap

## Reproducibility Notes

- Random seed is fixed (`SEED = 42`) for all sampling and model training.
- ESM-2 weights are downloaded automatically on first run to `~/.cache/torch/hub/`.
- Training set is subsampled to 8000 rows (`MAX_TRAIN_SAMPLES`) for speed; increase for higher RF accuracy.
- The RF predictor achieves R2 ≈ 0.30 on held-out validation — sufficient for ranking but not absolute prediction.
  Thermal sequences (Seq 2–3) are force-included because the training set contains very few multi-Cys variants,
  causing systematic RF underestimation for those designs.

## Project Structure

```
iGEM_GFP_Agent/
├── data/
│   ├── GFP_data.xlsx
│   ├── AAseqs.txt
│   └── 2WUR.pdb
├── scripts/
│   └── agent_main.py
├── outputs/
│   ├── final_submission.csv
│   ├── submission.csv
│   ├── valid_candidates.csv
│   └── run_log.txt
├── Exclusion_List.csv
└── README.md
```
