# Predicting the site of C–H borylation

```
python run_all.py --xtb <path to xtb>           # first run about 2 h; later runs reuse the slow steps saved in outputs/
python run_all.py --xtb <path to xtb> --fresh   # recompute everything
python run_all.py --precomputed                 # without xtb and Morfeus (see below)
```

`--xtb` can be left out if `xtb` is on the PATH or the `XTB` environment variable points to it.
`--workers N` limits the number of parallel xtb processes (default: all cores).

## Requirements

Python ≥ 3.10 with the packages in `requirements.txt` (including Morfeus for Sterimol), and the
xtb 6.7.1 program for the GFN2-xTB descriptors:

```
pip install -r requirements.txt
conda install -c conda-forge xtb=6.7.1     # into any conda environment; pass its xtb with --xtb
```

On Windows the program is `<env>\Library\bin\xtb.exe`.

The reported numbers were produced with scikit-learn 1.8.0; other versions can change the
tree ensembles slightly.

## Contents

| Path | What it is |
|---|---|
| `run_all.py` | The pipeline, sections 1–11 (see below) |
| `chbory/` | The project package: reading and cleaning data, site enumeration, labels, descriptors, models, evaluation |
| `data/CH_borylation_ChemAI_assessment.csv` | Course dataset (not to be redistributed, so not included here; place the course file at this path before running) |
| `data/new_reactions.csv` | The ten added reactions (Table S2); `new_reactions_sources.csv` gives the source and evidence for each |
| `data/precomputed_descriptors.csv` | The Sterimol and GFN2-xTB descriptors as used for the thesis; read only with `--precomputed`, otherwise used to check the new calculation |

All descriptors are computed by `run_all.py`: Sterimol L, B1, B5 with Morfeus and the Fukui indices,
C–H deprotonation energy and Wiberg C–H bond order with GFN2-xTB (`chbory/qm.py`); everything else
(topological counts, buried volume, SASA, Gasteiger charges, heteroatom and directing-group positions,
reaction conditions, rule score) with RDKit. The xtb results of each substrate are saved in
`outputs/qm_cache/`, so an interrupted run continues where it stopped.

## What `run_all.py` does

| Section | Step | Output |
|---|---|---|
| 1 | Merge the course data and the ten added reactions; clean (Table S1) | `outputs/data/reactions_merged_clean.csv` |
| 2 | Enumerate C–H sites, merge equivalent carbons, label borylated and major sites | `outputs/data/sites.csv` |
| 3 | Set aside 20% of reactions as the test set (by reaction, seed 42) | `outputs/data/test_reaction_ids.txt` |
| 4 | 57 candidate descriptors in 12 groups (Table 1), including Sterimol and xTB | `outputs/data/qm_descriptors.csv`, `outputs/data/candidate_descriptors.csv`, Fig. 1 |
| 5 | Baselines: random ranking and the steric rule | `baselines_training_set.csv` |
| 6 | Group-wise backward elimination with a fixed random forest | `selection_backward_elimination.csv`, Fig. 2 |
| 7 | Nine classifiers on the selected descriptors; tuning of the best (Table S3) | `model_comparison.csv`, `table_s3_tuning.csv`, Fig. 3 |
| 8 | Final model vs. baselines on the training set, by catalytic system | `fig4_ci_training_by_system.csv`, Fig. 4 |
| 9 | Final model on the test set (used once) | `fig5_ci_test.csv`, Fig. 5 |
| 10 | Prediction examples | Fig. 6 |
| 11 | Key numbers | `summary.csv` |

Tables are in `outputs/results/`, figures (PNG and PDF) in `outputs/figures/`. All choices
(descriptors, model, hyperparameters) use publication-grouped five-fold CV on the training set,
repeated with ten random assignments of publications to folds; the test set is used only in section 9.

The slow steps (RDKit 3D descriptors, Sterimol and xTB descriptors, backward elimination, model comparison, tuning, out-of-fold
predictions) are saved in `outputs/` and reused on the next run unless `--fresh` is given.
