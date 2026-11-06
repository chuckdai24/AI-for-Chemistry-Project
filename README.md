# Predicting the Site of C–H Borylation with a Random Forest: Gains over a Steric Rule

Chengzhi Dai, Casper van Koningsveld and Jordi Vermeer

Source code and data for the paper. A random forest ranks the C–H bonds of a substrate under given
reaction conditions and predicts the major borylation site across catalytic systems. Each site is
described by interpretable steric, electronic and directing-group descriptors, the reaction conditions
and a rule score; the model is compared with random ranking and a steric rule by top-1 accuracy.
`run_all.py` reproduces every number, table and figure of the paper, including the workflow schematic (Fig. 1),
with one command.

## Main results

637 reactions from 46 publications, 3,711 candidate C–H sites; 57 candidate descriptors in 12 groups,
of which backward elimination kept 30 in 7 groups.

| Top-1 accuracy | Random ranking | Steric rule | Random forest |
|---|---|---|---|
| Training set, publication-grouped CV | 0.21 | 0.37 | 0.58 |
| Training set, reaction-grouped CV | 0.21 | 0.37 | 0.81 |
| Test set (128 reactions) | 0.20 | 0.39 | 0.84 |

The gain over the steric rule is largest for directed reactions (0.84 vs. 0.13 in publication-grouped
CV); for undirected Ir catalysis it is not significant for unseen publications (0.54 vs. 0.46).

## Quick start

Requires Python ≥ 3.12 (tested with 3.13).

1. Download or clone this repository.
2. Install the packages: `pip install -r requirements.txt`
3. Run, from this folder:

```
python run_all.py --precomputed
```

This takes about 1.5 h and writes all tables, figures and the merged dataset to `outputs/`.
`--precomputed` reads the Sterimol and GFN2-xTB descriptors from `data/precomputed_descriptors.csv`,
so xtb is not needed; the results are identical to the paper.

## Recomputing the Sterimol and xTB descriptors

To calculate these descriptors as well, install the xtb 6.7.1 program and pass it with `--xtb`:

```
conda install -c conda-forge xtb=6.7.1          # into any conda environment
python run_all.py --xtb <path to xtb>           # first run about 2 h; later runs reuse the slow steps saved in outputs/
python run_all.py --xtb <path to xtb> --fresh   # recompute everything
```

On Windows the program is `<env>\Library\bin\xtb.exe`. `--xtb` can be left out if `xtb` is on the
PATH or the `XTB` environment variable points to it; `--workers N` limits the number of parallel xtb
processes (default: all cores). Sterimol uses Morfeus (in `requirements.txt`). The new values are
compared with `data/precomputed_descriptors.csv` in the output (largest difference about 1e-13).

The reported numbers were produced with the package versions in `requirements.txt`; other versions,
in particular of scikit-learn, can change the tree ensembles slightly.

## Data

| Path | What it is |
|---|---|
| `data/CH_borylation_ChemAI_assessment.csv` | Course dataset (744 product sites from 628 reactions), provided by the course and shared here with permission |
| `data/new_reactions.csv` | The ten reactions added from the literature (Table S2); `new_reactions_sources.csv` gives the source and evidence for each |
| `data/precomputed_descriptors.csv` | The Sterimol and GFN2-xTB descriptors as used in the paper; read only with `--precomputed`, otherwise used to check the new calculation |

## Code

| Path | What it is |
|---|---|
| `run_all.py` | The pipeline, sections 1–11 (see below) |
| `chbory/` | The package: reading and cleaning data, site enumeration, labels, descriptors, models, evaluation |

All descriptors are computed by `run_all.py`: Sterimol L, B1, B5 with Morfeus and the Fukui indices,
C–H deprotonation energy and Wiberg C–H bond order with GFN2-xTB (`chbory/qm.py`); everything else
(topological counts, buried volume, SASA, Gasteiger charges, heteroatom and directing-group positions,
reaction conditions, rule score) with RDKit. The xtb results of each substrate are saved in
`outputs/qm_cache/`, so an interrupted run continues where it stopped.

## What `run_all.py` does

| Section | Step | Paper | Output |
|---|---|---|---|
| 1 | Merge the course dataset and the ten added reactions; clean | Table S1 | `outputs/data/reactions_merged_clean.csv` |
| 2 | Enumerate C–H sites, merge equivalent carbons, label borylated and major sites | Section 2.1 | `outputs/data/sites.csv` |
| 3 | Set aside 20% of reactions as the test set (by reaction, seed 42) | Section 2.4 | `outputs/data/test_reaction_ids.txt` |
| 4 | 57 candidate descriptors in 12 groups, including Sterimol and xTB | Table 1, Fig. 2 | `outputs/data/qm_descriptors.csv`, `outputs/data/candidate_descriptors.csv`, `fig1_dataset` |
| 5 | Baselines: random ranking and the steric rule | Section 2.4 | `baselines_training_set.csv` |
| 6 | Group-wise backward elimination with a fixed random forest | Fig. 3 | `selection_backward_elimination.csv`, `fig2_backward_elimination` |
| 7 | Nine classifiers on the selected descriptors; tuning of the best | Fig. 4, Table S3 | `model_comparison.csv`, `table_s3_tuning.csv`, `fig3_model_comparison` |
| 8 | Final model vs. baselines on the training set, by catalytic system | Fig. 5 | `fig4_ci_training_by_system.csv`, `fig4_training_set` |
| 9 | Final model on the test set (used once) | Fig. 6 | `fig5_ci_test.csv`, `fig5_test_set` |
| 10 | Workflow schematic: dataset size, descriptor groups kept, and the site scores of a test reaction, all from this run | Fig. 1 | `fig0_workflow` |
| 11 | Key numbers | – | `summary.csv` |

Tables are in `outputs/results/`, figures (PNG and PDF) in `outputs/figures/`. The output file names
are numbered by pipeline step, not by figure in the paper (see the Paper column). All choices
(descriptors, model, hyperparameters) use publication-grouped five-fold CV on the training set,
repeated with ten random assignments of publications to folds; the test set is used only in section 9.

The slow steps (RDKit 3D descriptors, Sterimol and xTB descriptors, backward elimination, model comparison, tuning, out-of-fold
predictions) are saved in `outputs/` and reused on the next run unless `--fresh` is given.

## Citation

C. Dai, C. van Koningsveld and J. Vermeer, *Predicting the Site of C–H Borylation with a Random Forest:
Gains over a Steric Rule*, 2026.
