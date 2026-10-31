"""Run the whole pipeline: raw data -> merged dataset -> descriptors -> selection -> models -> figures.

Usage (from the repository root):
    python run_all.py --xtb <path to xtb>   # reuse slow steps already saved in outputs/
    python run_all.py --xtb <xtb> --fresh   # recompute everything (about 2 h)
    python run_all.py --precomputed         # without xtb/Morfeus: read the Sterimol and xTB
                                            # descriptors from data/precomputed_descriptors.csv

Sterimol needs Morfeus (pip install morfeus-ml) and the xTB descriptors need xtb 6.7.1
(conda install -c conda-forge xtb=6.7.1); see chbory/qm.py.

Inputs (data/):
    CH_borylation_ChemAI_assessment.csv  base dataset (not included, see README)
    new_reactions.csv                    the ten added reactions (Table S2)
    precomputed_descriptors.csv          the Sterimol and xTB descriptors as used for the paper
                                         (only read with --precomputed)

Outputs (outputs/):
    data/     merged and cleaned dataset, site table, descriptor table, test reaction ids
    results/  every table behind the numbers in the paper
    figures/  Fig. 1-6 (PNG, and PDF with editable text)
"""

from __future__ import annotations

import argparse
import ast
import itertools
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chbory import (  # noqa: E402
    CANDIDATE_GROUPS,
    add_candidate_descriptors,
    add_extended_features,
    backward_elimination,
    bootstrap_ci,
    build_feature_table,
    build_site_table,
    clean_reactions,
    compare_runs,
    grid_search,
    holdout_split,
    is_borylated,
    is_major,
    load_dataset,
    make_model,
    merge_datasets,
    out_of_fold_scores,
    predict_scores,
    publication_cv,
    publication_groups,
    selected_groups,
    steric_baseline,
    system_class,
    top1_by_reaction,
    tuning_grid,
)
from chbory.qm import find_xtb, qm_descriptors  # noqa: E402

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")  # UFF typing warnings for B and S(VI) during 3D embedding
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

DATA = HERE / "data"
OUT = HERE / "outputs"
OUT_DATA, RESULTS, FIGURES = OUT / "data", OUT / "results", OUT / "figures"
N_ASSIGNMENTS = 10  # random publication-to-fold assignments for publication-grouped CV
PAPER = [f"paper_{i}" for i in range(N_ASSIGNMENTS)]
SELECTION_MODEL = "random_forest"
TOLERANCE = 0.005
FRESH = False  # set from the command line
XTB = None  # path to xtb, or None with --precomputed
WORKERS = None  # parallel processes for the xTB calculations (default: all cores)

MODELS = {"logreg": "Logistic regression", "naive_bayes": "Naive Bayes", "knn": "k-nearest neighbours",
          "svm": "SVM (RBF)", "mlp": "MLP", "decision_tree": "Decision tree",
          "random_forest": "Random forest", "gradient_boosting": "Gradient boosting",
          "extra_trees": "Extra trees"}
SYSTEMS = ["directed", "undirected Ir", "other"]

# Figure style (Nature-like): deep blue for the model, greys for baselines;
# Arial 7 pt, no grid, left and bottom spines only. Widths: 7.2 in = double column, 3.5 in = single column
BLUE, LIGHT_BLUE = "#0F4D92", "#B4C0E4"
GREYS = ["#CFCECE", "#A3A3A3", "#767676"]
INK, MUTED = "#272727", "#4D4D4D"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
    "axes.titlesize": 7, "axes.titleweight": "bold", "axes.labelsize": 7, "axes.linewidth": 0.8,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "legend.fontsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def section(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}", flush=True)


def show(frame):
    print(frame.to_string(), flush=True)


def cached(path, compute):
    """Load a saved table, or compute and save it (always with --fresh)."""
    if path.exists() and not FRESH:
        print(f"loaded {path.relative_to(HERE)}")
        return pd.read_csv(path)
    compute().to_csv(path, index=False)
    print(f"saved {path.relative_to(HERE)}")
    # Read back: pandas can parse a float one unit in the last place off the value it wrote, and
    # gradient boosting is sensitive to that; this way a fresh run and a rerun get the same input
    return pd.read_csv(path)


def save_table(frame, name, **kwargs):
    frame.to_csv(RESULTS / name, **kwargs)
    print(f"saved outputs/results/{name}")


def save_figure(fig, name):
    """PNG for viewing, PDF with editable text for the paper."""
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved outputs/figures/{name}.png (+ .pdf)")


# ----------------------------------------------------------------------------------------------------
# 1-3. Data: merge, clean, sites, labels, test split
# ----------------------------------------------------------------------------------------------------
def load_data():
    section("1. Merge and clean the base dataset and the ten added reactions")
    course = load_dataset(DATA / "CH_borylation_ChemAI_assessment.csv")
    added = load_dataset(DATA / "new_reactions.csv")
    merged = merge_datasets(course, added)
    reactions = clean_reactions(merged)
    show(pd.DataFrame({"product rows": [len(merged), len(reactions)],
                       "reactions": [merged["rxn_id"].nunique(), reactions["rxn_id"].nunique()]},
                      index=["before cleaning", "after cleaning"]))
    print("removed:", sorted(set(merged["rxn_id"]) - set(reactions["rxn_id"])))
    print("reactions per source:", reactions.drop_duplicates("rxn_id")["source"].value_counts().to_dict())
    print("publications:", reactions["source_DOI"].nunique())
    reactions.to_csv(OUT_DATA / "reactions_merged_clean.csv", index=False)
    print("saved outputs/data/reactions_merged_clean.csv (one row per product, base + added reactions)")

    section("2. Site table and labels")
    sites = build_site_table(reactions)
    borylated = is_borylated(sites)
    balance = pd.DataFrame({"sites": sites.groupby("hybridization").size(),
                            "borylated": borylated.groupby(sites["hybridization"]).sum()})
    balance.loc["all"] = balance.sum()
    balance["share borylated"] = (balance["borylated"] / balance["sites"]).round(3)
    show(balance)
    n_sites = sites.groupby("rxn_id").size()
    n_major = is_major(sites).groupby(sites["rxn_id"]).sum()
    print(f"sites per reaction: median {n_sites.median():.0f}, range {n_sites.min()}-{n_sites.max()}")
    print("reactions without a major site:", int((n_major == 0).sum()),
          "| with two major sites:", n_major[n_major > 1].index.tolist())
    sites.assign(borylated=borylated, major=is_major(sites)).to_csv(OUT_DATA / "sites.csv", index=False)
    print("saved outputs/data/sites.csv (one row per candidate C-H site, with labels)")

    section("3. Test set: 20% of reactions, split by reaction before model development (seed 42)")
    train_idx, test_idx = holdout_split(sites)
    test_ids = sorted(sites.iloc[test_idx]["rxn_id"].unique())
    (OUT_DATA / "test_reaction_ids.txt").write_text("\n".join(test_ids) + "\n")
    train = sites.iloc[train_idx].reset_index(drop=True)
    test = sites.iloc[test_idx].reset_index(drop=True)
    show(pd.DataFrame(
        {"reactions": [train["rxn_id"].nunique(), test["rxn_id"].nunique()],
         "sites": [len(train), len(test)],
         "share borylated": [is_borylated(train).mean().round(3), is_borylated(test).mean().round(3)]},
        index=["train", "test"]))
    print("saved outputs/data/test_reaction_ids.txt")
    return reactions, sites, train_idx, test_idx, train, test


# ----------------------------------------------------------------------------------------------------
# 4. Candidate descriptors (57 in 12 groups), Table 1, Fig. 1
# ----------------------------------------------------------------------------------------------------
def build_descriptors(reactions, sites, train_idx, test_idx):
    section("4. Candidate descriptors (57 in 12 groups)")

    def compute_features():
        built = add_extended_features(build_feature_table(sites, reactions), sites)
        return pd.concat([sites[["rxn_id", "idx"]], built], axis=1)

    print("RDKit descriptors (3D conformers; about 5 min the first time)")
    features = cached(OUT_DATA / "rdkit_descriptors.csv", compute_features)
    assert (features[["rxn_id", "idx"]].to_numpy() == sites[["rxn_id", "idx"]].to_numpy()).all(), \
        "saved descriptors do not match the site table: run with --fresh"

    if XTB is None:
        print("Sterimol and xTB descriptors read from data/precomputed_descriptors.csv (--precomputed)")
        qm = pd.read_csv(DATA / "precomputed_descriptors.csv")
    else:
        print(f"Sterimol (Morfeus) and GFN2-xTB descriptors with {XTB}\n"
              "(about 30 min on 24 cores the first time; each substrate is saved in outputs/qm_cache/)")
        qm = cached(OUT_DATA / "qm_descriptors.csv",
                    lambda: qm_descriptors(sites, XTB, OUT / "qm_cache", WORKERS,
                                           unrounded=set(reactions.loc[reactions["source"] == "added", "rxn_id"])))
        reference = pd.read_csv(DATA / "precomputed_descriptors.csv").merge(qm, on=["rxn_id", "idx"])
        diff = max((reference[f"{c}_x"] - reference[f"{c}_y"]).abs().max() for c in qm.columns[2:])
        print(f"largest difference from the paper values (data/precomputed_descriptors.csv): {diff:.2g}")
    candidates = add_candidate_descriptors(features, sites, qm)
    print("deprotonation energy not computed for",
          int(qm.merge(sites[["rxn_id", "idx"]])["deprot"].isna().sum()), "sites (set to 0)")
    pd.concat([sites[["rxn_id", "idx"]], candidates], axis=1).to_csv(
        OUT_DATA / "candidate_descriptors.csv", index=False)
    print("saved outputs/data/candidate_descriptors.csv (57 candidate descriptors per site)")

    table1 = pd.DataFrame({"n": {k: len(v) for k, v in CANDIDATE_GROUPS.items()},
                           "descriptors": {k: ", ".join(v) for k, v in CANDIDATE_GROUPS.items()}})
    table1.loc["total", "n"] = table1["n"].sum()
    table1["n"] = table1["n"].astype(int)
    show(table1)
    save_table(table1, "table1_candidate_descriptors.csv", index_label="group")

    X_train = candidates.iloc[train_idx].reset_index(drop=True)
    X_test = candidates.iloc[test_idx].reset_index(drop=True)
    print("training matrix:", X_train.shape, "| test matrix:", X_test.shape)
    return features, X_train, X_test


def figure1(reactions, sites, features):
    conditions = features.groupby("rxn_id", sort=False).first()
    catalysts = reactions.drop_duplicates("rxn_id").set_index("rxn_id")["catalyst_1_smiles"]

    def system_name(row):
        if row["metal_free"] == 1:
            return "BBr$_3$\n(metal-free)" if row["boron_BBr3"] == 1 else "borane catalyst\n(metal-free)"
        named = next((m for m in ["Ir", "Pd", "Co", "Ni", "Pt"] if row[f"metal_{m}"] == 1), None)
        if named:
            return named
        # Metals without their own feature column (Fe in added_005): read the element from the catalyst
        catalyst = Chem.MolFromSmiles(catalysts[row.name], sanitize=False)
        return next(a.GetSymbol() for a in catalyst.GetAtoms()
                    if a.GetSymbol() not in ("C", "H", "O", "N", "P", "Cl", "Br"))

    systems = conditions.apply(system_name, axis=1).value_counts().sort_values()
    hyb = pd.crosstab(sites["hybridization"].map({"SP2": "sp²", "SP3": "sp³"}), is_borylated(sites))
    save_table(systems.rename("reactions").rename_axis("catalytic system").iloc[::-1].to_frame(),
               "fig1a_catalytic_systems.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax1.barh(systems.index, systems.values, color=BLUE, height=0.6)
    for y, v in enumerate(systems.values):
        ax1.text(v + 5, y, str(v), va="center", fontsize=6.5, color=INK)
    ax1.set_xlabel("Reactions")
    ax1.set_title("a  Catalytic systems", loc="left")

    hyb = hyb.loc[["sp²", "sp³"]]
    x = np.arange(len(hyb))
    for j, (label, column, color) in enumerate([("borylated", True, BLUE), ("not borylated", False, GREYS[0])]):
        xs = x + (j - 0.5) * 0.36
        ax2.bar(xs, hyb[column], 0.34, color=color, label=label)
        for xi, v, total in zip(xs, hyb[column], hyb.sum(axis=1)):
            text = f"{v}\n({v / total:.0%})" if column else str(v)
            ax2.text(xi, v + 30, text, ha="center", va="bottom", fontsize=6.5, color=INK)
    ax2.set_xticks(x, hyb.index)
    ax2.set_ylim(0, hyb.to_numpy().max() * 1.15)
    ax2.set_ylabel("Candidate C–H sites")
    ax2.set_title("b  Sites and labels", loc="left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, "fig1_dataset")


# ----------------------------------------------------------------------------------------------------
# 5. Baselines
# ----------------------------------------------------------------------------------------------------
def baselines(train, train_full):
    section("5. Baselines on the training set (random ranking, steric rule)")
    rxn_class = pd.Series(system_class(train_full), index=train.index).groupby(train["rxn_id"]).first()
    rules = {"random": np.zeros(len(train)),  # all sites tie: expected accuracy of a random pick
             "steric rule": steric_baseline(train_full)}
    rule_top1 = pd.DataFrame({name: top1_by_reaction(train, s) for name, s in rules.items()})
    table = rule_top1.groupby(rxn_class).mean().T
    table.insert(0, "all", rule_top1.mean())
    table.columns = [f"{c} (n={int((rxn_class == c).sum())})" if c != "all" else "all" for c in table.columns]
    show(table.round(3))
    save_table(table.round(3), "baselines_training_set.csv", index_label="baseline")
    return rule_top1, rxn_class


# ----------------------------------------------------------------------------------------------------
# 6. Descriptor selection: group-wise backward elimination with a fixed random forest (Fig. 2)
# ----------------------------------------------------------------------------------------------------
def select_descriptors(X_train, train, assignments, rule_top1):
    section("6. Backward elimination of descriptor groups (about 40 min the first time)")
    elimination = cached(RESULTS / "selection_backward_elimination.csv", lambda: backward_elimination(
        make_model(SELECTION_MODEL), X_train, train, CANDIDATE_GROUPS, assignments, tolerance=TOLERANCE,
        verbose=True))
    elimination["removed"] = elimination["removed"].astype(bool)
    selected = selected_groups(elimination)
    final_features = [c for g in selected for c in CANDIDATE_GROUPS[g]]

    path = elimination[elimination["removed"]][["round", "group", "n_features", "top1", "delta", "better_in"]]
    start = elimination.groupby("round")["current_top1"].first()
    print(f"all 12 groups (57 descriptors): top-1 {start.iloc[0]:.3f}")
    show(path.rename(columns={"group": "removed group", "top1": "top-1 after removal"}).round(3)
         .set_index("round"))
    print(f"selected {len(selected)} groups, {len(final_features)} descriptors:", selected)
    print("last round (every removal costs more than the tolerance):")
    show(elimination[elimination["round"] == elimination["round"].max()]
         [["group", "n_features", "top1", "delta", "better_in"]].sort_values("delta", ascending=False)
         .round(3).set_index("group"))
    figure2(elimination, path, start, selected, final_features, rule_top1)
    return selected, final_features


def figure2(elimination, path, start, selected, final_features, rule_top1):
    group_labels = {"topological": "Topological counts", "buried_volume": "Buried volume", "sasa": "H surface area",
                    "sterimol": "Sterimol", "gasteiger": "Gasteiger charges", "fukui": "Fukui indices",
                    "deprot": "Deprotonation energy", "wbo": "Wiberg bond order",
                    "heteroatoms": "Heteroatom positions", "directing": "Directing-group distances",
                    "conditions": "Reaction conditions", "rule": "Rule score"}
    factors = {"Steric": ["topological", "buried_volume", "sasa", "sterimol"],
               "Electronic": ["gasteiger", "fukui", "deprot", "wbo", "heteroatoms"],
               "Directing": ["directing"], "Conditions": ["conditions"], "Rule score": ["rule"]}
    order = [g for groups in factors.values() for g in groups]
    assert set(order) == set(CANDIDATE_GROUPS)
    rounds = sorted(elimination["round"].unique())
    delta = elimination.pivot(index="group", columns="round", values="delta").reindex(order)
    removed_in = path.set_index("group")["round"]
    block_starts = np.cumsum([0] + [len(g) for g in factors.values()])  # first row of each factor

    fig = plt.figure(figsize=(7.2, 4.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.45], hspace=0.42)
    top = outer[0].subgridspec(1, 4, width_ratios=[1.0, 0.16, 1.9, 0.06], wspace=0.04)
    ax1, ax2, cax = fig.add_subplot(top[0]), fig.add_subplot(top[2]), fig.add_subplot(top[3])
    ax3 = fig.add_subplot(outer[1])

    # a: descriptors per group; solid = kept, hatched = removed (round given)
    y = np.arange(len(order))
    for i, g in enumerate(order):
        n, gone = len(CANDIDATE_GROUPS[g]), g in removed_in.index
        ax1.barh(i, n, 0.62, color="white" if gone else GREYS[2], edgecolor=GREYS[2],
                 hatch="//////" if gone else None, lw=0.8)
        ax1.text(n + 0.4, i, f"removed, round {removed_in[g]}" if gone else "kept", va="center", fontsize=6,
                 color=MUTED if gone else INK, fontweight="normal" if gone else "bold")
    for factor, first in zip(factors, block_starts[:-1]):
        ax1.text(26.5, first, factor, ha="right", va="center", fontsize=6, style="italic", color=MUTED)
    for b in block_starts[1:-1]:
        ax1.axhline(b - 0.5, color=GREYS[0], lw=0.6)
    ax1.set_yticks(y, [f"{group_labels[g]} ({len(CANDIDATE_GROUPS[g])})" for g in order])
    ax1.set_ylim(len(order) - 0.5, -0.5)
    ax1.set_xlim(0, 27)
    ax1.set_xticks([0, 8, 16])
    ax1.set_xlabel("Descriptors per group")
    ax1.set_title("a  12 descriptor groups (57 descriptors)", loc="left")
    ax1.tick_params(axis="y", length=0)

    # b: Δ top-1 of removing each remaining group; blue = within the tolerance (may be removed), red = worse
    limit = np.nanmax(np.abs(delta.to_numpy()))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=-TOLERANCE, vmax=limit)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#F2F2F2")
    image = ax2.imshow(np.ma.masked_invalid(delta.to_numpy()), cmap=cmap, norm=norm, aspect="auto")
    for i, g in enumerate(order):
        for j, r in enumerate(rounds):
            value = delta.loc[g, r]
            if not np.isnan(value):
                ax2.text(j, i, f"{value:+.3f}".replace("-", "−"), ha="center", va="center", fontsize=5.5,
                         color="white" if abs(norm(value) - 0.5) > 0.35 else INK)
    for b in block_starts[1:-1]:
        ax2.axhline(b - 0.5, color="white", lw=2)
    for g, r in removed_in.items():
        ax2.add_patch(plt.Rectangle((rounds.index(r) - 0.5, order.index(g) - 0.5), 1, 1, fill=False,
                                    edgecolor=INK, lw=1.3))
    ax2.set_xticks(range(len(rounds)), rounds)
    ax2.set_yticks(y, [])
    ax2.set_xlabel("Round")
    ax2.set_title(f"b  Δ top-1 when a group is removed (removed if Δ ≥ −{TOLERANCE})", loc="left")
    ax2.spines[["left", "bottom"]].set_visible(False)
    ax2.tick_params(length=0)
    bar = fig.colorbar(image, cax=cax)
    bar.set_ticks([-0.06, -0.04, -0.02, -TOLERANCE, 0.02])
    bar.set_ticklabels(["−0.06", "−0.04", "−0.02", f"−{TOLERANCE}", "+0.02"])
    bar.set_label("Δ top-1 vs. set at start of round")
    bar.ax.tick_params(labelsize=6)

    # c: accuracy after each removal
    values = [start.iloc[0]] + path["top1"].tolist()
    labels = ["Start\n(12 groups)"] + ["− " + group_labels[g].replace(" ", "\n", 1) for g in path["group"]]
    ax3.plot(range(len(values)), values, color=BLUE, marker="o", ms=4, lw=1.2)
    for i, v in enumerate(values):
        ax3.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=6, color=INK)
    ax3.axhline(rule_top1["steric rule"].mean(), color=GREYS[2], ls="--", lw=0.8)
    ax3.text(len(values) - 1, rule_top1["steric rule"].mean() + 0.008, "steric rule", ha="right", fontsize=6,
             color=MUTED)
    ax3.set_xticks(range(len(values)), labels, fontsize=6)
    ax3.set_xlim(-0.4, len(values) - 0.6)
    ax3.set_ylim(0.35, max(values) + 0.045)
    ax3.set_ylabel("Top-1 accuracy")
    ax3.set_title(f"c  Removal path: {len(selected)} groups ({len(final_features)} descriptors) selected",
                  loc="left")
    save_figure(fig, "fig2_backward_elimination")


# ----------------------------------------------------------------------------------------------------
# 7. Model comparison (Fig. 3) and tuning of the best model (Table S3)
# ----------------------------------------------------------------------------------------------------
def compare_models(X_sel, train, assignments, rule_top1):
    section("7. Nine classifiers on the selected descriptors (about 15 min the first time)")

    def compute():
        rows = []
        for name in MODELS:
            model = make_model(name)
            row = {"model": name,
                   "top1_reaction_cv": top1_by_reaction(train, out_of_fold_scores(model, X_sel, train)).mean(),
                   **dict(zip(PAPER, publication_cv(model, X_sel, train, assignments)))}
            rows.append(row)
            print(f"{name:18s} publication CV {np.mean([row[c] for c in PAPER]):.3f}", flush=True)
        return pd.DataFrame(rows)

    model_cv = cached(RESULTS / "model_comparison.csv", compute)
    model_cv["top1_paper_cv"] = model_cv[PAPER].mean(axis=1)
    model_cv["paper_sd"] = model_cv[PAPER].std(axis=1)
    base = pd.DataFrame({"model": list(rule_top1), "top1_reaction_cv": rule_top1.mean().to_numpy(),
                         "top1_paper_cv": rule_top1.mean().to_numpy(), "paper_sd": 0.0})
    show(pd.concat([model_cv, base])[["model", "top1_paper_cv", "paper_sd", "top1_reaction_cv"]]
         .sort_values("top1_paper_cv", ascending=False).round(3).set_index("model"))
    best = model_cv.loc[model_cv["top1_paper_cv"].idxmax(), "model"]
    print("best model (publication-grouped CV):", best)

    # Fig. 3: publication-grouped CV only (the criterion used for every choice)
    cv = model_cv.sort_values("top1_paper_cv").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for i, row in cv.iterrows():
        runs = row[PAPER].astype(float)
        ax.plot([runs.min(), runs.max()], [i, i], color=LIGHT_BLUE, lw=0.8, zorder=1)
        ax.scatter(runs, np.full(len(PAPER), i), s=7, color=LIGHT_BLUE, linewidths=0, zorder=2)
        ax.text(runs.max() + 0.012, i, f"{row['top1_paper_cv']:.3f}", va="center", fontsize=6, color=INK)
    ax.scatter(cv["top1_paper_cv"], cv.index, s=28, color=BLUE, zorder=3)
    for (name, value), style in zip(rule_top1.mean().items(), [":", "--"]):
        ax.plot([value, value], [-0.6, len(cv) - 0.45], color=GREYS[2], ls=style, lw=0.8, zorder=0)
        ax.text(value, len(cv) - 0.4, name, ha="center", va="bottom", fontsize=6, color=MUTED)
    ax.set_yticks(cv.index, [MODELS[m] for m in cv["model"]])
    ax.set_ylim(-0.6, len(cv) + 0.05)
    ax.set_xlim(0.15, 0.7)
    ax.set_xlabel("Top-1 accuracy (publication-grouped CV)")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save_figure(fig, "fig3_model_comparison")
    return model_cv, best


def tune(best, X_sel, train, assignments, n_features):
    section("7.1 Tuning of the best model (Table S3)")
    grid = tuning_grid(best, n_features)

    def compute():
        table = grid_search(make_model(best), grid, X_sel, train, assignments)
        # one text column, so that None, strings and tuples come back exactly from the csv
        params = [repr(dict(zip(grid, values))) for values in itertools.product(*grid.values())]
        return pd.concat([pd.Series(params, name="params"), table[PAPER]], axis=1)

    tuning = cached(RESULTS / "tuning_runs.csv", compute)
    tuning["params"] = tuning["params"].map(ast.literal_eval)
    for k in grid:  # as text for the table, so that None shows as "None"
        tuning[k] = tuning["params"].map(lambda p: str(p[k]))
    defaults = {k: make_model(best).get_params()[k] for k in grid}
    is_default = tuning["params"].map(lambda p: p == defaults)
    default_runs = tuning.loc[is_default, PAPER].to_numpy(float)[0]
    tuning["top1_paper_cv"] = tuning[PAPER].mean(axis=1)
    tuning[["diff_vs_default", "better_in"]] = [compare_runs(r, default_runs) for r in tuning[PAPER].to_numpy(float)]
    tuning["default"] = is_default
    tuning = tuning.sort_values("top1_paper_cv", ascending=False).reset_index(drop=True)
    tuning["better_in"] = tuning["better_in"].astype(int)
    table_s3 = tuning[list(grid) + ["default", "top1_paper_cv", "diff_vs_default", "better_in"]].round(3)
    show(table_s3)
    save_table(table_s3, "table_s3_tuning.csv", index=False)

    top = tuning.iloc[0]
    adopt = top["diff_vs_default"] > 0.02 and top["better_in"] >= 9
    final_params = top["params"] if adopt else defaults
    print("adopted:" if adopt else "kept defaults:", final_params)
    return final_params, adopt


# ----------------------------------------------------------------------------------------------------
# 8. Comparison with the baselines on the training set (Fig. 4)
# ----------------------------------------------------------------------------------------------------
def paired_summary(frame, columns, gains, groups):
    """Mean and 95% CI (bootstrap over ``groups``) of each column and of each paired difference a − b."""
    values = frame[columns].to_numpy(float)
    diffs = np.column_stack([(frame[a] - frame[b]).to_numpy(float) for a, b in gains])

    def stat(idx):
        return np.r_[values[idx].mean(axis=0), diffs[idx].mean(axis=0)]

    low, high = bootstrap_ci(stat, groups)
    names = columns + [f"{a} − {b}" for a, b in gains]
    return pd.DataFrame({"mean": stat(np.arange(len(frame))), "low": low, "high": high}, index=names)


def comparison_bars(ax, tables, labels, colors, gains):
    """Grouped bars with 95% CIs; brackets mark each paired difference (* if its CI excludes zero, else n.s.)."""
    k = len(colors)
    width = 0.84 / k
    for i, table in enumerate(tables):
        xs = {m: i + (j - (k - 1) / 2) * width for j, m in enumerate(colors)}
        for m, color in colors.items():
            v, lo, hi = table.loc[m, ["mean", "low", "high"]]
            ax.bar(xs[m], v, width * 0.92, color=color, label=m if i == 0 else None, edgecolor="white",
                   linewidth=0.5)
            ax.errorbar(xs[m], v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2,
                        capthick=0.6)
            ax.text(xs[m], hi + 0.015, f"{v:.2f}", ha="center", fontsize=5.5, color=INK)
        top = table.loc[list(colors), "high"].max() + 0.09
        for a, b in gains:
            lo, hi = table.loc[f"{a} − {b}", ["low", "high"]]
            ax.plot([xs[b], xs[b], xs[a], xs[a]], [top - 0.02, top, top, top - 0.02], color=INK, lw=0.6)
            mark = "*" if lo > 0 or hi < 0 else "n.s."
            ax.text((xs[a] + xs[b]) / 2, top + 0.012, mark, ha="center", fontsize=7 if mark == "*" else 5.5,
                    color=INK)
            top += 0.09
    ax.set_xticks(range(len(tables)), labels)
    ax.set_xlim(-0.6, len(tables) - 0.4)


def two_panel_figure(left, right, ratios, ylabel, colors, gains, name):
    """Two panels sharing the y axis and one legend: left/right are (title, tables, tick labels)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, gridspec_kw={"width_ratios": ratios})
    for ax, (title, tables, labels) in zip(axes, [left, right]):
        comparison_bars(ax, tables, labels, colors, gains)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(0, 1.25)
    axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[1].tick_params(axis="y", left=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(colors), loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1,
               columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, name)


def training_comparison(best, final_params, X_sel, train, assignments, rule_top1, rxn_class, reactions):
    section("8. Final model against the baselines on the training set (Fig. 4)")

    def per_reaction_publication_cv():
        model = make_model(best).set_params(**final_params)
        runs = [top1_by_reaction(train, out_of_fold_scores(model, X_sel, train, groups=g)) for g in assignments]
        return pd.concat(runs, axis=1).mean(axis=1).rename("model").rename_axis("rxn_id").reset_index()

    def per_reaction_reaction_cv():
        model = make_model(best).set_params(**final_params)
        return (top1_by_reaction(train, out_of_fold_scores(model, X_sel, train)).rename("model")
                .rename_axis("rxn_id").reset_index())

    model_rxn = cached(RESULTS / "per_reaction_publication_cv.csv", per_reaction_publication_cv).set_index("rxn_id")
    model_rxn_rcv = cached(RESULTS / "per_reaction_reaction_cv.csv", per_reaction_reaction_cv).set_index("rxn_id")
    pub_cv, rxn_cv = f"{MODELS[best]}, publication-grouped CV", f"{MODELS[best]}, reaction-grouped CV"
    by_class = rule_top1.join(model_rxn["model"].rename(pub_cv)).join(model_rxn_rcv["model"].rename(rxn_cv))
    publication = reactions.drop_duplicates("rxn_id").set_index("rxn_id")["source_DOI"]

    columns = ["random", "steric rule", pub_cv, rxn_cv]
    gains = [(pub_cv, "steric rule"), (rxn_cv, "steric rule")]
    classes = {"all": by_class} | {c: by_class[rxn_class.reindex(by_class.index) == c] for c in SYSTEMS}
    train_ci = {c: paired_summary(f, columns, gains, publication.reindex(f.index).to_numpy())
                for c, f in classes.items()}
    table = pd.concat(train_ci, names=["group", "quantity"])
    print("n:", {c: len(f) for c, f in classes.items()}, "| publications:",
          {c: publication.reindex(f.index).nunique() for c, f in classes.items()})
    show(table.round(3))
    save_table(table.round(3), "fig4_ci_training_by_system.csv")

    colors = {"random": GREYS[0], "steric rule": GREYS[2], pub_cv: BLUE, rxn_cv: LIGHT_BLUE}
    two_panel_figure(("a  Entire training set", [train_ci["all"]], [f"all reactions\n(n = {len(by_class)})"]),
                     ("b  By catalytic system", [train_ci[c] for c in SYSTEMS],
                      [f"{c}\n(n = {len(classes[c])})" for c in SYSTEMS]),
                     [1, 3], "Top-1 accuracy (cross-validation)", colors, gains, "fig4_training_set")
    return model_rxn, model_rxn_rcv, train_ci, gains


# ----------------------------------------------------------------------------------------------------
# 9. Test set, used once (Fig. 5); 10. prediction examples (Fig. 6)
# ----------------------------------------------------------------------------------------------------
def test_evaluation(best, final_params, final_features, X_sel, y_train, X_test, test, test_full):
    section("9. Final model on the test set (used once; Fig. 5)")
    final_model = make_model(best).set_params(**final_params).fit(X_sel, y_train)
    test_scores = predict_scores(final_model, X_test[final_features])
    methods = {"random": np.zeros(len(test)), "steric rule": steric_baseline(test_full), MODELS[best]: test_scores}
    test_top1 = pd.DataFrame({name: top1_by_reaction(test, s) for name, s in methods.items()})
    major_hyb = test.loc[test.groupby("rxn_id")["regio_fraction"].idxmax()].set_index("rxn_id")["hybridization"]

    # 95% CIs: paired bootstrap over test reactions, as the test set was drawn by reaction
    gains = [(MODELS[best], "steric rule")]
    groups = {"all": test_top1} | {h: test_top1[major_hyb.reindex(test_top1.index) == h] for h in ("SP2", "SP3")}
    test_ci = {g: paired_summary(f, list(methods), gains, f.index.to_numpy()) for g, f in groups.items()}
    table = pd.concat(test_ci, names=["group", "quantity"])
    print("n:", {g: len(f) for g, f in groups.items()})
    show(table.round(3))
    save_table(table.round(3), "fig5_ci_test.csv")
    save_table(test_top1, "per_reaction_test.csv", index_label="rxn_id")

    colors = {"random": GREYS[0], "steric rule": GREYS[2], MODELS[best]: BLUE}
    two_panel_figure(("a  All test reactions", [test_ci["all"]], [f"all reactions\n(n = {len(test_top1)})"]),
                     ("b  By major-site type", [test_ci["SP2"], test_ci["SP3"]],
                      [f"C(sp²)–H\n(n = {len(groups['SP2'])})", f"C(sp³)–H\n(n = {len(groups['SP3'])})"]),
                     [1, 2], "Top-1 accuracy (test set)", colors, gains, "fig5_test_set")
    return test_scores, test_top1, test_ci


def figure6(best, test, test_full, test_scores, test_top1):
    section("10. Prediction examples (Fig. 6)")
    test_class = pd.Series(system_class(test_full), index=test.index).groupby(test["rxn_id"]).first()
    correct = test_top1[MODELS[best]]
    picks = {"directed, correct": correct[(test_class == "directed") & (correct == 1)].index[0],
             "Ir, correct": correct[(test_class == "undirected Ir") & (correct == 1)].index[0],
             "incorrect": correct[correct == 0].index[0]}
    mols, highlights, legends = [], [], []
    for label, rxn in picks.items():
        rows = test[test["rxn_id"] == rxn]
        mol = Chem.MolFromSmiles(rows["substrate_smiles_mapped"].iloc[0])
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        for idx, p in zip(rows["idx"], test_scores[rows.index]):
            mol.GetAtomWithIdx(int(idx)).SetProp("atomNote", f"{p:.2f}")
        mols.append(mol)
        highlights.append([int(i) for i in rows.loc[is_major(rows), "idx"]])
        legends.append(f"{label} ({rxn})")
    options = Draw.MolDrawOptions()
    options.annotationFontScale = 0.9
    options.legendFontSize = 20
    image = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(500, 420), legends=legends, drawOptions=options,
                                 highlightAtomLists=highlights, returnPNG=False,
                                 highlightAtomColors=[{i: (0.545, 0.812, 0.545) for i in h} for h in highlights])
    image.save(FIGURES / "fig6_examples.png")
    print("saved outputs/figures/fig6_examples.png:", picks)


# ----------------------------------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding="utf-8")  # "−" and "–" in the tables; the Windows console default is not UTF-8
    global FRESH, XTB, WORKERS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fresh", action="store_true", help="recompute every step, ignoring saved outputs")
    parser.add_argument("--xtb", help="path to the xtb program (default: XTB environment variable or PATH)")
    parser.add_argument("--precomputed", action="store_true",
                        help="read the Sterimol and xTB descriptors from data/precomputed_descriptors.csv")
    parser.add_argument("--workers", type=int, help="parallel processes for xTB (default: all cores)")
    args = parser.parse_args()
    FRESH, WORKERS = args.fresh, args.workers
    if not args.precomputed:
        XTB = find_xtb(args.xtb)
        import morfeus  # noqa: F401  (pip install morfeus-ml; or use --precomputed)
    for d in (OUT_DATA, RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    print("Python", sys.version.split()[0], "| scikit-learn", sklearn.__version__,
          "(the paper numbers were produced with scikit-learn 1.8.0)")

    reactions, sites, train_idx, test_idx, train, test = load_data()
    features, X_train, X_test = build_descriptors(reactions, sites, train_idx, test_idx)
    figure1(reactions, sites, features)
    train_full = features.iloc[train_idx].reset_index(drop=True)
    test_full = features.iloc[test_idx].reset_index(drop=True)
    y_train = is_borylated(train).to_numpy()

    rule_top1, rxn_class = baselines(train, train_full)
    assignments = publication_groups(train, reactions, N_ASSIGNMENTS)
    selected, final_features = select_descriptors(X_train, train, assignments, rule_top1)
    X_sel = X_train[final_features]
    model_cv, best = compare_models(X_sel, train, assignments, rule_top1)
    final_params, adopt = tune(best, X_sel, train, assignments, len(final_features))
    model_rxn, model_rxn_rcv, train_ci, train_gains = training_comparison(
        best, final_params, X_sel, train, assignments, rule_top1, rxn_class, reactions)
    test_scores, test_top1, test_ci = test_evaluation(
        best, final_params, final_features, X_sel, y_train, X_test, test, test_full)
    figure6(best, test, test_full, test_scores, test_top1)

    section("11. Summary")
    best_row = model_cv.set_index("model").loc[best]

    def with_ci(table, row):
        mean, low, high = table.loc[row, ["mean", "low", "high"]]
        return f"{mean:.3f} [{low:.3f}, {high:.3f}]"

    summary = pd.Series({
        "reactions (after cleaning, incl. added)": reactions["rxn_id"].nunique(),
        "publications": reactions["source_DOI"].nunique(),
        "training reactions": train["rxn_id"].nunique(),
        "test reactions": test["rxn_id"].nunique(),
        "candidate sites": len(sites),
        "share of sites borylated": round(is_borylated(sites).mean(), 3),
        "candidate descriptors": X_train.shape[1],
        "selected groups": ", ".join(selected),
        "selected descriptors": len(final_features),
        "random top-1 (training set)": round(rule_top1["random"].mean(), 3),
        "steric rule top-1 (training set)": round(rule_top1["steric rule"].mean(), 3),
        "best model": best,
        "best model top-1, publication-grouped CV (mean ± s.d.)":
            f"{best_row['top1_paper_cv']:.3f} ± {best_row['paper_sd']:.3f}",
        "best model top-1, reaction-grouped CV (defaults)": round(best_row["top1_reaction_cv"], 3),
        "tuned setting adopted": bool(adopt),
        "final model top-1, publication-grouped CV": round(model_rxn["model"].mean(), 3),
        "final model top-1, reaction-grouped CV": round(model_rxn_rcv["model"].mean(), 3),
        "final model top-1, test set (95% CI)": with_ci(test_ci["all"], MODELS[best]),
        "random top-1, test set": round(test_top1["random"].mean(), 3),
        "steric rule top-1, test set": round(test_top1["steric rule"].mean(), 3),
        **{f"training {g}: {a} − {b}": with_ci(train_ci[g], f"{a} − {b}")
           for g in ["all"] + SYSTEMS for a, b in train_gains},
        **{f"test {g}: {MODELS[best]} − steric rule": with_ci(test_ci[g], f"{MODELS[best]} − steric rule")
           for g in test_ci},
    }, name="value")
    show(summary.to_frame())
    save_table(summary.to_frame(), "summary.csv", index_label="quantity")
    print("\nDone. Dataset tables in outputs/data, results in outputs/results, figures in outputs/figures.")


if __name__ == "__main__":
    # Multithreaded histogram gradient boosting is not bit-reproducible between runs;
    # with one OpenMP thread it is deterministic and gives the paper numbers
    with threadpool_limits(limits=1, user_api="openmp"):
        main()
