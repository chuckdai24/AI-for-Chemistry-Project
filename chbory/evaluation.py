"""Splitting and metrics. All splits are grouped by reaction."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .models import RANDOM_STATE

MAJOR_THRESHOLD = 0.5


def is_major(sites: pd.DataFrame) -> pd.Series:
    """True for the experimental major site(s): ``regio_fraction >= 0.5``.

    Every reaction in the cleaned data has at least one such site; lit_pub_bo_185 has two
    (0.5 each).
    """
    return sites["regio_fraction"] >= MAJOR_THRESHOLD


def is_borylated(sites: pd.DataFrame) -> pd.Series:
    """True for every site that gave product (``regio_fraction > 0``): the training target."""
    return sites["regio_fraction"] > 0


def holdout_split(
    sites: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray]:
    """Split site rows into train and test indices, keeping each reaction on one side.

    ``test_size`` is the fraction of reactions (not of sites) in the test set.
    Output: ``(train_idx, test_idx)`` as positional indices into ``sites``.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return next(splitter.split(sites, groups=sites["rxn_id"]))


def split_by_ids(sites: pd.DataFrame, test_ids) -> tuple[np.ndarray, np.ndarray]:
    """Split site rows by a fixed list of test reactions (e.g. a frozen test set).

    Reactions not in ``test_ids`` (including any added later) go to the training set.
    Output: ``(train_idx, test_idx)`` as positional indices into ``sites``.
    """
    in_test = sites["rxn_id"].isin(set(test_ids)).to_numpy()
    return np.flatnonzero(~in_test), np.flatnonzero(in_test)


def publication_groups(
    sites: pd.DataFrame, reactions: pd.DataFrame, n_assignments: int = 10, n_splits: int = 5
) -> list[np.ndarray]:
    """Random assignments of publications to folds, for publication-grouped CV.

    With few publications, one fixed assignment gives a noisy estimate, so several are
    drawn (seed 0, 1, ...). Each assignment is one fold label per row of ``sites``, to be
    passed as ``groups`` to ``cross_validate`` or ``out_of_fold_scores``; all reactions of
    a publication (``source_DOI``) share a fold.
    """
    doi = sites["rxn_id"].map(reactions.groupby("rxn_id")["source_DOI"].first())
    papers = doi.unique()
    assignments = []
    for seed in range(n_assignments):
        order = np.random.default_rng(seed).permutation(papers)
        fold_of = {paper: i % n_splits for i, paper in enumerate(order)}
        assignments.append(doi.map(fold_of).to_numpy())
    return assignments


def top1_by_reaction(sites: pd.DataFrame, scores: np.ndarray) -> pd.Series:
    """Top-1 correctness per reaction: is the highest-scored site a major site?

    If several sites share the highest score, the result is the fraction of them that
    are major sites, i.e. the expected accuracy of picking one of them at random.
    This keeps rules with many ties (such as counting neighbours) from being favoured
    or penalised by the row order. Output: one value in [0, 1] per ``rxn_id``.
    """
    frame = pd.DataFrame(
        {"rxn_id": sites["rxn_id"].to_numpy(), "score": np.asarray(scores, dtype=float),
         "major": is_major(sites).to_numpy()}
    )
    best = frame.groupby("rxn_id", sort=False)["score"].transform("max")
    top = frame[frame["score"] == best]
    return top.groupby("rxn_id", sort=False)["major"].mean()


def top1_accuracy(sites: pd.DataFrame, scores: np.ndarray) -> float:
    """Fraction of reactions whose highest-scored site is the experimental major site.

    Inputs: the site table (needs ``rxn_id`` and ``regio_fraction``) and one score per row.
    Ties in score are handled as in ``top1_by_reaction``.
    """
    return float(top1_by_reaction(sites, scores).mean())


def site_metrics(sites: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    """Site-level metrics complementing top-1 accuracy.

    Both treat every site as one sample, pooled over reactions, with borylated sites
    (``regio_fraction > 0``) as positives:
    ``roc_auc`` (0.5 = random) and ``average_precision`` (random = share of positives).
    """
    y = is_borylated(sites)
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
    }


def out_of_fold_scores(
    model, X: pd.DataFrame, sites: pd.DataFrame, n_splits: int = 5, groups=None
) -> np.ndarray:
    """Score every site with a model that never saw its group (grouped k-fold).

    Same folds as ``cross_validate``; ``groups`` works the same way. Use the result with
    ``top1_by_reaction`` to break cross-validated accuracy down by subgroup, e.g. sp2
    and sp3 major sites.
    """
    y = is_borylated(sites).to_numpy()
    scores = np.empty(len(sites))
    for train, test in _folds(X, y, sites, n_splits, groups):
        fitted = clone(model).fit(X.iloc[train], y[train])
        scores[test] = fitted.predict_proba(X.iloc[test])[:, 1]
    return scores


def cross_validate(
    model, X: pd.DataFrame, sites: pd.DataFrame, n_splits: int = 5, groups=None
) -> pd.DataFrame:
    """Grouped k-fold cross-validation on the training set.

    Inputs: an unfitted scikit-learn classifier (e.g. from ``make_model``), the feature
    table and the site table, aligned row by row. The classifier is trained to recognise
    borylated sites (``regio_fraction > 0``); each fold is scored by its probability.
    GroupKFold keeps all sites of a group in the same fold. By default a group is a
    reaction; pass ``groups`` (one label per row, e.g. the source DOI) to keep whole
    papers together, which tests generalisation to new reaction systems.
    Output: one row per fold with ``top1_accuracy`` and the ``site_metrics``.
    """
    y = is_borylated(sites).to_numpy()
    rows = []
    for fold, (train, test) in enumerate(_folds(X, y, sites, n_splits, groups)):
        fitted = clone(model).fit(X.iloc[train], y[train])
        scores = fitted.predict_proba(X.iloc[test])[:, 1]
        test_sites = sites.iloc[test]
        rows.append(
            {"fold": fold, "n_reactions": test_sites["rxn_id"].nunique(),
             "top1_accuracy": top1_accuracy(test_sites, scores),
             **site_metrics(test_sites, scores)}
        )
    return pd.DataFrame(rows)


def publication_cv(model, X: pd.DataFrame, sites: pd.DataFrame, assignments) -> np.ndarray:
    """Top-1 accuracy of out-of-fold scores, once per publication-to-fold assignment.

    ``assignments`` is the output of ``publication_groups``. Output: one top-1 accuracy per
    assignment; their mean is the score used to compare settings.
    """
    return np.array(
        [top1_accuracy(sites, out_of_fold_scores(model, X, sites, groups=g)) for g in assignments]
    )


def compare_runs(new: np.ndarray, reference: np.ndarray) -> tuple[float, int]:
    """Compare two settings evaluated on the same assignments (``publication_cv``).

    Output: ``(delta, better_in)``, the difference of the means (new − reference) and the
    number of assignments on which ``new`` is strictly better.
    """
    new, reference = np.asarray(new, dtype=float), np.asarray(reference, dtype=float)
    return float(new.mean() - reference.mean()), int((new > reference).sum())


def bootstrap_ci(
    statistic, groups, n_boot: int = 5000, level: float = 0.95, random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap confidence interval, resampling whole groups with replacement.

    ``statistic`` maps an array of row positions (the rows of one resample, repeats allowed)
    to a number or a 1-D array. ``groups`` gives the group of every row: publications for a
    cluster bootstrap, or reactions. Paired comparisons put both methods in one statistic
    (e.g. the mean of model minus baseline), so they are resampled together.
    Output: ``(low, high)``, each a float or an array like the statistic.
    """
    _, codes = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(codes == k) for k in range(codes.max() + 1)]
    rng = np.random.default_rng(random_state)
    stats = [statistic(np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))]))
             for _ in range(n_boot)]
    low, high = np.percentile(stats, [50 * (1 - level), 50 * (1 + level)], axis=0)
    return low, high


def backward_elimination(
    model, X: pd.DataFrame, sites: pd.DataFrame, groups: dict[str, list[str]], assignments,
    tolerance: float = 0.005, verbose: bool = False,
) -> pd.DataFrame:
    """Remove descriptor groups one at a time while accuracy does not get worse.

    Starts from all ``groups`` (name -> columns of ``X``). Each round scores the current set
    minus each remaining group with ``publication_cv``. Of the groups whose removal changes
    the mean top-1 by at least ``-tolerance``, the one with the largest change is removed;
    the search stops when every removal costs more than ``tolerance`` (or one group is left).
    Output: one row per round and candidate group with ``round``, ``group``, ``n_features``
    (after removal), ``top1`` (mean over assignments), ``delta`` and ``better_in`` versus the
    current set, ``removed``, the current set's ``current_top1`` and one ``paper_i`` column
    per assignment. The groups never removed are the selection.
    """
    remaining = list(groups)
    current = publication_cv(model, X[[c for g in remaining for c in groups[g]]], sites, assignments)
    rows = []
    for round_ in range(1, len(groups)):
        trials = {}
        for name in remaining:
            columns = [c for g in remaining if g != name for c in groups[g]]
            trials[name] = (len(columns), publication_cv(model, X[columns], sites, assignments))
        deltas = {name: compare_runs(runs, current)[0] for name, (_, runs) in trials.items()}
        allowed = [name for name in remaining if deltas[name] >= -tolerance]
        removed = max(allowed, key=lambda name: deltas[name]) if allowed else None
        for name, (n, runs) in trials.items():
            delta, better = compare_runs(runs, current)
            rows.append({"round": round_, "group": name, "n_features": n, "top1": runs.mean(),
                         "delta": delta, "better_in": better, "removed": name == removed,
                         "current_top1": current.mean(),
                         **{f"paper_{i}": r for i, r in enumerate(runs)}})
        if verbose:
            print(f"round {round_}: current {current.mean():.3f}, removed {removed}"
                  + (f" (delta {deltas[removed]:+.3f})" if removed else ""))
        if removed is None or len(remaining) == 2:
            break
        remaining.remove(removed)
        current = trials[removed][1]
    return pd.DataFrame(rows)


def selected_groups(elimination: pd.DataFrame) -> list[str]:
    """The groups left after ``backward_elimination``: those of the last round not removed."""
    last = elimination[elimination["round"] == elimination["round"].max()]
    return last.loc[~last["removed"].astype(bool), "group"].tolist()


def grid_search(model, grid: dict[str, list], X: pd.DataFrame, sites: pd.DataFrame, assignments) -> pd.DataFrame:
    """Score every combination of hyperparameters in ``grid`` with ``publication_cv``.

    ``grid`` maps a parameter name of ``model`` (as for ``set_params``) to its values.
    Output: one row per combination with the parameter values and one ``paper_i`` column
    per assignment.
    """
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        runs = publication_cv(clone(model).set_params(**params), X, sites, assignments)
        rows.append({**params, **{f"paper_{i}": r for i, r in enumerate(runs)}})
    return pd.DataFrame(rows)


def _folds(X, y, sites, n_splits, groups):
    labels = sites["rxn_id"] if groups is None else groups
    return GroupKFold(n_splits=n_splits).split(X, y, groups=labels)
