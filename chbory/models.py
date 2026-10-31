"""Baselines and machine-learning models that score C–H sites."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def random_baseline(sites: pd.DataFrame, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Score every site randomly: the chance-level reference.

    Input: the site table. Output: one score per row (higher = more likely borylated),
    drawn uniformly from [0, 1) with a fixed seed so results are reproducible.
    """
    rng = np.random.default_rng(random_state)
    return rng.random(len(sites))


def steric_baseline(features: pd.DataFrame) -> np.ndarray:
    """Score sites with the textbook rule for Ir-catalysed borylation.

    Aromatic C–H bonds react before aliphatic ones, and among them the least hindered
    site wins. "Hindered" is the number of heavy atoms within two bonds of the site
    carbon (``n_heavy_d1 + n_heavy_d2``, at most 10, so the aromatic bonus of 10 always
    dominates). The rule is fixed in advance, not fitted. Ties are common and are
    handled by ``top1_accuracy``. Input: the feature table. Output: one score per row.
    """
    hindrance = features["n_heavy_d1"] + features["n_heavy_d2"]
    return (10 * features["is_aromatic"] - hindrance).to_numpy(dtype=float)


def is_directed(features: pd.DataFrame) -> pd.Series:
    """True for directed systems: Pd catalysts, or metal-free reactions with BBr3."""
    return (features["metal_Pd"] == 1) | ((features["metal_free"] == 1) & (features["boron_BBr3"] == 1))


def system_class(features: pd.DataFrame) -> np.ndarray:
    """Catalytic system of each row: "directed" (``is_directed``), "undirected Ir" or "other"."""
    return np.where(is_directed(features), "directed",
                    np.where(features["metal_Ir"] == 1, "undirected Ir", "other"))


def rule_score(features: pd.DataFrame) -> np.ndarray:
    """Score sites with the combined selectivity rules, fixed in advance (not fitted).

    The steric rule of ``steric_baseline``, plus 100 in directed systems (``is_directed``)
    for a C–H three or four bonds from a carbonyl O (the beta C–H of an amide or the
    ortho C–H of an anilide). The weights only order the terms: directing group >
    aromatic > hindrance. Input: the feature table. Output: one score per row.
    """
    score = steric_baseline(features).copy()
    score += 100 * (is_directed(features) & features["dist_carbonyl_O"].isin([3, 4])).to_numpy()
    return score


def make_model(name: str, random_state: int = RANDOM_STATE):
    """Return an unfitted scikit-learn classifier by name.

    Names: "logreg" (standardised logistic regression), "decision_tree" (depth ≤ 6),
    "random_forest", "extra_trees", "gradient_boosting", "knn" (k = 15), "svm"
    (RBF kernel), "mlp" (hidden layers 64 and 32) and "naive_bayes". Models that
    depend on feature scale are standardised first. Hyperparameters are fixed
    defaults, not tuned. All are trained to recognise borylated sites.
    """
    models = {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "decision_tree": lambda: DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, random_state=random_state
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, random_state=random_state, n_jobs=-1
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, random_state=random_state, n_jobs=-1
        ),
        "gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=random_state),
        "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        "svm": lambda: make_pipeline(
            StandardScaler(), SVC(probability=True, random_state=random_state)
        ),
        "mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True,
                random_state=random_state,
            ),
        ),
        "naive_bayes": lambda: make_pipeline(StandardScaler(), GaussianNB()),
    }
    if name not in models:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(models)}")
    return models[name]()


def tuning_grid(name: str, n_features: int) -> dict[str, list]:
    """Hyperparameter grid for ``grid_search`` around the ``make_model`` defaults.

    Tree ensembles (random forest, extra trees): ``min_samples_leaf`` 1, 2, 5, 10,
    ``max_features`` 1 to 4 (those below sqrt), sqrt, 20 or all (20 only if there are more
    than 20 features) and ``class_weight`` none or balanced, as in Table S3. The values below
    sqrt keep the default away from the edge of the grid. Other models: their main parameters.
    """
    if name in ("random_forest", "extra_trees"):
        below_sqrt = [k for k in range(1, 5) if k < int(np.sqrt(n_features))]
        return {"min_samples_leaf": [1, 2, 5, 10],
                "max_features": below_sqrt + ["sqrt"] + ([20] if n_features > 20 else []) + [None],
                "class_weight": [None, "balanced"]}
    grids = {
        "gradient_boosting": {"learning_rate": [0.03, 0.1, 0.3], "max_leaf_nodes": [15, 31, 63],
                              "min_samples_leaf": [5, 20, 50]},
        "logreg": {"logisticregression__C": [0.01, 0.1, 1, 10, 100]},
        "decision_tree": {"max_depth": [3, 6, 10, None], "min_samples_leaf": [1, 5, 20]},
        "knn": {"kneighborsclassifier__n_neighbors": [5, 15, 30, 60]},
        "svm": {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01, 0.1]},
        "mlp": {"mlpclassifier__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
                "mlpclassifier__alpha": [1e-4, 1e-3, 1e-2]},
        "naive_bayes": {"gaussiannb__var_smoothing": [1e-9, 1e-6, 1e-3]},
    }
    if name not in grids:
        raise ValueError(f"No tuning grid for {name!r}")
    return grids[name]


def predict_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Return a site score from a fitted model: the probability of being borylated."""
    return model.predict_proba(X)[:, 1]
