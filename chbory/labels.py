"""Building the site-level training table from reaction rows."""

from __future__ import annotations

import pandas as pd

from .sites import enumerate_sites

SITE_COLUMNS = [
    "rxn_id",
    "substrate_smiles_mapped",
    "map_num",
    "idx",
    "sym_class",
    "n_equivalent",
    "hybridization",
    "aromatic",
    "num_H",
    "regio_fraction",
]


def build_site_table(reactions: pd.DataFrame) -> pd.DataFrame:
    """Expand each reaction into one row per candidate C–H site (symmetry class).

    Input: cleaned reaction rows (output of ``clean_reactions``), one row per borylated site.
    Output: one row per (``rxn_id``, ``sym_class``) with the columns in ``SITE_COLUMNS``.

    Equivalent atoms (same ``sym_class``) form one site: their ``regio_fraction`` values
    are summed, and ``n_equivalent`` counts them. The representative atom (``map_num``,
    ``idx``) is the recorded atom with the smallest map number, or the smallest map
    number if no atom of the class was recorded; equivalent atoms share all descriptors,
    so the choice only affects which number is shown.
    Sites absent from the data gave no product and get ``regio_fraction = 0``.
    Raises ``ValueError`` if a recorded site is not a C–H carbon of the substrate.
    """
    tables = [
        _reaction_sites(rxn_id, rows) for rxn_id, rows in reactions.groupby("rxn_id", sort=False)
    ]
    return pd.concat(tables, ignore_index=True)


def _reaction_sites(rxn_id: str, rows: pd.DataFrame) -> pd.DataFrame:
    mapped = rows["substrate_smiles_mapped"].iloc[0]
    sites = enumerate_sites(mapped)

    observed = rows.groupby("site")["regio_fraction"].sum()
    unknown = sorted(set(observed.index) - set(sites["map_num"]))
    if unknown:
        raise ValueError(f"{rxn_id}: recorded site(s) {unknown} are not C–H carbons of {mapped}")
    sites["regio_fraction"] = sites["map_num"].map(observed).fillna(0.0)
    sites["recorded"] = sites["map_num"].isin(observed.index)

    # Recorded atoms first, then by map number, so .first() picks the representative atom
    sites = sites.sort_values(["sym_class", "recorded", "map_num"], ascending=[True, False, True])
    by_class = sites.groupby("sym_class", sort=False)
    table = by_class.first()
    table["regio_fraction"] = by_class["regio_fraction"].sum()
    table["n_equivalent"] = by_class.size()

    table = table.reset_index().sort_values("map_num")
    table["rxn_id"] = rxn_id
    table["substrate_smiles_mapped"] = mapped
    return table[SITE_COLUMNS]
