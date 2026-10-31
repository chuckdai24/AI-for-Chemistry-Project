"""Enumeration of candidate C–H sites in a substrate."""

from __future__ import annotations

import pandas as pd
from rdkit import Chem


def enumerate_sites(mapped_smiles: str) -> pd.DataFrame:
    """List every carbon bearing hydrogen, grouped by symmetry.

    Atoms with the same ``sym_class`` are topologically equivalent.
    Map numbers are removed before ranking, because they would otherwise
    make equivalent atoms look different.
    """
    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {mapped_smiles}")
    plain = Chem.Mol(mol)
    for atom in plain.GetAtoms():
        atom.SetAtomMapNum(0)
    ranks = list(Chem.CanonicalRankAtoms(plain, breakTies=False))

    rows = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C" and atom.GetTotalNumHs() > 0:
            rows.append(
                {
                    "map_num": atom.GetAtomMapNum(),
                    "idx": atom.GetIdx(),
                    "sym_class": ranks[atom.GetIdx()],
                    "hybridization": str(atom.GetHybridization()),
                    "aromatic": atom.GetIsAromatic(),
                    "num_H": atom.GetTotalNumHs(),
                }
            )
    return pd.DataFrame(rows)
