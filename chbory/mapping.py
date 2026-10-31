"""Atom-map conventions used by the dataset.

Convention (verified on all 744 rows of the provided data):
the SMILES is canonicalised with RDKit, then atoms are numbered 1..N in
canonical order. Hence ``site`` (an atom map number) equals the canonical
atom index + 1. Product SMILES are numbered independently, so product map
numbers do NOT correspond to substrate map numbers.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations_with_replacement

from rdkit import Chem

BPIN = Chem.MolFromSmiles("B1OC(C)(C)C(C)(C)O1")


def make_mapped(smiles: str) -> str:
    # Return the atom-mapped SMILES in the dataset convention.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return Chem.MolToSmiles(mol, canonical=False)


def find_site_atom(mapped_smiles: str, site: int):
    # Return (mol, atom) for the atom whose map number equals 'site'
    # atom is None if no atom carries that map number
    # atom.GetIdx() from RDKit, which is alos the order we used to create mapped Smiles Structure
    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {mapped_smiles}")
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == int(site):
            return mol, atom
    return mol, None


def _strip_maps(mol: Chem.Mol) -> Chem.Mol:
    # Map numbers make equivalent atoms look different, so they are removed before structures are compared.
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def find_sites(substrate_smiles: str, product_smiles: str, boryl: Chem.Mol = BPIN):
    """Find which substrate carbon(s) were borylated to give the product.

    Each C–H carbon of the substrate is replaced in turn by ``boryl``
    (Bpin by default); positions that reproduce the product are returned.
    Several hits mean symmetry-equivalent positions; any of them is valid.

    Returns ``(mapped_substrate_smiles, list_of_map_numbers)``.
    Products that are not simple mono-borylation with ``boryl``
    (diborylation, other boron esters, further transformations) give ``[]``.
    """
    mapped = make_mapped(substrate_smiles)
    sub = Chem.MolFromSmiles(mapped)
    target = Chem.CanonSmiles(product_smiles)
    hits = [
        atom.GetAtomMapNum()
        for atom in _ch_carbons(sub)
        if _borylate(sub, [atom.GetIdx()], boryl) == target
    ]
    return mapped, hits


def find_site_sets(substrate_smiles: str, product_smiles: str, boryl: Chem.Mol = BPIN):
    """Find which substrate carbons were borylated, for products with any number of boryl groups.

    One boryl group is attached per boron atom in the product; a carbon may receive
    more than one if it has enough H (geminal diborylation). Combinations that
    reproduce the product are returned as tuples of map numbers, so a diborylated
    product gives pairs such as ``(1, 12)``. Several tuples mean symmetry-equivalent
    choices; any of them is valid.

    Returns ``(mapped_substrate_smiles, list_of_tuples)``; ``[]`` if nothing matches
    (other boron esters, further transformations).
    """
    mapped = make_mapped(substrate_smiles)
    sub = Chem.MolFromSmiles(mapped)
    target_mol = Chem.MolFromSmiles(product_smiles)
    if target_mol is None:
        raise ValueError(f"Could not parse SMILES: {product_smiles}")
    target = Chem.MolToSmiles(target_mol)
    n_boron = sum(atom.GetSymbol() == "B" for atom in target_mol.GetAtoms())

    candidates = [atom.GetIdx() for atom in _ch_carbons(sub)]
    hits = []
    for combo in combinations_with_replacement(candidates, n_boron):
        if any(sub.GetAtomWithIdx(i).GetTotalNumHs() < n for i, n in Counter(combo).items()):
            continue
        if _borylate(sub, list(combo), boryl) == target:
            hits.append(tuple(sub.GetAtomWithIdx(i).GetAtomMapNum() for i in combo))
    return mapped, hits


def _ch_carbons(mol: Chem.Mol) -> list:
    # find all C with C-H bond
    return [a for a in mol.GetAtoms() if a.GetSymbol() == "C" and a.GetTotalNumHs() > 0]


def _borylate(sub: Chem.Mol, atom_idxs: list[int], boryl: Chem.Mol) -> str | None:
    """Replace one H on each listed atom by ``boryl``; return the canonical SMILES or None."""
    mol = sub
    for idx in atom_idxs:
        offset = mol.GetNumAtoms()
        rw = Chem.RWMol(Chem.CombineMols(mol, boryl))
        rw.AddBond(idx, offset, Chem.BondType.SINGLE)  # boron is atom 0 of ``boryl``
        mol = rw.GetMol()
    rw = Chem.RWMol(mol)
    # Mapped atoms are bracket atoms ([CH3:1]) whose H count is explicit,
    # so remove exactly the replaced H; setting 0 would turn CH3/CH2 into radicals.
    for idx, n in Counter(atom_idxs).items():
        atom = rw.GetAtomWithIdx(idx)
        atom.SetNumExplicitHs(sub.GetAtomWithIdx(idx).GetTotalNumHs() - n)
        atom.SetNoImplicit(True)
    candidate = _strip_maps(rw.GetMol())
    try:
        Chem.SanitizeMol(candidate)
    except Exception:
        return None
    return Chem.MolToSmiles(candidate)
