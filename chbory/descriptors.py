"""Descriptors for C–H sites and reaction conditions.

All descriptors are computed with RDKit from SMILES and are deterministic.
Site descriptors describe one candidate C–H carbon; condition descriptors are the
same for every site of a reaction. ``FEATURE_GROUPS`` and ``EXTENDED_GROUPS`` list the columns
that are computed here; ``CANDIDATE_GROUPS`` lists the 57 candidate descriptors in the 12 groups
used for descriptor selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFreeSASA

from .models import rule_score

# Topological distances (in bonds) are capped: beyond this, a group is "far away"
DIST_CAP = 10

HETEROATOMS = {7, 8, 16}  # N, O, S
CARBONYL = Chem.MolFromSmarts("[CX3]=[OX1]")
AMIDE_N = Chem.MolFromSmarts("[#7X3][CX3]=[OX1]")  # amide, carbamate and N-acyl (also aromatic N)

METALS = ["Ir", "Pd", "Co", "Ni", "Pt", "Rh", "Ru", "Cu", "Fe"]
BORON_SOURCES = {
    Chem.CanonSmiles("CC1(C)OB(B2OC(C)(C)C(C)(C)O2)OC1(C)C"): "B2pin2",
    Chem.CanonSmiles("CC1(C)OBOC1(C)C"): "HBpin",
    Chem.CanonSmiles("BrB(Br)Br"): "BBr3",
    Chem.CanonSmiles("B1Oc2ccccc2O1"): "HBcat",
}
BIPYRIDINE = Chem.MolFromSmarts("n1ccccc1-c1ccccn1")
PHENANTHROLINE = Chem.MolFromSmarts("c1cnc2c(c1)ccc1cccnc12")

FEATURE_GROUPS = {
    "site_type": ["is_sp3", "is_aromatic", "num_H", "n_equivalent", "in_ring", "ring_size"],
    "steric": ["n_heavy_d1", "n_heavy_d2", "n_heavy_d3", "ortho_substituents"],
    "electronic": [
        "charge_C", "charge_H", "alpha_heteroatom", "beta_heteroatom",
        "heteroaromatic_ring", "dist_aromatic_N",
    ],
    "directing": ["dist_carbonyl_O", "dist_amide_N"],
    "relative": [
        "num_H_rel", "n_heavy_d1_rel", "n_heavy_d2_rel", "n_heavy_d3_rel",
        "ortho_substituents_rel", "charge_C_rel", "charge_H_rel",
    ],
    "conditions": (
        [f"metal_{m}" for m in ["Ir", "Pd", "Co", "Ni", "Pt"]] + ["metal_free"]
        + [f"boron_{b}" for b in ["B2pin2", "HBpin", "BBr3", "HBcat", "other"]]
        + ["ligand_bipyridine", "ligand_phenanthroline", "ligand_other", "ligand_none"]
        + ["temperature_degC"]
    ),
}

# Added by ``add_extended_features`` on top of FEATURE_GROUPS
EXTENDED_GROUPS = {
    "steric_3d": ["vbur_C", "sasa_H", "vbur_C_rel", "sasa_H_rel"],
    "rule": ["rule_score", "rule_score_rel"],
}

def _with_rel(columns: list[str]) -> list[str]:
    return columns + [f"{c}_rel" for c in columns]


# The 57 candidate descriptors in 12 groups (Table 1). Descriptor selection starts from all of
# them and removes whole groups (``backward_elimination``); site_type is not a candidate.
CANDIDATE_GROUPS = {
    "topological": _with_rel(["n_heavy_d1", "n_heavy_d2", "n_heavy_d3", "ortho_substituents"]) + ["num_H_rel"],
    "buried_volume": _with_rel(["vbur_C"]),
    "sasa": _with_rel(["sasa_H"]),
    "sterimol": _with_rel(["sterimol_L", "sterimol_B1", "sterimol_B5"]),
    "gasteiger": _with_rel(["charge_C", "charge_H"]),
    "fukui": _with_rel(["fukui_plus_C", "fukui_minus_C", "fukui_zero_C"]),
    "deprot": _with_rel(["deprot"]),
    "wbo": _with_rel(["wbo_CH"]),
    "heteroatoms": ["alpha_heteroatom", "beta_heteroatom", "heteroaromatic_ring", "dist_aromatic_N"],
    "directing": FEATURE_GROUPS["directing"],
    "conditions": FEATURE_GROUPS["conditions"],
    "rule": EXTENDED_GROUPS["rule"],
}
CANDIDATE_FEATURES = [c for group in CANDIDATE_GROUPS.values() for c in group]
# Computed with xtb and Morfeus in qm.py (or read from data/precomputed_descriptors.csv)
PRECOMPUTED_GROUPS = ["sterimol", "fukui", "deprot", "wbo"]

# Buried volume: Bondi van der Waals radii (Å) scaled by 1.17 and a 3.5 Å sphere, as in SambVca
BONDI_RADII = {1: 1.10, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 14: 2.10, 15: 1.80,
               16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98}
VBUR_SCALE = 1.17
VBUR_RADIUS = 3.5
_axis = np.arange(-VBUR_RADIUS, VBUR_RADIUS, 0.15)
VBUR_GRID = np.stack(np.meshgrid(_axis, _axis, _axis, indexing="ij"), axis=-1).reshape(-1, 3)
VBUR_GRID = VBUR_GRID[np.linalg.norm(VBUR_GRID, axis=1) <= VBUR_RADIUS]


def site_descriptors(mapped_smiles: str, atom_idxs: list[int]) -> pd.DataFrame:
    """Describe the environment of C–H carbons in one substrate.

    Inputs: the (mapped) substrate SMILES and the atom *indices* (not map numbers) of
    the sites. The molecule is processed once for all sites.
    Output: one row per index, with the site_type (except ``n_equivalent``), steric,
    electronic and directing columns of ``FEATURE_GROUPS``.

    - steric: ``n_heavy_dk`` counts heavy atoms exactly k bonds away;
      ``ortho_substituents`` counts adjacent aromatic carbons without H (substituted or
      ring-fusion positions), 0 for non-aromatic sites.
    - electronic: Gasteiger charges of the carbon and (mean) of its hydrogens; N, O or S
      one (alpha) or two (beta) bonds away; whether the smallest ring of the site
      contains a heteroatom; bonds to the nearest aromatic N.
    - directing: bonds to the nearest carbonyl O and amide-type N.
    Distances are capped at ``DIST_CAP`` (also used when the group is absent).
    """
    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {mapped_smiles}")
    dist = Chem.GetDistanceMatrix(mol)
    ring_info = mol.GetRingInfo()

    mol_h = Chem.AddHs(mol)  # hydrogens are appended, so heavy-atom indices are unchanged
    AllChem.ComputeGasteigerCharges(mol_h)

    aromatic_n = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetIsAromatic()]
    carbonyl_o = [match[1] for match in mol.GetSubstructMatches(CARBONYL)]
    amide_n = [match[0] for match in mol.GetSubstructMatches(AMIDE_N)]
    heavy_shell = {k: (dist == k) for k in (1, 2, 3)}

    rows = []
    for idx in atom_idxs:
        atom = mol.GetAtomWithIdx(idx)
        neighbours = atom.GetNeighbors()
        hydrogens = [n for n in mol_h.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() == 1]
        smallest_ring = min(
            (ring for ring in ring_info.AtomRings() if idx in ring), key=len, default=()
        )
        rows.append(
            {
                "is_sp3": float(atom.GetHybridization() == Chem.HybridizationType.SP3),
                "is_aromatic": float(atom.GetIsAromatic()),
                "num_H": float(atom.GetTotalNumHs()),
                "in_ring": float(atom.IsInRing()),
                "ring_size": float(len(smallest_ring)),
                **{f"n_heavy_d{k}": float(shell[idx].sum()) for k, shell in heavy_shell.items()},
                "ortho_substituents": float(
                    sum(
                        n.GetIsAromatic() and n.GetSymbol() == "C" and n.GetTotalNumHs() == 0
                        for n in neighbours
                    )
                    if atom.GetIsAromatic()
                    else 0
                ),
                "charge_C": _charge(mol_h.GetAtomWithIdx(idx)),
                "charge_H": sum(_charge(h) for h in hydrogens) / len(hydrogens),
                "alpha_heteroatom": float(any(n.GetAtomicNum() in HETEROATOMS for n in neighbours)),
                "beta_heteroatom": float(
                    any(
                        mol.GetAtomWithIdx(int(j)).GetAtomicNum() in HETEROATOMS
                        for j in heavy_shell[2][idx].nonzero()[0]
                    )
                ),
                "heteroaromatic_ring": float(
                    atom.GetIsAromatic()
                    and any(mol.GetAtomWithIdx(j).GetAtomicNum() != 6 for j in smallest_ring)
                ),
                "dist_aromatic_N": _min_distance(dist, idx, aromatic_n),
                "dist_carbonyl_O": _min_distance(dist, idx, carbonyl_o),
                "dist_amide_N": _min_distance(dist, idx, amide_n),
            }
        )
    return pd.DataFrame(rows, index=list(atom_idxs))


def condition_descriptors(reaction: pd.Series) -> dict[str, float]:
    """Describe the reaction conditions of one reaction row.

    Input: one row of the (cleaned) reaction table.
    Output: one-hot metal of the catalyst (``metal_free`` if none), boron sources among
    the reagents, ligand class (bipyridine, phenanthroline, other, none) and temperature.
    Tetrafluoroborate salts are additives, not boron sources, and are ignored.
    """
    metals = {
        symbol
        for col in ["catalyst_1_smiles", "catalyst_2_smiles"]
        for symbol in _element_symbols(reaction.get(col))
        if symbol in METALS
    }
    boron = {
        _boron_source(reaction.get(col))
        for col in ["reagent_1_smiles", "reagent_2_smiles", "reagent_3_smiles", "reagent_4_smiles"]
    } - {None}
    ligand = _ligand_class(reaction.get("ligand_1_smiles"))

    features = {f"metal_{m}": float(m in metals) for m in ["Ir", "Pd", "Co", "Ni", "Pt"]}
    features["metal_free"] = float(not metals)
    features.update({f"boron_{b}": float(b in boron) for b in ["B2pin2", "HBpin", "BBr3", "HBcat", "other"]})
    features.update(
        {f"ligand_{c}": float(c == ligand) for c in ["bipyridine", "phenanthroline", "other", "none"]}
    )
    features["temperature_degC"] = float(reaction["temperature_degC"])
    return features


def build_feature_table(sites: pd.DataFrame, reactions: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptors for every row of the site table.

    Inputs: the output of ``build_site_table`` and the (cleaned) reaction table.
    Output: a DataFrame aligned row-for-row with ``sites`` (same index), with the columns
    of all ``FEATURE_GROUPS``. ``*_rel`` columns are the value minus its mean over the
    candidate sites of the same reaction, because sites compete within one molecule.
    """
    parts = [
        site_descriptors(group["substrate_smiles_mapped"].iloc[0], group["idx"].tolist())
        .set_index(group.index)
        for _, group in sites.groupby("rxn_id", sort=False)
    ]
    features = pd.concat(parts).loc[sites.index]
    features["n_equivalent"] = sites["n_equivalent"].astype(float)

    for column in [c.removesuffix("_rel") for c in FEATURE_GROUPS["relative"]]:
        mean = features[column].groupby(sites["rxn_id"]).transform("mean")
        features[f"{column}_rel"] = features[column] - mean

    first_rows = reactions.drop_duplicates("rxn_id").set_index("rxn_id")
    conditions = pd.DataFrame(
        {rxn_id: condition_descriptors(row) for rxn_id, row in first_rows.iterrows()}
    ).T
    features["rxn_id"] = sites["rxn_id"]
    features = features.join(conditions, on="rxn_id")

    columns = [c for group in FEATURE_GROUPS.values() for c in group]
    return features[columns]


def embed_3d(smiles: str) -> Chem.Mol | None:
    """One 3D conformer with hydrogens: ETKDGv3 (seed 42), then MMFF94 (UFF if MMFF lacks
    parameters, e.g. for boron). Heavy-atom indices follow the SMILES. None if embedding fails.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
    return mol


def steric_3d_descriptors(mapped_smiles: str, atom_idxs: list[int]) -> pd.DataFrame:
    """3D steric descriptors of C–H carbons, computed on one conformer (``embed_3d``).

    - ``vbur_C``: buried volume, the fraction of a 3.5 Å sphere around the carbon that lies
      inside the other atoms (Bondi radii × 1.17, as in SambVca); the carbon and its own
      hydrogens are left out. Computed on a 0.15 Å grid.
    - ``sasa_H``: mean solvent-accessible surface area of the carbon's hydrogens (FreeSASA,
      Bondi radii, 1.4 Å probe).
    Output: one row per index; NaN if the molecule cannot be embedded.
    """
    mol = embed_3d(mapped_smiles)
    if mol is None:
        return pd.DataFrame({"vbur_C": np.nan, "sasa_H": np.nan}, index=list(atom_idxs))
    xyz = mol.GetConformer().GetPositions()
    vdw = [BONDI_RADII.get(a.GetAtomicNum(), 2.0) for a in mol.GetAtoms()]
    radii = np.array(vdw) * VBUR_SCALE
    # Explicit radii: rdFreeSASA.classifyAtoms assigns 0 to every atom of a small molecule
    rdFreeSASA.CalcSASA(mol, vdw)

    rows = []
    for idx in atom_idxs:
        hydrogens = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() == 1]
        others = [i for i in range(mol.GetNumAtoms()) if i != idx and i not in hydrogens]
        points = VBUR_GRID + xyz[idx]
        dist = np.linalg.norm(points[:, None, :] - xyz[others][None, :, :], axis=2)
        rows.append({
            "vbur_C": float((dist <= radii[others]).any(axis=1).mean()),
            "sasa_H": float(np.mean([float(mol.GetAtomWithIdx(h).GetProp("SASA")) for h in hydrogens])),
        })
    return pd.DataFrame(rows, index=list(atom_idxs))


def add_extended_features(features: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Add the ``EXTENDED_GROUPS`` columns (3D sterics, rule score) to a feature table.

    Inputs: the output of ``build_feature_table`` and the site table it was built from.
    Output: a copy with the new columns, including ``*_rel`` versions (value minus the
    mean over the sites of the same reaction). The 3D descriptors are computed once per
    substrate; a substrate that cannot be embedded gets 0.
    """
    features = features.copy()
    parts = [
        steric_3d_descriptors(smiles, group["idx"].tolist()).set_index(group.index)
        for smiles, group in sites.groupby("substrate_smiles_mapped", sort=False)
    ]
    features[["vbur_C", "sasa_H"]] = pd.concat(parts).loc[sites.index].fillna(0.0)
    features["rule_score"] = rule_score(features)
    for column in ["vbur_C", "sasa_H", "rule_score"]:
        features[f"{column}_rel"] = features[column] - features[column].groupby(sites["rxn_id"]).transform("mean")
    return features


def add_candidate_descriptors(
    features: pd.DataFrame, sites: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    """The 57 candidate descriptors (``CANDIDATE_FEATURES``) of every site.

    Inputs: the output of ``add_extended_features``, the site table it was built from, and
    the Sterimol and xTB descriptors (``qm.qm_descriptors`` or ``data/precomputed_descriptors.csv``, one row per
    ``rxn_id`` and ``idx``), which supply the ``PRECOMPUTED_GROUPS``. Other columns of that
    file are ignored. Output: a table aligned row-for-row with ``sites`` with the columns of
    ``CANDIDATE_FEATURES``; a value the calculation could not give (NaN) is set to 0. Every
    site must have a row in ``candidates``.
    """
    columns = [c for name in PRECOMPUTED_GROUPS for c in CANDIDATE_GROUPS[name]]
    merged = sites[["rxn_id", "idx"]].merge(
        candidates[["rxn_id", "idx"] + columns], on=["rxn_id", "idx"], how="left", indicator=True
    )
    missing = merged["_merge"] != "both"
    if missing.any():
        raise ValueError(f"{int(missing.sum())} sites have no precomputed descriptors, "
                         f"e.g. {merged.loc[missing, 'rxn_id'].iloc[0]}")
    table = features.copy()
    table[columns] = merged[columns].fillna(0.0).to_numpy()
    return table[CANDIDATE_FEATURES]


def _charge(atom: Chem.Atom) -> float:
    """Gasteiger charge of an atom; 0 where RDKit has no parameters (e.g. B, Si)."""
    charge = atom.GetDoubleProp("_GasteigerCharge")
    return 0.0 if charge != charge else charge  # NaN check


def _min_distance(dist, idx: int, targets: list[int]) -> float:
    return float(min([dist[idx, t] for t in targets] + [DIST_CAP]))


def _parse(smiles, sanitize: bool = True) -> Chem.Mol | None:
    """Parse quietly: catalysts and salts in the condition columns often fail sanitisation."""
    if not isinstance(smiles, str):
        return None
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    RDLogger.EnableLog("rdApp.*")
    return mol


def _element_symbols(smiles) -> set[str]:
    mol = _parse(smiles, sanitize=False)
    return set() if mol is None else {a.GetSymbol() for a in mol.GetAtoms()}


def _boron_source(smiles) -> str | None:
    if "B" not in _element_symbols(smiles):
        return None
    mol = _parse(smiles)
    if mol is not None and Chem.MolToSmiles(mol) in BORON_SOURCES:
        return BORON_SOURCES[Chem.MolToSmiles(mol)]
    # A boron atom carrying four fluorines is a BF4- counter-ion (additive), not a boron source
    if any(
        a.GetSymbol() == "B" and sum(n.GetSymbol() == "F" for n in a.GetNeighbors()) >= 4
        for a in _parse(smiles, sanitize=False).GetAtoms()
    ):
        return None
    return "other"


def _ligand_class(smiles) -> str:
    if not isinstance(smiles, str):
        return "none"
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None and mol.HasSubstructMatch(PHENANTHROLINE):
        return "phenanthroline"
    if mol is not None and mol.HasSubstructMatch(BIPYRIDINE):
        return "bipyridine"
    return "other"
