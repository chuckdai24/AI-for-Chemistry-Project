"""Sterimol and GFN2-xTB descriptors of C–H sites (the ``PRECOMPUTED_GROUPS`` of descriptors.py).

These need two tools outside RDKit:

- xtb 6.7.1 (``conda install -c conda-forge xtb=6.7.1``), called as a program;
- Morfeus (``pip install morfeus-ml``) for Sterimol.

Per substrate (one 3D structure: ETKDGv3, seed 42, then MMFF94):

- Sterimol L, B1, B5 along the H -> C axis (Morfeus), mean over the H atoms of the site;
- Fukui indices f+, f-, f0 of the carbon (GFN2-xTB single point on the MMFF structure);
- Wiberg bond order of the C–H bond, mean over the H atoms (GFN2-xTB optimisation, ALPB THF);
- deprotonation energy E(anion) - E(neutral) in kcal/mol: one H of the site removed from the
  optimised neutral structure and the anion optimised (GFN2-xTB, ALPB THF).

Each descriptor also gets a ``*_rel`` column (value minus the mean over the sites of the reaction).
A calculation that fails gives NaN; ``add_candidate_descriptors`` sets it to 0.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

HARTREE = 627.509  # kcal/mol
QM_COLUMNS = ["sterimol_L", "sterimol_B1", "sterimol_B5", "fukui_plus_C", "fukui_minus_C", "fukui_zero_C",
              "deprot", "wbo_CH"]


def find_xtb(path: str | None = None) -> str:
    """The xtb program: ``path`` if given, else the XTB environment variable, else xtb on the PATH."""
    xtb = path or os.environ.get("XTB") or shutil.which("xtb")
    if xtb is None or not Path(xtb).exists() and shutil.which(xtb) is None:
        raise FileNotFoundError(
            "xtb not found. Install it (conda install -c conda-forge xtb=6.7.1) and pass --xtb <path to xtb>, "
            "or run with --precomputed to use data/precomputed_descriptors.csv")
    return xtb


def embed(smiles: str) -> Chem.Mol | None:
    """3D structure with hydrogens: ETKDGv3 (seed 42), then MMFF94 (UFF where MMFF has no parameters)."""
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


def _xtb(xtb: str, xyz: str, options: list[str], workdir: str) -> str | None:
    """Run xtb on one structure in ``workdir``; the output, or None if xtb did not finish normally."""
    Path(workdir, "mol.xyz").write_text(xyz)
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_STACKSIZE="500M")
    proc = subprocess.run([xtb, "mol.xyz", "--gfn", "2", *options], cwd=workdir, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, timeout=1800)
    log = proc.stdout + proc.stderr
    # "abnormal termination" also contains "normal termination"
    return None if "abnormal" in log or "normal termination" not in log else proc.stdout


def _energy(log: str) -> float:
    return float(log.split("TOTAL ENERGY")[-1].split()[0])


def fukui(mol: Chem.Mol, xtb: str) -> list[list[float]] | None:
    """Fukui indices (f+, f-, f0) of every atom: GFN2-xTB single point on the given structure."""
    charge = Chem.GetFormalCharge(mol)
    unpaired = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    with tempfile.TemporaryDirectory() as tmp:
        log = _xtb(xtb, Chem.MolToXYZBlock(mol), ["--vfukui", "--chrg", str(charge), "--uhf", str(unpaired)], tmp)
    if log is None or "Fukui functions:" not in log:
        return None
    lines = log.split("Fukui functions:")[1].splitlines()[2:2 + mol.GetNumAtoms()]
    return [[float(v) for v in line.split()[1:4]] for line in lines]


def acidity(mol: Chem.Mol, idxs: list[int], xtb: str) -> dict[int, tuple[float, float]]:
    """Deprotonation energy (kcal/mol) and mean Wiberg C–H bond order of each site (GFN2-xTB, ALPB THF)."""
    charge = Chem.GetFormalCharge(mol)
    solvent = ["--opt", "--alpb", "thf", "--uhf", "0"]
    nan = {idx: (np.nan, np.nan) for idx in idxs}
    with tempfile.TemporaryDirectory() as tmp:
        log = _xtb(xtb, Chem.MolToXYZBlock(mol), solvent + ["--chrg", str(charge)], tmp)
        if log is None or "TOTAL ENERGY" not in log:
            return nan
        e_neutral = _energy(log)
        wbo = {}
        for line in Path(tmp, "wbo").read_text().splitlines():
            if line.strip():
                i, j, w = line.split()
                wbo[(int(i) - 1, int(j) - 1)] = wbo[(int(j) - 1, int(i) - 1)] = float(w)
        lines = Path(tmp, "xtbopt.xyz").read_text().strip().splitlines()
    header, coords = lines[:2], lines[2:]
    result = {}
    for idx in idxs:
        hs = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() == 1]
        anion = [c for k, c in enumerate(coords) if k != hs[0]]
        with tempfile.TemporaryDirectory() as tmp:
            log = _xtb(xtb, f"{len(anion)}\n{header[1]}\n" + "\n".join(anion) + "\n",
                       solvent + ["--chrg", str(charge - 1)], tmp)
        deprot = np.nan if log is None or "TOTAL ENERGY" not in log else (_energy(log) - e_neutral) * HARTREE
        result[idx] = (deprot, sum(wbo.get((idx, h), 0.0) for h in hs) / len(hs))
    return result


def sterimol(mol: Chem.Mol, idxs: list[int]) -> dict[int, tuple[float, float, float]]:
    """Sterimol L, B1, B5 along each H -> C bond of a site (Morfeus), mean over the H atoms."""
    from morfeus import Sterimol

    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    xyz = mol.GetConformer().GetPositions()
    result = {}
    for idx in idxs:
        hs = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() == 1]
        values = [Sterimol(elements, xyz, h + 1, idx + 1, n_rot_vectors=720) for h in hs]  # 1-based
        result[idx] = tuple(float(np.mean([getattr(s, v) for s in values]))
                            for v in ["L_value", "B_1_value", "B_5_value"])
    return result


def molecule_descriptors(smiles: str, idxs: list[int], xtb: str, cache: str,
                         rounded: bool = True) -> dict[int, dict[str, float]]:
    """``QM_COLUMNS`` of the given sites of one substrate; saved in ``cache`` and read from there next time.

    With ``rounded`` the deprotonation energy, bond orders and Sterimol are calculated on the structure
    as saved in a mol file (coordinates to 4 decimals), otherwise on the unrounded structure; the Fukui
    indices always on the unrounded one. The thesis values used ``rounded`` for the course reactions
    and the unrounded structure for the ten added reactions, which were calculated later. The two
    differ by about 0.01 kcal/mol in most deprotonation energies, but an anion optimisation can end
    in a different minimum, so both are kept to reproduce the thesis exactly.
    """
    name = hashlib.md5(smiles.encode()).hexdigest()[:16] + ("" if rounded else "_unrounded")
    path = Path(cache, name + ".json")
    if path.exists():
        saved = json.loads(path.read_text())
        if all(str(i) in saved for i in idxs):
            return {i: saved[str(i)] for i in idxs}
    mol = embed(smiles)
    if mol is None:
        rows = {i: dict.fromkeys(QM_COLUMNS, np.nan) for i in idxs}
    else:
        saved = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol), removeHs=False) if rounded else mol
        f, acid, ster = fukui(mol, xtb), acidity(saved, idxs, xtb), sterimol(saved, idxs)
        rows = {}
        for i in idxs:
            rows[i] = dict(zip(["sterimol_L", "sterimol_B1", "sterimol_B5"], ster[i]))
            rows[i].update(zip(["fukui_plus_C", "fukui_minus_C", "fukui_zero_C"], f[i] if f else [np.nan] * 3))
            rows[i]["deprot"], rows[i]["wbo_CH"] = acid[i]
    path.write_text(json.dumps({str(i): r for i, r in rows.items()}))  # NaN is written as NaN
    return rows


def _job(args):
    return (args[0], args[4]), molecule_descriptors(*args)


def qm_descriptors(sites: pd.DataFrame, xtb: str, cache: Path, workers: int | None = None,
                   unrounded: set[str] = frozenset()) -> pd.DataFrame:
    """The Sterimol and xTB descriptors (with ``*_rel``) of every site, one row per ``rxn_id`` and ``idx``.

    ``unrounded``: reactions whose substrate is calculated on the unrounded structure (see
    ``molecule_descriptors``; for the thesis, the ten added reactions). Substrates are calculated in
    parallel (``workers`` processes, default all cores) and saved in ``cache`` (one JSON file per
    substrate), so an interrupted run continues where it stopped.
    """
    cache.mkdir(parents=True, exist_ok=True)
    rounded = ~sites["rxn_id"].isin(unrounded)
    jobs = [(s, sorted(int(i) for i in g["idx"].unique()), xtb, str(cache), r)
            for (s, r), g in sites.groupby([sites["substrate_smiles_mapped"], rounded], sort=False)]
    values = {}
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        for n, (job, rows) in enumerate(pool.map(_job, jobs), 1):
            values[job] = rows
            if n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} substrates", flush=True)
    table = pd.DataFrame([values[(s, r)][int(i)] for s, r, i in zip(sites["substrate_smiles_mapped"], rounded,
                                                                    sites["idx"])], index=sites.index)[QM_COLUMNS]
    for column in QM_COLUMNS:
        table[f"{column}_rel"] = table[column] - table[column].groupby(sites["rxn_id"]).transform("mean")
    return pd.concat([sites[["rxn_id", "idx"]], table], axis=1)
