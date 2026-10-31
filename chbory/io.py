"""Loading, merging and cleaning of the borylation dataset."""

from pathlib import Path

import pandas as pd
from rdkit import Chem

from .mapping import find_site_sets
from .sites import enumerate_sites

# 每一行 = 一个反应中的一个产物（一个被硼化的位点）
REQUIRED_COLUMNS = [
    # --- 标识 ---
    "rxn_id",  # 反应编号；同一反应的多个产物（多行）共用一个编号，不同数据源之间不能重复
    # --- 底物 ---
    "substrate_smiles",  # 底物（反应物）的 SMILES
    "substrate_smiles_mapped",  # 带原子编号的底物 SMILES：RDKit 规范化后按顺序编号 1..N，用来按 site 找原子
    # --- 反应条件 ---
    "temperature_degC",  # 反应温度（°C）；原始数据是文本，有 25 个反应写成 "50-60"
    "solvent_1_smiles",  # 溶剂 1；少数无溶剂反应为空
    "solvent_1_fraction",  # 溶剂 1 的比例；单一溶剂时为 1.0，部分为空
    "solvent_2_smiles",  # 溶剂 2（混合溶剂时才有）
    "solvent_2_fraction",  # 溶剂 2 的比例
    "catalyst_1_smiles",  # 催化剂，最常见的是 [Ir(COD)OMe]2；原始数据 56 行为空：53 行为无金属反应（如 BBr3），3 行漏填了 Ir 催化剂，已在 LITERATURE_CORRECTIONS 中补上
    "catalyst_2_smiles",  # 第二种催化剂，多数为空
    "ligand_1_smiles",  # 配体，例如 dtbpy（4,4'-二叔丁基联吡啶）
    "ligand_2_smiles",  # 第二种配体，多数为空
    "additive_1_smiles",  # 添加剂，多数为空
    "reagent_1_smiles",  # 试剂 1，通常是硼源：B2pin2（最常见）、BBr3、HBpin 等
    "reagent_2_smiles",  # 试剂 2-4：其他试剂，例如碱、频哪醇；多数为空
    "reagent_3_smiles",
    "reagent_4_smiles",
    # --- 结果 ---
    "total_yield",  # 该反应所有产物的总收率（0-1）；作业说明称可能缺失，课程数据中没有缺失
    "product_smiles",  # 这一行产物的 SMILES
    "product_smiles_mapped",  # 带原子编号的产物 SMILES；独立编号，与底物的编号不对应，不能用来找位点
    "site",  # 被硼化的碳在 substrate_smiles_mapped 中的原子编号（= 规范原子索引 + 1）
    "yield",  # 这一行产物的收率；yield = total_yield × regio_fraction
    "regio_fraction",  # 这一行产物占该反应所有硼化产物的比例；同一反应内加起来为 1，是区域选择性的预测目标
    "source_DOI",  # 来源文献的 DOI
]

# Entries that contradict their source publication, corrected from the publication itself
# (data/README.md, Table S1). Keys: rxn_id -> {column: corrected value}
MES_IR_BPIN3 = "Cc1cc(C)cc(C)c1.[Ir](B1OC(C)(C)C(C)(C)O1)(B1OC(C)(C)C(C)(C)O1)B1OC(C)(C)C(C)(C)O1"
BTFBPY = "FC(F)(F)c1ccnc(-c2cc(C(F)(F)F)ccn2)c1"
LITERATURE_CORRECTIONS = {
    # Catalyst missing; (Mes)Ir(Bpin)3 in J. Am. Chem. Soc. 2019, 141, 16479 (Table 1) and
    # ACS Catal. 2020, 10, 3415 (Table 2)
    "lit_pub_bo_15": {"catalyst_1_smiles": MES_IR_BPIN3},
    "lit_pub_bo_435": {"catalyst_1_smiles": MES_IR_BPIN3},
    "lit_pub_bo_438": {"catalyst_1_smiles": MES_IR_BPIN3},
    # Ligand missing; btfbpy, conditions C of Table 4 in J. Am. Chem. Soc. 2017, 139, 2825
    "lit_pub_bo_351": {"ligand_1_smiles": BTFBPY},
    "lit_pub_bo_354": {"ligand_1_smiles": BTFBPY},
    # Ir controls in the same Table 4: the recorded ortho-to-F product is the minor one
    # (ortho:meta 38:62, 36:64, 38:62); the major meta-to-F product is recorded instead
    "lit_pub_bo_336": {"site": 9, "product_smiles": "CN(C)S(=O)(=O)c1cc(F)cc(B2OC(C)(C)C(C)(C)O2)c1"},
    "lit_pub_bo_340": {"site": 9, "product_smiles": "COC(=O)C(C)c1cc(F)cc(B2OC(C)(C)C(C)(C)O2)c1"},
    "lit_pub_bo_349": {"site": 5, "product_smiles": "COc1cc(F)cc(B2OC(C)(C)C(C)(C)O2)c1"},
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the dataset CSV and check that all expected columns are present.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def merge_datasets(course: pd.DataFrame, added: pd.DataFrame) -> pd.DataFrame:
    """Combine the course data with our literature reactions.

    Inputs: two DataFrames as returned by ``load_dataset``.
    Output: one DataFrame with an extra ``source`` column ("course" or "added").
    Raises ``ValueError`` if a ``rxn_id`` appears in both inputs.
    """
    overlap = set(course["rxn_id"]) & set(added["rxn_id"])
    if overlap:
        raise ValueError(f"rxn_id values used in both datasets: {sorted(overlap)}")
    parts = [course.assign(source="course")]
    if len(added):  # concatenating an empty frame would only disturb the dtypes
        parts.append(added.assign(source="added"))
    return pd.concat(parts, ignore_index=True)


def clean_reactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions documented in data/README.md.

    1. Entries that contradict their source publication are corrected
       (``LITERATURE_CORRECTIONS``).
    2. ``temperature_degC`` becomes numeric; the range "50-60" is replaced by its midpoint.
    3. The added column ``n_boron`` records how many boron atoms each product carries.
    4. Rows with the same (rxn_id, site) are merged, ``regio_fraction`` and ``yield`` summed
       (stereoisomers that SMILES does not distinguish).
    5. Reactions with an incomplete label are removed: a product with several boron atoms
       whose borylated carbons are in different symmetry classes (e.g. one sp3 and one sp2
       carbon), since the single ``site`` column can name only one of them. Polyborylated
       products whose carbons are all equivalent (both ends of dodecane) are kept unchanged.

    ``regio_fraction`` still sums to 1 per reaction, because whole reactions are removed.
    Input is not modified; a cleaned copy is returned.
    """
    df = df.copy()
    for rxn_id, fixes in LITERATURE_CORRECTIONS.items():
        rows = df["rxn_id"] == rxn_id
        if rows.sum() > 1 and "site" in fixes:
            raise ValueError(f"{rxn_id}: site correction expects a single product row")
        for column, value in fixes.items():
            df.loc[rows, column] = value

    # "50-60" is the only range (25 reactions, one paper); any other non-numeric value should fail loudly
    df["temperature_degC"] = pd.to_numeric(df["temperature_degC"].replace({"50-60": "55"}))

    # Products without boron are kept: lit_pub_bo_853 reports the oxidised alcohol,
    # but its site is still the borylated carbon
    df["n_boron"] = df["product_smiles"].map(_count_boron)
    df = _merge_duplicate_sites(df)

    incomplete = {
        rxn_id
        for rxn_id, substrate, product, site, n_boron in zip(
            df["rxn_id"], df["substrate_smiles"], df["product_smiles"], df["site"], df["n_boron"]
        )
        if n_boron > 1 and _sites_differ(rxn_id, substrate, product, site)
    }
    return df[~df["rxn_id"].isin(incomplete)].reset_index(drop=True)


def _sites_differ(rxn_id: str, substrate: str, product: str, site: int) -> bool:
    """True if a polyborylated product is borylated at carbons of more than one symmetry class."""
    mapped, hits = find_site_sets(substrate, product)
    # Symmetry-equivalent atoms give several combinations; use one containing the recorded site
    chosen = next((hit for hit in hits if site in hit), None)
    if chosen is None:
        raise ValueError(f"{rxn_id}: site {site} is not among the borylated carbons of {product}")
    sym_class = enumerate_sites(mapped).set_index("map_num")["sym_class"]
    return sym_class.loc[list(chosen)].nunique() > 1


def _merge_duplicate_sites(df: pd.DataFrame) -> pd.DataFrame:
    """Merge rows with the same (rxn_id, site), e.g. diastereomers that SMILES does not distinguish."""
    keys = ["rxn_id", "site"]
    agg = {c: "first" for c in df.columns if c not in keys}
    agg["yield"] = "sum"
    agg["regio_fraction"] = "sum"
    agg["n_boron"] = "max"
    merged = df.groupby(keys, sort=False, as_index=False).agg(agg)
    return merged[df.columns]


def _count_boron(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return sum(atom.GetSymbol() == "B" for atom in mol.GetAtoms())
