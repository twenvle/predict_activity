from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import cclib


def _screening(df: pd.DataFrame, isomer: bool = False) -> pd.DataFrame:
    """Screening the SMILES string for validity and suitability for further processing."""
    for i in df.index:
        tf = True
        smiles = df.loc[i, "smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            df.drop(index=i, inplace=True)
            continue
        mol = Chem.AddHs(mol)

        # イオンや塩を形成しているものは除外
        ejects = [".", "-", "+"]
        for eject in ejects:
            if eject in smiles:
                df.drop(index=i, inplace=True)
                tf = False
                break
        if not tf:
            continue

        # 金属を含むものは除外
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if (
                (3 <= atomic_num <= 4)
                or (11 < atomic_num <= 13)
                or (19 <= atomic_num <= 32)
                or (37 <= atomic_num <= 51)
                or (55 <= atomic_num <= 84)
                or (87 <= atomic_num <= 118)
            ):
                df.drop(i, inplace=True)
                tf = False
                break
        if not tf:
            continue

        # 同位体を含むものは除外
        if isomer:
            if type(df.loc[i, "iso"]) == str:
                iso = df.loc[i, "iso"]
                if "2H" in iso or "3H" in iso or "13C" in iso or "14C" in iso:
                    df.drop(i, inplace=True)
                    continue

        phthalic_pattern = Chem.MolFromSmarts("c1ccc(C(=O)[OH])c(C(=O)[OH])c1")
        matches = mol.GetSubstructMatches(phthalic_pattern)
        if not matches:
            df.drop(i, inplace=True)
            continue

        if AllChem.EmbedMolecule(mol) == -1:
            df.drop(i, inplace=True)
            continue

    return df


def read(
    input_xlsx: Path | None = None,
    name: str | None = None,
    smiles: str | None = None,
    known: bool = True,
) -> pd.DataFrame:
    """Read the input excel file and return a DataFrame."""
    if input_xlsx:
        df = pd.read_excel(input_xlsx)
    elif name and smiles:
        df = pd.DataFrame({"cas": [name], "smiles": [smiles]})
    if known:
        df = df[["cas", "smiles", "yield", "conversion", "selectivity"]]
    df = _screening(df)
    return df


# name == main
# logging
# parser
# a molecule
