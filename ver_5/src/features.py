from __future__ import annotations


from pathlib import Path
import os
from typing import Any

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Geometry import Point3D
from rdkit import RDLogger

import cclib
import freesasa


atom = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


def _acid_idx(df: pd.DataFrame, benzene: bool = False) -> pd.DataFrame:
    """Identify the indices of acidic functional groups in the molecule represented by the SMILES string."""
    # Screening is already done in the preprocess.py

    for i in df.index:
        smiles = df.loc[i, "smiles"]
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        phthalic_pattern = Chem.MolFromSmarts("c1ccc(C(=O)[OH])c(C(=O)[OH])c1")
        matches_tuple = mol.GetSubstructMatches(phthalic_pattern)

        oh_list = []
        cooh_list = []
        matches = matches_tuple[0]
        for match in matches:
            atom = mol.GetAtomWithIdx(match)  # その番号の原子の情報
            symbol = atom.GetSymbol()  # その番号の原子

            # まだこの時点では水素の場所は検出されていないのでカルボン酸の水素を検出
            if symbol == "O":
                neighbors = atom.GetNeighbors()  # その原子に隣接する原子の情報
                elements = [n.GetSymbol() for n in neighbors]
                if "H" in elements and "C" in elements:
                    for neighbor in neighbors:
                        if neighbor.GetSymbol() == "H":
                            oh_list.append((atom.GetIdx(), neighbor.GetIdx()))

        for match in matches:
            atom = mol.GetAtomWithIdx(match)
            symbol = atom.GetSymbol()
            # カルボン酸の炭素を起点に他の原子を検出
            if symbol == "C":
                neighbors = atom.GetNeighbors()
                elements = [n.GetSymbol() for n in neighbors]
                if elements.count("O") >= 2:
                    for neighbor in neighbors:
                        if neighbor.GetSymbol() == "C":
                            c = neighbor.GetIdx()
                        elif neighbor.GetSymbol() == "O":
                            if len(neighbor.GetNeighbors()) == 1:
                                o_double = neighbor.GetIdx()
                            elif len(neighbor.GetNeighbors()) == 2:
                                for oh in oh_list:
                                    if oh[0] == neighbor.GetIdx():
                                        o_single = oh[0]
                                        h = oh[1]
                    # 自身の炭素原子、それに結合しているベンゼン環内の炭素原子、
                    # 炭素と2重結合している酸素原子、炭素と単結合している酸素原子、水素の順
                    cooh_list.append([atom.GetIdx(), c, o_double, o_single, h])
                df.loc[i, "cooh"] = cooh_list
    return df


def phthalicacid_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Identify the indices of the phthalic acid moiety in the molecule represented by the SMILES string."""
    for i in df.index:
        smiles = df.loc[i, "smiles"]
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        phthalic_pattern = Chem.MolFromSmarts("c1ccc(C(=O)[OH])c(C(=O)[OH])c1")
        matches_tuple = mol.GetSubstructMatches(phthalic_pattern)
        idx = matches_tuple[0]
        df.loc[i, "benzene_idx"] = idx

    return df


def homolumo(df: pd.DataFrame, logdata: Any, i: int) -> pd.DataFrame:
    homo_idx = logdata.homos[0]
    homo_ev = logdata.moenergies[0][homo_idx]
    lumo_ev = logdata.moenergies[0][homo_idx + 1]
    gap_ev = lumo_ev - homo_ev
    omega = ((lumo_ev + homo_ev) / 2) ** 2 / (lumo_ev - homo_ev)
    df.loc[i, "homo_ev"] = homo_ev
    df.loc[i, "lumo_ev"] = lumo_ev
    df.loc[i, "gap_ev"] = gap_ev
    df.loc[i, "omega"] = omega
    return df


def delta_g(df: pd.DataFrame, logdata: Any, logdata_ion: Any, i: int) -> pd.DataFrame:
    g_natural = logdata.freeenergy
    g_ion = logdata_ion.freeenergy
    delta_g_hartree = g_ion - g_natural
    df.loc[i, "delta_g_hartree"] = delta_g_hartree
    return df


def dipole_moment(df: pd.DataFrame, logdata: Any, i: int) -> pd.DataFrame:
    vector = logdata.moments[1]
    scalar = np.linalg.norm(vector)
    df.loc[i, "dipole_moment_debye"] = scalar
    return df


def molecular_volume(df: pd.DataFrame, logdata: Any, i: int) -> pd.DataFrame:
    atom_nums = logdata.atomnos
    coords = logdata.atomcoords[-1]
    geometory = ""
    geometory += f"{len(atom_nums)}\n"
    geometory += f"\n"
    for atom_num, (x, y, z) in zip(atom_nums, coords):
        geometory += f"{atom[atom_num]} {float(x):.8f} {float(y):.8f} {float(z):.8f}\n"
    mol = Chem.MolFromXYZBlock(geometory)
    volume = AllChem.ComputeMolVolume(mol)
    df.loc[i, "molecular_volume_A3"] = volume
    return df


def get_final_nbo_charges(logfile_path) -> list[float]:
    # NBO chargeはcclibを用いるとなぜか構造最適化前の
    # 値を取得してしまうので逆から探索する
    with open(logfile_path, "r") as f:
        lines = f.readlines()

    start_line = -1
    for i in range(len(lines) - 1, -1, -1):
        if "Summary of Natural Population Analysis" in lines[i]:
            start_line = i
            break

    if start_line == -1:
        return None

    charges = []
    for line in lines[start_line + 6 :]:
        if "---" in line or "Total" in line:
            break
        parts = line.split()
        if len(parts) >= 3:
            charges.append(float(parts[2]))
    return charges


def polar(df: pd.DataFrame, logdata: Any, i: int) -> pd.DataFrame:
    tensor = logdata.polarizabilities[-1]

    # 等方性分極率 (alpha_iso) = (XX + YY + ZZ) / 3
    # numpyのtrace関数を使うと対角成分の和 (XX+YY+ZZ) が簡単に計算できる
    alpha_iso = np.trace(tensor) / 3
    df.loc[i, "polar"] = alpha_iso
    return df


def sasa(df: pd.DataFrame, logdata: Any, i: int) -> pd.DataFrame:
    atom_nums = logdata.atomnos
    coords = logdata.atomcoords[-1]
    geometory = ""
    geometory += f"{len(atom_nums)}\n"
    geometory += f"\n"
    pt = Chem.GetPeriodicTable()

    coords_flat = []
    for coord in coords:
        coords_flat.extend(coord)

    radii = [pt.GetRvdw(int(z)) for z in atom_nums]

    result = freesasa.calcCoord(coords_flat, radii)
    atom_sasa_list = [result.atomArea(j) for j in range(len(atom_nums))]

    matches = df.loc[i, "cooh"]
    target = set()
    for match in matches:
        for idx in match:
            target.add(idx)

    partial_sasa = sum(atom_sasa_list[j] for j in target)
    df.loc[i, "sasa"] = partial_sasa
    return df


def get_data(df: pd.DataFrame, path: Path, delta_g: bool = False) -> pd.DataFrame:
    path_ion = "in/" + path + "_ion/logfile"
    path = rf"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\data\in\{path}"
    for i in range(len(df)):
        cas = df.loc[i, "cas"]
        h1 = df.loc[i, "cooh"][1][4]
        h2 = df.loc[i, "cooh"][0][4]
        o_double1 = df.loc[i, "cooh"][1][2]
        o_double2 = df.loc[i, "cooh"][0][2]
        filepath = Path(f"{path}/sub_{cas}.log")
        if not filepath.exists():
            continue
        logdata = cclib.io.ccread(str(filepath))

        # delta_g
        if delta_g:
            filepath_ion = Path(f"{path_ion}/sub_{cas}_ion.log")
            if not filepath_ion.exists():
                continue
            data_ion = cclib.io.ccread(str(filepath_ion))
            df = delta_g(df, logdata, data_ion, i)

        # homolumo
        df = homolumo(df, logdata, i)

        # dipole moment
        df = dipole_moment(df, logdata, i)

        # molecular volume
        df = molecular_volume(df, logdata, i)

        nbo_charges = get_final_nbo_charges(f"{path}/sub_{cas}.log")
        # nbo charge(coo"h")
        h_nbo1 = nbo_charges[h1]
        h_nbo2 = nbo_charges[h2]
        h_nbo = max(h_nbo1, h_nbo2)
        df.loc[i, "h_nbo_charge"] = h_nbo

        # nbo charge(c"o"oh)
        o_nbo1 = nbo_charges[o_double1]
        o_nbo2 = nbo_charges[o_double2]
        o_nbo = max(o_nbo1, o_nbo2)
        df.loc[i, "o_nbo_charge"] = o_nbo

        # polarizabilities
        df = polar(df, logdata, i)

        # sasa
        df = sasa(df, logdata, i)

    return df


def sub_idx(df):
    df["3and6"] = 0
    df["4and5"] = 0
    for i in range(len(df)):
        smiles = df.loc[i, "smiles"]
        c_acid1 = df.loc[i, "cooh"][0][0]
        c_acid2 = df.loc[i, "cooh"][1][0]
        c_benzene1 = df.loc[i, "cooh"][0][1]
        c_benzene2 = df.loc[i, "cooh"][1][1]
        benzene_tuple = df.loc[i, "benzene"]
        benzene_list = [c_acid2, c_acid1, c_benzene2, c_benzene1]

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        while len(benzene_list) != 8:
            atom = mol.GetAtomWithIdx(benzene_list[-1])
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            for neighbor in neighbors:
                if neighbor not in benzene_list and neighbor in benzene_tuple:
                    benzene_list.append(neighbor)

        idx = ["3and6", "4and5", "4and5", "3and6"]
        for j in range(4, 8):
            atom = mol.GetAtomWithIdx(benzene_list[j])
            neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
            if "H" in neighbors:
                df.loc[i, f"{idx[j-4]}"] += 0
            else:
                df.loc[i, f"{idx[j-4]}"] += 1
    return df


def get_descriptors(df):
    for i in range(len(df)):
        smiles = df.loc[i, "smiles"]
        mol = Chem.MolFromSmiles(smiles)

        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        df.loc[i, "logp"] = logp
        df.loc[i, "hbd"] = hbd
        df.loc[i, "hba"] = hba

    return df
