from __future__ import annotations

from pathlib import Path

import cclib
import freesasa
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski

ATOM = {
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


def acid_idx(row: pd.Series) -> list[list[int]]:
    """Identify the indices of acidic functional groups in the molecule represented by the SMILES string."""
    # Screening is already done in the preprocess.py

    smiles = row["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    phthalic_pattern = Chem.MolFromSmarts("c1ccc(C(=O)[OH])c(C(=O)[OH])c1")
    matches_tuple = mol.GetSubstructMatches(phthalic_pattern)

    oh_list = []
    cooh_list = []
    c = o_double = o_single = h = None
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
    if None in (c, o_double, o_single, h):
        raise ValueError("Acidic functional group not found in the molecule.")
    return cooh_list


def phthalicacid_idx(row: pd.Series) -> tuple:
    """Identify the indices of the phthalic acid moiety in the molecule represented by the SMILES string."""
    smiles = row["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    phthalic_pattern = Chem.MolFromSmarts("c1ccc(C(=O)[OH])c(C(=O)[OH])c1")
    matches_tuple = mol.GetSubstructMatches(phthalic_pattern)
    phthalicacid_tuple = matches_tuple[0]
    return phthalicacid_tuple


def homolumo(logdata: cclib.io.ccread) -> dict:
    homo_idx = logdata.homos[0]
    homo_ev = logdata.moenergies[0][homo_idx]
    lumo_ev = logdata.moenergies[0][homo_idx + 1]
    gap_ev = lumo_ev - homo_ev
    omega = ((lumo_ev + homo_ev) / 2) ** 2 / (lumo_ev - homo_ev)
    return {
        "homo_ev": homo_ev,
        "lumo_ev": lumo_ev,
        "gap_ev": gap_ev,
        "omega": omega,
    }


def delta_g(logdata: cclib.io.ccread, logdata_ion: cclib.io.ccread) -> dict:
    g_natural = logdata.freeenergy
    g_ion = logdata_ion.freeenergy
    delta_g_hartree = g_ion - g_natural
    return {"delta_g_hartree": delta_g_hartree}


def dipole_moment(logdata: cclib.io.ccread) -> dict:
    vector = logdata.moments[1]
    scalar = np.linalg.norm(vector)
    return {"dipole_moment_debye": scalar}


def molecular_volume(logdata: cclib.io.ccread) -> dict:
    atom_nums = logdata.atomnos
    coords = logdata.atomcoords[-1]
    geometry = ""
    geometry += f"{len(atom_nums)}\n"
    geometry += f"\n"
    for atom_num, (x, y, z) in zip(atom_nums, coords):
        geometry += f"{ATOM[atom_num]} {float(x):.8f} {float(y):.8f} {float(z):.8f}\n"
    mol = Chem.MolFromXYZBlock(geometry)
    volume = AllChem.ComputeMolVolume(mol)
    return {"molecular_volume_A3": volume}


def get_final_nbo_charges(logfile_path: Path, cooh_list: list[list[int]]) -> dict:
    # NBO chargeはcclibを用いると構造最適化前の
    # 値を取得してしまうので逆から探索する
    h1 = cooh_list[1][4]
    h2 = cooh_list[0][4]
    o_double1 = cooh_list[1][2]
    o_double2 = cooh_list[0][2]

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

    h_nbo1 = charges[h1]
    h_nbo2 = charges[h2]
    h_nbo = max(h_nbo1, h_nbo2)

    o_nbo1 = charges[o_double1]
    o_nbo2 = charges[o_double2]
    o_nbo = max(o_nbo1, o_nbo2)

    return {"h_nbo_charge": h_nbo, "o_nbo_charge": o_nbo}


def polar(logdata: cclib.io.ccread) -> dict:
    tensor = logdata.polarizabilities[-1]

    # 等方性分極率 (alpha_iso) = (XX + YY + ZZ) / 3
    # numpyのtrace関数を使うと対角成分の和 (XX+YY+ZZ) が簡単に計算できる
    alpha_iso = np.trace(tensor) / 3
    return {"polar": alpha_iso}


def sasa(logdata: cclib.io.ccread, cooh_list: list[list[int]]) -> dict:
    atom_nums = logdata.atomnos
    coords = logdata.atomcoords[-1]
    geometry = ""
    geometry += f"{len(atom_nums)}\n"
    geometry += f"\n"
    pt = Chem.GetPeriodicTable()

    coords_flat = []
    for coord in coords:
        coords_flat.extend(coord)

    radii = [pt.GetRvdw(int(z)) for z in atom_nums]

    result = freesasa.calcCoord(coords_flat, radii)
    atom_sasa_list = [result.atomArea(j) for j in range(len(atom_nums))]

    matches = cooh_list
    target = set()
    for match in matches:
        for idx in match:
            target.add(idx)

    partial_sasa = sum(atom_sasa_list[j] for j in target)
    return {"sasa": partial_sasa}


def sub_idx(smiles: str, cooh_list: list[list[int]], phthalicacid_tuple: tuple) -> dict:
    c_acid1 = cooh_list[0][0]
    c_acid2 = cooh_list[1][0]
    c_benzene1 = cooh_list[0][1]
    c_benzene2 = cooh_list[1][1]
    benzene_tuple = phthalicacid_tuple
    benzene_list = [c_acid2, c_acid1, c_benzene2, c_benzene1]

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    while len(benzene_list) != 8:
        atom = mol.GetAtomWithIdx(benzene_list[-1])
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        for neighbor in neighbors:
            if neighbor not in benzene_list and neighbor in benzene_tuple:
                benzene_list.append(neighbor)

    three_six = 0
    four_five = 0
    for j in range(4, 8):
        atom = mol.GetAtomWithIdx(benzene_list[j])
        neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
        if not "H" in neighbors:
            if j == 4 or j == 7:
                three_six += 1
            elif j == 5 or j == 6:
                four_five += 1
    return {"3and6": three_six, "4and5": four_five}


def get_rdkit_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)

    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    return {"logp": logp, "hbd": hbd, "hba": hba}


def calc_features(
    cas: str,
    smiles: str,
    logfile_path: Path,
    logdata: cclib.io.ccread,
    delta_g_bool: bool,
    cooh_list: list[list[int]],
    phthalicacid_tuple: tuple,
) -> dict:
    result = {}
    result.update(sasa(logdata, cooh_list))
    result.update(polar(logdata))
    result.update(get_final_nbo_charges(logfile_path, cooh_list))
    result.update(molecular_volume(logdata))
    result.update(dipole_moment(logdata))
    if delta_g_bool:
        logdata_ion = cclib.io.ccread(logfile_path.parent / f"sub_{cas}_ion.log")
        result.update(delta_g(logdata, logdata_ion))
    result.update(homolumo(logdata))
    result.update(sub_idx(smiles, cooh_list, phthalicacid_tuple))
    result.update(get_rdkit_descriptors(smiles))
    return result


def get_data(
    df: pd.DataFrame, name: str, delta_g_bool: bool = False, known: bool = True
) -> pd.DataFrame:
    base_path = Path(__file__).resolve().parent.parent.parent / "data/in/"
    result = []
    valid_indices = []
    invalid_indices = []
    for idx, row in df.iterrows():
        cas = row["cas"]
        smiles = row["smiles"]
        cooh_list = acid_idx(row)
        phthalicacid_tuple = phthalicacid_idx(row)
        path_pattern = (
            f"known/sub_{cas}.log" if known else f"unknown/*/logfile/sub_{cas}.log"
        )
        logfile_path = list(base_path.glob(path_pattern))
        if len(logfile_path) == 0:
            invalid_indices.append(idx)
            continue
        if not logfile_path[0].is_file():
            invalid_indices.append(idx)
            continue
        logfile_path = logfile_path[0]
        logdata = cclib.io.ccread(str(logfile_path))
        result.append(
            calc_features(
                cas,
                smiles,
                logfile_path,
                logdata,
                delta_g_bool,
                cooh_list,
                phthalicacid_tuple,
            )
        )
        valid_indices.append(idx)

    df_valid = df.loc[valid_indices].reset_index(drop=True)
    df_invalid = df.loc[invalid_indices].reset_index(drop=True)
    features_df = pd.DataFrame(result).reset_index(drop=True)
    df_valid = pd.concat([df_valid.reset_index(drop=True), features_df], axis=1)

    if known:
        df_valid.to_csv(f"../data/known/features/{name}.csv", index=False)
        df_invalid.to_csv(f"../data/known/features/{name}_invalid.csv", index=False)
    elif not known:
        df_valid.to_csv(f"../data/unknown/features/{name}.csv", index=False)
        df_invalid.to_csv(f"../data/unknown/features/{name}_invalid.csv", index=False)

    return df_valid
