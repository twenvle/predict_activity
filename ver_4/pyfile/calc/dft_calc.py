import numpy as np
import pandas as pd
import cclib
import freesasa
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors


def homolumo(df, data, i):
    homo_idx = data.homos[0]
    homo_ev = data.moenergies[0][homo_idx]
    lumo_ev = data.moenergies[0][homo_idx + 1]
    gap_ev = lumo_ev - homo_ev
    omega = ((lumo_ev + homo_ev) / 2) ** 2 / (lumo_ev - homo_ev)
    df.loc[i, "homo_ev"] = homo_ev
    df.loc[i, "lumo_ev"] = lumo_ev
    df.loc[i, "gap_ev"] = gap_ev
    df.loc[i, "omega"] = omega
    return df


def delta_g(df, data, data_ion, i):
    g_natural = data.freeenergy
    g_ion = data_ion.freeenergy
    delta_g_hartree = g_ion - g_natural
    df.loc[i, "delta_g_hartree"] = delta_g_hartree
    return df


def dipole_moment(df, data, i):
    vector = data.moments[1]
    scalar = np.linalg.norm(vector)
    df.loc[i, "dipole_moment_debye"] = scalar
    return df


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


def molecular_volume(df, data, i):
    atom_nums = data.atomnos
    coords = data.atomcoords[-1]
    geometory = ""
    geometory += f"{len(atom_nums)}\n"
    geometory += f"\n"
    for atom_num, (x, y, z) in zip(atom_nums, coords):
        geometory += f"{atom[atom_num]} {float(x):.8f} {float(y):.8f} {float(z):.8f}\n"
    mol = Chem.MolFromXYZBlock(geometory)
    volume = AllChem.ComputeMolVolume(mol)
    df.loc[i, "molecular_volume_A3"] = volume
    return df


def get_final_nbo_charges(logfile_path):
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


def polar(df, data, i):
    tensor = data.polarizabilities[-1]

    # 等方性分極率 (alpha_iso) = (XX + YY + ZZ) / 3
    # numpyのtrace関数を使うと対角成分の和 (XX+YY+ZZ) が簡単に計算できる
    alpha_iso = np.trace(tensor) / 3
    df.loc[i, "polar"] = alpha_iso
    return df


def sasa(df, data, i):
    atom_nums = data.atomnos
    coords = data.atomcoords[-1]
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


def get_data(df, path, delta_g=False):
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
        data = cclib.io.ccread(str(filepath))

        # delta_g
        if delta_g:
            filepath_ion = Path(f"{path_ion}/sub_{cas}_ion.log")
            if not filepath_ion.exists():
                continue
            data_ion = cclib.io.ccread(str(filepath_ion))
            df = delta_g(df, data, data_ion, i)

        # homolumo
        df = homolumo(df, data, i)

        # dipole moment
        df = dipole_moment(df, data, i)

        # molecular volume
        df = molecular_volume(df, data, i)

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
        df = polar(df, data, i)

        # sasa
        df = sasa(df, data, i)

    return df
