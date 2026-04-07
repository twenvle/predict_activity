import numpy as np
import pandas as pd
import cclib
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors


def homolumo(df, path):
    path = "in/" + path + "/logfile"
    for i in range(len(df)):
        cas = df.loc[i, "cas"]
        filepath = Path(f"{path}/sub_{cas}.log")
        if not filepath.exists():
            continue
        data = cclib.io.ccread(str(filepath))
        homo_idx = data.homos[0]
        homo_ev = data.moenergies[0][homo_idx]
        lumo_ev = data.moenergies[0][homo_idx + 1]
        gap_ev = lumo_ev - homo_ev
        df.loc[i, "homo_ev"] = homo_ev
        df.loc[i, "lumo_ev"] = lumo_ev
        df.loc[i, "gap_ev"] = gap_ev
    return df


def delta_g(df, path):
    path_ion = "in/" + path + "_ion/logfile"
    path = "in/" + path + "/logfile"
    for i in range(len(df)):
        cas = df.loc[i, "cas"]
        filepath = Path(f"{path}/sub_{cas}.log")
        filepath_ion = Path(f"{path_ion}/sub_{cas}_ion.log")
        if not filepath.exists() or not filepath_ion.exists():
            continue
        data = cclib.io.ccread(str(filepath))
        data_ion = cclib.io.ccread(str(filepath_ion))
        g_natural = data.freeenergy
        g_ion = data_ion.freeenergy
        delta_g_hartree = g_ion - g_natural
        df.loc[i, "delta_g_hartree"] = delta_g_hartree
    return df


def dipole_moment(df, path):
    path = "in/" + path + "/logfile"
    for i in range(len(df)):
        cas = df.loc[i, "cas"]
        filepath = Path(f"{path}/sub_{cas}.log")
        if not filepath.exists():
            continue
        data = cclib.io.ccread(str(filepath))
        vector = data.moments[1]
        scalar = np.linalg.norm(vector)
        df.loc[i, "dipole_moment_debye"] = scalar
    return df


atom = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


def molecular_volume(df, path):
    path = "in/" + path + "/logfile"
    for i in range(len(df)):
        cas = df.loc[i, "cas"]
        filepath = Path(f"{path}/sub_{cas}.log")
        if not filepath.exists():
            continue
        data = cclib.io.ccread(str(filepath))
        atom_nums = data.atomnos
        coords = data.atomcoords[-1]
        geometory = ""
        geometory += f"{len(atom_nums)}\n"
        geometory += f"\n"
        for atom_num, (x, y, z) in zip(atom_nums, coords):
            geometory += (
                f"{atom[atom_num]} {float(x):.8f} {float(y):.8f} {float(z):.8f}\n"
            )
        mol = Chem.MolFromXYZBlock(geometory)
        volume = AllChem.ComputeMolVolume(mol)
        df.loc[i, "molecular_volume_A3"] = volume
    return df


def nbo_charge(df, path):
    path = "in/" + path + "/logfile"
    for i in range(len(df)):
        cas = df.loc[i, "cas"]
        h1 = df.loc[i, "h1"]
        h2 = df.loc[i, "h2"]
        filepath = Path(f"{path}/sub_{cas}.log")
        if not filepath.exists():
            continue
        data = cclib.io.ccread(str(filepath))
        nbo_charges = data.atomcharges["natural"]
        nbo1 = nbo_charges[h1]
        nbo2 = nbo_charges[h2]
        nbo = max(nbo1, nbo2)
        df.loc[i, "nbo_charge"] = nbo
    return df


def get_data(df, path):
    path_ion = "in/" + path + "_ion/logfile"
    path = "in/" + path + "logfile"
    for i in range(len(df)):
        cas = df.loc[i, "cas"]
        h1 = df.loc[i, "h1"]
        h2 = df.loc[i, "h2"]
        filepath = Path(f"{path}/sub_{cas}.log")
        filepath_ion = Path(f"{path_ion}/sub_{cas}_ion.log")
        if not filepath.exists() or not filepath_ion.exists():
            continue
        data = cclib.io.ccread(str(filepath))
        data_ion = cclib.io.ccread(str(filepath_ion))

        # delta_g
        g_natural = data.freeenergy
        g_ion = data_ion.freeenergy
        delta_g_hartree = g_ion - g_natural
        df.loc[i, "delta_g_hartree"] = delta_g_hartree

        # homolumo
        homo_idx = data.homos[0]
        homo_ev = data.moenergies[0][homo_idx]
        lumo_ev = data.moenergies[0][homo_idx + 1]
        gap_ev = lumo_ev - homo_ev
        df.loc[i, "homo_ev"] = homo_ev
        df.loc[i, "lumo_ev"] = lumo_ev
        df.loc[i, "gap_ev"] = gap_ev

        # dipole moment
        vector = data.moments[1]
        scalar = np.linalg.norm(vector)
        df.loc[i, "dipole_moment_debye"] = scalar

        # molecular volume
        atom_nums = data.atomnos
        coords = data.atomcoords[-1]
        geometory = ""
        geometory += f"{len(atom_nums)}\n"
        geometory += f"\n"
        for atom_num, (x, y, z) in zip(atom_nums, coords):
            geometory += (
                f"{atom[atom_num]} {float(x):.8f} {float(y):.8f} {float(z):.8f}\n"
            )
        mol = Chem.MolFromXYZBlock(geometory)
        volume = AllChem.ComputeMolVolume(mol)
        df.loc[i, "molecular_volume_A3"] = volume

        # nbo charge
        nbo_charges = data.atomcharges["natural"]
        nbo1 = nbo_charges[h1]
        nbo2 = nbo_charges[h2]
        nbo = max(nbo1, nbo2)
        df.loc[i, "nbo_charge"] = nbo
    return df
