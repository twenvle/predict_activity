from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

input_gjf_path = Path(__file__).parent.parent.parent / "out/dft_file/format/base.gjf"
input_sh_path = Path(__file__).parent.parent.parent / "out/dft_file/format/base.sh"

mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
AllChem.UFFOptimizeMolecule(mol)
conf = mol.GetConformer()
coordinates = []
for atom in mol.GetAtoms():
    pos = conf.GetAtomPosition(atom.GetIdx())
    coordinates.append([pos.x, pos.y, pos.z])
coordinates = np.array(coordinates)

output_gjf_path = Path(__file__).parent.parent.parent / "out/dft_file/test.gjf"
output_sh_path = Path(__file__).parent.parent.parent / "out/dft_file/test.sh"

for file_type in ["gjf", "sh"]:
    if file_type == "gjf":
        input_path = input_gjf_path
        output_path = output_gjf_path
    elif file_type == "sh":
        input_path = input_sh_path
        output_path = output_sh_path

    with input_path.open("r", newline="\n") as input_file, output_path.open(
        "w", newline="\n"
    ) as output_file:
        for line in input_file:
            new_line = line.replace("---change---", "sub_" + "test")
            output_file.write(new_line)

        if file_type == "gjf":
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                idx = atom.GetIdx()
                x, y, z = coordinates[idx]
                output_file.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")
