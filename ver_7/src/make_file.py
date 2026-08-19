from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

input_gjf_path = Path(__file__).parent.parent / "out/dft_file/format/base.gjf"
input_sh_path = Path(__file__).parent.parent / "out/dft_file/format/base.sh"


def gjf_and_sh(
    df: pd.DataFrame, dir_name: str = "", file_range: list[int, int] | None = None
) -> None:
    """Generate .gjf and .sh files for Gaussian calculations."""
    if dir_name:
        output_dir = Path(__file__).parent.parent / f"out/dft_file/{dir_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
    if file_range:
        first = file_range[0]
        last = file_range[1]
        output_dir = output_dir / f"{first}_{last}"
        output_dir.mkdir(parents=True, exist_ok=True)
        df = df.iloc[first - 1 : last]
    all_sh = ""

    for i in df.index:
        cas = df.loc[i, "cas"]
        smiles = df.loc[i, "smiles"]
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        if AllChem.EmbedMolecule(mol) == -1:
            return None
        AllChem.UFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        coordinates = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coordinates.append([pos.x, pos.y, pos.z])
        coordinates = np.array(coordinates)

        output_gjf_path = output_dir / f"sub_{cas}.gjf"
        output_sh_path = output_dir / f"sub_{cas}.sh"

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
                    new_line = line.replace("---change---", "sub_" + cas)
                    output_file.write(new_line)

                if file_type == "gjf":
                    for atom in mol.GetAtoms():
                        symbol = atom.GetSymbol()
                        idx = atom.GetIdx()
                        x, y, z = coordinates[idx]
                        output_file.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")
                    output_file.write("\n")
                    output_file.write("$nbo bndidx $end\n")

        if i != df.index[-1]:
            all_sh += f"qsub -g tga-ynabae sub_{cas}.sh\n"
        elif i == df.index[-1]:
            all_sh += f"qsub -g tga-ynabae sub_{cas}.sh"

        with open(output_dir / "all.sh", "w", newline="\n") as f:
            f.write(all_sh)
